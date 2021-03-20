import torch
from torch import nn, optim
from utils import load_data, ClassType
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from transforms import aug_transform, clean_transform
import json, os

sns.set()


def train(model, device, train_loader, optimizer, weights, pos_ind=0):
    """
	Train model for one epoch.
	Returns (average loss per batch, accuracy, sensitivity).
	"""
    model.train()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())
    loss_sum = 0
    correct = 0
    pos_total = 0
    pos_correct = 0
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        pred = model(data)
        correct += sum((torch.argmax(pred, dim=1) - target) == 0).item()
        pos_total += sum(target == pos_ind).item()
        pos = sum(torch.argmax(pred, dim=1)[target == pos_ind] == pos_ind)
        pos_correct += pos if type(pos) == int else pos.item()
        loss = loss_fn(pred, target)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    accuracy = correct / len(train_loader.dataset)
    sensitivity = pos_correct / pos_total if pos_total else 0
    return loss_sum / len(train_loader), accuracy, sensitivity


def test(model, device, test_loader, weights, pos_ind=0):
    """
	Test model.
	Returns (average loss per batch, accuracy, sensitivity).
	"""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())
    loss_sum = 0
    correct = 0
    pos_total = 0
    pos_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            pred = model(data)
            correct += sum((torch.argmax(pred, dim=1) - target) == 0).item()
            pos_total += sum(target == pos_ind).item()
            pos = sum(torch.argmax(pred, dim=1)[target == pos_ind] == pos_ind)
            pos_correct += pos if type(pos) == int else pos.item()
            loss_sum += loss_fn(pred, target).item()

    sensitivity = pos_correct / pos_total if pos_total else 0
    return loss_sum / len(test_loader), correct / len(test_loader.dataset), sensitivity


def get_progress(arch_name):
    """
	Obtain the progress (test/train loss and accuracy) curves for a trained model.
	"""
    with open("progress.json") as f:
        data = json.load(f)
    if arch_name not in data:
        print(
            f"'{arch_name}' is not a valid name. Available names are: {', '.join(data.keys())}"
        )
        return
    else:
        return data[arch_name]


def plot_progress(arch_name):
    """
	Plot the progress (test/train loss and accuracy) curves for a trained model.
	"""
    progress = get_progress(arch_name)
    if not progress:
        return
    test_loss = progress["test_loss"]
    test_acc = progress["test_acc"]
    train_loss = progress["train_loss"]
    train_acc = progress["train_acc"]
    x = np.arange(len(test_loss)) + 1
    plt.plot(x, test_loss, label="Test Loss")
    plt.plot(x, train_loss, label="Train Loss")
    plt.plot(x, test_acc, label="Test Accuracy")
    plt.plot(x, train_acc, label="Train Accuracy")
    plt.title(arch_name)
    plt.legend()
    plt.show()


def append_progress(arch_name, progress):
    if os.path.isfile("progress.json"):
        with open("progress.json") as f:
            data = json.load(f)
    else:
        data = {}
    with open("progress.json", "w+") as f:
        if arch_name not in data:
            data[arch_name] = {key: [] for key in progress.keys()}
        for key in progress.keys():
            data[arch_name][key] += progress[key]
        json.dump(data, f, indent="\t")


def clear_progress(arch_name):
    with open("progress.json", "r") as f:
        data = json.load(f)
    with open("progress.json", "w+") as f:
        del data[arch_name]
        json.dump(data, f, indent="\t")


def load_model(arch, load, device):
    model = arch.to(device)
    if load:
        model.load_state_dict(torch.load(load))
    return model


def print_table(tr_loss, tr_acc, tr_sens, te_loss, te_acc, te_sens):
    na = "    N/A"
    tr_acc = f"{round(100 * tr_acc, 3):06.3f}%" if tr_acc != None else na
    te_acc = f"{round(100 * te_acc, 3):06.3f}%" if te_acc != None else na
    tr_loss = f"{round(tr_loss, 5):07.5f}" if tr_loss != None else na
    te_loss = f"{round(te_loss, 5):07.5f}" if te_loss != None else na
    tr_sens = f"{round(100 * tr_sens, 3):06.3f}%" if tr_sens != None else na
    te_sens = f"{round(100 * te_sens, 3):06.3f}%" if te_sens != None else na
    row = f"+{'-' * 8}+{'-' * 8}+{'-' * 8}+{'-' * 8}+"
    print(row)
    print(f"|{' ' * 8}|    Acc.|    Loss|   Sens.|")
    print(row)
    print(f"|   Train| {tr_acc}| {tr_loss}| {tr_sens}|")
    print(f"|    Test| {te_acc}| {te_loss}| {te_sens}|")
    print(row)


def run(
    arch,
    class_type,
    N_EPOCH=0,
    L_RATE=5e-5,
    W_DECAY=0.01,
    L_WEIGHTS=None,
    pos_index=0,
    name=None,
    prop=None,
    test_with="test",
    load=None,
    save=None,
    use_scheduler=False,
):
    """
	Train or evaluate a model.
	arch: class of model architecture
	class_type: 'inf' for ClassType.NORMAL_INFECTED, 'cov' for ClassType.COVID_NONCOVID, and 'all' for ClassType.THREE_CLASS
	N_EPOCH: number of epochs to train for.
	L_RATE: learning rate parameter.
	W_DECAY: regularization parameter.
	L_WEIGHTS: cross-entropy loss weights.
    pos_index: index of positive case.
    name: name of the model to record in progress.
	prop: proportion of training dataset to utilize (for bagging).
	test_with: which test dataset to use. one of 'val' or 'test'.
	load: location of model file to load, if any.
	save: location of model file to save to, if any.
	"""
    class_type = class_type.lower()
    if class_type == "inf":
        class_type = ClassType.NORMAL_INFECTED
        if not L_WEIGHTS:
            L_WEIGHTS = [1 / 2, 1 / 2]
    elif class_type == "cov":
        class_type = ClassType.COVID_NONCOVID
        if not L_WEIGHTS:
            L_WEIGHTS = [1 / 2, 1 / 2]
    elif class_type == "all":
        class_type = ClassType.THREE_CLASS
        if not L_WEIGHTS:
            L_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
    else:
        raise ValueError(
            "class_type should be 'inf' for normal/infected, 'cov' for covid/non-covid, or 'all' for three classes"
        )

    if not name:
        name = arch.__class__.__name__

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = load_data(
        kind="train",
        class_type=class_type,
        transform=aug_transform,
        prop=prop,
        shuffle=True,
        batch_size=32,
    )
    test_loader = load_data(
        kind=test_with.lower(),
        class_type=class_type,
        transform=clean_transform,
        shuffle=True,
        batch_size=32,
    )

    model = load_model(arch, load, device)
    if N_EPOCH:
        optimizer = optim.AdamW(model.parameters(), lr=L_RATE, weight_decay=W_DECAY)
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    if N_EPOCH:
        print(f"Beginning training of {name} with")
        print(f"    N_EPOCH  {N_EPOCH}")
        print(f"     L_RATE  {L_RATE}")
        print(f"    W_DECAY  {W_DECAY}\n")
    else:
        print(f"Evaluating {name}\n")

    progress = []
    init_loss, init_acc, init_sens = test(model, device, test_loader, L_WEIGHTS)
    print_table(None, None, None, init_loss, init_acc, init_sens)

    best_metric = 0
    for epoch in range(N_EPOCH):
        print(f"\n--- Beginning Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}")
        tr_loss, tr_acc, tr_sens = train(
            model, device, train_loader, optimizer, L_WEIGHTS
        )
        print(f"--- Completed Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}\n")
        te_loss, te_acc, te_sens = test(model, device, test_loader, L_WEIGHTS)
        print_table(tr_loss, tr_acc, tr_sens, te_loss, te_acc, te_sens)
        progress.append((te_loss, te_acc, te_sens, tr_loss, tr_acc, tr_sens))
        if use_scheduler:
            scheduler.step(te_loss)
        if save:
            metric = (te_acc + te_sens) + abs(te_sens - te_acc) / 2
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), save)
                print(f"Model saved in '{save}' @ {str(datetime.now())}")

    if N_EPOCH:
        test_loss, test_acc, test_sens, train_loss, train_acc, train_sens = zip(
            *progress
        )
        append_progress(
            name,
            {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_sens": test_sens,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_sens": train_sens,
            },
        )
        print(f"\nCompleted training of {name} with")
        print(f"    N_EPOCH  {N_EPOCH}")
        print(f"     L_RATE  {L_RATE}")
        print(f"    W_DECAY  {W_DECAY}\n")

