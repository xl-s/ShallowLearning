import torch
from torch import nn, optim
from utils import load_data, ClassType
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transforms import aug_transform, clean_transform
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

class Conv2dPad(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dPad, self).__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 5, 3),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(),

            nn.Conv2d(5, 10, 3),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),

            nn.Conv2d(10, 15, 3),
            nn.BatchNorm2d(15),
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(27, 27)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(10935, 512, bias=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        for layer in self.convolutional:
            x = layer(x)

        for layer in self.classifier:
            x = layer(x)
        return x

    def predict(self, inp, transform=clean_transform):
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        if transform: inp = transform(inp)
        inp = inp.to(device)
        out = self.forward(inp)
        return nn.functional.softmax(out, dim=1)[0]

    def classify(self, loc, top=5):
        prediction = self.predict(loc)
        _, indices = torch.sort(prediction, descending=True)
        return [(labels[ind.item()], prediction[ind.item()].item()) for ind in indices[:top]]


def train(model, device, train_loader, optimizer, loss_wt=None):
    """
	Train model for one epoch.
	Returns (average loss per batch, accuracy).
	"""
    model.train()
    # loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([0.673, 1.945]).cuda())
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([loss_wt, 1-loss_wt]).cuda())

    # loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0
    correct = 0

    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0


    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        pred = model(data)
        pred_class = torch.argmax(pred, dim=1)

        correct += sum((pred_class - target) == 0).item()
        false_positives += sum((target - pred_class) == 1).item()
        false_negatives += sum((pred_class - target)==1).item()
        true_positives += sum(pred_class + target == 0).item()
        true_negatives += sum(pred_class + target == 2).item()
        loss = loss_fn(pred, target)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    accuracy = correct / len(train_loader.dataset)
    sensitivity = true_positives/(true_positives + false_negatives)
    specificity = true_negatives/(true_negatives + false_positives)
    f1 = 2*true_positives/(2*true_positives + false_positives + false_negatives)

    metrics = {"loss": loss_sum/len(train_loader),
               "accuracy": accuracy,
               "sensitivity": sensitivity,
               "specificity": specificity,
               "f1-score": f1}
    return metrics


def test(model, device, test_loader, loss_wt=None, plot=False):
    """
	Test model.
	Returns (average loss per batch, accuracy).
	"""
    model.eval()

    correct = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    # 0.674, 1.945
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([loss_wt, 1-loss_wt]).cuda())
    # loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            pred = model(data)
            pred_class = torch.argmax(pred, dim=1)
            correct += sum((pred_class - target) == 0).item()
            false_positives += sum((target - pred_class) == 1).item()
            false_negatives += sum((pred_class - target) == 1).item()
            true_positives += sum(pred_class + target == 0).item()
            true_negatives += sum(pred_class + target == 2).item()
            loss_sum += loss_fn(pred, target).item()

    accuracy = correct / len(test_loader.dataset)
    sensitivity = true_positives/(true_positives + false_negatives)
    specificity = true_negatives/(true_negatives + false_positives)
    f1 = 2*true_positives/(2*true_positives + false_positives + false_negatives)

    print(f"Test Accuracy/Loss: {round(100 * accuracy, 3)}%, {round(loss_sum / len(test_loader), 5)}")

    metrics = {"loss": loss_sum/len(test_loader),
               "accuracy": accuracy,
               "sensitivity": sensitivity,
               "specificity": specificity,
               "f1-score": f1}

    return metrics


def load_model(load, device):
    model = CNN().to(device)
    if load: model.load_state_dict(torch.load(load))
    return model


def concat_dict(inp):
    dd = defaultdict(list)
    for d in inp:
        for key, value in d.items():
            dd[key].append(value)
    return dd


def run(N_EPOCH, L_RATE, W_DECAY, class_type, load=None, save=None, loss_wt=None):
    """
	Train or evaluate a model.
	N_EPOCH: number of epochs to train for.
	L_RATE: learning rate parameter.
	W_DECAY: regularization parameter.
	class_type: one of ClassType.NORMAL_INFECTED or ClassType.COVID_NONCOVID.
	load: location of model file to load, if any.
	save: location of model file to save to, if any.
	"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = load_data(kind="train", class_type=class_type, transform=aug_transform, shuffle=True, batch_size=32)
    test_loader = load_data(kind="test", class_type=class_type, transform=clean_transform, shuffle=True, batch_size=32)
    val_loader = load_data(kind="val", class_type=class_type, transform=clean_transform, shuffle=True, batch_size=8)

    model = load_model(load, device)
    optimizer = optim.AdamW(model.parameters(), lr=L_RATE, weight_decay=W_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    # print(model)

    all_train_metrics = []
    all_test_metrics = []
    for epoch in range(N_EPOCH):
        test_metrics = test(model, device, test_loader, loss_wt=loss_wt)
        print(f"--- Beginning Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}")
        train_metrics = train(model, device, train_loader, optimizer, loss_wt=loss_wt)
        print(f"--- Completed Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}")
        print("training metrics", train_metrics)
        print("validation metrics", test_metrics)

        all_train_metrics.append(train_metrics)
        all_test_metrics.append(test_metrics)

        scheduler.step()
        scheduler.get_last_lr()

    # dump all training and test metrics into dataframes
    all_train_metrics = pd.DataFrame(concat_dict(all_train_metrics))
    all_train_metrics.round(5)
    all_test_metrics = pd.DataFrame(concat_dict(all_test_metrics))
    all_test_metrics.round(5)

    val_metrics = test(model, device, val_loader, loss_wt=loss_wt)

    if save:
        torch.save(model.state_dict(), save)

    return all_train_metrics, all_test_metrics, val_metrics


if __name__ == "__main__":
    lr_grid = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    loss_wt = [0.1, 0.3, 0.5, 0.7, 0.9]

    all_train = []
    all_test = []

    for wt in loss_wt:
        print(f"Running loss weight: {wt}")
        train_metrics, test_metrics, val_metrics = run(8, 5e-6, 0.1, ClassType.NORMAL_INFECTED, save="base.model", loss_wt=wt)

        avg_train = train_metrics.tail(4).mean(axis=0).to_frame().transpose()
        avg_test = test_metrics.tail(4).mean(axis=0).to_frame().transpose()

        all_train.append(avg_train)
        all_test.append(avg_test)

    all_train = pd.concat(all_train, ignore_index=True)
    all_test = pd.concat(all_test, ignore_index=True)

    print(all_train)
    print(all_test)

    all_train["Loss Weight"] = loss_wt
    all_test["Loss Weight"] = loss_wt

    all_train.to_json("train_lossWT_more.json")
    all_test.to_json("test_lossWT_more.json")

