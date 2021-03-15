import torch
from torch import nn, optim
from utils import load_data, ClassType
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from transforms import aug_transform, clean_transform, aug_ext_transform, ext_transform
import json, os

sns.set()


def train(model, device, train_loader, optimizer):
	"""
	Train model for one epoch.
	Returns (average loss per batch, accuracy).
	"""
	model.train()
	loss_fn = nn.CrossEntropyLoss()
	loss_sum = 0
	correct = 0
	for data, target in tqdm(train_loader):
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		pred = model(data)
		correct += sum((torch.argmax(pred, dim=1) - target) == 0).item()
		loss = loss_fn(pred, target)
		loss_sum += loss.item()
		loss.backward()
		optimizer.step()

	accuracy = correct / len(train_loader.dataset)
	return loss_sum / len(train_loader), accuracy


def test(model, device, test_loader):
	"""
	Test model.
	Returns (average loss per batch, accuracy).
	"""
	model.eval()
	correct = 0
	loss_fn = nn.CrossEntropyLoss()
	loss_sum = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)

			pred = model(data)
			correct += sum((torch.argmax(pred, dim=1) - target) == 0).item()
			loss_sum += loss_fn(pred, target).item()

	return loss_sum / len(test_loader), correct / len(test_loader.dataset)


def get_progress(arch_name):
	with open("progress.json") as f:
		data = json.load(f)
	if arch_name not in data:
		print(
			f"'{arch_name}' is not a valid name. Available names are: {', '.join(data.keys())}"
		)
		return
	else:
		return data[arch_name]


def get_metrics():
	with open("progress.json") as f:
		data = json.load(f)

	def metrics(values):
		arr = np.array(values["test_acc"])
		return {
			"mean-5": arr[-5:].mean(),
			"std-5": arr[-5:].std(),
			"mean-10": arr[-10:].mean(),
			"std-10": arr[-10:].std(),
			"mean-15": arr[-15:].mean(),
			"std-15": arr[-15:].std(),
		}

	data = pd.DataFrame({k: metrics(v) for k, v in data.items()}).transpose()
	return data


def plot_progress(arch_name):
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


def print_table(tr_loss, tr_acc, te_loss, te_acc):
	na = "    N/A"
	tr_acc = f"{round(100 * tr_acc, 3):.3f}%" if tr_acc else na
	te_acc = f"{round(100 * te_acc, 3):.3f}%" if te_acc else na
	tr_loss = f"{round(tr_loss, 5):.5f}" if tr_loss else na
	te_loss = f"{round(te_loss, 5):.5f}" if te_loss else na
	row = f"+{'-' * 8}+{'-' * 8}+{'-' * 8}+"
	print(row)
	print(f"|{' ' * 8}|    Acc.|    Loss|")
	print(row)
	print(f"|   Train| {tr_acc}| {tr_loss}|")
	print(f"|    Test| {te_acc}| {te_loss}|")
	print(row)


def run(
	arch,
	class_type,
	N_EPOCH=0,
	L_RATE=0.0001,
	W_DECAY=0.01,
	name=None,
	prop=None,
	test_with="val",
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
	prop: proportion of training dataset to utilize (for bagging).
	test_with: which test dataset to use. one of 'val' or 'test'.
	load: location of model file to load, if any.
	save: location of model file to save to, if any.
	"""
	class_type = class_type.lower()
	if class_type == "inf":
		class_type = ClassType.NORMAL_INFECTED
	elif class_type == "cov":
		class_type = ClassType.COVID_NONCOVID
	elif class_type == "all":
		class_type = ClassType.THREE_CLASS
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
	init_loss, init_acc = test(model, device, test_loader)
	print_table(None, None, init_loss, init_acc)

	best_acc = 0
	for epoch in range(N_EPOCH):
		print(f"\n--- Beginning Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}")
		tr_loss, tr_acc = train(model, device, train_loader, optimizer)
		print(f"--- Completed Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}\n")
		te_loss, te_acc = test(model, device, test_loader)
		print_table(tr_loss, tr_acc, te_loss, te_acc)
		progress.append((te_loss, te_acc, tr_loss, tr_acc))
		if use_scheduler:
			scheduler.step(te_loss)
		if save:
			if te_acc > best_acc:
				best_acc = te_acc
				torch.save(model.state_dict(), save)
				print(f"Model saved in '{save}' @ {str(datetime.now())}")

	if N_EPOCH:
		test_loss, test_acc, train_loss, train_acc = zip(*progress)
		append_progress(
			name,
			{
				"test_loss": test_loss,
				"test_acc": test_acc,
				"train_loss": train_loss,
				"train_acc": train_acc,
			},
		)
		print(f"\nCompleted training of {name} with")
		print(f"    N_EPOCH  {N_EPOCH}")
		print(f"     L_RATE  {L_RATE}")
		print(f"    W_DECAY  {W_DECAY}\n")

