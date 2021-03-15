import torch
from torch import nn, optim
from utils import load_data, ClassType
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transforms import aug_transform, clean_transform


class Conv2dPad(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dPad, self).__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.block1 = nn.Sequential(
            Conv2dPad(1, 5, 5),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(),
            Conv2dPad(5, 10, 5),
            nn.BatchNorm2d(10),
            nn.LeakyReLU()
        )

        self.block2 = nn.Sequential(
            Conv2dPad(10, 15, 5),
            nn.BatchNorm2d(15),
            nn.LeakyReLU(),
            Conv2dPad(15, 20, 5),
            nn.BatchNorm2d(20),
            nn.LeakyReLU()
        )


        self.aggregate = nn.Sequential(
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(18000, 4096),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        res = x
        highway = nn.Conv2d(1, 10, 1, bias=False).cuda()
        res = highway(res).cuda()

        # print(f"res.size() = {res.size()}")

        for layer in self.block1:
            x = layer(x)
        # print(f"x.size() = {x.size()}")
        x += res

        res2 = x
        highway = nn.Conv2d(10, 20, 1, bias=False).cuda()
        res2 = highway(res2).cuda()
        for layer in self.block2:
            x = layer(x)
        x += res2

        for layer in self.aggregate:
            x = layer(x)
            # print(x.size())
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


def test(model, device, test_loader, plot=False):
    """
	Test model.
	Returns (average loss per batch, accuracy).
	"""
    model.eval()

    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0
    # displaySet = False
    # display = np.zeros([24, 150, 150])
    # dispaly_pred = np.zeros(24)
    # dispaly_target = np.zeros(24)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            pred = model(data)
            correct += sum((torch.argmax(pred, dim=1) - target) == 0).item()
            loss_sum += loss_fn(pred, target).item()

        # if not exampleSet:
        # 	for i in range(24):
        # 		display_data[i] = data[i][0].to("cpu").numpy()
        # 		display_pred[i] = pred[i].to("cpu").numpy()
        # 		display_target[i] = target[i].to("cpu").numpy()
        # 		displaySet = True

    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy/Loss: {round(100 * accuracy, 3)}%, {round(loss_sum / len(test_loader), 5)}")

    # if not plot: return
    # for i in range(10):
    # 	plt.subplot(5,4,i+1)
    # 	data = (example_data[i] - example_data[i].min()) * (example_data[i].max() - example_data[i].min())
    # 	plt.imshow(data, cmap='gray', interpolation='none')
    # 	corr = int(example_pred[i]) == int(example_target[i])
    # 	plt.title()
    # 	plt.title(labels[int(example_pred[i])] + (" ✔" if corr else " ✖"))
    # 	plt.xticks([])
    # 	plt.yticks([])
    # plt.show()

    return loss_sum / len(test_loader), accuracy


def load_model(load, device):
    model = ResBlock().to(device)
    if load: model.load_state_dict(torch.load(load))
    return model


def run(N_EPOCH, L_RATE, W_DECAY, class_type, load=None, save=None):
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

    print(model)
    progress = []
    for epoch in range(N_EPOCH):
        te_loss, te_acc = test(model, device, val_loader)
        print(f"--- Beginning Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}")
        tr_loss, tr_acc = train(model, device, train_loader, optimizer)
        print(f"--- Completed Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}")
        print(f"Train Loss: {round(tr_loss, 5)}")
        progress.append((te_loss, te_acc, tr_loss, tr_acc))
        scheduler.step()
        scheduler.get_last_lr()

    test(model, device, test_loader)

    if save: torch.save(model.state_dict(), save)
    return progress


if __name__ == "__main__":
    # Example:
    run(10, 1e-3, 0.1, ClassType.NORMAL_INFECTED, save="infected-covid-res02-adamW.model")

"""
Architecture type: two binary classifiers.
We expect two specific classifiers to give a better performance than a
single three-class model since they may be better fitted to each of their problems.

Being a medical identification problem, we prefer models which have more false positives.

Layers to try:
- BatchNorm
- DropOut
- Inception
- Residual Connections
"""
