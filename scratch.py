import torch
from torch import nn, optim
from transforms import aug_transform, clean_transform
from utils import load_data, ClassType
from tqdm import tqdm
from datetime import datetime


# res input size = [31, 1, 150, 150]
# x output size = [32, 20, 146, 146]

class Conv2dPad(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dPad, self).__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class ResStack(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super(ResStack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.downsampling = downsampling

        self.highway = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.activate = nn.LeakyReLU(inplace=True)

        self.blocks = nn.Identity()

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

    def forward(self, x):
        res = x

        if self.should_apply_shortcut:
            res = self.highway(x)
        x = self.blocks(x)
        x += res
        x = self.activate(x)
        return x


class ResBlock(ResStack):
    def __init__(self, in_channels, out_channels,  *args, **kwargs):
        super(ResBlock, self).__init__(in_channels, out_channels, *args, **kwargs)

        self.expansion = 1
        self.blocks = nn.Sequential(
            Conv2dPad(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            Conv2dPad(self.out_channels, self.expanded_channels, 3),
            nn.BatchNorm2d(self.expanded_channels),
            nn.LeakyReLU()
        )


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResBlock, n=1, *args, **kwargs):
        super().__init__()

        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * 2,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

        self.aggregate = nn.Sequential(
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Linear(18000, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )


    def forward(self, x):
        x = self.blocks(x)
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
    model = ResLayer(1, 20, n=3).to(device)
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
    optimizer = optim.Adam(model.parameters(), lr=L_RATE, weight_decay=W_DECAY)

    print(model)
    progress = []
    for epoch in range(N_EPOCH):
        te_loss, te_acc = test(model, device, val_loader)
        print(f"--- Beginning Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}")
        tr_loss, tr_acc = train(model, device, train_loader, optimizer)
        print(f"--- Completed Epoch {epoch + 1}/{N_EPOCH} @ {str(datetime.now())}")
        print(f"Train Loss: {round(tr_loss, 5)}")
        progress.append((te_loss, te_acc, tr_loss, tr_acc))

    test(model, device, test_loader)

    if save: torch.save(model.state_dict(), save)
    return progress


if __name__ == "__main__":
    # Example:
    run(5, 10e-6, 10e-3, ClassType.NORMAL_INFECTED, save="infected-covid.model")







