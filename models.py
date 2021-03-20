from torch import nn
from learn import load_model
import torch
from torch.nn.functional import softmax


labels = {
    "inf": {"infected": 0, "normal": 1},
    "cov": {"covid": 0, "non-covid": 1},
    "all": {"infected/covid": 0, "infected/non-covid": 1, "normal": 2},
}
labels_inv = {
    "inf": {0: "infected", 1: "normal"},
    "cov": {0: "covid", 1: "non-covid"},
    "all": {0: "infected/covid", 1: "infected/non-covid", 2: "normal"},
}


def collapse_max(x):
    """
	Collapses x into a tensor with all-zeros except for the max value,
	which will be 1.
	"""
    return torch.zeros_like(x).scatter(
        -1, torch.argmax(x, axis=1).reshape((x.shape[0], 1)), 1
    )


class Convolutional(nn.Module):
    """
	Basic convolutional module with a dynamic number of layers/channels.
	"""

    def __init__(
        self,
        out_size,
        layers=[],
        act_fn=nn.LeakyReLU,
        kernel_size=3,
        stride=1,
        padding=1,
        batchnorm=False,
        dropout=False,
    ):
        super(Convolutional, self).__init__()
        modules = []

        def append(in_c, out_c):
            modules.append(
                nn.Conv2d(
                    in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding
                )
            )
            if batchnorm:
                modules.append(nn.BatchNorm2d(out_c))
            modules.append(act_fn())
            if dropout:
                modules.append(nn.Dropout2d(p=0.2))

        if layers:
            append(1, layers[0])
        for ind in range(len(layers) - 1):
            append(layers[ind], layers[ind + 1])
        self.layers = nn.Sequential(
            *modules, nn.AdaptiveMaxPool2d((out_size, out_size)), nn.Flatten()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Classifier(nn.Module):
    """
	Basic fully connected module with a dynamic number of layers.
	"""

    def __init__(
        self,
        in_features,
        out_features=2,
        layers=[],
        act_fn=nn.LeakyReLU,
        batchnorm=False,
        dropout=False,
    ):
        super(Classifier, self).__init__()
        modules = []

        def append(in_f, out_f):
            modules.append(nn.Linear(in_f, out_f))
            if batchnorm:
                modules.append(nn.BatchNorm1d(out_f))
            modules.append(act_fn())
            if dropout:
                modules.append(nn.Dropout(p=0.2))

        if layers:
            append(in_features, layers[0])
        for ind in range(len(layers) - 1):
            append(layers[ind], layers[ind + 1])
        self.layers = nn.Sequential(
            *modules, nn.Linear(layers[-1] if layers else in_features, out_features)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class CNN(nn.Module):
    """
	Basic convolutional network comprising of one convolutional module
	and one fully connected module.
	"""

    def __init__(
        self,
        out_features=2,
        intermediate_size=27,
        conv=[5, 10, 15],
        conv_activation=True,
        stride=1,
        padding=1,
        kernel_size=5,
        fc=[512],
        batchnorm=True,
        dropout=False,
    ):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            Convolutional(
                intermediate_size,
                layers=conv,
                act_fn=nn.LeakyReLU if conv_activation else nn.Identity,
                batchnorm=batchnorm,
                dropout=dropout,
                stride=stride,
                padding=padding,
                kernel_size=kernel_size,
            ),
            Classifier(
                (conv[-1] if conv else 1) * intermediate_size ** 2,
                out_features=out_features,
                layers=fc,
                batchnorm=batchnorm,
                dropout=dropout,
            ),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Ensemble(nn.Module):
    """
	Wrapper class for ensemble model.
	Only meant to be used in evaluation mode.
	"""

    def __init__(self, params, files):
        super(Ensemble, self).__init__()
        self.models = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for ind, (param, files) in enumerate(zip(params, files)):
            for file in files:
                self.models.append(load_model(CNN(**param), file, device))
        for model in self.models:
            model.eval()

    def forward(self, x):
        preds = sum([collapse_max(model(x)) for model in self.models])
        return softmax(preds, dim=1)


def ensemble_for(class_type):
    if class_type == "inf":
        return Ensemble(
            params=[{"conv": [5, 10, 15], "kernel_size": 5}],
            files=[[f"trained_models/inf-{ind+1}.model" for ind in range(10)]],
        )
    elif class_type == "cov":
        return Ensemble(
            params=[{"conv": [5, 10, 15], "kernel_size": 5}],
            files=[[f"trained_models/cov-{ind+1}.model" for ind in range(10)]],
        )
    elif class_type == "all":
        return Ensemble(
            params=[{"conv": [5, 10, 15], "out_features": 3, "kernel_size": 5}],
            files=[[f"trained_models/all-{ind+1}.model" for ind in range(5)]],
        )
    else:
        raise ValueError("class_type must be one of 'cov', 'inf', or 'all'")


class Assembly(nn.Module):
    """
	Assembly model that acts as a two-part binary classifier, given predictors
	for normal/infected and covid/non-covid models.
	Only meant to be used in evaluation mode.
	"""

    def __init__(self, pred_mode="prob", inf=None, cov=None):
        super(Assembly, self).__init__()
        self.inf = ensemble_for("inf") if not inf else inf
        self.cov = ensemble_for("cov") if not cov else cov
        self.pred_mode = pred_mode

    def forward(self, x):
        """
		'pred'	Pr(norm) = softmax(inf(x, norm))
				Pr(covd) = softmax(inf(x, infc)) * softmax(cov(x, covd))
				Pr(ncov) = softmax(inf(x, infc)) * softmax(cov(x, ncov))
				Note that this may return an 'infected' result even if
				inf predicts normal if it is with low probability and
				either 'covid' or 'non-covid' has a strong probability.
		'tree'	If softmax(inf(x, norm)) > 0.5:
					Pr(norm) = 1
					Pr(covd) = 0
					Pr(ncov) = 0
				else:
					Pr(norm) = 0
					If softmax(cov, (x, covd)) > 0.5:
						Pr(covd) = 1
						Pr(ncov) = 0
					else:
						Pr(covd) = 0
						Pr(ncov) = 1
		"""
        if self.pred_mode.lower() == "prob":
            inf_out = softmax(self.inf(x), dim=1)
            cov_out = softmax(self.cov(x), dim=1)
        elif self.pred_mode.lower() == "tree":
            inf_out = collapse_max(self.inf(x))
            cov_out = collapse_max(self.cov(x))
        else:
            raise ValueError("pred_mode must be one of 'prob' or 'tree'")

        inf_prob, norm_prob = (
            inf_out[:, labels["inf"]["infected"]],
            inf_out[:, labels["inf"]["normal"]],
        )
        cov_prob, ncov_prob = (
            cov_out[:, labels["cov"]["covid"]],
            cov_out[:, labels["cov"]["non-covid"]],
        )
        cov_prob *= inf_prob
        ncov_prob *= inf_prob

        probs = [None, None, None]
        probs[labels["all"]["infected/covid"]] = cov_prob
        probs[labels["all"]["infected/non-covid"]] = ncov_prob
        probs[labels["all"]["normal"]] = norm_prob

        return torch.stack(probs, dim=1)

