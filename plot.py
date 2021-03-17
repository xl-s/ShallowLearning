import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.axes as ax

def plot_metrics(json_pth, param):
    name = json_pth.strip(".json")
    df = pd.read_json(json_pth)
    # df["acc_sens"] = df[["accuracy", "sensitivity"]].mean(axis=1)
    df["acc_sens"] = (df["accuracy"] + df["sensitivity"] - 0.5 * np.abs((df["accuracy"]-df["sensitivity"])))/2

    if param == "Learning Rate" or param == "Weight Decay":
        df[param] = np.log(df[param])

    plt.plot(param, "accuracy", data=df, color="red")
    plt.plot(param, "loss", data=df, color="yellow")
    plt.plot(param, "sensitivity", data=df, color="green")
    plt.plot(param, "specificity", data=df, color="blue")
    plt.plot(param, "f1-score", data=df, color="purple")
    plt.plot(param, "acc_sens", data=df, color="pink")

    for _ in df[param]:
        plt.vlines(df[param], -0.1, 1.1, color="lightgray", linestyles="dashed")

    plt.legend()
    plt.savefig(name)
    plt.show()

plot_metrics("test_LR.json", "Learning Rate")