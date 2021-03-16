import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.axes as ax

def plot_metrics(json_pth):
    name = json_pth.strip(".json")
    df = pd.read_json(json_pth)
    # df[""] = np.log(df["Learning Rate"])
    df["acc_sens"] = df[["accuracy", "sensitivity"]].mean(axis=1)

    plt.plot("Loss Weight", "accuracy", data=df, color="red")
    plt.plot("Loss Weight", "loss", data=df, color="yellow")
    plt.plot("Loss Weight", "sensitivity", data=df, color="green")
    plt.plot("Loss Weight", "specificity", data=df, color="blue")
    plt.plot("Loss Weight", "f1-score", data=df, color="purple")
    plt.plot("Loss Weight", "acc_sens", data=df, color="pink")

    for _ in df["Loss Weight"]:
        plt.vlines(df["Loss Weight"], -0.1, 1.1, color="lightgray", linestyles="dashed")

    plt.legend()
    plt.savefig(name)
    plt.show()



plot_metrics("test_lossWT_more.json")