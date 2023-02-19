import csv
import matplotlib.pyplot as plt
import numpy as np
import os


def _moving_average(x, w):
    return x
    # Other move is "valid", but changes the size of the output array
    return np.convolve(x, np.ones(w), "same") / w


def _plot_metric(in_dir, metric, x_metric, smoothing_window):
    label_names = []
    x_data = {}
    y_data = {}

    for file_name in os.listdir(in_dir):
        full_file_name = os.path.join(in_dir, file_name)
        if os.path.isdir(full_file_name):
            progress_file_name = os.path.join(full_file_name, "progress.csv")

            if os.path.exists(progress_file_name):
                file_name_tokens = file_name.split("_")
                label_name = file_name_tokens[3]
                label_names.append(label_name)

                x_data[label_name] = []
                y_data[label_name] = []

                with open(progress_file_name, newline='') as csv_file:
                    reader = csv.reader(csv_file, delimiter=',')

                    header = None

                    for row in reader:
                        if header is None:
                            header = {value: key for (key, value) in enumerate(row)}

                            if x_metric not in header or metric not in header:
                                return
                        elif len(row) > 0:
                            x_data[label_name].append(float(row[header[x_metric]]))
                            y_data[label_name].append(float(row[header[metric]]))

    print(label_names)
    fig, ax = plt.subplots(1, 1)
    for name in label_names:
        if len(y_data[name]) == 0:
            continue
        print("plotting...")
        ax.plot(x_data[name], _moving_average(y_data[name], w=smoothing_window), label=name, alpha=0.5)

    plt.legend()
    plt.title(metric)
    plt.show()


# Plots the reward and success training curves of all ray tune runs in a given directory
def plot_rllib_runs(in_dir, metrics=["episode_reward_mean"], x_metric="training_iteration", smoothing_window=10):
    for metric in metrics:
        _plot_metric(in_dir, metric, x_metric, smoothing_window)


if __name__ == "__main__":
    plot_rllib_runs("/Users/joe/Dropbox/School & Work/PostDoc/Student Projects/Sophie/results/none/PPO_2022-09-13_16-01-19")