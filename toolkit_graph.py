from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit




def exponential_model(x, a, b, c):
    return a * np.exp(b * (-x)) + c




def scatter_loss(datasets, x_limits=None, y_limits=None, title: str = 'Experiment', use_expo_fit: bool = True):
    plt.figure(figsize=(10, 7))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (x_data, y_data, label) in enumerate(datasets):
        if use_expo_fit:
            try:
                popt, _ = curve_fit(exponential_model, x_data, y_data, maxfev=10000)
            except RuntimeError as e:
                print(f"Fit did not converge for {label}: {e}")
                continue
            a, b, c = popt

            x_fit = np.linspace(np.min(x_data), np.max(x_data), 500)
            y_fit = exponential_model(x_fit, *popt)

        color = colors[i % len(colors)]

        plt.plot(x_data, y_data, marker='o', linestyle='-', label=f"{label} Data", color=color, alpha=0.6)

        if use_expo_fit:
            plt.plot(x_fit, y_fit, label=f"{label} Exp. Fit", color=color, linewidth=2)

    plt.xlabel('Timestamp')
    plt.ylabel('Loss')
    plt.title(title)

    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)

    plt.legend()
    plt.grid(True)
    plt.show()




def plot_loss(experiment_instances, title=None, x_range=None, y_range=None) -> None:
    keys = []
    x_max = -np.inf
    y_max = -np.inf

    for instance in experiment_instances:
        keys.append((instance.timestamps, instance.losses))

        current_x_max = np.max(instance['timestamps'])
        current_y_max = np.max(instance['losses'])
        if current_x_max > x_max:
            x_max = current_x_max
        if current_y_max > y_max:
            y_max = current_y_max

    x_limits: Tuple[float, float] = x_range if x_range is not None else (0, x_max)
    y_limits: Tuple[float, float] = y_range if y_range is not None else (0, y_max)

    if title is None:
        title = 'Experiment Graph'

    scatter_loss(datasets=keys, x_limits=x_limits, y_limits=y_limits, title=title)
