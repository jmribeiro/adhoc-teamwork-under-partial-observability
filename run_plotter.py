from collections import defaultdict

import numpy as np
import argparse
import yaaf
import matplotlib.pyplot as plt

from run_adhoc import is_adhoc
from run_pretrain import load_environment_config
from yaaf.visualization import standard_error


def plot_confidence_bar(names, means, std_devs, N, title, x_label, y_label, confidence, show=False, filename=None, colors=None, yscale=None):
    names = [name.replace(" ", "\n") for name in names]
    errors = [standard_error(std_devs[i], N[i], confidence) for i in range(len(means))]
    fig, ax = plt.subplots()
    x_pos = np.arange(len(names))
    ax.bar(x_pos, means, yerr=errors, align='center', alpha=0.5, color=colors if colors is not None else "gray", ecolor='black', capsize=10)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.yaxis.grid(True)
    if yscale is not None:
        plt.yscale(yscale)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    plt.close()

def plot_confidence_errors(x, y, yerr, title, xlabel, ylabel, color, show, filename=None):
    ls = 'dotted'
    plt.errorbar(x, y, yerr=yerr, ls=ls, capsize=10, marker="o", color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, [int(x[i]) for i in range(x.size)])
    plt.grid()
    if show:
        plt.show()
    if filename is not None:
        plt.savefig(filename)
    plt.close()

def load_results(environment, noise, results, task=-1):
    runs = f"{results}/{environment}-{int(noise*100)}%/runs"
    accumulated_rewards = defaultdict(lambda: [])
    decision_times = defaultdict(lambda: [])
    for file in yaaf.files(runs):
        if ".npy" in file:
            filename = f"{runs}/{file}"
            try:
                agent, model_id, trial_id = file.split("-")
            except ValueError:
                agent, library_size, model_id, trial_id = file.split("-")
                assert is_adhoc(agent), f"Corrupt result file {runs}/{file}"
            model_id = int(model_id.replace("v", ""))
            if task != -1 and model_id != task:
                continue
            result = np.load(filename)
            rewards, _decision_times = result[0], result[1]
            accumulated_reward = rewards.sum()
            accumulated_rewards[agent].append(accumulated_reward)
            decision_times[agent] += list(_decision_times)
    return accumulated_rewards, decision_times

def plot_task(environment, config, task, results, plots, confidence, no_show, no_save):

    runs = f"{results}/runs"

    if not yaaf.isdir(runs):
        print(f"{environment}-{int(config['noise']*100)}% has no results.", flush=True)
    else:
        accumulated_rewards = defaultdict(lambda: [])
        decision_times = defaultdict(lambda: [])
        for file in yaaf.files(runs):
            if ".npy" in file:
                filename = f"{runs}/{file}"
                try:
                    agent, model_id, trial_id = file.split("-")
                except ValueError:
                    agent, library_size, model_id, trial_id = file.split("-")
                    assert is_adhoc(agent), f"Corrupt result file {runs}/{file}"
                model_id = int(model_id.replace("v", ""))
                if task != -1 and model_id != task:
                    continue
                result = np.load(filename)
                rewards, _decision_times = result[0], result[1]
                accumulated_reward = rewards.sum()
                accumulated_rewards[agent].append(accumulated_reward)
                decision_times[agent] += list(_decision_times)

        yaaf.mkdir(f"{plots}/{environment}")
        yaaf.mkdir(f"{plots}/{environment}/decision times")
        task = "all_tasks" if task == -1 else f"task {task}"

        # Plot Reward
        names = [f"{agent}\n(N={np.array(accumulated_rewards[agent]).size})" for agent in accumulated_rewards]
        means = [np.array(accumulated_rewards[agent]).mean() for agent in accumulated_rewards]
        stds = [np.array(accumulated_rewards[agent]).mean() for agent in accumulated_rewards]
        N = [np.array(accumulated_rewards[agent]).size for agent in accumulated_rewards]
        if task == "all_tasks":
            filename = f"{plots}/{environment}.png" if not no_save else None
        else:
            filename = f"{plots}/{environment}/{environment}_{task}.png" if not no_save else None
        plot_confidence_bar(names, means, stds, N, f"{environment}", "Agents", f"Reward Accumulated in {config['horizon']} steps", confidence, not no_show, filename)

        # Plot Decision Times
        names = [f"{agent}\n(N={np.array(decision_times[agent]).size})" for agent in decision_times]
        means = [np.array(decision_times[agent]).mean() for agent in decision_times]
        stds = [np.array(decision_times[agent]).mean() for agent in decision_times]
        N = [np.array(decision_times[agent]).size for agent in decision_times]
        filename = f"{plots}/{environment}/decision times/{environment}_{task}.png" if not no_save else None
        plot_confidence_bar(names, means, stds, N, f"{environment}", "Agents", f"Average Decision Time", confidence, not no_show, filename, yscale="log")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("environment", type=str)
    parser.add_argument("--resources", default="resources/environments", type=str)
    parser.add_argument("--results", default="resources/results", type=str)
    parser.add_argument("--plots", default="resources/plots", type=str)
    parser.add_argument("--config", default="config.csv", type=str)
    parser.add_argument("--confidence", default=0.95, type=float)
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--task", default=-1, type=int)

    opt = parser.parse_args()
    config = load_environment_config(opt.config, opt.environment)

    directory = f"{opt.results}/{opt.environment}-{int(config['noise']*100)}%"
    if opt.task == -1:
        for task in range(config["models"]):
            plot_task(opt.environment, config, task, directory, opt.plots, opt.confidence, opt.no_show, opt.no_save)
        plot_task(opt.environment, config, opt.task, directory, opt.plots, opt.confidence, opt.no_show, opt.no_save)
    else:
        plot_task(opt.environment, config, opt.task, directory, opt.plots, opt.confidence, opt.no_show, opt.no_save)
