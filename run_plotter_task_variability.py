
import numpy as np
import argparse
import yaaf

from run_plotter import plot_confidence_bar
from run_pretrain import load_environment_config
from yaaf.visualization import LinePlot


def load_results_per_library_size(environment, config, results):

    horizon = config["horizon"]
    noise = config["noise"]
    max_library_size = config["maxlib"]
    directory = f"{results}/{environment}-{int(noise * 100)}%/runs"

    # Instead of defaultdict for sorting
    results_per_library_size = {}
    for library_size in range(2, max_library_size + 1):
        results_per_library_size[library_size] = []

    print(f"Total {len(results_per_library_size)} libraries")
    for file in yaaf.files(directory):
        agent_library_size, task, trial = file.split("-")
        agent, library_size = agent_library_size.split("_")
        library_size = int(library_size)
        task = int(task.replace("v", ""))
        trial = int(trial.replace("t", "").replace(".npy", ""))
        rewards, decision_times = np.load(f"{directory}/{file}")
        accumulated_reward = rewards.sum()
        results_per_library_size[library_size].append(accumulated_reward)

    return results_per_library_size

def plot_task_variability(environment, config, results, plots, confidence, no_show, no_save):

    results_per_library_size = load_results_per_library_size(environment, config, results)

    names = []
    means = []
    stds = []
    N = []
    for library_size in results_per_library_size:
        results = np.array(results_per_library_size[library_size])
        mean, std, n = results.mean(), results.std(), results.size
        names.append(str(library_size))
        means.append(mean)
        stds.append(std)
        N.append(n)

    filename = f"{plots}/{environment}-task-variability" if not no_save else None

    plot_confidence_bar(names, means, stds, N, "", "Library Size", "Accumulated Reward", confidence, not no_show, filename, ["green" for _ in range(len(results_per_library_size))])

def plot_task_variability_line(environment, config, results, plots, confidence, no_show, no_save):

    results_per_library_size = load_results_per_library_size(environment, config, results)
    num_libraries = len(results_per_library_size)

    MAX_VAL_IT = 95.62
    MIN_RAN_AG = 5.88

    N = 32 # FIXME When its done

    runs = []
    relative_runs = []
    for r in range(N):
        run = np.zeros(num_libraries)
        relative_run = np.zeros(num_libraries)
        for l, library_size in enumerate(results_per_library_size):
            results = np.array(results_per_library_size[library_size])
            value = results[r]
            relative_value = (value - MIN_RAN_AG) / (MAX_VAL_IT - MIN_RAN_AG)
            run[l] = value
            relative_run[l] = relative_value
            print(f"{relative_value},",end="")
        print()
        runs.append(np.array(run))
        relative_runs.append(np.array(relative_run))

    plot = LinePlot("", "Library Size", "Avg. Accumulated Reward", num_libraries, confidence=confidence)
    for run in runs:
        plot.add_run("", run)
    plot.show()

    relative_plot = LinePlot("", "Library Size", "Avg. Accumulated Reward", num_libraries, confidence=confidence)
    for run in relative_runs:
        relative_plot.add_run("", run)
    relative_plot.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("environment", type=str)
    parser.add_argument("--results", default="resources/results_task_variability", type=str)
    parser.add_argument("--plots", default="resources/plots", type=str)
    parser.add_argument("--config", default="config.csv", type=str)
    parser.add_argument("--confidence", default=0.95, type=float)
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--no_save", action="store_true")

    opt = parser.parse_args()
    config = load_environment_config(opt.config, opt.environment)

    plot_task_variability_line(opt.environment, config, opt.results, opt.plots, opt.confidence, opt.no_show, opt.no_save)
    #plot_task_variability(opt.environment, config, opt.results, opt.plots, opt.confidence, opt.no_show, opt.no_save)