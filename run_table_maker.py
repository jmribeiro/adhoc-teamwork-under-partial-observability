import argparse
import datetime

import yaml

import yaaf
import csv
import numpy as np

from run_plotter import load_results
from run_pretrain import load_environment_config

def all_environments(results):
    _environments = yaaf.subdirectories(results)
    _environments.sort()
    environments = []
    for environment in _environments:
        parts = environment.split("-")
        environment = "-".join([part for part in parts if "%" not in part])
        environments.append(environment)
    return environments

def make_rewards_table(filename, environments, agents, config, results):
    with open(filename, 'w+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Environment'] + [agent.capitalize() for agent in agents])
        for environment in environments:
            try:
                env_config = load_environment_config(config, environment)
                noise = env_config["noise"]
                rewards, times = load_results(environment, noise, results)
                row = [environment]
                for agent in agents:
                    value = np.array(rewards[agent])
                    mean = np.round(value.mean(), 2)
                    std = np.round(value.std(), 2)
                    row.append(f"{mean}(±{std})")
                writer.writerow(row)
            except:
                continue

def timedeltaf(timedelta):
    h, m, sms = str(timedelta).split(":")
    try:
      s, ms = sms.split(".")
      ms = float(f"0.{ms}")
    except ValueError:
      s = 0
      ms = 0.0
    return f"{h}h {m}m {s}s {round(ms, 2)}ms"
    
def make_times_table(filename, environments, config, results):
    with open(filename, 'w+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Environment', "Avg. Setup Time", "Avg. Solve Time"])
        for environment in environments:
            try:

                env_config = load_environment_config(config, environment)
                noise = env_config["noise"]
                environment_noise_directory = f"{results}/{environment}-{int(noise*100)}%"
                setup_times = []
                solve_times = []

                for file in yaaf.files(environment_noise_directory):

                    if ".yaml" in file and "meta_perseus" in file:
                        # Process Solve time
                        with open(f"{environment_noise_directory}/{file}") as file:
                            meta_perseus = yaml.load(file, Loader=yaml.FullLoader)
                            solve_times.append(meta_perseus["Setup Time"])
                    elif ".yaml" in file and "meta_" in file:
                        # Process Setup time
                        with open(f"{environment_noise_directory}/{file}") as file:
                            meta = yaml.load(file, Loader=yaml.FullLoader)
                            setup_times.append(meta["Setup Time"])
                    else:
                        continue

                try:
                    setup_times = np.array(setup_times)
                    setup_mean = datetime.timedelta(seconds=(setup_times.mean()))
                    setup_std = datetime.timedelta(seconds=(setup_times.std()))
                    setup = f"{timedeltaf(setup_mean)}(±{timedeltaf(setup_std)})"
                except:
                    setup = "Not Setup"

                try:
                    solve_times = np.array(solve_times)
                    solve_mean = datetime.timedelta(seconds=(solve_times.mean()))
                    solve_std = datetime.timedelta(seconds=(solve_times.std()))
                    solve = f"{timedeltaf(solve_mean)}(±{timedeltaf(solve_std)})"
                except:
                    solve = "Not Solved"

                row = [environment, setup, solve]
                writer.writerow(row)

            except:
                continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--environments', nargs='+', default=[])
    parser.add_argument('--agents', nargs='+',
                        default=[
                            "Value Iteration",
                            "BOPA",
                            "Perseus",
                            "PerseusRandom",
                            "ATPO",
                            "Random Agent",
                        ])
    parser.add_argument("--resources", default="resources/environments", type=str)
    parser.add_argument("--results", default="resources/results", type=str)
    parser.add_argument("--tables", default="resources/tables", type=str)
    parser.add_argument("--config", default="config.csv", type=str)
    parser.add_argument("--no_show", action="store_true")

    opt = parser.parse_args()
    environments = opt.environments if len(opt.environments) > 0 else all_environments(opt.results)

    yaaf.mkdir(opt.tables)
    make_rewards_table(f"{opt.tables}/results.csv", environments, opt.agents, opt.config, opt.results)
    make_times_table(f"{opt.tables}/times.csv", environments, opt.config, opt.results)
