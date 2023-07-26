import argparse
import datetime
from collections import defaultdict
import yaml

import yaaf
from run_adhoc import ran_trials
from run_pretrain import load_environment_config
import numpy as np

def describe(environment, resources, results, base_noise, breakdown):

    noises = []
    states = []
    actions = []
    observations = []
    models = defaultdict(lambda: 0)
    policies = defaultdict(lambda: 0)
    setup_times = defaultdict(lambda: {})
    policy_times = defaultdict(lambda: {})

    for environment_directory in yaaf.subdirectories(results):

        if environment in environment_directory:

            parts = environment_directory.split("-")
            noise = parts[-1].replace("%", "")
            noise = int(noise) / 100

            for yaml_file in yaaf.files(f"{results}/{environment_directory}"):

                model_id = yaml_file.split("_v")[1].replace(".yaml", "")

                if "perseus" in yaml_file:

                    policies[noise] += 1
                    with open(f"{results}/{environment_directory}/meta_perseus_v{model_id}.yaml") as file:
                        meta = yaml.load(file, Loader=yaml.FullLoader)
                        policy_times[noise][f"{environment}-v{model_id}"] = meta["Setup Time"]

                else:
                    models[noise] += 1
                    if noise not in noises:
                        noises.append(noise)
                    with open(f"{results}/{environment_directory}/meta_v{model_id}.yaml") as file:
                        meta = yaml.load(file, Loader=yaml.FullLoader)
                        states.append(meta["|X|"])
                        actions.append(meta["|A|"])
                        observations.append(meta["|Z|"])
                        setup_times[noise][f"{environment}-v{model_id}"] = meta["Setup Time"]

    if len(noises) > 0:

        noises.sort()

        print(f"Environment: {environment}")
        states = list(set(states))
        actions = list(set(actions))
        observations = list(set(observations))

        assert len(actions) == 1
        actions = actions[0]

        assert len(observations) == 1
        observations = observations[0]

        if len(states) == 1: states = states[0]

        _setup_times = np.array([setup_times[base_noise][env] for env in setup_times[base_noise]])
        mean_setup_time = _setup_times.mean() if _setup_times.size > 0 else None

        _solve_times = np.array([policy_times[base_noise][env] for env in policy_times[base_noise]])
        mean_solve_time = _solve_times.mean() if _solve_times.size > 0 else None

        print(f"\t|X|: {states} |A|: {actions} |Z|: {observations} |M|: {len(setup_times[base_noise])} |eps|: {len(noises)}")

        if mean_setup_time is not None:
            print(f"\tAvg Setup Time: {datetime.timedelta(seconds=mean_setup_time)}")

        if mean_solve_time is not None:
            print(f"\tAvg Solve Time: {datetime.timedelta(seconds=mean_solve_time)}")

        print("\tTrials: ", end="")
        for agent in ["Value Iteration", "Perseus", "Random Agent", "ATPO"]:
            print(f"{agent}: {ran_trials(environment, base_noise, agent, results)} ", end="")
        print()

        if breakdown:
            for yaml_file in sorted(setup_times[base_noise]):
                setup_time = datetime.timedelta(seconds=setup_times[base_noise][yaml_file])
                print(f"\t - {yaml_file} (Time to setup: {str(setup_time).split('.')[0]},", end="")
                if yaml_file in policy_times[base_noise]:
                    solve_time = datetime.timedelta(seconds=policy_times[base_noise][yaml_file])
                    print(f" Time to solve: {str(solve_time).split('.')[0]})")
                else:
                    print(f" Not Solved!)")

    else:
        print(f"Environment {environment} not setup")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", type=str, default="all")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--resources", default="resources/environments", type=str)
    parser.add_argument("--results", default="resources/results", type=str)
    parser.add_argument("--config", default="config.csv", type=str)
    parser.add_argument("--breakdown", action="store_true")

    opt = parser.parse_args()

    if opt.environment == "all":
        environments = yaaf.subdirectories(opt.results)
        environments.sort()
        for environment in environments:
            parts = environment.split("-")
            environment = "-".join([part for part in parts if "%" not in part])
            config = load_environment_config(opt.config, environment)
            base_noise = config["noise"]
            describe(environment, opt.resources, opt.results, base_noise, opt.breakdown)
            print()
    else:
        config = load_environment_config(opt.config, opt.environment)
        base_noise = config["noise"]
        describe(opt.environment, opt.resources, opt.results, base_noise, opt.breakdown)
