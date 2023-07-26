import argparse
import random
import numpy as np

import yaaf
from run_adhoc import FACTORY, run_trial, save_trial, load_from_cache
from run_pretrain import load_environment_config


def run(agent_name, environment, config, num_trials, pomdp_cache):

    pomdps = []
    for model_id in range(config["models"]):
        load_controllers = agent_name == "ATPO"
        print(f"\t\tLoading {environment}-{int(config['noise']*100)}%-v{model_id}", end=" ", flush=True)
        try:
            pomdp = load_from_cache(pomdp_cache, model_id, environment, config, "resources/environments", "resources/results", load_controllers)
        except FileNotFoundError:
            print(f"Run pretrain for {environment}", flush=True)
            exit(-1)

        print(f"Done!", flush=True)
        pomdps.append(pomdp)

    results = []

    for trial in range(num_trials):

        print(f"\t\tTrial {trial+1}/{num_trials}", flush=True, end=" ")

        m = random.choice(range(len(pomdps)))
        env = pomdps[m]
        agent = FACTORY[agent_name](env, config, pomdps)
        rewards, decision_times, beliefs = run_trial(agent, env, config["horizon"])
        trial_result = np.array([rewards, decision_times])
        save_trial(environment, agent_name, m, config["noise"], trial_result, "resources/belief_results")

        beliefs = np.array(beliefs)
        results.append((m, beliefs))

        print("Done!", flush=True)

    return results, pomdp_cache


def save_beliefs(agent, environment, noise, beliefs, results):
    environment_name = f"{environment}-{int(noise * 100)}%"
    directory = f"{results}/{environment_name}"
    yaaf.mkdir(directory)
    for m, belief in beliefs:
        file = f"{directory}/{agent}_{random.getrandbits(32)}"
        np.save(file, (m, belief), allow_pickle=True)


def load_beliefs(agent, environment, noise, results):
    environment_name = f"{environment}-{int(noise * 100)}%"
    directory = f"{results}/{environment_name}"
    beliefs = []
    try:
        for file in yaaf.files(directory):
            if ".npy" in file and agent in file:
                m, belief = np.load(f"{directory}/{file}", allow_pickle=True)
                beliefs.append((m, belief))
    except FileNotFoundError:
        pass
    return beliefs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('environment', type=str)
    opt = parser.parse_args()

    config = "config.csv"
    resources = "resources"
    belief_resources = "resources/beliefs"
    num_trials = 32

    agents = [
        "BOPA",
        "ATPO",
    ]

    environment = opt.environment

    print(f"{environment}", flush=True)

    cache = {}

    env_config = load_environment_config(config, environment)

    for agent in agents:

        print(f"\t{agent}", flush=True)

        beliefs = load_beliefs(agent, environment, env_config["noise"], belief_resources)
        needed_trials = max(0, num_trials - len(beliefs))
        print(f"\t\tHave {len(beliefs)} trials, need {needed_trials}", flush=True)

        if needed_trials > 0:
            more_beliefs, cache = run(agent, environment, env_config, needed_trials, cache)
            assert len(more_beliefs) == needed_trials
            save_beliefs(agent, environment, env_config["noise"], more_beliefs, belief_resources)
            beliefs += more_beliefs
