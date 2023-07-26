import argparse
import random
import time

import numpy as np

from backend.agents import AITKNoHorizonAgent
from run_adhoc import FACTORY, save_trial, ran_trials
from run_pretrain import load_environment_config, load_pomdp, create_pomdp, check_perseus_policy, create_perseus_policy
from yaaf import Timestep

def load_or_create(cache, environment, model_id, config, resources, results):

    if model_id not in cache:

        try:
            pomdp = load_pomdp(environment, model_id, config["noise"], resources, results)
        except FileNotFoundError:
            print(f"\t\t{environment}-v{model_id} -> Not in disk, creating... ", flush=True, end="")
            pomdp, time = create_pomdp(environment, model_id, config["noise"], resources, results)
            print(f" -> Created! ({time})", flush=True)

        try:
            check_perseus_policy(pomdp, config, resources, results)
            controller = AITKNoHorizonAgent.load_cached_or_compute(pomdp, config)
        except FileNotFoundError:
            print(f"\t\t{environment}-v{model_id} Perseus Policy -> Not in disk, creating... ", flush=True, end="")
            controller, time = create_perseus_policy(environment, pomdp, config, resources, results)
            print(f" -> Created! ({time})", flush=True)

        pomdp.controller = controller
        cache[model_id] = pomdp
    else:
        pomdp = cache[model_id]

    return pomdp


def run(environment, trials, config, resources, results):

    horizon = config["horizon"]
    possible_models = config["maxlib"]

    pomdp_cache = {}

    for library_size in range(2, possible_models + 1):

        agent_name = f"ATPO_{library_size}"
        print(f"Library: {library_size} models", flush=True)
        
        for trial in range(trials):

            print(f"\tTrial #{trial + 1}/{trials}")
            current_model_id = random.choice(list(range(possible_models)))

            trials_saved = ran_trials(environment, config["noise"], agent_name+"-", results)
            print(f"Total of {trials_saved} for {agent_name}", flush=True)
            if trial < trials_saved:

                # Don't run
                print("\t\t-> Already ran", flush=True)

            else:

                # Run
                pomdp = load_or_create(pomdp_cache, environment, current_model_id, config, resources, results)

                library = [pomdp]
                candidate_models = list(range(possible_models))
                del candidate_models[candidate_models.index(current_model_id)]
                other_model_ids = random.sample(candidate_models, library_size-1)
                assert current_model_id not in other_model_ids

                for other_model_id in other_model_ids:
                    other_pomdp = load_or_create(pomdp_cache, environment, other_model_id, config, resources, results)
                    library.append(other_pomdp)

                agent = FACTORY["ATPO"](pomdp, config, library)
                rewards, decision_times = run_trial(agent, pomdp, horizon)
                result = np.array([rewards, decision_times])
                save_trial(environment, agent_name, current_model_id, config["noise"], result, results)

                print("\t\t-> Ran", flush=True)

def run_trial(agent, pomdp, horizon):

    rewards = np.zeros(horizon)
    decision_times = np.zeros(horizon)

    pomdp.reset()
    agent.reset()

    for step in range(horizon):

        action = agent.action(None)
        next_obs, reward, terminal, info = pomdp.step(action)
        timestep = Timestep(None, action, None, next_obs, None, {})

        start = time.time()
        agent.reinforcement(timestep)
        end = time.time()

        # Register
        rewards[step] = reward
        decision_time = end - start
        decision_times[step] = decision_time

    return rewards, decision_times


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("environment", type=str)
    parser.add_argument("--trials", default=32, type=int)
    parser.add_argument("--resources", default="resources/environments", type=str)
    parser.add_argument("--results", default="resources/results_task_variability", type=str)
    parser.add_argument("--config", default="config.csv", type=str)

    opt = parser.parse_args()

    config = load_environment_config(opt.config, opt.environment)
    run(opt.environment, opt.trials, config, opt.resources, opt.results)
