import argparse
import random
import time

import numpy as np

import yaaf
from run_pretrain import load_environment_config, load_pomdp
from yaaf import Timestep
from yaaf.agents import ValueIteration, RandomAgent
from backend.agents import ATPO, AITKNoHorizonAgent, BOPA, PerseusRandom

FACTORY = {
    "Value Iteration": lambda pomdp, params, pomdps=None: ValueIteration(pomdp),
    "Perseus": lambda pomdp, params, pomdps=None: AITKNoHorizonAgent(pomdp, params),
    "ATPO": lambda pomdp, params, pomdps=None: ATPO(pomdps, params),
    "Random Agent": lambda pomdp, params, pomdps=None: RandomAgent(pomdp.num_actions),
    "BOPA": lambda pomdp, params, pomdps=None: BOPA(pomdps),
    "PerseusRandom": lambda pomdp, params, pomdps=None: PerseusRandom(pomdps, params),

}

def is_adhoc(agent_name):
    return "ATPO" in agent_name or "Assistant" in agent_name or "BOPA" in agent_name or "PerseusRandom" in agent_name

def load_from_cache(cache, model_id, environment, config, resources, results, load_controller):
    if model_id not in cache:
        print(f"Loading pomdp {environment}-{int(config['noise']*100)}%-v{model_id}", flush=True)
        pomdp = load_pomdp(environment, model_id, config["noise"], resources, results)
        if load_controller:
            print(f"Loading controller {environment}-{int(config['noise'] * 100)}%-v{model_id}", flush=True)
            pomdp.controller = AITKNoHorizonAgent.load_cached_or_compute(pomdp, config)
        cache[model_id] = pomdp
    else:
        pomdp = cache[model_id]
    # Dict is not a primitive type right (its passed by reference)
    # If so, no need to return it below here v
    return pomdp

def run(environment, agents, trials, rerun, config, resources, results):

    horizon = config["horizon"]

    pomdp_cache = {}
    load_controllers = "Perseus" in agents or "ATPO" in agents or "PerseusRandom" in agents #(True in [is_adhoc(agent) for agent in agents])
    for trial in range(trials):
        possible_models = config["models"]

        #
        if environment == "overcooked":
            model_id = 3
        else:
            model_id = random.choice(list(range(possible_models)))

        print(f"Trial #{trial + 1}/{trials} (chosen model: {model_id})")

        for agent_name in agents:
            print(f"\t- {agent_name}", flush=True, end="")
            trials_saved = ran_trials(environment, config["noise"], agent_name, results)
            if not rerun and trial < trials_saved:
                print(" -> Already ran", flush=True)
                continue
            else:
                pomdp = load_from_cache(pomdp_cache, model_id, environment, config, resources, results, load_controllers)
                if is_adhoc(agent_name):
                    library_size = possible_models
                    library = [pomdp]
                    candidate_models = list(range(library_size))
                    del candidate_models[candidate_models.index(model_id)]
                    other_model_ids = random.sample(candidate_models, library_size-1)
                    assert model_id not in other_model_ids
                    for other_model_id in other_model_ids:
                        other_pomdp = load_from_cache(pomdp_cache, other_model_id, environment, config, resources, results, load_controllers)
                        library.append(other_pomdp)
                    agent = FACTORY[agent_name](pomdp, config, library)
                else:
                    agent = FACTORY[agent_name](pomdp, config)
                if not rerun:
                    rewards, decision_times, _ = run_trial(agent, pomdp, horizon)
                    result = np.array([rewards, decision_times])
                    save_trial(environment, agent_name, model_id, config["noise"], result, results)
                print(" -> Ran", flush=True)

def run_trial(agent, pomdp, horizon):

    full_observability = agent.name in ["Value Iteration", "BOPA"]

    rewards = np.zeros(horizon)
    decision_times = np.zeros(horizon)
    beliefs = []

    if full_observability:
        print(" full observability", end=" ")
        state = pomdp.reset()
        if agent.name == "BOPA":
            agent.reset(state)
    else:
        pomdp.reset()
        agent.reset()

    if is_adhoc(agent.name):
        beliefs.append(agent.beliefs_over_pomdps)

    for step in range(horizon):

        if full_observability:
            action = agent.action(state)
            next_obs, reward, terminal, info = pomdp.step(action)
            next_state = pomdp.state
            timestep = Timestep(state, action, reward, next_state, terminal, info)
            state = next_state
        else:
            action = agent.action(None)
            next_obs, reward, terminal, info = pomdp.step(action)
            timestep = Timestep(None, action, None, next_obs, None, {})

        start = time.time()
        agent.reinforcement(timestep)
        end = time.time()

        # Register
        if is_adhoc(agent.name):
            beliefs.append(agent.beliefs_over_pomdps)
        rewards[step] = reward
        decision_time = end - start
        decision_times[step] = decision_time

    return rewards, decision_times, beliefs

def ran_trials(environment, noise, agent, results):
    try:
        trials = 0
        directory = f"{results}/{environment}-{int(noise*100)}%/runs"
        for file in yaaf.files(directory):
            if agent in file and ".npy" in file:
                trials += 1
        return trials
    except FileNotFoundError:
        return 0

def save_trial(environment, agent, model_id, noise, result, results):
    directory = f"{results}/{environment}-{int(noise*100)}%/runs"
    filename = f"{directory}/{agent}-v{model_id}-t{random.getrandbits(32)}.npy"
    yaaf.mkdir(directory)
    np.save(filename, result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("environment", type=str)
    parser.add_argument("--trials", default=32, type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--resources", default="resources/environments", type=str)
    parser.add_argument("--results", default="resources/results", type=str)
    parser.add_argument("--config", default="config.csv", type=str)
    parser.add_argument('--agents', nargs='+',
                        default=[
                            "Value Iteration",
                            "Perseus",
                            "Random Agent",
                            "ATPO",
                            "PerseusRandom"
                        ])

    opt = parser.parse_args()

    config = load_environment_config(opt.config, opt.environment)
    agents = [agent.replace("_", " ") for agent in opt.agents]
    run(opt.environment, agents, opt.trials, opt.rerun, config, opt.resources, opt.results)
