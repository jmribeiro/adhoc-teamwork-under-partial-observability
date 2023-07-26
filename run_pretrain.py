import argparse
import datetime
import time

import pandas as pd
import yaml

import yaaf
from backend.environments import factory
from backend.pomdp import PartiallyObservableMarkovDecisionProcess as POMDP
from backend.agents import AITKAgent

def create_pomdp(environment, model_id, noise, resources, results):

    start = time.time()
    pomdp = factory[environment](model_id, noise)
    end = time.time()
    model_setup_time = end - start

    pomdp.save(f"{resources}/{environment}-{int(noise*100)}%")
    yaaf.mkdir(f"{results}/{environment}-{int(noise*100)}%")

    meta = {"|X|": pomdp.num_states, "|A|": pomdp.num_actions, "|Z|": pomdp.num_observations, "Setup Time": model_setup_time}
    with open(f"{results}/{environment}-{int(noise*100)}%/meta_v{model_id}.yaml", 'w+') as file:
        yaml.dump(meta, file, default_flow_style=False)

    meta = {"|X|": pomdp.num_states, "|A|": pomdp.num_actions, "|Z|": pomdp.num_observations, "Setup Time": model_setup_time}
    with open(f"{resources}/{environment}-{int(noise * 100)}%/meta_v{model_id}.yaml", 'w+') as file:
        yaml.dump(meta, file, default_flow_style=False)

    return pomdp, datetime.timedelta(seconds=model_setup_time)

def load_pomdp(environment, model_id, noise, resources, results):
    directory = f"{resources}/{environment}-{int(noise*100)}%/{environment}-v{model_id}"
    pomdp = POMDP.load(directory)
    with open(f"{resources}/{environment}-{int(noise * 100)}%/meta_v{model_id}.yaml", 'r') as file:
        meta = yaml.load(file, Loader=yaml.FullLoader)

    yaaf.mkdir(f"{results}/{environment}-{int(noise*100)}%")
    with open(f"{results}/{environment}-{int(noise*100)}%/meta_v{model_id}.yaml", 'w+') as file:
        yaml.dump(meta, file, default_flow_style=False)
    return pomdp

def create_perseus_policy(environment, pomdp, config, resources, results):
    start = time.time()
    policy = AITKAgent.compute_policy(pomdp, "Perseus", config)
    end = time.time()
    offline_planning_time = end - start
    policy.save(f"{resources}/{environment}-{int(config['noise']*100)}%/{pomdp.spec.id}/perseus_policy_b{config['beliefs']}_t{config['tolerance']}_h{config['horizon']}")
    config["Setup Time"] = offline_planning_time

    yaaf.mkdir(f"{results}/{environment}-{int(config['noise']*100)}%")
    with open(f"{results}/{environment}-{int(config['noise']*100)}%/meta_perseus_v{pomdp.spec.id.split('-v')[1]}.yaml", 'w+') as file: yaml.dump(config, file, default_flow_style=False)

    yaaf.mkdir(f"{resources}/{environment}-{int(config['noise'] * 100)}%")
    with open(f"{resources}/{environment}-{int(config['noise'] * 100)}%/meta_perseus_v{pomdp.spec.id.split('-v')[1]}.yaml", 'w+') as file: yaml.dump(config, file, default_flow_style=False)

    return policy, datetime.timedelta(seconds=offline_planning_time)

def check_perseus_policy(pomdp, config, resources, results):

    id = pomdp.spec.id
    environment = id.split("-v")[0]
    model_id = id.split("-v")[1]

    directory = f"{resources}/{environment}-{int(config['noise']*100)}%"
    if yaaf.isdir(directory):

        with open(f"{directory}/meta_perseus_v{model_id}.yaml") as file:
            meta = yaml.load(file, Loader=yaml.FullLoader)

        if config["beliefs"] != meta["beliefs"] or config["tolerance"] != config["tolerance"] or config["horizon"] != config["horizon"]:
            print(f"Warning: Different PERSEUS hyperparameters found in cached policy {directory}", flush=True)
    else:
        raise FileNotFoundError()

    directory = f"{results}/{environment}-{int(config['noise']*100)}%"

    if yaaf.isdir(directory):

        try:
            with open(f"{directory}/meta_perseus_v{model_id}.yaml") as file:
                meta = yaml.load(file, Loader=yaml.FullLoader)

            if config["beliefs"] != meta["beliefs"] or config["tolerance"] != config["tolerance"] or config["horizon"] != config["horizon"]:
                print(f"Warning: Different PERSEUS hyperparameters found in cached policy {directory}", flush=True)
        except FileNotFoundError:
            yaaf.mkdir(directory)
            with open(f"{directory}/meta_perseus_v{pomdp.spec.id.split('-v')[1]}.yaml", 'w+') as file:
                yaml.dump(config, file, default_flow_style=False)
    else:
        yaaf.mkdir(directory)
        with open(f"{directory}/meta_perseus_v{pomdp.spec.id.split('-v')[1]}.yaml", 'w+') as file:
            yaml.dump(config, file, default_flow_style=False)


def load_environment_config(filename, environment):

    config = pd.read_csv(filename, sep=",")

    if environment in config["environment"].to_list():
        config = config[config['environment'] == environment]
    else:
        config = config[config['environment'] == "default"]

    config = config.to_dict()
    new_config = {}
    for key in config:
        flat_key = list(config[key].keys())[0]
        new_key = key.replace(" ", "")
        new_config[new_key] = config[key][flat_key]

    return new_config

def pretrain(environment, config, resources, results, solve=True):

    possible_pomdps = config["models"]

    for model_id in range(possible_pomdps):

        print(f"{environment}-v{model_id}: POMDP", flush=True, end="")
        try:
            pomdp = load_pomdp(environment, model_id, config["noise"], resources, results)
            print(" -> Cached!")
        except FileNotFoundError:
            print(" -> Not in cache, creating... ", flush=True, end="")
            pomdp, time = create_pomdp(environment, model_id, config["noise"], resources, results)
            print(f" -> Created! ({time})")

        if solve:
            print(f"{environment}-v{model_id}: Perseus Policy", flush=True, end="")
            try:
                check_perseus_policy(pomdp, config, resources, results)
                print(" -> Cached!")
            except FileNotFoundError:
                print(" -> Not in cache, creating... ", flush=True, end="")
                _, time = create_perseus_policy(environment, pomdp, config, resources, results)
                print(f" -> Created! ({time})")

        del pomdp
        print(flush=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("environment", type=str)
    parser.add_argument("--resources", default="resources/environments", type=str)
    parser.add_argument("--results", default="resources/results", type=str)
    parser.add_argument("--config", default="config.csv", type=str)
    parser.add_argument("--task-only", action="store_true")

    opt = parser.parse_args()

    config = load_environment_config(opt.config, opt.environment)
    pretrain(opt.environment, config, opt.resources, opt.results, not opt.task_only)

    print("Done!", flush=True)
