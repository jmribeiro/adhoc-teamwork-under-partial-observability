import pandas as pd
from backend.environments import factory
from run_pretrain import load_environment_config


def solve_setup_all_models(environment, config):
    possible_pomdps = config["models"]
    print(environment, flush=True)
    for model_id in range(possible_pomdps):
        print(f"\t{environment}-v{model_id}", flush=True, end=" - ")
        pomdp = factory[environment](model_id, config["noise"])
        try:
            pomdp.convert_to_ai_toolbox_model()
            print("Done!", flush=True)
        except Exception as e:
            print(e, flush=True)

if __name__ == '__main__':
    full_config = pd.read_csv("../config.csv", sep=",")
    blacklist = ["overcooked"]
    whitelist = ["isr", "nju", "ntu"]
    for e, environment in enumerate(full_config["environment"]):
        if environment not in blacklist and ((len(whitelist) > 0 and environment in whitelist) or len(whitelist) == 0):
            config = load_environment_config("../config.csv", environment)
            solve_setup_all_models(environment, config)
