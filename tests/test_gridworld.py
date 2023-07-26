from backend.environments.gridworld import create
from run_adhoc import FACTORY
from run_pretrain import load_environment_config
from test_utils import run_trial


world_size = (3, 3)
render = True

for task in range(4):

    environment = "overcooked"

    config = load_environment_config("config.csv", environment)

    horizon = config["horizon"]

    pomdp = create(0, config["noise"], world_size, "gridworld")

    agent = FACTORY["Value Iteration"](pomdp, config)
    print(run_trial(agent, pomdp, horizon, render)[0].sum())

    agent = FACTORY["Perseus"](pomdp, config)
    print(run_trial(agent, pomdp, horizon, render)[0].sum())



