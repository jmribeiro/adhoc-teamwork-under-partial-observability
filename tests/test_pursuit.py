from backend.environments.pursuit import create
from run_adhoc import FACTORY
from run_pretrain import load_environment_config
from test_utils import run_trial

grid = 3
render = True
part = "task"
noise = 0.2

if part == "task":
    possible = 2
elif part == "teammate":
    possible = 4
elif part == "both":
    possible = 8
else:
    raise ValueError()

for model_id in range(possible):

    world_size = (grid, grid)

    environment = f"pursuit-{part}"
    config = load_environment_config("../config.csv", environment)

    horizon = config["horizon"]

    pomdp = create(model_id, noise, world_size, environment)

    agent = FACTORY["Value Iteration"](pomdp, config)
    print(run_trial(agent, pomdp, horizon, render)[0].sum())

    agent = FACTORY["Perseus"](pomdp, config)
    print(run_trial(agent, pomdp, horizon, render)[0].sum())
