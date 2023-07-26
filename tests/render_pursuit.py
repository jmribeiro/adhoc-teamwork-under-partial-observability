from backend.environments.pursuit import create, POSSIBLE_MODELS
from run_adhoc import FACTORY
from run_pretrain import load_environment_config
from test_utils import run_trial

grid = 4
task = "n/s"
teammate = "teammate aware"
model_id = POSSIBLE_MODELS.index((task, teammate))

world_size = (grid, grid)
max_view_distance = 1

environment = f"pursuit{world_size[0]}x{world_size[1]}"
config = load_environment_config("../config.csv", environment)

horizon = config["horizon"]

pomdp = create(model_id, max_view_distance, world_size, environment)

agent = FACTORY["Value Iteration"](pomdp, config)
print(run_trial(agent, pomdp, horizon, render=True)[0].sum())
