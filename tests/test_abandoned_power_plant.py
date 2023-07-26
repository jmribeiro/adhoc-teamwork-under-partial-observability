from backend.environments.abandoned_power_plant import create
from run_adhoc import FACTORY
from run_pretrain import load_environment_config
from test_utils import run_trial

render = True

for task in range(2):

    environment = "abandoned_power_plant"
    config = load_environment_config("config.csv", environment)
    horizon = config["horizon"]
    pomdp = create(task, 0.00)
    agent = FACTORY["Value Iteration"](pomdp, config)
    print(run_trial(agent, pomdp, horizon, render)[0].sum())
    #agent = FACTORY["Perseus"](pomdp, config)
    #print(run_trial(agent, pomdp, horizon, render)[0].sum())
