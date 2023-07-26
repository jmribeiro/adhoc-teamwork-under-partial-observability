from backend.environments.overcooked import generate_mmdp, draw_state

from backend.mmdp import MultiAgentMarkovDecisionProcess
from run_adhoc import FACTORY
from run_pretrain import load_environment_config, load_pomdp, create_pomdp
from test_utils import run_trial

def test_mmdp():
    render = True
    mmdp: MultiAgentMarkovDecisionProcess = generate_mmdp("optimal policy")
    agent = FACTORY["Value Iteration"](mmdp, {}, {})

    state = mmdp.reset()
    if render:
        draw_state(state)
        print()

    for _ in range(30):
        ja = agent.action(state)
        joint_action = mmdp.joint_action_space[ja]
        action = joint_action[0]
        next_state, reward, terminal, info = mmdp.step(action)
        state = next_state
        if render:
            draw_state(state)
            print(reward)
            print()

def test_pomdp():

    render = True

    for task in range(4):

        environment = "overcooked"

        config = load_environment_config("../config.csv", environment)

        horizon = config["horizon"]
        noise = config["noise"]

        try:
            pomdp = load_pomdp("overcooked", task, noise, "../resources/environments", "../resources/results")
        except:
            pomdp, _ = create_pomdp("overcooked", task, noise, "../resources/environments", "../resources/results")

        agent = FACTORY["Value Iteration"](pomdp, config)
        print(run_trial(agent, pomdp, horizon, render)[0].sum())

        agent = FACTORY["Perseus"](pomdp, config)
        print(run_trial(agent, pomdp, horizon, render)[0].sum())

if __name__ == '__main__':
    test_mmdp()
