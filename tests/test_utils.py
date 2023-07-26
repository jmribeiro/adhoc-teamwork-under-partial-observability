import time

from yaaf import Timestep
import numpy as np

def run_trial(agent, pomdp, horizon, render=False):

    full_observability = agent.name in ["Value Iteration"]

    rewards = np.zeros(horizon)
    decision_times = np.zeros(horizon)

    if full_observability:
        state = pomdp.reset()
    else:
        pomdp.reset()
        agent.reset()

    if render:
        pomdp.render()

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

        if render:
            pomdp.render()
            print(pomdp.state)
            print(next_obs)
            print()

        start = time.time()
        agent.reinforcement(timestep)
        end = time.time()

        # Register
        rewards[step] = reward
        decision_time = end - start
        decision_times[step] = decision_time

    return rewards, decision_times
