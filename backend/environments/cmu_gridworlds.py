import itertools

import numpy as np

import yaaf
from backend.environments.cmu_gridworld_games import layouts
from backend.environments.cmu_gridworld_games.layouts import TAGS, BLOCKED, GOAL, START, OPEN, possible_tasks
from backend.environments.overcooked import GreedyTeammate
from backend.mmdp import MultiAgentMarkovDecisionProcess
from backend.pomdp import PartiallyObservableMarkovDecisionProcess
from yaaf.environments.markov import MarkovDecisionProcess

def create(model_id, noise, environment):
    layout = layouts.load(environment)
    possible_models = possible_tasks(layout)
    task = possible_models[model_id]
    layout = layout_for_goal(task, layout)
    pomdp = CMUGridWorldPOMDP(environment, model_id, layout, noise)
    return pomdp

ABSORBENT = [-1, -1, -1, -1]
UP, DOWN, LEFT, RIGHT, STAY = range(5)
ACTION_MEANINGS = [
    "up",
    "down",
    "left",
    "right",
    "stay"
]
ACTION_SPACE = tuple(range(len(ACTION_MEANINGS)))
JOINT_ACTION_SPACE = list(itertools.product(ACTION_SPACE, repeat=2))

class CMUGridWorldPOMDP(PartiallyObservableMarkovDecisionProcess):

    def __init__(self, environment, model_id, layout, noise):

        mdp = generate_mdp(layout, model_id, environment)

        observations = self._generate_observations()
        observation_probabilities = self._generate_observation_probabilities(layout, mdp.states, mdp.action_meanings, observations, noise)

        id = f"{environment}-v{model_id}"

        super().__init__(id,
                         mdp.states, tuple(range(len(mdp.action_meanings))), observations,
                         mdp.transition_probabilities, observation_probabilities,
                         mdp.rewards, mdp.gamma, mdp.miu,
                         action_meanings=mdp.action_meanings)

        self.layout = layout

    def _generate_observations(self):
        observations = [
            np.array((up, down, left, right))
            for up in range(3)
            for down in range(3)
            for left in range(3)
            for right in range(3)
        ]
        return observations

    def _generate_observation_probabilities(self, layout, states, action_meanings, observations, noise):

        def get_possible_observations(next_state):

            # If we are in the absorbent state, see nothing in the four directions
            if tuple(next_state) == tuple(ABSORBENT):
                observations_per_direction = [
                    {0: 1.0},
                    {0: 1.0},
                    {0: 1.0},
                    {0: 1.0}
                ]
            else:

                num_rows, num_columns = layout.shape
                x0, y0, x1, y1 = next_state

                # Walls
                walls = [
                    y0 == 0 or ((y0 - 1) >= 0 and layout[(y0 - 1), x0] == BLOCKED),                             # Wall up
                    y0 == num_rows - 1 or ((y0 + 1) <= num_rows - 1 and layout[(y0 + 1), x0] == BLOCKED),       # Wall down
                    x0 == 0 or ((x0 - 1) >= 0 and layout[y0, (x0 - 1)] == BLOCKED),                             # Wall left
                    x0 == num_columns - 1 or ((x0 + 1) <= num_columns - 1 and layout[y0, (x0 + 1)] == BLOCKED), # Wall right
                ]

                # Agent
                adjacent_agent = [
                    y0 == y1 + 1 and x0 == x1,  # Agent up
                    y0 == y1 - 1 and x0 == x1,  # Agent down
                    y0 == y1 and x0 == x1 + 1,  # Agent left
                    y0 == y1 and x0 == x1 - 1   # Agent right
                ]
                assert np.array(adjacent_agent).sum() <= 1.0, "Agent seems to be detected in more than one direction"
                teammate_direction = adjacent_agent.index(True) if True in adjacent_agent else None

                observations_per_direction = []
                for direction, there_is_wall in enumerate(walls):

                    there_is_teammate = teammate_direction is not None and direction == teammate_direction

                    if there_is_wall:
                        possibilities = {
                            0: noise / 2,      # Detect nothing (error)
                            1: 1.0 - noise,    # Detect wall
                            2: noise / 2,      # Detect teammate as wall (error)
                        }

                    elif there_is_teammate:
                        possibilities = {
                            0: noise / 2,      # Detect nothing (error)
                            1: noise / 2,      # Detect wall instead of teammate (error)
                            2: 1.0 - noise     # Detect teammate
                        }
                    else:   # There's nothing
                        possibilities = {0: 1.0}    # Detect nothing

                    observations_per_direction.append(possibilities)

            return observations_per_direction

        num_actions = len(action_meanings)
        num_states = len(states)
        num_observations = len(observations)

        O = np.zeros((num_actions, num_states, num_observations))
        for a in range(num_actions):
            for y in range(num_states):
                possible_observations_per_direction = get_possible_observations(states[y])
                for up_flag, up_prob in possible_observations_per_direction[0].items():
                    for down_flag, down_prob in possible_observations_per_direction[1].items():
                        for left_flag, left_prob in possible_observations_per_direction[2].items():
                            for right_flag, right_prob in possible_observations_per_direction[3].items():
                                probability = up_prob * down_prob * left_prob * right_prob
                                observation = np.array([up_flag, down_flag, left_flag, right_flag])
                                z = yaaf.ndarray_index_from(observations, observation)
                                O[a, y, z] = probability
        return O

    def render(self, mode='human'):
        print_state(self.state, self.layout)

def generate_mdp(layout, model_id, environment):

    mmdp = generate_mmdp(layout, model_id, environment)

    # Reduce P
    P = np.zeros((mmdp.num_disjoint_actions, mmdp.num_states, mmdp.num_states))
    for a0 in range(mmdp.num_disjoint_actions):
        for x in range(mmdp.num_states):
            state = mmdp.states[x]
            pi1 = mmdp.teammate.policy(state)
            for y in range(mmdp.num_states):
                for a1 in range(mmdp.num_disjoint_actions):
                    ja = JOINT_ACTION_SPACE.index((a0, a1))
                    P[a0, x, y] += mmdp.P[ja, x, y] * pi1[a1]

#    P = np.around(P, 3)
    # Reduce R
    R = generate_rewards(mmdp.states, ACTION_MEANINGS, layout)

    mdp = MarkovDecisionProcess("nju-mdp-v0", mmdp.states, tuple(range(len(ACTION_MEANINGS))), P, R, mmdp.gamma, mmdp.miu, action_meanings=ACTION_MEANINGS)
    return mdp

def generate_mmdp(layout, model_id, environment):
    states = generate_states(layout)
    P_mmdp = generate_mmdp_transition_probabilities(states, ACTION_MEANINGS, layout)
    R_mmdp = generate_mmdp_rewards(states, layout)
    gamma = 0.95
    miu = generate_initial_state_distribution(states, layout)
    mmdp = MultiAgentMarkovDecisionProcess(f"{environment}-mmdp-v{model_id}", 2, states, ACTION_SPACE, P_mmdp, R_mmdp, gamma, miu, ACTION_MEANINGS)
    teammate = GreedyTeammate(mmdp)
    mmdp.add_teammate(teammate)
    return mmdp

def generate_states(layout):
    num_rows, num_columns = layout.shape
    states = []
    for x0 in range(num_columns):
        for y0 in range(num_rows):
            if layout[y0, x0] == BLOCKED: continue
            for x1 in range(num_columns):
                for y1 in range(num_rows):
                    if layout[y1, x1] == BLOCKED: continue
                    collision = x0 == x1 and y0 == y1
                    if collision: continue
                    state = np.array([x0, y0, x1, y1])
                    states.append(state)
    states = states + [np.array(ABSORBENT)]
    return states

def generate_mmdp_transition_probabilities(states, action_meanings, layout):

    num_rows, num_columns = layout.shape
    num_states = len(states)

    def next_cell(column, row, action_meaning):

        next_row, next_column = row, column

        if action_meaning == "up":
            next_row = max(0, next_row - 1)
        elif action_meaning == "down":
            next_row = min(num_rows - 1, next_row + 1)
        elif action_meaning == "left":
            next_column = max(0, next_column - 1)
        elif action_meaning == "right":
            next_column = min(num_columns - 1, next_column + 1)

        if layout[next_row, next_column] == BLOCKED:
            return column, row
        else:
            return next_column, next_row

    def compute_next_state(state, action_meaning_0, action_meaning_1):

        x0, y0, x1, y1 = state

        is_goal_state = solved(state, layout)
        is_reset_state = tuple(state) == tuple(ABSORBENT)

        # Absorbent Transition
        if is_goal_state or is_reset_state:
            return np.array(ABSORBENT)

        # Regular Transition
        else:

            next_x0, next_y0 = next_cell(x0, y0, action_meaning_0)
            next_x1, next_y1 = next_cell(x1, y1, action_meaning_1)

            # Teammate moves first
            teammate_collision = next_x1 == x0 and next_y1 == y0
            if teammate_collision: next_x1, next_y1 = x1, y1

            # Agent moves next
            agent_collision = next_x0 == next_x1 and next_y0 == next_y1
            if agent_collision: next_x0, next_y0 = x0, y0

            next_state = np.array([next_x0, next_y0, next_x1, next_y1])
            return np.array(next_state)

    num_actions = len(JOINT_ACTION_SPACE)
    P = np.zeros((num_actions, num_states, num_states))
    for a0, action_meaning_0 in enumerate(action_meanings):
        for a1, action_meaning_1 in enumerate(action_meanings):
            for x, state in enumerate(states):
                ja = JOINT_ACTION_SPACE.index((a0, a1))
                next_state = compute_next_state(state, action_meaning_0, action_meaning_1)
                y = yaaf.ndarray_index_from(states, next_state)
                P[ja, x, y] = 1.0

    return P

def generate_mmdp_rewards(states, layout):
    num_actions = len(JOINT_ACTION_SPACE)
    num_states = len(states)
    # Any movement costs -1
    R = np.full((num_states, num_actions), -1.0)
    # Actions on prey captured yield 100
    for x, state in enumerate(states):
        if solved(state, layout):
            R[x, :] = 100.0
    # Actions on reset yield 0
    y_reset = yaaf.ndarray_index_from(states, np.array(ABSORBENT))
    R[y_reset, :] = 0.0
    return R

def generate_initial_state_distribution(states, layout):
    initial_states = []
    for x, state in enumerate(states):
        x0, y0, x1, y1 = state
        agent = x0, y0
        teammate = x1, y1
        different_cells = agent != teammate
        agent_start = layout[y0, x0] == START
        teammate_start = layout[y1, x1] == START
        if different_cells and agent_start and teammate_start:
            initial_states.append(x)
    miu = np.zeros(len(states))
    for x in initial_states:
        miu[x] = 1 / len(initial_states)
    return miu

def generate_rewards(states, action_meanings, layout):
    num_actions = len(action_meanings)
    num_states = len(states)
    # Any movement costs -1
    R = np.full((num_states, num_actions), -1.0)
    # Actions on prey captured yield 100
    for x, state in enumerate(states):
        if solved(state, layout):
            R[x, :] = 100.0
    # Actions on reset yield 0
    y_reset = yaaf.ndarray_index_from(states, np.array(ABSORBENT))
    R[y_reset, :] = 0.0
    return R

def solved(state, layout):
    x0, y0, x1, y1 = state
    agent = x0, y0
    teammate = x1, y1
    return agent != teammate and layout[y0, x0] == GOAL and layout[y1, x1] == GOAL

def print_state(state, layout):
    x0, y0, x1, y1 = state
    num_rows, num_columns = layout.shape
    for y in range(num_rows):
        print("|", end="")
        for x in range(num_columns):
            if x0 == x and y0 == y and x1 == x and y1 == y:
                print(f"»|", end="")
            elif x0 == x and y0 == y:
                print(f"0|", end="")
            elif x1 == x and y1 == y:
                print(f"1|", end="")
            else:
                print(f"{TAGS[layout[y, x]]}|", end="")
        print()

def print_layout(layout):
    num_rows, num_columns = layout.shape
    for y in range(num_rows):
        print("|", end="")
        for x in range(num_columns):
            print(f"{TAGS[layout[y, x]]}|", end="")
        print()
    print()

def layout_for_goal(goal, layout):
    new_layout = np.copy(layout)
    ga_x, ga_y, gb_x, gb_y = goal
    num_rows, num_columns = layout.shape
    for x in range(num_columns):
        for y in range(num_rows):
            if layout[y, x] == GOAL and (x != ga_x and y != ga_y) and (x != gb_x and y != gb_y):
                new_layout[y, x] = OPEN
            else:
                new_layout[y, x] = layout[y, x]
    return new_layout

def test_mdp(model_id, environment):

    layout = layouts.load(environment)
    possible_models = possible_tasks(layout)
    task = possible_models[model_id]
    layout = layout_for_goal(task, layout)

    from run_adhoc import FACTORY
    print_layout(layout)
    print()

    env = generate_mdp(layout, model_id, environment)
    agent = FACTORY["Value Iteration"](env, {})
    state = env.reset()
    print_state(state, layout)
    print()

    terminal = False
    while not terminal:
        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        state = next_state
        print_state(state, layout)
        print(f"{env.action_meanings[action]}")
        print(f"{state}")
        print()

def test_mmdp(model_id, environment):

    layout = layouts.load(environment)
    possible_models = possible_tasks(layout)
    task = possible_models[model_id]
    layout = layout_for_goal(task, layout)

    from run_adhoc import FACTORY
    env = generate_mmdp(layout, model_id, environment)

    print_layout(layout)
    print()

    agent = FACTORY["Value Iteration"](env, {})
    state = env.reset()
    print_state(state, layout)
    print()

    terminal = False
    while not terminal:
        joint_action = agent.action(state)
        next_state, reward, terminal, info = env.joint_step(joint_action)

        print("###############\n")

        print_state(state, layout)
        print(f"{ACTION_MEANINGS[JOINT_ACTION_SPACE[joint_action][0]], ACTION_MEANINGS[JOINT_ACTION_SPACE[joint_action][1]]}")
        print(f"{state}")
        print()
        print_state(next_state, layout)
        print(f"{next_state}")
        print()

        state = next_state

def test_pomdp(model_id, noise, environment):

    #layout = layouts.load(environment)
    #possible_models = possible_tasks(layout)
    #task = possible_models[model_id]
    #layout = layout_for_goal(task, layout)
    #env = CMUGridWorldPOMDP(environment, model_id, layout, noise)

    pomdp = create(model_id, noise, environment)

    from run_adhoc import FACTORY
    print_layout(pomdp.layout)
    print()

    agent = FACTORY["Value Iteration"](pomdp, {})
    perseus = FACTORY["Perseus"](pomdp, {"horizon": 75, "beliefs": 5000, "tolerance": 0.01, "noise": noise})
    pomdp.reset()
    state = pomdp.state
    print_state(state, pomdp.layout)
    print()

    terminal = False
    while not terminal:
        action = agent.action(state)
        next_obs, reward, terminal, info = pomdp.step(action)
        state = pomdp.state
        print_state(state, pomdp.layout)
        print(f"{pomdp.action_meanings[action]}")
        print(f"{state}")
        print(f"{next_obs}")
        print()

def print_possible_tasks(environment):
    tasks = possible_tasks(environment)
    print(f"{environment} -> {len(tasks)}")
    for task in tasks:
        print(task)


if __name__ == '__main__':
    model_id, environment = 0, "isr"
    noise = 0.00
    #test_mdp(model_id, environment)
    #for environment in ["nju", "ntu", "isr"]: print_possible_tasks(environment)
    #test_pomdp(model_id, noise, environment)
    print_possible_tasks("pentagon")
    np.iscl