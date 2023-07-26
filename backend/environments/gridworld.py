import random

import numpy as np
import yaaf
from random import randint
from typing import Tuple
from backend.pomdp import PartiallyObservableMarkovDecisionProcess

def create(model_id, noise, world_size, name="gridworld"):
    gridsize = world_size[0]
    assert world_size[0] == world_size[1]
    goal = goal_from_id(model_id, gridsize)
    id = f"{name}-v{model_id}"
    pomdp = GridWorldPOMDP(id, world_size, goal, noise)
    return pomdp

##########################
# Environment code below #
##########################

ABSORBENT = [-1, -1, -1, -1]
SEED = 239758   # makes sure the randomly generated goals are always the same (nothing to do with the evaluation itself)
MAX_TASKS = 64
PERSEUS_FAILS_TO_SOLVE = {
    3: [],
    4: [],
    5: [],
    6: []
}
def generate_mdp(world_size, noise, goals):
    states = generate_state_space(world_size[1], world_size[0])
    action_meanings = ("up", "down", "left", "right", "stay")
    transition_probabilities = generate_transition_probabilities_cube(world_size, states, action_meanings, noise, goals)
    rewards_matrix = generate_rewards_matrix(states, action_meanings, goals)
    miu = generate_miu(states, goals)
    return states, action_meanings, transition_probabilities, rewards_matrix, miu

def generate_state_space(num_rows, num_columns):
    states = [
        np.array((x_agent, y_agent, x_teammate, y_teammate))
        for x_agent in range(num_columns) for y_agent in range(num_rows)
        for x_teammate in range(num_columns) for y_teammate in range(num_rows)
    ] + [np.array(ABSORBENT)]
    return states

def generate_transition_probabilities_cube(world_size, states, action_meanings, movement_failure_probability, goals):

    def greedy_direction(source, target):
        num_columns, num_rows = world_size[1], world_size[0]
        dx_forward = max(0, source[0] - target[0])
        dx_backward = min(target[0] - source[0], num_columns - 1)
        dy_forward = max(0, source[1] - target[1])
        dy_backward = min(target[1] - source[1], num_rows - 1)
        if dx_forward < dx_backward: return "right"
        elif dx_backward < dx_forward: return "left"
        elif dy_forward < dy_backward: return "down"
        elif dy_backward < dy_forward: return "up"
        else: return "stay"

    def closest_button(source):
        from scipy.spatial.distance import cityblock
        button_a, button_b = goals[:2], goals[2:]
        distance_a, distance_b = cityblock(source, button_a), cityblock(source, button_b)
        return button_a if distance_a <= distance_b else button_b

    def get_next_cell(current_cell, action_meaning):
        x, y = current_cell
        num_rows, num_columns = world_size
        last_row, last_column = num_rows - 1, num_columns - 1
        if action_meaning == "up": next_cell = x, max(y - 1, 0)
        elif action_meaning == "down": next_cell = x, min(y + 1, last_row)
        elif action_meaning == "left": next_cell = max(x - 1, 0), y
        elif action_meaning == "right": next_cell = min(x + 1, last_column), y
        else: next_cell = current_cell
        return next_cell

    def get_possible_next_states(state, action_meaning):

        button_a, button_b = goals[:2], goals[2:]
        goal_states = [np.array(button_a + button_b), np.array(button_b + button_a)]
        current_agent_cell = tuple(state[:2])
        current_teammate_cell = tuple(state[2:])

        is_goal_state = np.array_equal(state, goal_states[0]) or np.array_equal(state, goal_states[1])
        is_reset_state = np.array_equal(state, np.array(ABSORBENT))

        # Absorbent Transition
        if is_goal_state or is_reset_state:
            agent_possibilities = {(-1, -1): 1.0}
            teammate_possibilities = {(-1, -1): 1.0}

        # Regular Transition
        else:

            # Agent
            next_agent_cell = get_next_cell(current_agent_cell, action_meaning)
            if next_agent_cell == current_agent_cell:
                agent_possibilities = {
                    current_agent_cell: 1.0
                }
            else:
                agent_possibilities = {
                    current_agent_cell: movement_failure_probability,
                    next_agent_cell: 1.0 - movement_failure_probability
                }

            # Teammate
            closest_button_to_teammate = closest_button(current_teammate_cell)
            teammate_action_meaning = greedy_direction(current_teammate_cell, closest_button_to_teammate)
            next_teammate_cell = get_next_cell(current_teammate_cell, teammate_action_meaning)
            if next_teammate_cell == current_teammate_cell:
                teammate_possibilities = {next_teammate_cell: 1.0}
            else:
                teammate_possibilities = {
                    current_teammate_cell: movement_failure_probability,
                    next_teammate_cell: 1.0 - movement_failure_probability
                }

        # Independent events
        possible_next_states = {}
        for agent_cell, agent_movement_probability in agent_possibilities.items():
            for teammate_cell, teammate_movement_probability in teammate_possibilities.items():
                invalid = (agent_cell == (-1, -1) and teammate_cell != (-1, -1)) or (agent_cell != (-1, -1) and teammate_cell == (-1, -1))
                if not invalid:
                    next_state_tuple = agent_cell + teammate_cell
                    transition_probability = agent_movement_probability * teammate_movement_probability
                    possible_next_states[next_state_tuple] = transition_probability

        return possible_next_states

    num_states = len(states)
    num_actions = len(action_meanings)

    P = np.zeros((num_actions, num_states, num_states))
    for a in range(num_actions):
        for x in range(num_states):
            state = states[x]
            action_meaning = action_meanings[a]
            possible_next_states = get_possible_next_states(state, action_meaning)
            for next_state_tuple, transition_probability in possible_next_states.items():
                next_state = np.array(next_state_tuple)
                y = yaaf.ndarray_index_from(states, next_state)
                P[a, x, y] = transition_probability

    return P

def generate_rewards_matrix(states, action_meanings, goals):
    num_actions = len(action_meanings)
    num_states = len(states)
    R = np.full((num_states, num_actions), -1.0)
    button_a, button_b = goals[:2], goals[2:]
    goal_states = [np.array(button_a + button_b), np.array(button_b + button_a)]
    for goal_state in goal_states:
        y = yaaf.ndarray_index_from(states, goal_state)
        R[y, :] = 100.0
    y_reset = yaaf.ndarray_index_from(states, np.array(ABSORBENT))
    R[y_reset, :] = 0.0
    return R

def generate_miu(states, goals):
    starting_states = []
    button_a, button_b = goals[:2], goals[2:]
    goal_states = [(button_a + button_b), (button_b + button_a)]
    for x, state in enumerate(states):
        ended = tuple(state) in goal_states or tuple(state) == tuple(ABSORBENT)
        if not ended:
            starting_states.append(x)
    miu = np.zeros(len(states))
    for x in starting_states:
        miu[x] = 1 / len(starting_states)
    return miu

class GridWorldPOMDP(PartiallyObservableMarkovDecisionProcess):

    def __init__(self,
                 id,
                 world_size: Tuple,
                 goals: Tuple,
                 noise: float,
                 discount_factor: float = 0.95):

        states, action_meanings, transition_probabilities, rewards_matrix, miu = generate_mdp(world_size, noise, goals)
        observations = self._generate_observations()
        observation_probabilities = self._generate_observation_probabilities(world_size, states, action_meanings, observations, noise)

        super(GridWorldPOMDP, self).__init__(id,
                         states, tuple(range(len(action_meanings))), observations,
                         transition_probabilities, observation_probabilities,
                         rewards_matrix, discount_factor, miu,
                         action_meanings=action_meanings)

        self._world_size = world_size
        self._goals = goals

    def _generate_observations(self):
        observations = [
            np.array((up, down, left, right))
            for up in range(3)
            for down in range(3)
            for left in range(3)
            for right in range(3)
        ]
        return observations

    def _generate_observation_probabilities(self, world_size, states, action_meanings, observations, sensor_failure_probability):

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
                num_rows, num_columns = world_size
                agent_cell, teammate_cell = next_state[:2], next_state[2:4]
                (x1, y1), (x2, y2) = agent_cell, teammate_cell

                # Walls
                walls = [
                    y1 == 0,                # Wall up
                    y1 == num_rows - 1,     # Wall down
                    x1 == 0,                # Wall left
                    x1 == num_columns - 1   # Wall right
                ]

                # Agent
                adjacent_agent = [
                    y1 == y2 + 1 and x1 == x2,  # Agent up
                    y1 == y2 - 1 and x1 == x2,  # Agent down
                    y1 == y2 and x1 == x2 + 1,  # Agent left
                    y1 == y2 and x1 == x2 - 1   # Agent right
                ]
                assert np.array(adjacent_agent).sum() <= 1.0, "Agent seems to be detected in more than one direction"
                teammate_direction = adjacent_agent.index(True) if True in adjacent_agent else None

                observations_per_direction = []
                for direction, there_is_wall in enumerate(walls):

                    there_is_teammate = teammate_direction is not None and direction == teammate_direction

                    if there_is_wall:
                        possibilities = {
                            0: sensor_failure_probability / 2,      # Detect nothing (error)
                            1: 1.0 - sensor_failure_probability,    # Detect wall
                            2: sensor_failure_probability / 2,      # Detect teammate as wall (error)
                        }

                    elif there_is_teammate:
                        possibilities = {
                            0: sensor_failure_probability / 2,      # Detect nothing (error)
                            1: sensor_failure_probability / 2,      # Detect wall instead of teammate (error)
                            2: 1.0 - sensor_failure_probability     # Detect teammate
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
        draw(self.state, self._world_size, self._goals)

def goal_from_id(model_id, gridsize):
    assert model_id < MAX_TASKS
    if model_id in PERSEUS_FAILS_TO_SOLVE[gridsize]:
        model_id = 32 + (PERSEUS_FAILS_TO_SOLVE[gridsize].index(model_id) + 1)
    possible_goals = all_possible_goals(gridsize)
    random.seed(SEED)
    always_the_same_goals = random.sample(possible_goals, MAX_TASKS)
    goal = always_the_same_goals[model_id]
    return goal

def all_possible_goals(gridsize):
    # Dumb way of getting all possible goals
    possible_goals = list(set([random_goal(gridsize) for _ in range(99999)]))
    possible_goals.sort()
    return possible_goals

def random_goal(gridsize):
    x1, y1 = randint(0, gridsize-1), randint(0, gridsize-1)
    x2, y2 = x1, y1
    while x2 == x1 and y2 == y1:
        x2, y2 = randint(0, gridsize - 1), randint(0, gridsize - 1)
    return x1, y1, x2, y2

def draw(state, world_size, initial_positions=None, goals=None):
    num_rows, num_columns = world_size
    display = ""
    display += f" {' '.join([f' {i} ' for i in range(num_columns)])}\n"
    for row in range(num_rows):
        display += draw_row_border(num_columns)
        display += "|"
        for col in range(num_columns):
            cell = col, row
            cell_display = draw_cell(cell, state, initial_positions, goals)
            display += " " if col == 0 else "+ "
            display += f"{cell_display} "
        display += f"| {row}\n"
    display += draw_row_border(num_columns)
    print(display)

def draw_cell(cell, state, initial_positions=None, goals=None):

    x, y = cell
    agent_x, agent_y = tuple(state[:2])
    teammate_x, teammate_y = tuple(state[2:])
    if agent_x == x and agent_y == y: return 'A'
    if teammate_x == x and teammate_y == y: return 'T'

    if initial_positions is not None:
        start_agent_x, start_agent_y = initial_positions[:2]
        start_teammate_x, start_teammate_y = initial_positions[2:]
        if start_agent_x == x and start_agent_y == y: return 'o'
        if start_teammate_x == x and start_teammate_y == y: return '*'

    if goals is not None:
        button_a_x, button_a_y = goals[:2]
        button_b_x, button_b_y = goals[2:]
        if button_a_x == x and button_a_y == y or button_b_x == x and button_b_y == y:
            return "G"

    return " "

def draw_row_border(columns):
    border = ""
    for y in range(columns): border += "+---"
    border += "+\n"
    return border

