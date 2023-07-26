import random
from itertools import product

import numpy as np

import cv2

from backend.mmdp import MultiAgentMarkovDecisionProcess
from backend.pomdp import PartiallyObservableMarkovDecisionProcess
from yaaf.agents import Agent, RandomAgent
from yaaf.policies import deterministic_policy

import yaaf
from backend.environments.overcooked_ai_py.mdp.actions import Direction
from backend.environments.overcooked_ai_py.mdp.overcooked_mdp import Recipe, PlayerState, ObjectState, SoupState
from backend.environments.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

def create(model_id, noise, name):
    assert model_id < len(TEAMMATE_FACTORY.keys())
    assert noise == 0.00
    possible_teammates = list(TEAMMATE_FACTORY.keys())
    teammate = possible_teammates[model_id]
    id = f"{name}-v{model_id}"
    pomdp = OvercookedPOMDP(id, teammate)
    return pomdp

##########################
# Environment code below #
##########################

class OvercookedPOMDP(PartiallyObservableMarkovDecisionProcess):

    def __init__(self,
                 id: str,
                 teammate,
                 discount_factor: float = 0.95):

        states, action_meanings, transition_probabilities, rewards_matrix, miu = generate_mdp(teammate)
        observations = self._generate_observations(states)
        observation_probabilities = self._generate_observation_probabilities(states, action_meanings, observations)

        super().__init__(id,
                         states, tuple(range(len(action_meanings))), observations,
                         transition_probabilities, observation_probabilities,
                         rewards_matrix, discount_factor, miu,
                         action_meanings=action_meanings)
        self.teammate = teammate

    @staticmethod
    def observation_fn(state):
        return state

    def _generate_observations(self, states):
        return states

    def _generate_observation_probabilities(self, states, action_meanings, observations):

        num_actions = len(action_meanings)
        num_states = len(states)
        num_observations = len(observations)

        O = np.zeros((num_actions, num_states, num_observations))
        for a in range(num_actions):
            for y in range(num_states):
                next_state = states[y]
                z = yaaf.ndarray_index_from(states, next_state)
                O[a, y, z] = 1.0

        return O

    def render(self, mode='human'):

        Recipe.configure({})
        vis = StateVisualizer()

        def make_hand(object, position):
            if object != "soup":
                return ObjectState(object, position)
            else:
                return SoupState(position, ingredients=[ObjectState("onion", position) for _ in range(3)])

        p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered = self.state
        GRID = [
            ["X", "X", "X", "P", "X"],
            ["O", " ", "X", " ", "X"],
            ["D", " ", "X", " ", "X"],
            ["X", "X", "X", "S", "X"],
        ]
        P1_TOP, P1_BOTTOM = (1, 1), (1, 2)
        P2_TOP, P2_BOTTOM = (3, 1), (3, 2)
        TOP_BALCONY, BOTTOM_BALCONY = (2, 1), (2, 2)
        PAN = (3, 0)

        # Positions
        if p1 == TOP:
            p1_pos = P1_TOP
        elif p1 == BOTTOM:
            p1_pos = P1_BOTTOM
        else:
            raise ValueError()
        if p2 == TOP:
            p2_pos = P2_TOP
        elif p2 == BOTTOM:
            p2_pos = P2_BOTTOM
        else:
            raise ValueError()

        # Hands
        if hand_1 == NOTHING:
            p1_hand = None
        elif hand_1 == ONION:
            p1_hand = make_hand("onion", p1_pos)
        elif hand_1 == PLATE:
            p1_hand = make_hand("dish", p1_pos)
        else:
            raise ValueError()
        if hand_2 == NOTHING:
            p2_hand = None
        elif hand_2 == ONION:
            p2_hand = make_hand("onion", p2_pos)
        elif hand_2 == PLATE:
            p2_hand = make_hand("dish", p2_pos)
        elif hand_2 == SOUP:
            p2_hand = make_hand("soup", p2_pos)
        else:
            raise ValueError()

        objects = {}
        if top_balcony == NOTHING:
            pass
        elif top_balcony == ONION:
            objects[TOP_BALCONY] = ObjectState("onion", TOP_BALCONY)
        elif top_balcony == PLATE:
            objects[TOP_BALCONY] = ObjectState("dish", TOP_BALCONY)
        else:
            raise ValueError()
        if bottom_balcony == NOTHING:
            pass
        elif bottom_balcony == ONION:
            objects[BOTTOM_BALCONY] = ObjectState("onion", BOTTOM_BALCONY)
        elif bottom_balcony == PLATE:
            objects[BOTTOM_BALCONY] = ObjectState("dish", BOTTOM_BALCONY)
        else:
            raise ValueError()
        if soup_pan > 0:
            objects[PAN] = SoupState(PAN, ingredients=[ObjectState("onion", PAN) for _ in range(soup_pan)],
                                     cooking_tick=soup_pan, cook_time=3)
        elif soup_pan == -1:
            objects[PAN] = SoupState(PAN, ingredients=[ObjectState("onion", PAN) for _ in range(3)],
                                     cooking_tick=3, cook_time=3)

        p1 = PlayerState(p1_pos, Direction.SOUTH, p1_hand)
        p2 = PlayerState(p2_pos, Direction.SOUTH, p2_hand)
        players = (p1, p2)

        vis.render_clean_state(players, objects, GRID, None, show=True)

TEAMMATE_FACTORY = {
    "optimal policy": lambda mmdp, index: GreedyTeammate(mmdp, index),
    "upper": lambda mmdp, index: UpperTeammate(mmdp),
    "downer": lambda mmdp, index: DownerTeammate(mmdp),
    "random policy": lambda mmdp, index: RandomAgent(mmdp.num_disjoint_actions),
}
UP, DOWN, NOOP, ACT = range(4)
ACTION_MEANINGS = [
    "up",
    "down",
    "noop",
    "act",
]
ACTION_SPACE = tuple(range(len(ACTION_MEANINGS)))
JOINT_ACTION_SPACE = list(product(ACTION_SPACE, repeat=2))
TOP, BOTTOM = range(2)
ONION_SUPPLY, PLATE_SUPPLY = range(2)
NOTHING, ONION, PLATE = range(3)
SOUP = -1

def generate_mdp(teammate):

    mmdp = generate_mmdp(teammate)
    states, miu = mmdp.states, mmdp.miu
    action_meanings = ACTION_MEANINGS

    num_states = mmdp.num_states
    num_actions = len(action_meanings)
    P_mmdp = mmdp.P
    P = np.zeros((num_actions, num_states, num_states))
    for a0 in range(num_actions):
        for x in range(num_states):
            state = states[x]
            pi1 = mmdp.teammate.policy(state)
            for y in range(num_states):
                for a1 in range(num_actions):
                    ja = JOINT_ACTION_SPACE.index((a0, a1))
                    P[a0, x, y] += P_mmdp[ja, x, y] * pi1[a1]

    R = generate_rewards(states)

    return states, action_meanings, P, R, miu

def generate_mmdp(teammate):
    states = generate_state_space()
    gamma = 0.95
    P_mmdp = generate_mmdp_transition_probabilities_cube(states)
    R = generate_mmdp_rewards(states)
    miu = generate_miu(states)
    id = list(TEAMMATE_FACTORY.keys()).index(teammate)
    mmdp = MultiAgentMarkovDecisionProcess(f"overcooked-mmdp-v{id}", 2, states, ACTION_SPACE, P_mmdp, R, gamma, miu, ACTION_MEANINGS)
    if teammate is not None:
        teammate = TEAMMATE_FACTORY[teammate](mmdp, 1)
        mmdp.add_teammate(teammate)
    return mmdp

def generate_state_space():

    states = [
        np.array((p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered))
        for p1 in range(2)              # Top, bottom
        for p2 in range(2)              # Top, bottom
        for hand_1 in range(3)          # Nothing, Plate, Onion
        for hand_2 in range(-1, 3)      # Nothing, Plate, Onion, Soup
        for top_balcony in range(3)     # Nothing, Plate, Onion
        for bottom_balcony in range(3)  # Nothing, Plate, Onion
        for soup_pan in range(-1, 3)    # Nothing, 1, 2, 3, Cooked
        for soup_delivered in range(2)  # yes/no
    ]

    return states

def generate_mmdp_transition_probabilities_cube(states):
    
    def transition(state, action_meaning_1, action_meaning_2):

        p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered = state
        
        if not soup_delivered:
                
            def next_position(current_position, action_meaning):
                if current_position == TOP and action_meaning == "down":
                    return BOTTOM
                elif current_position == BOTTOM and action_meaning == "up":
                    return TOP
                else:
                    return current_position
                
            # next_p1
            next_p1 = next_position(p1, action_meaning_1)
    
            # next_p2
            next_p2 = next_position(p2, action_meaning_2)
    
            # I set these to know if a1 has dropped anything before a2
            next_top_balcony = None
            next_bottom_balcony = None
            next_soup_delivered = 0
            
            # next_m1
            if action_meaning_1 == "act":
                if hand_1 == NOTHING:
                    # Grab
                    next_hand_1 = ONION if p1 == TOP else PLATE
                else:
                    # Drop
                    if p1 == TOP and top_balcony == NOTHING:
                        next_top_balcony = hand_1
                        next_hand_1 = NOTHING
                    elif p1 == BOTTOM and bottom_balcony == NOTHING:
                        next_bottom_balcony = hand_1
                        next_hand_1 = NOTHING
                    else:
                        next_hand_1 = hand_1
            else:
                next_hand_1 = hand_1
    
            # next_m2
            next_soup_pan = soup_pan
            if action_meaning_2 == "act":
                if hand_2 == NOTHING:
                    # Grab
                    if p2 == TOP and top_balcony != NOTHING:
                        next_hand_2 = top_balcony
                        next_top_balcony = NOTHING
                    elif p2 == BOTTOM and bottom_balcony != NOTHING:
                        next_hand_2 = bottom_balcony
                        next_bottom_balcony = NOTHING
                    else:
                        next_hand_2 = NOTHING
                elif hand_2 == PLATE and p2 == TOP and soup_pan == SOUP:
                    next_hand_2 = SOUP
                    next_soup_pan = NOTHING
                elif hand_2 == ONION and p2 == TOP and soup_pan != SOUP and soup_pan < 3:
                    next_hand_2 = NOTHING
                    next_soup_pan = SOUP if (soup_pan + 1) == 3 else (soup_pan + 1)
                elif hand_2 == SOUP:
                    if p2 == BOTTOM:
                        next_hand_2 = NOTHING
                        next_soup_delivered = 1
                    else:
                        next_hand_2 = SOUP
                else:
                    # Drop
                    if p2 == TOP and top_balcony == NOTHING and next_top_balcony is None:
                        next_top_balcony = hand_2
                        next_hand_2 = NOTHING
                    elif p2 == BOTTOM and bottom_balcony == NOTHING and next_bottom_balcony is None:
                        next_bottom_balcony = hand_2
                        next_hand_2 = NOTHING
                    else:
                        next_hand_2 = hand_2
            else:
                next_hand_2 = hand_2
    
            next_top_balcony = next_top_balcony if next_top_balcony is not None else top_balcony
            next_bottom_balcony = next_bottom_balcony if next_bottom_balcony is not None else bottom_balcony
    
            next_state = np.array((next_p1, next_p2, next_hand_1, next_hand_2, next_top_balcony, next_bottom_balcony, next_soup_pan, next_soup_delivered))
            y = yaaf.ndarray_index_from(states, next_state)
            return y
        else:
            next_state = np.array((p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, 0))
            y = yaaf.ndarray_index_from(states, next_state)
            return y
        
    num_states = len(states)
    num_actions = len(JOINT_ACTION_SPACE)
    P = np.zeros((num_actions, num_states, num_states))
    for a1, action_meaning_1 in enumerate(ACTION_MEANINGS):
        for a2, action_meaning_2 in enumerate(ACTION_MEANINGS):
            for x, state in enumerate(states):
                ja = JOINT_ACTION_SPACE.index((a1, a2))
                y = transition(state, action_meaning_1, action_meaning_2)
                P[ja, x, y] = 1.0
    return P

def generate_mmdp_rewards(states):
    num_states = len(states)
    num_actions = len(JOINT_ACTION_SPACE)
    R = np.zeros((num_states, num_actions))
    for a1, action_meaning_1 in enumerate(ACTION_MEANINGS):
        for a2, action_meaning_2 in enumerate(ACTION_MEANINGS):
            for x, state in enumerate(states):
                ja = JOINT_ACTION_SPACE.index((a1, a2))
                p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered = state
                if action_meaning_2 == "act" and hand_2 == SOUP and p2 == BOTTOM:
                    R[x, ja] = 15.0
                else:
                    R[x, ja] = -1.0
    return R

def generate_rewards(states):
    num_states = len(states)
    num_actions = len(ACTION_MEANINGS)
    R = np.zeros((num_states, num_actions))
    for x, state in enumerate(states):
        p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered = state
        if soup_delivered:
            R[x, :] = 15
        else:
            R[x, :] = -1.0
    return R

def generate_miu(states):

    def valid_initial_state(state):
        p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered = state
        return\
            hand_1 == NOTHING and \
            hand_2 == NOTHING and \
            top_balcony == NOTHING and \
            bottom_balcony == NOTHING and \
            soup_pan == NOTHING and \
            soup_delivered == 0

    num_states = len(states)
    miu = np.zeros(num_states)
    initial_states = []
    for x, state in enumerate(states):
        if valid_initial_state(state):
            initial_states.append(x)

    num_initial_states = len(initial_states)
    for x in initial_states:
        miu[x] = 1 / num_initial_states

    return miu

def draw_state(state):

    p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered = state

    if p1 == TOP:
        p1_bot = "_"
        if hand_1 == NOTHING:
            p1_top = "n"
        elif hand_1 == ONION:
            p1_top = "o"
        elif hand_1 == PLATE:
            p1_top = "p"
        else:
            raise ValueError("Invalid item")
    else:
        p1_top = "_"
        if hand_1 == NOTHING:
            p1_bot = "n"
        elif hand_1 == ONION:
            p1_bot = "o"
        elif hand_1 == PLATE:
            p1_bot = "p"
        else:
            raise ValueError("Invalid item")

    if p2 == TOP:
        p2_bot = "_"
        if hand_2 == NOTHING:
            p2_top = "n"
        elif hand_2 == ONION:
            p2_top = "o"
        elif hand_2 == PLATE:
            p2_top = "p"
        elif hand_2 == SOUP:
            p2_top = "s"
        else:
            raise ValueError("Invalid Item")
    else:
        p2_top = "_"
        if hand_2 == NOTHING:
            p2_bot = "n"
        elif hand_2 == ONION:
            p2_bot = "o"
        elif hand_2 == PLATE:
            p2_bot = "p"
        elif hand_2 == SOUP:
            p2_bot = "s"
        else:
            raise ValueError("Invalid Item")

    if top_balcony == NOTHING:
        balcony_top = "_"
    elif top_balcony == ONION:
        balcony_top = "o"
    elif top_balcony == PLATE:
        balcony_top = "p"
    else:
        raise ValueError("Impossible item")

    if bottom_balcony == NOTHING:
        balcony_bot = "_"
    elif bottom_balcony == ONION:
        balcony_bot = "o"
    elif bottom_balcony == PLATE:
        balcony_bot = "p"
    else:
        raise ValueError("Impossible item")

    state_render = f"****{soup_pan}*" + "\n" + \
                   f"*>{p1_top}{balcony_top}{p2_top}*" + "\n" + \
                   f"*>{p1_bot}{balcony_bot}{p2_bot}*" + "\n" + \
                   f"****v*"

    state_render = state_render.replace("_", " ")

    print(state_render, flush=True)

def render(state, filename, show=False):

    Recipe.configure({})
    vis = StateVisualizer()

    def make_hand(object, position):
        if object != "soup":
            return ObjectState(object, position)
        else:
            return SoupState(position, ingredients=[ObjectState("onion", position) for _ in range(3)])

    p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered = state
    GRID = [
        ["X", "X", "X", "P", "X"],
        ["O", " ", "X", " ", "X"],
        ["D", " ", "X", " ", "X"],
        ["X", "X", "X", "S", "X"],
    ]
    P1_TOP, P1_BOTTOM = (1, 1), (1, 2)
    P2_TOP, P2_BOTTOM = (3, 1), (3, 2)
    TOP_BALCONY, BOTTOM_BALCONY = (2, 1), (2, 2)
    PAN = (3, 0)

    # Positions
    if p1 == TOP: p1_pos = P1_TOP
    elif p1 == BOTTOM: p1_pos = P1_BOTTOM
    else: raise ValueError()
    if p2 == TOP: p2_pos = P2_TOP
    elif p2 == BOTTOM: p2_pos = P2_BOTTOM
    else: raise ValueError()

    # Hands
    if hand_1 == NOTHING: p1_hand = None
    elif hand_1 == ONION: p1_hand = make_hand("onion", p1_pos)
    elif hand_1 == PLATE: p1_hand = make_hand("dish", p1_pos)
    else: raise ValueError()
    if hand_2 == NOTHING: p2_hand = None
    elif hand_2 == ONION: p2_hand = make_hand("onion", p2_pos)
    elif hand_2 == PLATE: p2_hand = make_hand("dish", p2_pos)
    elif hand_2 == SOUP: p2_hand = make_hand("soup", p2_pos)
    else: raise ValueError()

    objects = {}
    if top_balcony == NOTHING: pass
    elif top_balcony == ONION: objects[TOP_BALCONY] = ObjectState("onion", TOP_BALCONY)
    elif top_balcony == PLATE: objects[TOP_BALCONY] = ObjectState("dish", TOP_BALCONY)
    else: raise ValueError()
    if bottom_balcony == NOTHING: pass
    elif bottom_balcony == ONION: objects[BOTTOM_BALCONY] = ObjectState("onion", BOTTOM_BALCONY)
    elif bottom_balcony == PLATE: objects[BOTTOM_BALCONY] = ObjectState("dish", BOTTOM_BALCONY)
    else: raise ValueError()
    if soup_pan > 0:
        objects[PAN] = SoupState(PAN, ingredients=[ObjectState("onion", PAN) for _ in range(soup_pan)],
                                 cooking_tick=soup_pan, cook_time=3)
    elif soup_pan == -1:
        objects[PAN] = SoupState(PAN, ingredients=[ObjectState("onion", PAN) for _ in range(3)],
                                 cooking_tick=3, cook_time=3)

    p1 = PlayerState(p1_pos, Direction.SOUTH, p1_hand)
    p2 = PlayerState(p2_pos, Direction.SOUTH, p2_hand)
    players = (p1, p2)
    if filename is not None:
        parts = filename.split("/")[:-1]
        directory = "/".join(parts)
        yaaf.mkdir(directory)
    vis.render_clean_state(players, objects, GRID, filename, show)

def make_video(episode_directory, fps=5):

    img_array = []
    step = 0
    while True:
        filename = f"{episode_directory}/frames/t{step}.png"
        img = cv2.imread(filename)
        if img is not None:
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
            step += 1
        else:
            break

    out = cv2.VideoWriter(f'{episode_directory}/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

class GreedyTeammate(Agent):

    def __init__(self, mmdp, index=1):
        super(GreedyTeammate, self).__init__("Greedy Teammate")
        self.mmdp = mmdp
        self._policy = mmdp.disjoint_policies[index]

    def policy(self, state):
        x = self.mmdp.state_index(state)
        return self._policy[x]

    def _reinforce(self, timestep):
        pass

class UpperTeammate(GreedyTeammate):

    def __init__(self, mmdp, stubbornness=0.9):
        super(UpperTeammate, self).__init__(mmdp)
        self.stubbornness = stubbornness

    @property
    def name(self):
        return "Upper Teammate"

    def policy(self, state):
        p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered = state
        roll = random.randint(0, 1)
        if hand_2 == NOTHING and roll <= self.stubbornness:
            if p2 == TOP:
                action = ACT
            else:
                action = UP
            return deterministic_policy(action, self.mmdp.num_disjoint_actions)
        else:
            return super().policy(state)

class DownerTeammate(GreedyTeammate):

    def __init__(self, mmdp, stubbornness=0.9):
        super(DownerTeammate, self).__init__(mmdp)
        self.stubbornness = stubbornness

    @property
    def name(self):
        return "Downer Teammate"

    def policy(self, state):
        p1, p2, hand_1, hand_2, top_balcony, bottom_balcony, soup_pan, soup_delivered = state
        roll = random.randint(0, 1)
        if hand_2 == NOTHING and roll <= self.stubbornness:
            if p2 == BOTTOM:
                action = ACT
            else:
                action = DOWN
            return deterministic_policy(action, self.mmdp.num_disjoint_actions)
        else:
            return super().policy(state)
