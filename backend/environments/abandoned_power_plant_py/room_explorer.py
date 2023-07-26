from itertools import product

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from backend.pomdp import PartiallyObservableMarkovDecisionProcess
from yaaf import ndarray_index_from
from yaaf.environments.markov import MarkovDecisionProcess

from typing import Sequence

import yaaf

class EnvironmentReckonPOMDP(PartiallyObservableMarkovDecisionProcess):

    def __init__(self,
                 adjacency_matrix: np.ndarray,
                 explorable_nodes: Sequence[int],
                 noise: float,
                 discount_factor: float = 0.90,
                 initial_state: np.ndarray = np.array([0, 0, 1, 0, 0]),
                 node_meanings: Sequence[str] = ("door", "baxter", "single workbench", "double workbench", "table"),
                 id="environment-reckon-pomdp-v1"):


        self._load_under(adjacency_matrix, explorable_nodes, noise, discount_factor, initial_state, node_meanings)

        self._adjacency_matrix = adjacency_matrix
        self._explorable_nodes = explorable_nodes
        self._dead_reckoning_failure_probability = noise
        self._movement_failure_probability = noise
        self._transmission_failure_probability = noise
        self._initial_state = initial_state
        self.node_meanings = node_meanings

        # States
        states = self._underlying_mmdp.states

        # Actions
        action_meanings = tuple(list(self._underlying_mmdp.individual_action_meanings) + ["locate human", "locate robot"])
        actions = tuple(range(len(action_meanings)))

        # Transitions
        transition_probabilities = self.generate_transition_probabilities(
            states, actions, adjacency_matrix, explorable_nodes, noise, action_meanings
        )

        # Observations
        observations = self.generate_observations(adjacency_matrix)
        observation_probabilities = self.generate_observation_probabilities(
            states, actions, observations, adjacency_matrix, action_meanings, noise
        )

        # Reward
        reward_matrix = self._underlying_mmdp.generate_reward_matrix(states, actions)

        # Miu
        initial_state_distribution = np.zeros(len(states))
        initial_state_distribution[self.state_index_from(states, initial_state)] = 1

        super(EnvironmentReckonPOMDP, self).__init__(
            id, states, actions, observations,
            transition_probabilities, observation_probabilities, reward_matrix,
            discount_factor, initial_state_distribution, action_meanings=action_meanings)

    # ################## #
    # Auxiliary for init #
    # ################## #

    def generate_transition_probabilities(self, states, actions, adjacency_matrix, explorable_nodes, noise, action_meanings):

        num_actions = len(actions)
        num_explorable_nodes = len(explorable_nodes)
        num_states = len(states)
        P = np.zeros((num_actions, num_states, num_states))

        for x, state in enumerate(states):

            absorbent = (state == -1).all()
            solved = (state[2:] == 1).all()

            if absorbent:
                P[:, x, x] = 1.0
            elif solved:
                next_state = np.zeros_like(state)
                for i in range(next_state.size):
                    next_state[i] = -1
                y = yaaf.ndarray_index_from(states, next_state)
                P[:, x, y] = 1.0
            else:

                node = self.state_index_from(states, state)
                robot_node, human_node, already_explored_bits = state[0], state[1], state[2:]

                for a in range(num_actions):

                    a_robot = a
                    a_human = self._optimal_teammate_actions[node]

                    if "move" in action_meanings[a_robot]:
                        adjacencies = np.where(adjacency_matrix[robot_node]==1)[0]
                        downgrade_to_lower_index = int(a_robot) >= len(adjacencies)
                        a_robot = 0 if downgrade_to_lower_index else a_robot
                        next_robot_node = adjacencies[a_robot]
                        robot_transitions = {robot_node: noise, next_robot_node: 1.0 - noise}
                    else:
                        robot_transitions = {robot_node: 1.0}

                    if "move" in action_meanings[a_human]:
                        adjacencies = np.where(adjacency_matrix[human_node]==1)[0]
                        downgrade_to_lower_index = int(a_human) >= len(adjacencies)
                        a_human = 0 if downgrade_to_lower_index else a_human
                        next_human_node = adjacencies[a_human]
                        human_transitions = {next_human_node: 1.0}
                    else:
                        human_transitions = {human_node: 1.0}

                    for next_robot_node, robot_prob in robot_transitions.items():
                        for next_human_node, human_prob in human_transitions.items():
                            next_state = np.array([next_robot_node, next_human_node]+[0 for _ in range(num_explorable_nodes)])
                            # If any moved into explorable location, set to 1 the explored bit
                            for explorable_node in explorable_nodes:
                                if next_robot_node == explorable_node or next_human_node == explorable_node:
                                    bit_to_turn_on = explorable_nodes.index(explorable_node) + 2
                                    next_state[bit_to_turn_on] = 1.0
                            for b, bit in enumerate(already_explored_bits):
                                b = b + 2
                                if bit == 1:
                                    next_state[b] = 1
                            y = self.state_index_from(states, next_state)
                            prob = robot_prob * human_prob
                            P[a, node, y] = prob  # Independent events

                for a_robot in range(num_actions):
                    sum = P[a_robot, node].sum()
                    assert sum == 1.0, f"Action {a_robot} ({action_meanings[a_robot]}) on state {node} ({states[node]}) doesnt have transition (sums to {sum})"

        return P

    def generate_observations(self, adjacency_matrix):
        num_nodes = adjacency_matrix.shape[0]
        observations = list(np.arange(-1, num_nodes).reshape(-1, 1))
        return observations

    def generate_observation_probabilities(self, states, actions, observations, adjacency_matrix, action_meanings, noise):

        num_actions = len(actions)
        num_states = len(states)
        num_observations = len(observations)
        num_nodes = adjacency_matrix.shape[0]

        O = np.zeros((num_actions, num_states, num_observations))

        for a in range(num_actions):

            for y in range(num_states):

                next_state = states[y]
                absorbent = (next_state == -1).all()

                if absorbent:
                    obs = np.array([-1])
                    z = yaaf.ndarray_index_from(observations, obs)
                    O[a, y, z] = 1.0

                else:

                    robot_node, human_node = next_state[0], next_state[1]

                    if "locate" in action_meanings[a]:

                        correct_node = int(human_node if "human" in action_meanings[a] else robot_node)

                        probs = {correct_node: 1.0 - noise}

                        for other_node in range(-1, num_nodes):
                            if other_node != correct_node:
                                probs[other_node] = noise / num_nodes

                        aux_probs = np.array(list(probs.values()))
                        aux_probs /= aux_probs.sum()

                        for n, node in enumerate(probs):
                            probs[node] = aux_probs[n]

                    else:

                        correct_node = int(robot_node)
                        adjacent_nodes = np.where(adjacency_matrix[robot_node] == 1)[0]
                        num_adjacent_nodes = len(adjacent_nodes)

                        probs = {correct_node: 1.0 - noise}

                        for other_node in range(-1, num_adjacent_nodes):
                            if other_node != correct_node:
                                probs[other_node] = noise / num_adjacent_nodes

                        aux_probs = np.array(list(probs.values()))
                        aux_probs /= aux_probs.sum()

                        for n, node in enumerate(probs):
                            probs[node] = aux_probs[n]

                    for obs, prob in probs.items():
                        observation = np.array([obs])
                        z = self.state_index_from(observations, observation)
                        O[a, y, z] = prob

        return O

    # ########## #
    # Draw utils #
    # ########## #

    def show_topological_map(self):
        self._underlying_mmdp.draw_state(self._initial_state, self.spec.id)

    def render(self, mode='human'):
        self._underlying_mmdp.draw_state(self.state, title=f"{self.spec.id}\n{self.state}")

    def _generate_underlying_mmdp(self, adjacency_matrix, nodes_of_interest, movement_failure_probability, discount_factor, initial_state, node_meanings):
        return EnvironmentReckonMMDP(adjacency_matrix, nodes_of_interest, movement_failure_probability, discount_factor, initial_state, node_meanings)

    def _load_under(self, adjacency_matrix, explorable_nodes, movement_failure_probability, discount_factor, initial_state, node_meanings):
        self._underlying_mmdp = self._generate_underlying_mmdp(adjacency_matrix, explorable_nodes, movement_failure_probability, discount_factor, initial_state, node_meanings)
        self._optimal_joint_policy = self._underlying_mmdp.policy
        self._optimal_teammate_actions = []
        for joint_policy_state in self._optimal_joint_policy:
            optimal_joint_action = joint_policy_state.argmax()
            meanings = self._underlying_mmdp.action_meanings[optimal_joint_action]
            teammate_meaning = meanings[-1]
            optimal_teammate_action = self._underlying_mmdp.individual_action_meanings.index(teammate_meaning)
            self._optimal_teammate_actions.append(optimal_teammate_action)
        self._optimal_teammate_actions = np.array(self._optimal_teammate_actions)

class EnvironmentReckonMMDP(MarkovDecisionProcess):

    def __init__(self,
                 adjacency_matrix: np.ndarray,
                 explorable_nodes: Sequence[int],
                 noise: float,
                 discount_factor: float = 0.90,
                 initial_state: np.ndarray = np.array([0, 0, 1, 0, 0]),
                 node_meanings: Sequence[str] = ("door", "baxter", "single workbench", "double workbench", "table"),
                 id="environment-reckon-mmdp-v1"):

        self._adjacency_matrix = adjacency_matrix
        self._explorable_nodes = explorable_nodes
        self._movement_failure_probability = noise
        self._node_meanings = node_meanings
        self._initial_state = initial_state

        # States
        states = self.generate_states(adjacency_matrix, explorable_nodes)

        # Actions
        self.individual_action_meanings = self.generate_action_meanings()
        self._joint_actions = list(np.array(list(product(range(len(self.individual_action_meanings)), repeat=2))))
        actions = tuple(range(len(self._joint_actions)))
        action_meanings = tuple(product(self.individual_action_meanings, repeat=2))

        # Transitions
        transition_probabilities = self.generate_transition_probabilities(
            states, self._joint_actions, adjacency_matrix, explorable_nodes, noise, self.individual_action_meanings
        )

        # Reward
        reward_matrix = self.generate_reward_matrix(states, actions)

        # Miu
        initial_state_distribution = np.zeros(len(states))
        initial_state_distribution[self.state_index_from(states, initial_state)] = 1

        super(EnvironmentReckonMMDP, self).__init__(
            id, states, actions, transition_probabilities, reward_matrix,
            discount_factor, initial_state_distribution,
            action_meanings=action_meanings)

    # ################## #
    # Auxiliary for init #
    # ################## #

    def generate_states(self, adjacency_matrix, explorable_nodes):
        num_explorable_nodes = len(explorable_nodes)
        num_nodes = adjacency_matrix.shape[0]
        num_agents = 2
        possible_bit_combinations = list(product(range(num_agents), repeat=num_explorable_nodes))
        states = [
            np.array([x_robot, x_human, *explorable_node_bits])
            for x_robot in range(num_nodes)
            for x_human in range(num_nodes)
            for explorable_node_bits in possible_bit_combinations
        ]
        valid_states = []
        for x_robot in range(num_nodes):
            for x_human in range(num_nodes):
                for x, state in enumerate(states):
                    if state[0] == x_robot and state[1] == x_human:
                        explorable_bits = state[num_agents:]
                        robot_in_explorable = x_robot in explorable_nodes
                        human_in_explorable = x_human in explorable_nodes
                        invalid_robot_node_state = robot_in_explorable and explorable_bits[explorable_nodes.index(x_robot)] == 0
                        invalid_human_node_state = human_in_explorable and explorable_bits[explorable_nodes.index(x_human)] == 0
                        if not invalid_robot_node_state and not invalid_human_node_state:
                            valid_states.append(x)
        states = [states[x] for x in valid_states]
        return states + [np.array([-1, -1] + [-1 for _ in range(num_explorable_nodes)])]

    def generate_action_meanings(self):
        individual_action_meanings = (
            "move to lower-index node",
            "move to second-lower-index node",
            "move to third-lower-index node",
            "stay",
        )
        return individual_action_meanings

    def generate_transition_probabilities(self, states, joint_actions, adjacency_matrix, explorable_nodes, noise, individual_action_meanings):

        num_joint_actions = len(joint_actions)
        num_explorable_nodes = len(explorable_nodes)
        num_states = len(states)
        P = np.zeros((num_joint_actions, num_states, num_states))

        for x, state in enumerate(states):

            absorbent = (state == -1).all()
            solved = (state[2:] == 1).all()

            if absorbent:
                P[:, x, x] = 1.0
            elif solved:
                next_state = np.zeros_like(state)
                for i in range(next_state.size):
                    next_state[i] = -1
                y = yaaf.ndarray_index_from(states, next_state)
                P[:, x, y] = 1.0
            else:

                x_robot, x_human, already_explored_bits = state[0], state[1], state[2:]

                for a in range(num_joint_actions):
                    a_robot, a_human = joint_actions[a]
                    if "move" in individual_action_meanings[a_robot]:
                        adjacencies = np.where(adjacency_matrix[x_robot]==1)[0]
                        downgrade_to_lower_index = int(a_robot) >= len(adjacencies)
                        a_robot = 0 if downgrade_to_lower_index else a_robot
                        next_x_robot = adjacencies[a_robot]
                        robot_transitions = {x_robot: noise, next_x_robot: 1.0 - noise}
                    else:
                        robot_transitions = {x_robot: 1.0}
                    if "move" in individual_action_meanings[a_human]:
                        adjacencies = np.where(adjacency_matrix[x_human]==1)[0]
                        downgrade_to_lower_index = int(a_human) >= len(adjacencies)
                        a_human = 0 if downgrade_to_lower_index else a_human
                        next_x_human = adjacencies[a_human]
                        human_transitions = {next_x_human: 1.0}
                    else:
                        human_transitions = {x_human: 1.0}
                    for next_x_robot, robot_prob in robot_transitions.items():
                        for next_x_human, human_prob in human_transitions.items():
                            next_state = np.array([next_x_robot, next_x_human]+[0 for _ in range(num_explorable_nodes)])
                            # If any moved into explorable location, set to 1 the explored bit
                            for explorable_node in explorable_nodes:
                                if next_x_robot == explorable_node or next_x_human == explorable_node:
                                    bit_to_turn_on = explorable_nodes.index(explorable_node) + 2
                                    next_state[bit_to_turn_on] = 1.0
                            for b, bit in enumerate(already_explored_bits):
                                b = b + 2
                                if bit == 1:
                                    next_state[b] = 1
                            y = self.state_index_from(states, next_state)
                            P[a, x, y] = robot_prob * human_prob  # Independent events
        return P

    def generate_reward_matrix(self, states, actions):
        num_states = len(states)
        num_actions = len(actions)
        R = np.zeros((num_states, num_actions))
        for x, state in enumerate(states):
            solved = (state[2:] == 1).all()
            absorbent = (state == -1).all()
            if solved: R[x, :] = 100.0
            elif absorbent: R[x, :] = 0.00
            else: R[x, :] = -1.0
        return R

    # ########## #
    # Draw utils #
    # ########## #

    def show_topological_map(self):
        self.draw_state(self._initial_state, self.spec.id)

    def draw_state(self, state, title=None):
        graph = nx.DiGraph()
        labels = {}
        x_robot, x_human, explored_bits = state[0], state[1], state[2:]
        num_nodes = self._adjacency_matrix.shape[0]
        colors = []
        for n in range(num_nodes):
            graph.add_node(n)
            label = self._node_meanings[n].replace(' ', '\n')
            labels[n] = f"[{n}: {label}]"
            if x_robot == n:
                labels[n] += f"\nR"
            if x_human == n:
                label = "\nH" if 'R' not in labels[n] else ', H'
                labels[n] += f"{label}"

            if n in self._explorable_nodes:
                if explored_bits[self._explorable_nodes.index(n)] == 1:
                    colors.append("lightgreen")
                    labels[n] = labels[n].replace("]", "]\n(explored)")
                else:
                    colors.append("lightgray")
                    labels[n] = labels[n].replace("]", "]\n(unexplored)")
            else:
                colors.append("white")

        rows, cols = np.where(self._adjacency_matrix == 1)
        graph.add_edges_from(zip(rows.tolist(), cols.tolist()))
        plt.figure()
        if not hasattr(self, "_node_draw_pos"):
            self._node_draw_pos = nx.spring_layout(graph)
        fig = nx.draw_networkx_nodes(graph, node_color=colors, pos=self._node_draw_pos, node_size=8000)
        fig.set_edgecolor('k')
        plt.gcf().set_size_inches(10, 10, forward=True)
        nx.draw_networkx_edges(graph, self._node_draw_pos, width=1.0, node_size=8000, arrowsize=10)
        nx.draw_networkx_labels(graph, self._node_draw_pos, labels, font_size=12)
        plt.axis('off')
        plt.title(title or f"State: {state}")
        plt.show()
        plt.close()

    def render(self, mode='human'):
        self.draw_state(self.state, title=f"{self.spec.id}\n{self.state}")

    def joint_action_index(self, joint_action):
        return ndarray_index_from(self._joint_actions, joint_action)