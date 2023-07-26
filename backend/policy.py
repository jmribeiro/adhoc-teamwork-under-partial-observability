from random import choice
from typing import List
import numpy as np
import yaaf

class Node:

    def __init__(self, values, action, observations):
        self.values = [round(value, 5) for value in values]
        self.action = action
        self.observations = observations

    def __eq__(self, other):
        return self.values == other.values and self.action == other.action and self.observations == other.observations

class Policy:

    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 num_observations: int,
                 value_function: List[List[Node]]):

        self._num_states = num_states
        self._num_actions = num_actions
        self._num_observations = num_observations
        self._value_function = value_function
        self.H = len(value_function) - 1

    def actions(self, horizon=-1):
        nodes: List[Node] = self._value_function[horizon]
        actions = [node.action for node in nodes]
        return actions

    def alpha_vectors(self, horizon=-1):
        nodes: List[Node] = self._value_function[horizon]
        alpha_vectors = [node.values for node in nodes]
        return alpha_vectors

    def sampleAction(self, belief: List[float], sample=True):
        best_actions, ids = self.sampleActionHorizon(belief, -1, sample)
        return best_actions, ids

    def sampleActionHorizon(self, belief, horizon, sample=True):
        nodes: List[Node] = self._value_function[horizon]
        best_nodes_ids = self.findBestAtPoint(belief, nodes)
        if sample:
            id = choice(best_nodes_ids)
            action = nodes[id].action
            return action, id
        else:
            return [node.action for node in [nodes[id] for id in best_nodes_ids]], best_nodes_ids

    def sampleActionObservation(self, id, z, horizon):
        nodes: List[Node] = self._value_function[horizon + 1]
        successor_node_id = nodes[id].observations[z]
        successor_node_action = self._value_function[horizon][successor_node_id].action
        return successor_node_action, successor_node_id

    def bestNode(self, belief, sample=True):
        return self.bestNodeHorizon(belief, -1, sample)

    def bestNodeHorizon(self, belief, horizon, sample=True):
        nodes: List[Node] = self._value_function[horizon]
        best_nodes_ids = self.findBestAtPoint(belief, nodes)
        if sample:
            id = choice(best_nodes_ids)
            node = nodes[id]
            return node, id
        else:
            return [nodes[id] for id in best_nodes_ids], best_nodes_ids

    @staticmethod
    def findBestAtPoint(belief, nodes):
        best = -np.inf
        arg_best = []
        for i, ventry in enumerate(nodes):
            values = np.array(ventry.values)
            current = np.dot(belief, values)
            if current > best:
                best = current
                arg_best = [i]
            elif current == best:
                arg_best.append(i)
        return arg_best

    @staticmethod
    def convert_value_function(value_function):
        value_function = [nodes for nodes in value_function]
        new_value_function = []
        for i, nodes in enumerate(value_function):
            new_nodes = []
            nodes = [node for node in nodes]
            for j, node in enumerate(nodes):
                values = node.values
                action = node.action
                observations = node.observations
                new_nodes.append(Node(list(values), int(action), list(observations)))
            new_value_function.append(new_nodes)
        return new_value_function

    @staticmethod
    def load_value_function(directory):
        value_function = []
        vlists = yaaf.subdirectories(directory)
        vlists.sort(key=int)
        if len(vlists) == 0:
            raise FileNotFoundError(directory)
        for vlist_file in vlists:
            if yaaf.isdir(directory+"/"+vlist_file):
                vlist_dir = directory+"/"+vlist_file
                vlist = []
                nodes = yaaf.subdirectories(vlist_dir)
                nodes.sort(key=int)
                for ventry_file in nodes:
                    if yaaf.isdir(directory+"/"+vlist_file+"/"+ventry_file):
                        values = list(np.load(directory+"/"+vlist_file+"/"+ventry_file+"/values.npy"))
                        action = int(np.load(directory+"/"+vlist_file+"/"+ventry_file+"/action.npy"))
                        observations = list(np.load(directory+"/"+vlist_file+"/"+ventry_file+"/observations.npy"))
                        ventry = Node(values, action, observations)
                        vlist.append(ventry)
                value_function.append(vlist)
        return value_function

    @staticmethod
    def load(pomdp, directory):
        value_function = Policy.load_value_function(directory)
        policy = Policy(pomdp.num_states, pomdp.num_actions, pomdp.num_observations, value_function)
        return policy

    def save(self, directory):

        value_function = [vlist for vlist in self._value_function]

        for i, vlist in enumerate(value_function):

            ventries = [ventry for ventry in vlist]

            for j, ventry in enumerate(ventries):
                current_directory = f"{directory}/{i}/{j}"
                yaaf.mkdir(current_directory)

                values = ventry.values
                action = ventry.action
                observations = ventry.observations

                np.save(f"{current_directory}/values.npy", values)
                np.save(f"{current_directory}/action.npy", action)
                np.save(f"{current_directory}/observations.npy", observations)
