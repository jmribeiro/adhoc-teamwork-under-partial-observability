from abc import ABC

import numpy as np

import yaaf
from yaaf import Timestep
from yaaf.environments.markov import MarkovDecisionProcess


class MultiAgentMarkovDecisionProcess(MarkovDecisionProcess, ABC):

    def __init__(self, name,
                 num_agents,
                 states,
                 disjoint_action_space,
                 transition_probabilities,
                 rewards,
                 discount_factor,
                 miu,
                 action_descriptions=None):

        self._num_agents = num_agents
        self._num_disjoint_actions = len(disjoint_action_space)
        self._num_joint_actions = self._num_disjoint_actions ** self._num_agents
        joint_action_space = self._setup_joint_action_space(self._num_agents, disjoint_action_space)
        assert len(joint_action_space) == self._num_joint_actions
        super(MultiAgentMarkovDecisionProcess, self).__init__(name, states, joint_action_space,
                                                              transition_probabilities, rewards, discount_factor, miu,
                                                              action_meanings=action_descriptions)
        self._teammates = []

    @property
    def teammate(self):
        return self._teammates[0]

    def add_teammate(self, teammate):
        assert len(self._teammates) < self._num_agents - 1, "Maximum number of agents reached"
        self._teammates.append(teammate)

    def set_teammate(self, teammate):
        self._teammates.clear()
        self.add_teammate(teammate)

    def step(self, action):
        state = self.state
        teammates_actions = [teammate.action(self.state) for teammate in self._teammates]
        joint_action = tuple([action] + teammates_actions)
        ja = self._actions.index(joint_action)
        next_state, reward, terminal, info = super().step(ja)
        [teammate.reinforcement(Timestep(state, joint_action[t], reward, next_state, terminal, info)) for t, teammate in enumerate(self._teammates)]
        info["ja"] = ja
        return next_state, reward, terminal, info

    def joint_step(self, joint_action):
        ja = joint_action
        next_state, reward, terminal, info = super().step(ja)
        info["ja"] = ja
        return next_state, reward, terminal, info

    @property
    def disjoint_policies(self):
        if not hasattr(self, "_disjoint_policies"):
            self._disjoint_policies = np.zeros((self.num_agents, self.num_states, self.num_disjoint_actions))
            policy = self.policy
            for x in range(len(self.states)):
                for ja, probability in enumerate(policy[x]):
                    joint_action = self._actions[ja]
                    for agent, action in enumerate(joint_action):
                        self._disjoint_policies[agent, x, action] += probability
        return self._disjoint_policies

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def num_teammates(self):
        return self._num_agents - 1

    @property
    def num_joint_actions(self):
        """ Alias for super().num_actions """
        return self.num_actions

    @property
    def joint_action_space(self):
        return self._actions

    @property
    def num_disjoint_actions(self):
        return self._num_disjoint_actions

    @staticmethod
    def _setup_joint_action_space(num_agents, disjoint_action_space):

        joint_action_space = []

        for _ in range(num_agents):

            auxiliary = []

            if len(joint_action_space) == 0: # First action

                for a0 in range(len(disjoint_action_space)):
                    auxiliary.append([a0])

            else:

                for a in joint_action_space:

                    for a0 in range(len(disjoint_action_space)):
                        new_a = a + [a0]
                        auxiliary.append(tuple(new_a))

            joint_action_space = auxiliary

        return tuple(joint_action_space)

    @staticmethod
    def load(directory):
        name = str(np.load(f"{directory}/name.npy"))
        X = np.load(f"{directory}/X.npy")
        A = list(np.load(f"{directory}/A.npy"))
        P = np.load(f"{directory}/P.npy")
        R = np.load(f"{directory}/R.npy")
        gamma = float(np.load(f"{directory}/gamma.npy"))
        num_agents = int(np.load(f"{directory}/num_agents.npy"))
        miu = np.load(f"{directory}/miu.npy")
        return MultiAgentMarkovDecisionProcess(name, num_agents, X, tuple(range(len(A))), P, R, gamma, miu, A)

    def save(self, directory):
        yaaf.mkdir(f"{directory}/{self.spec.id}")
        pomdp = {
            "name": self.spec.id,
            "X": self.states,
            "A": self.action_meanings,
            "P": self.P,
            "R": self.R,
            "gamma": self.gamma,
            "num_agents": self.num_agents,
            "miu": self.miu
        }
        for var in pomdp:
            np.save(f"{directory}/{self.spec.id}/{var}", pomdp[var])
