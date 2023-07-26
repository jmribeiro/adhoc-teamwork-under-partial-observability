import abc
from abc import ABC

import numpy as np

from backend.policy import Policy
from yaaf import Timestep, normalize, uniform_distribution
from yaaf.agents import Agent
from yaaf.environments.markov import MarkovDecisionProcess
from yaaf.policies import greedy_policy, deterministic_policy, sample_action

from backend.pomdp import PartiallyObservableMarkovDecisionProcess

# ########## #
# Heuristics #
# ########## #

class TEQMDPAgent(Agent):

    def __init__(self, pomdp):

        super(TEQMDPAgent, self).__init__("Transition Entropy QMDP Agent")

        teqmdp = self.teqmdp = self.teqmdp(pomdp)
        self.pomdp = pomdp

        self.q_values = pomdp.q_values
        self._policy = pomdp.policy

        self.information_q_values = teqmdp.q_values
        self.information_policy = teqmdp.policy

        self.num_actions = pomdp.num_actions

        self.reset()

    def reset(self):
        self.belief = self.pomdp.miu
        self._policy = self.compute_next_policy()

    def compute_next_policy(self):
        Q = self.q_values
        Q_info = self.information_q_values
        q_values = self.transition_entropy_q_values(self.belief, Q, Q_info)
        return greedy_policy(q_values)

    def policy(self, observation=None):
        return self._policy

    def _reinforce(self, timestep):
        a = timestep.action
        next_obs = timestep.next_observation
        self.belief = self.pomdp.belief_update(self.belief, a, next_obs)
        self._policy = self.compute_next_policy()

    @staticmethod
    def teqmdp(pomdp):
        return MarkovDecisionProcess(
            name=pomdp.spec.id.replace("pomdp", "teqmdp"),
            states=pomdp.states,
            actions=pomdp.actions,
            transition_probabilities=pomdp.transition_probabilities,
            rewards=TEQMDPAgent.information_reward(pomdp),
            discount_factor=pomdp.discount_factor,
            initial_state_distribution=pomdp.initial_state_distribution,
            state_meanings=pomdp.state_meanings,
            action_meanings=pomdp.action_meanings
        )

    @staticmethod
    def information_gain(pomdp):
        dH = np.zeros((pomdp.num_actions, pomdp.num_observations))
        for a in range(pomdp.num_actions):
            Pa_sum = np.sum(pomdp.P[a], axis=0)
            mu = np.diag(Pa_sum).dot(pomdp.O[a]).T
            msum = np.sum(mu, axis=1)
            for z in range(pomdp.num_observations):
                if msum[z] > 0:
                    aux = 0
                    for x in range(pomdp.num_states):
                        mu[z, x] = mu[z, x] / msum[z]
                        if mu[z, x] > 0:
                            aux += mu[z, x] * np.log(mu[z, x]) / np.log(pomdp.num_states)
                    dH[a, z] = 1 + aux
                else:
                    dH[a, z] = 0
        return dH

    @staticmethod
    def reward_gain(pomdp):
        d_r = np.zeros((pomdp.num_actions, pomdp.num_observations))
        for a in range(pomdp.num_actions):
            Pa_sum = np.sum(pomdp.P[a], axis=0) / pomdp.num_states
            for z in range(pomdp.num_observations):
                d_r[a, z] = np.max((Pa_sum * pomdp.O[a, :, z])[None, :].dot(pomdp.R), axis=1)
        return d_r

    @staticmethod
    def information_reward(pomdp):
        rG = np.zeros((pomdp.num_states, pomdp.num_actions))
        G = TEQMDPAgent.information_gain(pomdp) #* TEQMDPAgent.reward_gain(pomdp)
        for a in range(pomdp.num_actions):
            rG[:, a] = pomdp.P[a].dot(pomdp.O[a]).dot(G[a, :, None])[:, 0]
        return rG

    @staticmethod
    def normalized_entropy(belief):
        H = 0
        num_states = len(belief)
        for x in range(num_states):
            if belief[x] > 0:
                H -= belief[x] * np.log(belief[x]) / np.log(num_states)
        return H

    @staticmethod
    def transition_entropy_q_values(belief, q_values, information_q_values):
        H = TEQMDPAgent.normalized_entropy(belief)
        TEQ = belief.dot((1 - H) * q_values + H * information_q_values)
        return TEQ

class QMDPAgent(Agent):

    def __init__(self, pomdp: PartiallyObservableMarkovDecisionProcess):
        super(QMDPAgent, self).__init__("QMDP")
        self._pomdp = pomdp
        self._q_values = pomdp.q_values
        self.reset()

    def reset(self):
        self._belief = self._pomdp.miu

    def policy(self, observation=None):
        q_values = self._belief.dot(self._q_values)
        return greedy_policy(q_values)

    def _reinforce(self, timestep: Timestep):
        _, action, _, next_obs, terminal, _ = timestep
        if terminal:
            self._belief = self._pomdp.miu
        else:
            self._belief = self._pomdp.belief_update(self._belief, action, next_obs)

class MLSAgent(Agent):

    def __init__(self, pomdp: PartiallyObservableMarkovDecisionProcess):
        super(MLSAgent, self).__init__("MLS")
        self._pomdp = pomdp
        self._q_values = pomdp.q_values
        self.reset()

    def reset(self):
        self._belief = self._pomdp.miu

    def policy(self, observation=None):
        most_likely_state = self._belief.argmax()
        q_values = self._q_values[most_likely_state]
        return greedy_policy(q_values)

    def _reinforce(self, timestep: Timestep):
        _, action, _, next_obs, _, _ = timestep
        self._belief = self._pomdp.belief_update(self._belief, action, next_obs)

# ########################## #
# Point Based Solvers Agents #
# ########################## #

class AITKAgent(Agent):

    def __init__(self, pomdp, parameters, cache=True):
        super().__init__("Perseus")
        self._pomdp = pomdp
        if hasattr(pomdp, "controller"):
            self._controller = pomdp.controller
        else:
            self._controller = self.load_cached_or_compute(pomdp, parameters) if cache else self.compute_policy(pomdp, "Perseus", parameters)
        self._parameters = parameters
        self.alpha_vectors = np.array(self._controller.alpha_vectors())
        self.actions = self._controller.actions()
        self.num_actions = pomdp.num_actions
        self.reset()

    def reset(self):
        self._b = self._pomdp.miu.tolist()
        self._a, self._ID = self._controller.sampleAction(self._b)
        self._t = self._parameters["horizon"]

    def policy(self, observation=None):
        return deterministic_policy(self._a, self._pomdp.num_actions)

    def action(self, observation=None):
        return self._a

    def _reinforce(self, timestep):
        a = timestep.action
        z = self._pomdp.observation_index(timestep.next_observation)
        self._t = max(0, self._t - 1)
        self._b = self._pomdp.belief_update(np.array(self._b), a, timestep.next_observation).tolist()
        if self._t > self._controller.H:
            self._a, self._ID = self._controller.sampleActionHorizon(self._b, self._controller.H)
        else:
            self._a, self._ID = self._controller.sampleActionObservation(self._ID, z, self._t)

    @staticmethod
    def compute_value_function(pomdp, name, parameters):
        from AIToolbox import POMDP
        from AIToolbox.POMDP import POMDP_VFun, VList
        model = pomdp.convert_to_ai_toolbox_model()
        if name == "Perseus":
            solver = POMDP.PERSEUS(parameters["beliefs"], parameters["horizon"], parameters["tolerance"])
            solution = solver(model, pomdp.R.min())
        elif name == "Witness":
            solver = POMDP.Witness(parameters["horizon"], parameters["tolerance"])
            solution = solver(model)
        elif name == "Incremental Pruning":
            solver = POMDP.IncrementalPruning(parameters["horizon"], parameters["tolerance"])
            solution = solver(model)
        else:
            raise ValueError(f"Invalid solver {name}")
        return solution

    @staticmethod
    def compute_policy(pomdp, solver_name, parameters):
        cpp_value_function = AITKAgent.compute_value_function(pomdp, solver_name, parameters)[1]
        python_value_function = Policy.convert_value_function(cpp_value_function)
        policy = Policy(pomdp.num_states, pomdp.num_actions, pomdp.num_observations, python_value_function)
        return policy

    @staticmethod
    def load_cached_or_compute(pomdp, parameters):
        environment = pomdp.spec.id.split('-v')[0]
        noise = parameters["noise"]
        directory = f"resources/environments/{environment}-{int(noise*100)}%/{pomdp.spec.id}/perseus_policy_b{parameters['beliefs']}_t{parameters['tolerance']}_h{parameters['horizon']}"
        try:
            policy = Policy.load(pomdp, directory)
        except FileNotFoundError:
            print(f"Warning: Policy for {environment}-{int(noise*100)}%/{pomdp.spec.id} was not cached.", flush=True)
            policy = AITKAgent.compute_policy(pomdp, "Perseus", parameters)
            policy.save(directory)
        return policy

class AITKNoHorizonAgent(AITKAgent):

    def __init__(self, pomdp, parameters, cache=True):
        super().__init__(pomdp, parameters, cache)

    def sample_action(self):
        alpha_vectors = self.alpha_vectors
        belief_values = self._b @ alpha_vectors.T
        argmaxes = np.where(belief_values == belief_values.max())[0]
        greedy_actions = list(set([self.actions[int(i)] for i in argmaxes]))
        num_greedy_actions = len(greedy_actions)
        probability = 1 / num_greedy_actions
        policy = np.zeros(self.num_actions)
        for a in greedy_actions:
            policy[a] = probability
        return sample_action(policy)

    def reset(self):
        self._b = self._pomdp.miu.tolist()
        self._a = self.sample_action()
        self._t = self._parameters["horizon"]

    def _reinforce(self, timestep):
        a = timestep.action
        z = self._pomdp.observation_index(timestep.next_observation)
        self._b = self._pomdp.belief_update(self._b, a, z).tolist()
        self._a = self.sample_action()

# ############# #
# Ad Hoc Agents #
# ############# #

class ATPO(Agent):

    def __init__(self, pomdps, parameters):
        super().__init__("ATPO")
        self.pomdps = pomdps
        self.num_pomdps = len(pomdps)
        self.num_actions = pomdps[0].num_actions
        self.num_states = pomdps[0].num_states
        self.alpha_vectors = []
        self.actions = []
        self.policies = []
        for pomdp in pomdps:
            if hasattr(pomdp, "controller"):
                policy = pomdp.controller
            else:
                policy = AITKAgent.load_cached_or_compute(pomdp, parameters)
            alpha_vectors = np.array(policy.alpha_vectors())
            actions = policy.actions()
            self.alpha_vectors.append(alpha_vectors)
            self.actions.append(actions)
            self.policies.append(policy)
        self.reset()

    def reset(self):
        self.beliefs_over_pomdps = uniform_distribution(self.num_pomdps)
        self.beliefs_over_states = [pomdp.miu for pomdp in self.pomdps]
        self._policy = self.compute_next_policy()

    def policy(self, observation=None):
        return self._policy

    def _reinforce(self, timestep):
        _, a, _, next_obs, _, _ = timestep
        self.update_beliefs_over_pomdps(a, next_obs)
        self.update_beliefs_over_states(a, next_obs)
        self._policy = self.compute_next_policy()
        return {"beliefs over pomdps": self.beliefs_over_pomdps}

    def compute_next_policy(self):

        policy = np.zeros(self.num_actions)

        for k, pomdp in enumerate(self.pomdps):

            belief = self.beliefs_over_states[k]
            alpha_vectors = self.alpha_vectors[k]
            belief_values = belief @ alpha_vectors.T
            argmaxes1 = np.where(belief_values == belief_values.max())[0]
            
            greedy_actions = [self.actions[k][int(i)] for i in argmaxes1]
            greedy_actions = list(set(greedy_actions))
            num_greedy_actions = len(greedy_actions)
            probability = 1 / num_greedy_actions

            model_policy = np.zeros(self.num_actions)
            for a in greedy_actions: model_policy[a] = probability
            policy += model_policy * self.beliefs_over_pomdps[k]

        return normalize(policy)

    def update_beliefs_over_pomdps(self, a, next_obs):
        accumulators = np.zeros(self.num_pomdps)
        for k, pomdp in enumerate(self.pomdps):
            belief = self.beliefs_over_states[k]
            z = pomdp.observation_index(next_obs)
            accumulator = 0
            for x in range(pomdp.num_states):
                dot = pomdp.O[a, :, z].dot(pomdp.P[a, x])
                accumulator += dot * belief[x]
            accumulator *= self._policy[a] * self.beliefs_over_pomdps[k]
            accumulators[k] = accumulator
        self.beliefs_over_pomdps = normalize(accumulators)

    def update_beliefs_over_states(self, a, next_obs):
        for k, pomdp in enumerate(self.pomdps):
            self.beliefs_over_states[k] = pomdp.belief_update(self.beliefs_over_states[k], a, next_obs)

class PerseusRandom(Agent):

    def __init__(self, pomdps, parameters):
        super().__init__("PerseusRandom")
        self.pomdps = pomdps
        self.num_pomdps = len(pomdps)
        self.num_actions = pomdps[0].num_actions
        self.num_states = pomdps[0].num_states
        self.alpha_vectors = []
        self.actions = []
        self.policies = []
        for pomdp in pomdps:
            if hasattr(pomdp, "controller"):
                policy = pomdp.controller
            else:
                policy = AITKAgent.load_cached_or_compute(pomdp, parameters)
            alpha_vectors = np.array(policy.alpha_vectors())
            actions = policy.actions()
            self.alpha_vectors.append(alpha_vectors)
            self.actions.append(actions)
            self.policies.append(policy)
        self.reset()

    def reset(self):
        self.beliefs_over_pomdps = uniform_distribution(self.num_pomdps)
        self.beliefs_over_states = [pomdp.miu for pomdp in self.pomdps]
        self._policy = self.compute_next_policy()

    def policy(self, observation=None):
        return self._policy

    def _reinforce(self, timestep):
        _, a, _, next_obs, _, _ = timestep
        self.update_beliefs_over_states(a, next_obs)
        self._policy = self.compute_next_policy()
        return {"beliefs over pomdps": self.beliefs_over_pomdps}

    def compute_next_policy(self):

        policy = np.zeros(self.num_actions)

        for k, pomdp in enumerate(self.pomdps):

            belief = self.beliefs_over_states[k]
            alpha_vectors = self.alpha_vectors[k]
            belief_values = belief @ alpha_vectors.T
            argmaxes1 = np.where(belief_values == belief_values.max())[0]
            
            greedy_actions = [self.actions[k][int(i)] for i in argmaxes1]
            greedy_actions = list(set(greedy_actions))
            num_greedy_actions = len(greedy_actions)
            probability = 1 / num_greedy_actions

            model_policy = np.zeros(self.num_actions)
            for a in greedy_actions: model_policy[a] = probability
            policy += model_policy * self.beliefs_over_pomdps[k]

        return normalize(policy)

    def update_beliefs_over_states(self, a, next_obs):
        for k, pomdp in enumerate(self.pomdps):
            self.beliefs_over_states[k] = pomdp.belief_update(self.beliefs_over_states[k], a, next_obs)


class AdHocTeammateLibraryAgent(Agent, ABC):

    def __init__(self, name, mmdp, teammates, index):
        super(AdHocTeammateLibraryAgent, self).__init__(name)
        self.mmdp = mmdp
        self.teammates = teammates
        self.num_teammates = len(teammates)
        self.priors = uniform_distribution(self.num_teammates)
        self.index = index
        self.joint_action_space = [(ja[0], ja[1]) for ja in mmdp.joint_action_space]
        self.num_disjoint_actions = mmdp.num_disjoint_actions
        self.num_joint_actions = self.mmdp.policy.shape[1]

    def policy(self, state):
        x = self.mmdp.state_index(state)
        num_actions = self.num_disjoint_actions
        q_values = np.zeros((self.num_teammates, num_actions))
        for m, teammate in enumerate(self.teammates):
            pi = teammate.policy(state)
            for a0 in range(num_actions):
                for a1 in range(num_actions):
                    joint_action = (a0, a1) if self.index == 0 else (a1, a0)
                    ja = self.joint_action_space.index(joint_action)
                    q_values[m, a0] += (self.mmdp.q_values[x, ja] * pi[a1])
        best_response = np.zeros(num_actions)
        for a0 in range(num_actions):
            for m in range(self.num_teammates):
                best_response[a0] += (q_values[m, a0] * self.priors[m])
        return greedy_policy(best_response)

    def _reinforce(self, timestep):

        # Get actions
        ja = timestep.info["ja"]

        # Get transition
        state = timestep.observation
        next_state = timestep.next_observation
        x, y = self.mmdp.state_index(state), self.mmdp.state_index(next_state)

        # Update
        priors = np.array([self.compute_prior(x, y, ja, teammate.policy(state)) for teammate in self.teammates])
        priors *= self.priors
        priors = normalize(priors)

        self.priors = priors
        info = {"priors": self.priors}
        return info

    @abc.abstractmethod
    def compute_prior(self, x, y, ja, pi):
        pass


class ATPOTeammates(AdHocTeammateLibraryAgent):

    def __init__(self, mmdp, teammates, index: int = 0):
        super(ATPOTeammates, self).__init__("ATPO", mmdp, teammates, index)

    def compute_prior(self, x, y, ja, pi):
        # Only views its own action
        a0 = self.mmdp.joint_action_space[ja][self.index]
        prior = 0.0
        for a1 in range(self.num_disjoint_actions):
            prior += self.mmdp.P[self.joint_action_space.index((a0, a1) if self.index == 0 else (a1, a0)), x, y] * pi[a1]
        return prior


class Assistant(AdHocTeammateLibraryAgent):

    def __init__(self, mmdp, teammates, index: int = 0):
        super(Assistant, self).__init__("Assistant", mmdp, teammates, index)

    def compute_prior(self, x, y, ja, pi):
        # Accesses both action and teammates'
        a1 = self.mmdp.joint_action_space[ja][1 if self.index == 0 else 0]
        prior = pi[a1] * self.mmdp.P[ja, x, y]
        return prior


class MLSBOPA(Agent):

    def __init__(self, pomdps):
        super(MLSBOPA, self).__init__("MLSBOPA")
        self.pomdps = pomdps
        self.num_pomdps = len(pomdps)
        self.num_actions = pomdps[0].num_actions
        self.num_states = pomdps[0].num_states
        self.qvalues = [pomdp.policy for pomdp in pomdps]
        self.reset()

    def reset(self):
        self.beliefs_over_pomdps = uniform_distribution(self.num_pomdps)
        self.beliefs_over_states = [pomdp.miu for pomdp in self.pomdps]
        self._policy = self.compute_next_policy()

    def policy(self, observation=None):
        return self._policy

    def _reinforce(self, timestep):
        _, a, _, next_obs, _, _ = timestep
        self.update_beliefs_over_pomdps(a, next_obs)
        self.update_beliefs_over_states(a, next_obs)
        self._policy = self.compute_next_policy()
        return {"beliefs over pomdps": self.beliefs_over_pomdps}

    def compute_next_policy(self):
        policy = np.zeros(self.num_actions)
        for k, pomdp in enumerate(self.pomdps):
            belief = self.beliefs_over_states[k]
            mls = belief.argmax()
            qvalues = self.qvalues[k][mls]
            model_policy = greedy_policy(qvalues)
            policy += model_policy * self.beliefs_over_pomdps[k]
        return normalize(policy)

    def update_beliefs_over_pomdps(self, a, next_obs):
        accumulators = np.zeros(self.num_pomdps)
        for k, pomdp in enumerate(self.pomdps):
            belief = self.beliefs_over_states[k]
            z = pomdp.observation_index(next_obs)
            accumulator = 0
            for x in range(pomdp.num_states):
                dot = pomdp.O[a, :, z].dot(pomdp.P[a, x])
                accumulator += dot * belief[x]
            accumulator *= self._policy[a] * self.beliefs_over_pomdps[k]
            accumulators[k] = accumulator
        self.beliefs_over_pomdps = normalize(accumulators)

    def update_beliefs_over_states(self, a, next_obs):
        for k, pomdp in enumerate(self.pomdps):
            self.beliefs_over_states[k] = pomdp.belief_update(self.beliefs_over_states[k], a, next_obs)

class BOPA(Agent):

    def __init__(self, pomdps):
        super(BOPA, self).__init__("BOPA")
        self.pomdps = pomdps
        self.num_pomdps = len(pomdps)
        self.num_actions = pomdps[0].num_actions
        self.num_states = pomdps[0].num_states   
        self.qvalues = [pomdp.q_values for pomdp in pomdps]

    def reset(self, state):
        self.beliefs_over_pomdps = uniform_distribution(self.num_pomdps)
        self._policy = self.compute_next_policy(state)

    def policy(self, observation=None):
        return self._policy

    def _reinforce(self, timestep):
        state, a, _, next_state, _, _ = timestep
        self.update_beliefs_over_pomdps(state, a, next_state)
        self._policy = self.compute_next_policy(next_state)
        return {"beliefs over pomdps": self.beliefs_over_pomdps}

    def compute_next_policy(self, state):
        policy = np.zeros(self.num_actions)
        for k, pomdp in enumerate(self.pomdps):
            x = pomdp.state_index(state)
            qvalues_m = self.qvalues[k]
            q_x = qvalues_m[x]
            model_policy = greedy_policy(q_x)
            policy += model_policy * self.beliefs_over_pomdps[k]
        return normalize(policy)

    def update_beliefs_over_pomdps(self, state, a, next_state):
        
        p_next = np.zeros(self.num_pomdps)

        for k, pomdp in enumerate(self.pomdps):

            x = pomdp.state_index(state)
            y = pomdp.state_index(next_state)
            Pm = pomdp.P[a, x, y]

            pm_t = self.beliefs_over_pomdps[k]
            
            pm_next = Pm * pm_t
            p_next[k] = pm_next

        self.beliefs_over_pomdps = normalize(p_next)


# ########## #
# DEPRECATED #
# ########## #

class CppAITKAgent(Agent):

    def __init__(self, pomdp, solver_name, parameters):
        from AIToolbox import POMDP
        from AIToolbox.POMDP import POMDP_VFun, VList
        super().__init__(solver_name)
        self.pomdp = pomdp
        solution = AITKAgent.compute_value_function(pomdp, solver_name, parameters)
        value_function: POMDP_VFun = solution[1]
        self.parameters = parameters
        self.policy = POMDP.Policy(pomdp.num_states, pomdp.num_actions, pomdp.num_observations, value_function)
        self.reset()

    def reset(self):
        self.b = self.pomdp.miu.tolist()
        self.a, self.ID = self.policy.sampleAction(self.b, self.parameters["horizon"])
        self.t = self.parameters["horizon"]

    def policy(self, observation=None):
        return deterministic_policy(self.a, self.pomdp.num_actions)

    def action(self, observation=None):
        return self.a

    def _reinforce(self, timestep):
        a = timestep.action
        z = self.pomdp.observation_index(timestep.next_observation)
        self.t = max(0, self.t - 1)
        self.b = self.pomdp.belief_update(np.array(self.b), a, timestep.next_observation).tolist()
        if self.t > self.policy.getH():
            self.a, self.ID = self.policy.sampleAction(self.b, self.policy.getH())
        else:
            self.a, self.ID = self.policy.sampleAction(self.ID, z, self.t)
