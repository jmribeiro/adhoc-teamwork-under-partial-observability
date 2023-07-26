import numpy as np

from typing import Sequence, Optional

import yaaf
from yaaf import ndarray_index_from
from yaaf.environments.markov import MarkovDecisionProcess as MDP

class PartiallyObservableMarkovDecisionProcess(MDP):

    def __init__(self, name: str,
                 states: Sequence[np.ndarray],
                 actions: Sequence[int],
                 observations: Sequence[np.ndarray],
                 transition_probabilities: np.ndarray,
                 observation_probabilities: np.ndarray,
                 reward_matrix: np.ndarray,
                 discount_factor: float,
                 initial_state_distribution: np.ndarray,
                 state_meanings: Optional[Sequence[str]] = None,
                 action_meanings: Optional[Sequence[str]] = None,
                 observation_meanings: Optional[Sequence[str]] = None):

        self.validate_pomdp(states, actions, observations, observation_probabilities, observation_meanings)

        self._observations = observations
        self._num_observations = len(observations)
        self._observation_meanings = observation_meanings or ["<UNK>" for _ in range(self._num_observations)]
        self._observation_probabilities = observation_probabilities

        super(PartiallyObservableMarkovDecisionProcess, self).__init__(
            name,
            states, actions,
            transition_probabilities, reward_matrix,
            discount_factor, initial_state_distribution,
            state_meanings, action_meanings)

        self._belief = self.miu
        self._observation = None

    # ########## #
    # OpenAI Gym #
    # ########## #

    def reset(self, belief=None):
        self._belief = belief if belief is not None else self._miu
        x = np.random.choice(range(self.num_states), p=self._belief)
        initial_state = self.states[x]
        self._state = initial_state
        self._observation = None
        return initial_state

    def step(self, action):
        next_state, reward, is_terminal, info = super().step(action)
        next_observation = self.observe(next_state, action)
        self._belief = self.belief_update(self._belief, action, next_observation)
        self._observation = next_observation
        return next_observation, reward, is_terminal, info

    # ############## #
    # Main Interface #
    # ############## #

    def observe(self, state, previous_action):
        a = previous_action
        y = self.state_index(state)
        z = np.random.choice(range(self.num_observations), p=self.O[a, y])
        return self.observations[z]

    # ##### #
    # Other #
    # ##### #

    @property
    def observations(self):
        return self._observations

    @property
    def num_observations(self):
        return self._num_observations

    @property
    def observation_meanings(self):
        return self._observation_meanings

    @property
    def O(self):
        return self._observation_probabilities

    @property
    def observation(self):
        return self._observation

    def observation_index(self, observation=None):
        if observation is None and self._observation is not None:
            return self.observation_index(self._observation)
        elif observation is None and self._observation is None:
            return None
        else:
            return ndarray_index_from(self._observations, observation)

    @staticmethod
    def observation_index_from(observations, observation):
        return ndarray_index_from(observations, observation)

    @property
    def belief(self):
        return self._belief

    @property
    def most_likely_state(self):
        return self.states[self.belief.argmax()]

    def belief_update(self, belief, action, next_observation):
        z = self.observation_index_from(self.observations, next_observation) if not isinstance(next_observation, int) else next_observation
        next_belief = np.dot(belief, self.P[action] * self.O[action, :, z])
        next_belief = yaaf.normalize(next_belief)
        return next_belief

    def convert_to_ai_toolbox_model(self):

        from AIToolbox import POMDP

        P = np.zeros((self.num_states, self.num_actions, self.num_states))
        for x in range(self.num_states):
            for a in range(self.num_actions):
                P[x, a] = self.P[a, x]

        O = np.zeros((self.num_states, self.num_actions, self.num_observations))
        for y in range(self.num_states):
            for a in range(self.num_actions):
                O[y, a] = self.O[a, y]

        R = np.zeros((self.num_states, self.num_actions, self.num_states))
        for x in range(self.num_states):
            for a in range(self.num_actions):
                R[x, a] = self.R[x, a]

        model = POMDP.Model(self.num_observations, self.num_states, self.num_actions)

        fit = False
        decimals = 7
        while not fit and decimals > 0:
            _P = np.around(P, decimals)
            try:
                model.setTransitionFunction(P.tolist())
                fit = True
                #print(f"{decimals}", end=" ")
            except:
                #print(f"!{decimals}", end=" ")
                decimals -= 1

        model.setRewardFunction(R.tolist())
        model.setObservationFunction(O.tolist())
        model.setDiscount(self.gamma)

        model.miu = self.miu.tolist()
        model.num_actions = self.num_actions
        model.num_states = self.num_states
        model.num_observations = self.num_observations

        model = POMDP.SparseModel(model)

        return model

    # ########### #
    # Persistence #
    # ########### #

    @staticmethod
    def load_cassandra(filename):
        name = filename.split("/")[-1].replace(".", "-").lower() + "-v1"  # Needed for OpenAI Gym
        state_meanings, action_meanings, observation_meanings, P, O, R, discount_factor, miu = parse_cassandra_pomdp_file(filename)
        state_space = np.arange(len(state_meanings))
        action_space = np.arange(len(action_meanings))
        observation_space = np.arange(len(observation_meanings))
        return PartiallyObservableMarkovDecisionProcess(
            name, state_space, action_space, observation_space,
            P, O, R, discount_factor, miu,
            state_meanings=tuple(state_meanings),
            action_meanings=tuple(action_meanings),
            observation_meanings=tuple(observation_meanings)
        )

    def save(self, directory):
        yaaf.mkdir(f"{directory}/{self.spec.id}")
        pomdp = {
            "name": self.spec.id,
            "X": self.states,
            "Z": self.observations,
            "A": self.action_meanings,
            "P": self.P,
            "O": self.O,
            "R": self.R,
            "gamma": self.gamma,
            "miu": self.miu
        }
        for var in pomdp:
            np.save(f"{directory}/{self.spec.id}/{var}", pomdp[var])

    @staticmethod
    def load(directory):
        name = str(np.load(f"{directory}/name.npy"))
        X = np.load(f"{directory}/X.npy")
        Z = np.load(f"{directory}/Z.npy")
        A = list(np.load(f"{directory}/A.npy"))
        P = np.load(f"{directory}/P.npy")
        O = np.load(f"{directory}/O.npy")
        R = np.load(f"{directory}/R.npy")
        gamma = float(np.load(f"{directory}/gamma.npy"))
        miu = np.load(f"{directory}/miu.npy")
        return PartiallyObservableMarkovDecisionProcess(name, X, range(len(A)), Z, P, O, R, gamma, miu, action_meanings=A)

    # ##### #
    # Valid #
    # ##### #

    @staticmethod
    def validate_pomdp(states, actions, observations, observation_probabilities, observation_meanings, tolerance=0.001):

        num_states = len(states)
        num_actions = len(actions)
        num_observations = len(observations)

        assert observation_probabilities.shape == (num_actions, num_states, num_observations)
        for a in range(num_actions):
            for y in range(num_states):
                assert np.isclose(observation_probabilities[a, y].sum(), 1.0, tolerance), \
                    f"Observation probabilities for action #{a} and next_state #{y} don't add up to 1.0"
        if observation_meanings is not None:
            assert len(observation_meanings) == num_observations, \
                f"Size of non-empty observation meanings {len(observation_meanings)} must match the number of observations {num_observations}"

# ################ #
# Cassandra Parser #
# ################ #

PREAMBLE_LABELS = ('discount:', 'values:', 'states:', 'actions:', 'observations:')
INITIAL_LABELS = ('start:', 'start')

def is_equal(x, y, tolerance=1e-8):
    return abs(x - y) < tolerance

def is_valid_item(item, item_list):
    return item in item_list or (isinstance(item, int) and 0 <= item < len(item_list))

def get_index(item, item_list):
    if not is_valid_item(item, item_list):
        raise ValueError('get_index -> invalid item')
    if item in item_list:
        return item_list.index(item)
    return item

def is_stochastic_matrix(m, n, p):
    if not isinstance(p, np.ndarray):
        return False
    if p.shape != (m, n):
        return False
    if not all(is_equal(np.sum(p, axis=1), 1)):
        #print(np.sum(p, axis=1))
        return False
    return True

def process_tokens(fname):
    f = open(fname, 'r')
    t = f.readlines()
    f.close()

    # Clean lines and delete comment lines

    tokens = []
    for lin in t:

        # Remove comments

        if lin[0] == '#':
            continue

        # Clean white space characters and end-of-line comments

        aux = lin.split()
        if aux:
            for j in range(len(aux)):
                if aux[j][0] == '#':
                    aux = aux[:j]
                    break
            tokens += [aux]

    return tokens

def parse_index(element, domain):
    try:
        x = int(element)
    except ValueError:
        x = element

    if is_valid_item(x, domain):
        return get_index(x, domain)
    else:
        raise ValueError('parse: invalid element')

def process_preamble(tok):
    """ process_preamble : list -> dict

        process_preamble(l) goes over the list l searching for the POMDP prb information and fills in a dictionary
        structure with that info. Preamble lines are removed from l (destructively). The returned dictionary
        includes the fields:

        . states       -> a list of entities corresponding to the POMDP states.
        . actions      -> a list of entities corresponding to the POMDP actions.
        . observations -> a list of entities corresponding to the POMDP observations.
        . discount     -> the value of the discount for the POMDP.
        . cost         -> +1 or -1, depending on whether the POMDP is defined as rewards or costs.

        If any of the fields is not provided, the function raises an exception. """

    fields = ['states', 'actions', 'observations', 'discount', 'cost']
    prb = dict()

    i = 0
    while i < len(tok):
        line = tok[i]

        if line[0] == 'states:':
            if len(line) == 1:
                line = line + tok[i + 1]
                del (tok[i + 1])

            if len(line) > 2:
                prb['states'] = line[1:]
            elif line[1].isdigit():
                prb['states'] = list(range(int(line[1])))
            else:
                raise ValueError('process_preamble -> invalid specification')
            del (tok[i])

        elif line[0] == 'actions:':
            if len(line) > 2 or len(line) == 1:
                if len(line) == 1:
                    line = line + tok[i + 1]
                    del (tok[i + 1])
                prb['actions'] = line[1:]
            elif line[1].isdigit():
                prb['actions'] = list(range(int(line[1])))
            else:
                raise ValueError('process_preamble -> invalid specification')
            del (tok[i])

        elif line[0] == 'observations:':
            if len(line) > 2 or len(line) == 1:
                if len(line) == 1:
                    line = line + tok[i + 1]
                    del (tok[i + 1])
                prb['observations'] = line[1:]
            elif line[1].isdigit():
                prb['observations'] = list(range(int(line[1])))
            else:
                raise ValueError('process_preamble -> invalid specification')
            del (tok[i])

        elif line[0] == 'discount:':
            if len(line) > 2:
                raise ValueError('process_preamble -> invalid specification')

            prb['discount'] = float(line[1])
            del (tok[i])

        elif line[0] == 'values:':
            if len(line) > 2:
                raise ValueError('process_preamble -> invalid specification')

            if line[1] == 'reward':
                prb['cost'] = -1
            elif line[1] == 'cost':
                prb['cost'] = 1
            else:
                raise ValueError('process_preamble -> invalid specification')
            del (tok[i])

        else:
            i += 1

    for fld in fields:
        if fld not in prb:
            raise ValueError('process_preamble -> incomplete specification (missing %s)' % fld)

    #print('Preamble: processed successfully.')

    return prb

def process_start(tok, states):
    """ process_start : list x tuple -> np.array

        process_start(l, t) goes over the list l and processes the information regarding the POMDP's initial
        distribution over the states in t. It returns a 1D numpy array with that distribution. If no distribution is
        specified, the uniform distribution is returned."""

    # Initial state or distribution over X

    sdist = None
    i = 0

    while i < len(tok):
        line = tok[i]

        if line[0] == 'start:':

            # Distribution

            if len(line) == 1:
                line = line + tok[i + 1]
                del (tok[i + 1])

            if len(line) > 2:
                if sdist is None:
                    sdist = []

                for p in line[1:]:
                    try:
                        sdist += [float(p)]
                    except:
                        raise ValueError('process_start -> invalid specification')

                if len(sdist) != len(states) or sum(sdist) != 1:
                    raise ValueError('process_start -> invalid specification')
                else:
                    sdist = np.array(sdist)

            # Uniform

            elif line[1] == 'uniform':
                sdist = np.ones(len(states)) / len(states)

            # Single state

            else:
                # Figure out state

                try:
                    sdist = parse_index(line[1], states)
                except ValueError:
                    raise ValueError('process_start -> invalid specification')

            del (tok[i])

        # Distribution over subset of X

        elif line[0] == 'start':
            if len(line) < 3:
                raise ValueError('process_start -> invalid specification')

            if tok[1] == 'include:':
                sdist = np.zeros(len(states))

            elif tok[1] == 'exclude:':
                sdist = np.ones(len(states))

            else:
                raise ValueError('process_start -> invalid specification')

            for x in line[2:]:
                try:
                    x = int(x)
                except ValueError:
                    pass

                if not is_valid_item(x, states):
                    raise ValueError('process_start -> invalid specification')
                else:
                    sdist[get_index(x, states)] = 1 - sdist[get_index(x, states)]

            del (tok[i])

        else:
            i += 1

    if sdist is None:
        sdist = np.ones(len(states)) / len(states)

    if not is_equal(sum(sdist), 1.0):
        raise ValueError('process_start -> invalid specification')

    #print('Start: processed successfully.')

    return np.array(sdist)

def process_stochastic_matrix(tok, label, rows, columns, actions):
    """ process_stochastic_matrix : list x str x list x list x list -> list

        process_stochastic_matrix(l, s, x, y, a) goes over the list l searching for the stochastic matrix with
        label s. The matrix is indexed by the elements of x in its rows and the elements of y in its columns. The
        elements in a correspond to the actions according to which the matrix will be built. Probability lines are
        removed from l (destructively). Each element of the returned list corresponds to a stochastic matrix for
        the corresponding action (the matrices are ordered according to the actions in a).

        If any of the single-action probabilities is not provided, the function raises an exception. """

    m = [None] * len(actions)

    i = 0
    while i < len(tok):
        line = tok[i]

        if line[0] == label:
            if len(line) == 7:
                if line[2] != ':' or line[4] != ':':
                    raise ValueError('process_stochastic_matrix -> invalid specification')

                # Identify action

                if line[1] == '*':
                    act = actions
                else:
                    try:
                        act = [parse_index(line[1], actions)]
                    except ValueError:
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                if line[3] == '*':
                    row = rows
                else:
                    try:
                        row = [parse_index(line[3], rows)]
                    except ValueError:
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                if line[5] == '*':
                    column = columns
                else:
                    try:
                        column = [parse_index(line[5], columns)]
                    except ValueError:
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                for a in act:
                    a_idx = get_index(a, actions)
                    if m[a_idx] is None:
                        m[a_idx] = np.zeros((len(rows), len(columns)))

                    for x in row:
                        x_idx = get_index(x, rows)

                        for y in column:
                            y_idx = get_index(y, columns)
                            m[a_idx][x_idx][y_idx] = np.double(line[6])

                del (tok[i])

            elif len(line) == 4:
                if line[2] != ':':
                    raise ValueError('process_stochastic_matrix -> invalid specification')

                # Identify action

                if line[1] == '*':
                    act = actions
                else:
                    try:
                        act = [parse_index(line[1], actions)]
                    except ValueError:
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                if line[3] == '*':
                    row = rows
                else:
                    try:
                        row = [parse_index(line[3], rows)]
                    except ValueError:
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                try:
                    values = tok[i + 1]
                    del (tok[i + 1])
                except IndexError:
                    raise ValueError('process_stochastic_matrix -> invalid specification')

                if values == ['uniform']:
                    values = [1 / len(columns)] * len(columns)

                for a in act:
                    a_idx = get_index(a, actions)
                    if m[a_idx] is None:
                        m[a_idx] = np.zeros((len(rows), len(columns)))

                    for x in row:
                        x_idx = get_index(x, rows)

                        for y_idx in range(len(columns)):
                            try:
                                m[a_idx][x_idx][y_idx] = np.double(values[y_idx])
                            except IndexError:
                                raise ValueError('process_stochastic_matrix -> invalid specification')

                del (tok[i])

            elif len(line) == 2:

                # Identify action

                if line[1] == '*':
                    act = actions
                else:
                    try:
                        act = [parse_index(line[1], actions)]
                    except ValueError:
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                try:
                    if tok[i + 1] == ['uniform']:
                        values = np.ones((len(rows), len(columns))) / len(columns)
                        del (tok[i + 1])
                    elif tok[i + 1] == ['identity']:
                        values = np.eye(len(rows), len(columns))
                        del (tok[i + 1])
                    else:
                        values = tok[i + 1:i + len(rows) + 1]
                        del (tok[i + 1:i + len(rows) + 1])

                except IndexError:
                    raise ValueError('process_stochastic_matrix -> invalid specification')

                for a in act:
                    a_idx = get_index(a, actions)
                    if m[a_idx] is None:
                        m[a_idx] = np.zeros((len(rows), len(columns)))

                    for x_idx in range(len(rows)):
                        for y_idx in range(len(columns)):
                            try:
                                m[a_idx][x_idx][y_idx] = np.double(values[x_idx][y_idx])
                            except IndexError:
                                raise ValueError('process_stochastic_matrix -> invalid specification')

                del (tok[i])
            else:
                raise ValueError('process_stochastic_matrix -> invalid specification')
        else:
            i += 1

    for a in range(len(actions)):
        if not is_stochastic_matrix(len(rows), len(columns), m[a]):
            #print('action:', a)
            raise ValueError('process_stochastic_matrix -> invalid specification (incomplete or incorrect P)')

    #print('Stochastic matrix %s processed successfully.' % label)

    return np.array(m)

def process_reward(tok, states, actions, observations, p, o, inv=1):
    """ process_reward : list x list x list x list x list x list x int -> np.array

        process_reward(l, s, a, z, p, o, inv) goes over the list l searching for a cost matrix. The matrix is 4D,
        depending on actions (a), states (s), next states and observations (z). The function averages out both
        next states and observations to provide as a result a cost matrix that is a function of s and a. For the
        next state and observation to be averaged out, the function requires the transition and observation
        probabilities (p and o). Finally, the cost is also normalized, for which reason it is also required the bit
        inv (+1 or -1) to turn rewards into costs in the [0,1] interval.

        If the reward is mis-specified in the list tok, the function raises an exception. """

    cost = np.zeros((len(states), len(actions)))

    i = 0
    while i < len(tok):
        line = tok[i]

        if line[0] == 'R:':
            if len(line) >= 9:
                if line[2] != ':' or line[4] != ':' or line[6] != ':':
                    raise ValueError('process_reward -> invalid specification')

                # Identify action

                if line[1] == '*':
                    act = actions
                else:
                    try:
                        act = [parse_index(line[1], actions)]
                    except ValueError:
                        raise ValueError('process_reward -> invalid specification')

                if line[3] == '*':
                    state_init = states
                else:
                    try:
                        state_init = [parse_index(line[3], states)]
                    except ValueError:
                        raise ValueError('process_reward -> invalid specification')

                if line[5] == '*':
                    state_end = states
                else:
                    try:
                        state_end = [parse_index(line[5], states)]
                    except ValueError:
                        raise ValueError('process_reward -> invalid specification')

                if line[7] == '*':
                    obs = observations
                else:
                    try:
                        obs = [parse_index(line[7], observations)]
                    except ValueError:
                        raise ValueError('process_reward -> invalid specification')

                for a in act:
                    a_idx = get_index(a, actions)
                    for x in state_init:
                        x_idx = get_index(x, states)
                        c_aux = cost[x_idx][a_idx]

                        for y in state_end:
                            y_idx = get_index(y, states)

                            for z in obs:
                                z_idx = get_index(z, observations)
                                c_aux += float(line[8]) * p[a_idx][x_idx][y_idx] * o[a_idx][y_idx][z_idx]

                        cost[x_idx][a_idx] = c_aux

                del (tok[i])

            elif len(line) == 6:
                if line[2] != ':' or line[4] != ':':
                    raise ValueError('process_reward -> invalid specification')

                # Identify action

                if line[1] == '*':
                    act = actions
                else:
                    try:
                        act = [parse_index(line[1], actions)]
                    except ValueError:
                        raise ValueError('process_reward -> invalid specification')

                if line[3] == '*':
                    state_init = states
                else:
                    try:
                        state_init = [parse_index(line[3], states)]
                    except ValueError:
                        raise ValueError('process_reward -> invalid specification')

                if line[5] == '*':
                    state_end = states
                else:
                    try:
                        state_end = [parse_index(line[5], states)]
                    except ValueError:
                        raise ValueError('process_reward -> invalid specification')

                try:
                    values = tok[i + 1]
                    del (tok[i + 1])
                except IndexError:
                    raise ValueError('process_reward -> invalid specification')

                for a in act:
                    a_idx = get_index(a, actions)
                    for x in state_init:
                        x_idx = get_index(x, states)
                        c_aux = cost[x_idx][a_idx]

                        for y in state_end:
                            y_idx = get_index(y, states)

                            for z_idx in range(len(observations)):
                                c_aux += float(values[z_idx]) * p[a_idx][x_idx][y_idx] * o[a_idx][y_idx][z_idx]

                        cost[x_idx][a_idx] = c_aux

                del (tok[i])

            elif len(line) == 4:
                if line[2] != ':':
                    raise ValueError('process_reward -> invalid specification')

                # Identify action

                if line[1] == '*':
                    act = actions
                else:
                    try:
                        act = [parse_index(line[1], actions)]
                    except ValueError:
                        raise ValueError('process_reward -> invalid specification')

                if line[3] == '*':
                    state_init = states
                else:
                    try:
                        state_init = [parse_index(line[3], states)]
                    except ValueError:
                        raise ValueError('process_reward -> invalid specification')

                try:
                    values = tok[i + 1:i + len(states) + 1]
                    del (tok[i + 1:i + len(states) + 1])
                except IndexError:
                    raise ValueError('process_reward -> invalid specification')

                for a in act:
                    a_idx = get_index(a, actions)
                    for x in state_init:
                        x_idx = get_index(x, states)
                        c_aux = 0

                        for y_idx in range(len(states)):
                            for z_idx in range(len(observations)):
                                c_aux += float(values[y_idx][z_idx]) * p[a_idx][x_idx][y_idx] * \
                                         o[a_idx][y_idx][z_idx]

                        cost[x_idx][a_idx] = c_aux

                del (tok[i])

        else:
            i += 1

    cost *= inv

    if inv == -1:
        for a in range(len(actions)):
            m = cost.max().max()
            n = cost.min().min()
            cost = (cost - n) / (m - n)

    return cost

def parse_cassandra_pomdp_file(fname):
    tokens = process_tokens(fname)
    preamble = process_preamble(tokens)
    states = preamble["states"]
    actions = preamble["actions"]
    observations = preamble['observations']
    discount_factor = preamble["discount"]
    P = process_stochastic_matrix(tokens, 'T:', states, states, actions)
    O = process_stochastic_matrix(tokens, 'O:', states, observations, actions)
    R = process_reward(tokens, states, actions, observations, P, O)
    init = process_start(tokens, states)
    #print('File', fname, 'processed successfully.')
    return states, actions, observations, P, O, R, discount_factor, init