import itertools

import numpy as np

X, O, S, G = range(4)
BLOCKED, OPEN, START, GOAL = range(4)
TAGS = ["x", " ", "S", "G"]
TAG_MEANINGS = ["Blocked", "Open", "Start", "Goal"]

ntu = np.array([
    [X, X, X, S, X, X],
    [X, G, O, O, G, X],
    [S, O, X, X, O, X],
    [X, O, X, X, O, S],
    [X, G, O, O, G, X],
    [X, X, S, X, X, X]
])

nju = np.array([
    [X, X, X, S, X, X],
    [X, G, O, O, O, X],
    [S, O, X, X, G, X],
    [X, G, O, O, O, S]
])

isr = np.array([
    [X, O, X, S, X, O, X, O, X],
    [O, O, O, O, O, G, O, O, O],
    [X, O, X, X, X, X, X, O, X],
    [O, O, X, X, X, X, X, O, S],
    [X, O, X, X, X, X, X, O, X],
    [S, O, X, X, X, X, X, O, O],
    [X, O, X, G, X, X, X, O, X],
    [G, O, O, O, O, O, O, O, O],
    [X, X, O, O, X, O, X, O, X],
    [X, X, O, S, X, X, X, X, X],
])

mit = np.array([
    [X, X, S, X, O, X, O, X, X, X, X, X, O, X, G, X, X],
    [X, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, X],
    [S, O, X, O, X, X, X, O, X, X, O, X, X, X, X, O, G],
    [X, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, X],
    [X, X, S, X, O, X, O, X, X, X, X, X, O, X, G, X, X],
])

pentagon = np.array([
    [X, S, O, O, O, O, O, O, O, S, X],
    [X, O, X, X, X, O, X, X, X, O, X],
    [X, O, X, X, X, O, X, X, X, O, X],
    [X, O, X, X, O, O, O, G, X, O, X],
    [X, O, O, O, O, X, O, O, O, O, G],
    [X, O, X, X, O, O, O, X, X, O, X],
    [X, O, X, X, X, O, X, X, X, O, X],
    [X, O, X, X, X, O, O, X, X, O, X],
    [G, O, O, O, O, O, O, O, O, O, X],
])

cit = np.array([
    [X, X, X, O, O, O, O, O, O, O, O, O, O, X],
    [X, X, X, X, O, X, X, X, X, X, X, X, O, X],
    [X, X, X, X, O, O, O, X, X, X, X, O, O, X],
    [O, X, X, X, O, X, O, O, O, O, O, O, O, X],
    [O, O, O, O, O, X, O, X, X, X, G, X, O, X],
    [O, X, X, X, X, X, O, O, G, X, X, X, O, X],
    [O, X, X, X, X, X, X, O, X, X, X, X, O, X],
    [O, X, X, X, X, X, O, O, O, O, O, O, O, G],
    [O, X, X, X, X, X, O, X, X, X, O, X, X, X],
    [O, X, X, X, O, O, O, O, O, O, O, X, X, X],
    [O, O, O, O, O, X, X, X, X, X, S, X, X, X],
    [S, X, X, X, S, X, X, X, X, X, X, X, X, X],
])

def load(environment):
    if environment == "nju":
        return nju
    elif environment == "ntu":
        return ntu
    elif environment == "isr":
        return isr
    elif environment == "mit":
        return mit
    elif environment == "pentagon":
        return pentagon
    elif environment == "cit":
        return cit
    else:
        raise ValueError(f"Invalid environment {environment}")

def possible_tasks(layout):

    if isinstance(layout, str):
        layout = load(layout)

    goals = []
    num_rows, num_columns = layout.shape
    for x in range(num_columns):
        for y in range(num_rows):
            if layout[y, x] == GOAL:
                goals.append((x, y))
    goals = list(itertools.combinations(goals, r=2))
    _goals = []
    for goal in goals:
        goal_a, goal_b = goal
        _goals.append((goal_a[0], goal_a[1], goal_b[0], goal_b[1]))
    return _goals
