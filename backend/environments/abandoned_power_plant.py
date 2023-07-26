import numpy as np

from backend.environments.abandoned_power_plant_py.room_cleaner import GarbageCollectionPOMDP
from backend.environments.abandoned_power_plant_py.room_explorer import EnvironmentReckonPOMDP


def create(model_id, noise):

    adjacency_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ])

    goals = [
        [0, 1, 4],  # explorable nodes
        [2, 4]      # dirty nodes
    ]
    factory = [
        EnvironmentReckonPOMDP,
        GarbageCollectionPOMDP
    ]

    return factory[model_id](adjacency_matrix, goals[model_id], noise, id=f"abandoned_power_plant-v{model_id}")
