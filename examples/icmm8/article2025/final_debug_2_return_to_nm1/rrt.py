import numpy as np
import random
from typing import Tuple


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


def distance(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)


def steer(from_node, to_node, step_size):
    d = distance(from_node, to_node)
    if d < step_size:
        return Node(to_node.x, to_node.y, from_node)
    else:
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        return Node(from_node.x + step_size * np.cos(theta),
                    from_node.y + step_size * np.sin(theta),
                    from_node)


def is_collision_free(node_from, node_to, is_restricted, step=0.5):
    """
    Checks if any of the [node_from, node_to] points are not intersecting
    the restricted area.
    """
    d = distance(node_from, node_to)
    steps = int(d / step)
    for i in range(steps + 1):
        x = node_from.x + (node_to.x - node_from.x) * i / steps
        y = node_from.y + (node_to.y - node_from.y) * i / steps
        if is_restricted(x, y):  # punkt w przeszkodzie
            return False
    return True


def extract_path(node):
    path = []
    while node:
        path.append([node.x, node.y])
        node = node.parent
    return path[::-1]


def rrt(start: Tuple[float, float], goal: Tuple[float, float],
        is_restricted, n_points=1000, step_size=1.0, max_iter=5000,
        x_range=(0, 100), y_range=(0, 100)):
    start_node = Node(*start)
    goal_node = Node(*goal)
    tree = [start_node]

    for _ in range(max_iter):
        rnd = Node(random.uniform(*x_range), random.uniform(*y_range))
        nearest = min(tree, key=lambda node: distance(node, rnd))
        new_node = steer(nearest, rnd, step_size)

        if not is_restricted(new_node.x, new_node.y) and is_collision_free(nearest, new_node, is_restricted):
            tree.append(new_node)
            if distance(new_node, goal_node) < step_size:
                if is_collision_free(new_node, goal_node, is_restricted):
                    goal_node.parent = new_node
                    path = extract_path(goal_node)
                    return interpolate_path(path, n_points), tree
    return None, tree


def interpolate_path(path, n_points):
    path = np.array(path)
    dists = np.cumsum([0] + [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))])
    total_length = dists[-1]
    new_dists = np.linspace(0, total_length, n_points)
    new_path = []
    for nd in new_dists:
        i = np.searchsorted(dists, nd) - 1
        i = max(0, min(i, len(path) - 2))
        t = (nd - dists[i]) / (dists[i+1] - dists[i] + 1e-9)
        new_point = (1 - t) * path[i] + t * path[i+1]
        new_path.append(new_point.tolist())
    return np.array(new_path)
