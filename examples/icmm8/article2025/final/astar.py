import heapq
import numpy as np


def astar(start, goal, is_restricted, step=0.01, goal_tol=2.0):
    """
    start, goal: (x, y) w przestrzeni ciągłej
    is_restricted: funkcja (x, y) -> bool, True jeśli przeszkoda
    step: rozdzielczość siatki (np. 0.1)
    """
    # Move (0, 0) to the start point
    start = np.asarray(start).squeeze()[:2]
    goal = np.asarray(goal).squeeze()[:2]
    center = start

    def discretize(point):
        point = np.asarray(point)
        point = point - center
        return np.round(point / step)

    def continuous(point):
        point = np.asarray(point)
        point = point * step
        return point + center

    start_d = np.round(discretize(start)).astype(int)  # grid point
    goal_d = discretize(goal)   # grid point

    def heuristic(a, b):
        return np.sum(np.abs(a-b))

    def make_tuple(array):
        return tuple(array.tolist())

    neighbors = np.asarray([(0, 1), (0, -1), (1, 0), (-1, 0)])

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start_d, goal_d), 0, start_d))

    came_from = {}
    g_score = {make_tuple(start_d): 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        current = np.asarray(current)

        if np.linalg.norm(current-goal_d) < goal_tol:
            path = [goal]
            current = make_tuple(current)
            while current in came_from:
                path.append(continuous(current))
                current = came_from[current]
            path.append(continuous(start_d))
            return path[::-1]

        for dd in neighbors:
            neighbor = current + dd
            current_t = make_tuple(current)
            neighbor_t = make_tuple(neighbor)

            cx, cy = continuous(neighbor)
            if is_restricted(cx, cy):
                continue

            tentative_g = g_score[current_t] + 1

            if tentative_g < g_score.get(neighbor_t, float('inf')):
                came_from[neighbor_t] = current_t
                g_score[neighbor_t] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_d)
                f_score = float(f_score)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor_t))
    return None


def interpolate_path(path, n_points):
    """
    Interpoluje ścieżkę do zadanej liczby punktów.
    path: lista punktów [(x,y), ...]
    n_points: liczba punktów wynikowych
    """
    if not path or n_points < 2:
        return path

    pts = np.array(path)
    # długości odcinków
    seg_vecs = pts[1:] - pts[:-1]
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)
    total_length = np.sum(seg_lengths)

    if total_length == 0:
        return [tuple(pts[0])] * n_points

    # pozycje docelowe wzdłuż ścieżki
    target_dists = np.linspace(0, total_length, n_points)

    result = []
    seg_cum = np.cumsum(seg_lengths)
    seg_start = np.insert(seg_cum[:-1], 0, 0.0)

    for d in target_dists:
        idx = np.searchsorted(seg_cum, d)
        if idx >= len(seg_lengths):
            result.append(tuple(pts[-1]))
        else:
            t = (d - seg_start[idx]) / seg_lengths[idx]
            point = pts[idx] + t * seg_vecs[idx]
            result.append(tuple(point))

    return result