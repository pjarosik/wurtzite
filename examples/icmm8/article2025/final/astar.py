import heapq


def astar(start, goal, is_restricted):
    """
    start, goal: (x, y)
    is_restricted: funkcja (x, y) -> bool, True jeśli przeszkoda
    """

    # Heurystyka: odległość Manhattan (dla ruchu w 4 kierunkach)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Ruchy: góra, dół, lewo, prawo
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))

    came_from = {}  # śledzenie ścieżki
    g_score = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            # Odtwórz ścieżkę
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if is_restricted(*neighbor):
                continue

            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return None  # Brak ścieżki