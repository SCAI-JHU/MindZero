from collections import deque

import numpy as np

from .env import ACTION_DELTAS


def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def in_bounds(pos, width, height):
    return 0 <= pos[0] < width and 0 <= pos[1] < height


def get_next_pos(pos, action, width, height):
    if action not in ACTION_DELTAS:
        return pos
    dx, dy = ACTION_DELTAS[action]
    next_pos = (pos[0] + dx, pos[1] + dy)
    if in_bounds(next_pos, width, height):
        return next_pos
    return pos


def min_dist_to_adjacent(pos, target_pos, width, height):
    min_dist = float("inf")
    for dx, dy in ACTION_DELTAS.values():
        adj_pos = (target_pos[0] + dx, target_pos[1] + dy)
        if in_bounds(adj_pos, width, height):
            dist = manhattan_dist(pos, adj_pos)
            min_dist = min(min_dist, dist)
    return min_dist


def boltzmann_policy(q_values, temperature):
    actions = list(q_values.keys())
    q_array = np.array([q_values[a] for a in actions])

    logits = q_array / temperature
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    return {actions[i]: probs[i] for i in range(len(actions))}


def discounted_return(num_steps, gamma, action_cost):
    # * Return = -sum_{i=0}^{n-1} gamma^i * action_cost
    # * = -action_cost * (1 - gamma^n) / (1 - gamma)  [geometric series]
    if num_steps == 0:
        return 0.0
    return -action_cost * (1 - gamma**num_steps) / (1 - gamma)


def build_shortest_paths(width, height, blocked):
    free_cells = [
        (x, y)
        for x in range(width)
        for y in range(height)
        if (x, y) not in blocked
    ]
    all_dist = {}
    for start in free_cells:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ACTION_DELTAS.values():
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and nxt not in blocked
                    and nxt not in dist
                ):
                    dist[nxt] = dist[(x, y)] + 1
                    queue.append(nxt)
        all_dist[start] = dist
    return all_dist
