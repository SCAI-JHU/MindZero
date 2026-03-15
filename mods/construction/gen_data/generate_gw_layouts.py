import os
import random
from collections import deque


def generate_layouts(layout_dir, seed=0):
    width, height = _load_base_size(os.path.join(layout_dir, "0.txt"))
    rng = random.Random(seed)

    for i in range(1, 11):
        blocked = _sample_blocked(width, height, 10, rng)
        _write_layout(layout_dir, i, width, height, blocked)

    for i in range(11, 21):
        blocked = _sample_blocked(width, height, 15, rng)
        _write_layout(layout_dir, i, width, height, blocked)

    for i in range(21, 31):
        blocked = _sample_blocked(width, height, 20, rng)
        _write_layout(layout_dir, i, width, height, blocked)

    for i in range(31, 51):
        blocked = _generate_maze_blocked(width, height, rng)
        _write_layout(layout_dir, i, width, height, blocked)


def _load_base_size(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]
    assert lines, "Base layout is empty"
    width = len(lines[0])
    height = len(lines)
    assert all(len(line) == width for line in lines), "Base layout rows must align"
    return width, height


def _write_layout(layout_dir, layout_id, width, height, blocked):
    lines = []
    for y in range(height - 1, -1, -1):
        line = []
        for x in range(width):
            line.append("#" if (x, y) in blocked else ".")
        lines.append("".join(line))
    path = os.path.join(layout_dir, f"{layout_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _sample_blocked(width, height, count, rng):
    max_attempts = 500
    all_cells = [(x, y) for x in range(width) for y in range(height)]
    for _ in range(max_attempts):
        blocked = set(rng.sample(all_cells, count))
        if _is_connected(width, height, blocked):
            return blocked
    assert False, "Failed to sample connected layout"


def _is_connected(width, height, blocked):
    free_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in blocked]
    if not free_cells:
        return False
    start = free_cells[0]
    visited = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if 0 <= nx < width and 0 <= ny < height and nxt not in blocked and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return len(visited) == len(free_cells)


def _generate_maze_blocked(width, height, rng):
    blocked = {(x, y) for x in range(width) for y in range(height)}
    cell_w = (width - 1) // 2
    cell_h = (height - 1) // 2
    assert cell_w > 0 and cell_h > 0, "Layout too small for maze"

    visited = set()
    stack = [(0, 0)]
    visited.add((0, 0))
    _carve_cell(blocked, 0, 0)

    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cell_w and 0 <= ny < cell_h and (nx, ny) not in visited:
                neighbors.append((nx, ny))
        if not neighbors:
            stack.pop()
            continue

        nx, ny = rng.choice(neighbors)
        visited.add((nx, ny))
        _carve_passage(blocked, cx, cy, nx, ny)
        _carve_cell(blocked, nx, ny)
        stack.append((nx, ny))

    return blocked


def _carve_cell(blocked, cx, cy):
    x = 2 * cx + 1
    y = 2 * cy + 1
    blocked.discard((x, y))


def _carve_passage(blocked, cx, cy, nx, ny):
    x1 = 2 * cx + 1
    y1 = 2 * cy + 1
    x2 = 2 * nx + 1
    y2 = 2 * ny + 1
    blocked.discard((x2, y2))
    blocked.discard(((x1 + x2) // 2, (y1 + y2) // 2))


if __name__ == "__main__":
    layouts_dir = os.path.join(os.path.dirname(__file__), "layouts")
    generate_layouts(layouts_dir, seed=0)

