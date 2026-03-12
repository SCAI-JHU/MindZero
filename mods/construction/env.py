import math
import os

import numpy as np
from PIL import Image, ImageDraw


STAY, UP, DOWN, LEFT, RIGHT = "stay", "up", "down", "left", "right"
PICK, PUT = "pick", "put"
TELEPORT = "teleport"

ACTION_DELTAS = {
    UP: (0, 1),
    DOWN: (0, -1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}

MOVE_ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTIONS = [STAY, UP, DOWN, LEFT, RIGHT, PICK, PUT]

COLOR_NAMES = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "pink",
    "yellow",
    "brown",
    "cyan",
    "gray",
]

COLOR_RGBS = [
    [255, 0, 0],
    [0, 150, 255],
    [0, 200, 0],
    [255, 165, 0],
    [148, 0, 211],
    [255, 192, 203],
    [255, 255, 0],
    [165, 42, 42],
    [0, 255, 255],
    [128, 128, 128],
]

COLOR_TO_RGB = {name: rgb for name, rgb in zip(COLOR_NAMES, COLOR_RGBS)}

SHAPE_NAMES = ["circle", "square", "triangle", "star", "pentagon"]


class ConstructionEnv:
    def __init__(
        self,
        grid_size=(10, 10),
        num_blocks=8,
        max_steps=100,
        seed=None,
        layout_id=0,
        blocked_positions=None,
        object_colors=None,
        object_shapes=None,
    ):
        self.layout_id = layout_id
        self.grid_size = grid_size
        if blocked_positions is not None:
            self.width, self.height = grid_size
            self.blocked = {tuple(pos) for pos in blocked_positions}
            self.free_cells = [
                (x, y)
                for x in range(self.width)
                for y in range(self.height)
                if (x, y) not in self.blocked
            ]
            self.layout_id = None
        else:
            (
                self.width,
                self.height,
                self.blocked,
                self.free_cells,
            ) = self._load_layout(layout_id, grid_size)
            self.grid_size = (self.width, self.height)
        self.num_blocks = num_blocks
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        if object_colors is None:
            object_colors = COLOR_NAMES[: self.num_blocks]
        if object_shapes is None:
            object_shapes = [
                SHAPE_NAMES[i % len(SHAPE_NAMES)] for i in range(self.num_blocks)
            ]
        assert len(object_colors) == self.num_blocks, "object_colors length mismatch"
        assert len(object_shapes) == self.num_blocks, "object_shapes length mismatch"
        self.object_colors = list(object_colors)
        self.object_shapes = list(object_shapes)
        self.object_labels = [
            f"{self.object_colors[i]} {self.object_shapes[i]}"
            for i in range(self.num_blocks)
        ]

        # ^ load agent sprites
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        human_img_path = os.path.join(assets_dir, "human_agent.png")
        helper_img_path = os.path.join(assets_dir, "helper_agent.png")
        self.human_sprite = self._load_sprite(human_img_path)
        self.helper_sprite = self._load_sprite(helper_img_path)

        assert len(self.free_cells) >= self.num_blocks + 2, (
            "Not enough free cells for agents and blocks"
        )

        from .utils import build_shortest_paths

        self.shortest_paths = build_shortest_paths(
            self.width, self.height, self.blocked
        )

        self.reset()

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def reset(self, human_goal=None):
        sampled = self.rng.choice(len(self.free_cells), size=2, replace=False)
        agent_positions = [self.free_cells[i] for i in sampled]
        self.human_pos = np.array(agent_positions[0])
        self.helper_pos = np.array(agent_positions[1])
        self.block_positions = self._sample_block_positions(agent_positions)

        self.human_holding = None
        self.helper_holding = None

        if human_goal is None:
            goal_ids = self.rng.choice(self.num_blocks, size=2, replace=False)
            self.human_goal = tuple(sorted(goal_ids))
        else:
            self.human_goal = tuple(sorted(human_goal))

        self.time_step = 0
        self.human_traj = [tuple(self.human_pos)]
        self.helper_traj = [tuple(self.helper_pos)]
        return self._get_obs()

    def export_config(self):
        return {
            "grid_size": list(self.grid_size),
            "num_objects": self.num_blocks,
            "object_colors": list(self.object_colors),
            "object_shapes": list(self.object_shapes),
            "num_blocks": len(self.blocked),
            "block_pos": [list(pos) for pos in sorted(self.blocked)],
            "max_steps": self.max_steps,
        }

    def get_state(self):
        human_holding = (
            self.object_labels[int(self.human_holding)]
            if self.human_holding is not None
            else None
        )
        helper_holding = (
            self.object_labels[int(self.helper_holding)]
            if self.helper_holding is not None
            else None
        )
        object_pos = {
            self.object_labels[i]: self.block_positions[i].tolist()
            for i in range(self.num_blocks)
        }
        return {
            "human_pos": self.human_pos.tolist(),
            "helper_pos": self.helper_pos.tolist(),
            "object_pos": object_pos,
            "human_holding": human_holding,
            "helper_holding": helper_holding,
            "human_goal": [
                self.object_labels[int(self.human_goal[0])],
                self.object_labels[int(self.human_goal[1])],
            ],
            "time_step": self.time_step,
        }

    def set_state(self, state):
        self.human_pos = np.array(state["human_pos"])
        self.helper_pos = np.array(state["helper_pos"])
        object_pos = state["object_pos"]
        if self.object_labels[0] in object_pos:
            self.block_positions = [
                np.array(object_pos[label]) for label in self.object_labels
            ]
        else:
            color_pos = {label.split(" ")[0]: pos for label, pos in object_pos.items()}
            self.block_positions = [
                np.array(color_pos[label.split(" ")[0]]) for label in self.object_labels
            ]
        label_to_index = {label: i for i, label in enumerate(self.object_labels)}
        color_to_index = {}
        for i, color in enumerate(self.object_colors):
            color_to_index.setdefault(color, i)

        def resolve_index(label):
            if label is None:
                return None
            if label in label_to_index:
                return label_to_index[label]
            if label in color_to_index:
                return color_to_index[label]
            parts = label.split(" ")
            if len(parts) >= 2:
                candidate = " ".join(parts[:2])
                if candidate in label_to_index:
                    return label_to_index[candidate]
                if parts[0] in color_to_index:
                    return color_to_index[parts[0]]
            return None

        self.human_holding = resolve_index(state["human_holding"])
        self.helper_holding = resolve_index(state["helper_holding"])
        goal_a = resolve_index(state["human_goal"][0])
        goal_b = resolve_index(state["human_goal"][1])
        assert goal_a is not None and goal_b is not None, "Invalid human_goal labels"
        self.human_goal = (goal_a, goal_b)
        self.time_step = state.get("time_step", 0)
        self.human_traj = [tuple(self.human_pos)]
        self.helper_traj = [tuple(self.helper_pos)]

    def _sample_block_positions(self, agent_positions):
        max_attempts = 200
        for _ in range(max_attempts):
            candidates = [pos for pos in self.free_cells if pos not in agent_positions]
            self.rng.shuffle(candidates)
            chosen = []
            for pos in candidates:
                if all(
                    abs(pos[0] - other[0]) + abs(pos[1] - other[1]) > 1
                    for other in chosen
                ):
                    chosen.append(pos)
                    if len(chosen) == self.num_blocks:
                        return [np.array(p) for p in chosen]
        assert False, "Failed to sample non-adjacent blocks"

    def _load_layout(self, layout_id, grid_size):
        if layout_id is None:
            width, height = grid_size
            blocked = set()
            free_cells = [
                (x, y)
                for x in range(width)
                for y in range(height)
                if (x, y) not in blocked
            ]
            return width, height, blocked, free_cells

        layouts_dir = os.path.join(os.path.dirname(__file__), "layouts")
        layout_path = os.path.join(layouts_dir, f"{layout_id}.txt")
        with open(layout_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip() != ""]

        height = len(lines)
        assert height > 0, "Layout file must have at least one row"
        width = len(lines[0])
        assert width > 0, "Layout file must have at least one column"
        assert all(len(line) == width for line in lines), "Layout rows must align"

        blocked = set()
        for row, line in enumerate(lines):
            y = height - 1 - row
            for x, ch in enumerate(line):
                assert ch in (".", "#"), "Layout must use '.' and '#'"
                if ch == "#":
                    blocked.add((x, y))

        free_cells = [
            (x, y) for x in range(width) for y in range(height) if (x, y) not in blocked
        ]
        return width, height, blocked, free_cells

    def step(self, actions):
        action_human, action_helper = actions
        teleport_target = None
        if isinstance(action_helper, (list, tuple)):
            action_helper, teleport_target = action_helper[0], action_helper[1]
        self.time_step += 1

        # ^ 1. update positions
        self.human_pos = self._get_next_pos(self.human_pos, action_human)
        self.human_traj.append(tuple(self.human_pos))
        if action_helper == TELEPORT:
            target = None
            if teleport_target is not None:
                if (
                    isinstance(teleport_target, (list, tuple))
                    and len(teleport_target) == 2
                ):
                    target = tuple(teleport_target)
                else:
                    target = tuple(self.block_positions[int(teleport_target)])
            if target is None:
                self.helper_traj.append(tuple(self.helper_pos))
            else:
                self.helper_pos = np.array(target)
                self.helper_traj.append(None)
                self.helper_traj.append(tuple(self.helper_pos))
                if self.helper_holding is None:
                    self._try_pickup("helper")
        else:
            self.helper_pos = self._get_next_pos(self.helper_pos, action_helper)
            self.helper_traj.append(tuple(self.helper_pos))

        # ^ 2. sync carried blocks
        if self.human_holding is not None:
            self.block_positions[self.human_holding] = self.human_pos.copy()
        if self.helper_holding is not None:
            self.block_positions[self.helper_holding] = self.helper_pos.copy()

        # ^ 3. handle pick/put actions with priority
        if action_human == PICK:
            self._try_pickup("human")
        if action_helper == PICK:
            self._try_pickup("helper")
        if action_human == PUT:
            self._try_put("human")
        if action_helper == PUT:
            self._try_put("helper")

        # ^ 4. check termination
        goal_achieved = self._is_goal_achieved()
        done = goal_achieved or self.time_step >= self.max_steps

        info = {
            "human_goal": self.human_goal,
            "goal_achieved": goal_achieved,
            "timeout": self.time_step >= self.max_steps,
        }

        return self._get_obs(), done, info

    def _get_next_pos(self, pos, action):
        if action in ACTION_DELTAS:
            dx, dy = ACTION_DELTAS[action]
            next_pos = pos + np.array([dx, dy])
            if self._is_free(next_pos):
                return next_pos
        return pos.copy()

    def _in_bounds(self, pos):
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def _is_free(self, pos):
        return self._in_bounds(pos) and tuple(pos) not in self.blocked

    def _try_pickup(self, agent):
        pos = self.human_pos if agent == "human" else self.helper_pos
        holding = self.human_holding if agent == "human" else self.helper_holding
        if holding is not None:
            return

        # 只拾取地面物体
        candidates = [
            bid
            for bid in range(self.num_blocks)
            if np.array_equal(self.block_positions[bid], pos)
            and bid not in (self.human_holding, self.helper_holding)
        ]

        if candidates:
            block_id = min(candidates)
            if agent == "human":
                self.human_holding = block_id
            else:
                self.helper_holding = block_id
            self.block_positions[block_id] = pos.copy()

    def _try_put(self, agent):
        pos = self.human_pos if agent == "human" else self.helper_pos
        holding = self.human_holding if agent == "human" else self.helper_holding

        if holding is None:
            return

        occupied = any(
            np.array_equal(self.block_positions[bid], pos)
            and bid not in (self.human_holding, self.helper_holding)
            for bid in range(self.num_blocks)
        )
        if occupied:
            return

        if agent == "human":
            self.block_positions[holding] = pos.copy()
            self.human_holding = None
        else:
            self.block_positions[holding] = pos.copy()
            self.helper_holding = None

    def _is_goal_achieved(self):
        id_a, id_b = self.human_goal
        if self._who_carries(id_a) is not None or self._who_carries(id_b) is not None:
            return False
        pos_a, pos_b = self.block_positions[id_a], self.block_positions[id_b]
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]) == 1

    def get_dist(self, pos_a, pos_b):
        pos_a = tuple(pos_a)
        pos_b = tuple(pos_b)
        dist_map = self.shortest_paths.get(pos_a)
        if dist_map is None:
            return float("inf")
        return dist_map.get(pos_b, float("inf"))

    def min_dist_to_adjacent(self, pos, target_pos):
        min_dist = float("inf")
        for dx, dy in ACTION_DELTAS.values():
            adj = (target_pos[0] + dx, target_pos[1] + dy)
            if self._in_bounds(adj) and adj not in self.blocked:
                min_dist = min(min_dist, self.get_dist(pos, adj))
        return min_dist

    def get_human_dist_to_block(self, block_id):
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid block_id: {block_id}")
        return self.get_dist(self.human_pos, self.block_positions[block_id])

    def _get_obs(self):
        return {
            "agents": [
                {
                    "id": 0,
                    "name": "Human",
                    "pos": self.human_pos.tolist(),
                    "holding": self.human_holding,
                },
                {
                    "id": 1,
                    "name": "Helper",
                    "pos": self.helper_pos.tolist(),
                    "holding": self.helper_holding,
                },
            ],
            "blocks": [
                {
                    "id": i,
                    "pos": self.block_positions[i].tolist(),
                    "carried_by": self._who_carries(i),
                }
                for i in range(self.num_blocks)
            ],
            "time_step": self.time_step,
        }

    def _who_carries(self, block_id):
        if self.human_holding == block_id:
            return 0
        if self.helper_holding == block_id:
            return 1
        return None

    def render(
        self,
        mode="ascii",
        show_goal=False,
        show_block_ids=True,
        show_traj=True,
        traj_len=-1,
    ):
        if mode == "ascii":
            return self._render_ascii(show_goal)
        elif mode == "rgb_array":
            return self._render_rgb(show_block_ids, show_traj, traj_len)
        raise ValueError(f"Unknown render mode: {mode}")

    def _render_ascii(self, show_goal):
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        entities = {}

        # ^ 0. mark obstacles
        for x, y in self.blocked:
            grid[y][x] = "#"

        # ^ 1. collect blocks on ground
        for block_id in range(self.num_blocks):
            if self._who_carries(block_id) is None:
                x, y = self.block_positions[block_id]
                char = str(block_id) if block_id < 10 else chr(ord("a") + block_id - 10)
                if show_goal and block_id in self.human_goal:
                    char = char.upper()
                entities.setdefault((x, y), []).append(char)

        # ^ 2. collect agents
        hx, hy = self.human_pos
        human_str = f"H{self.human_holding}" if self.human_holding is not None else "H"
        entities.setdefault((hx, hy), []).append(human_str)

        px, py = self.helper_pos
        helper_str = (
            f"P{self.helper_holding}" if self.helper_holding is not None else "P"
        )
        entities.setdefault((px, py), []).append(helper_str)

        # ^ 3. place sorted entities in grid
        for (x, y), entity_list in entities.items():
            sorted_entities = sorted(entity_list, key=self._entity_sort_key)
            grid[y][x] = "&".join(sorted_entities)

        # ^ 4. format output
        lines = [
            " ".join(f"{cell:>5}" for cell in grid[y])
            for y in range(self.height - 1, -1, -1)
        ]

        header = f"Step {self.time_step}"
        if show_goal:
            header += (
                f" | Goal: blocks {self.human_goal[0]} & {self.human_goal[1]} adjacent"
            )

        return header + "\n" + "\n".join(lines)

    def _entity_sort_key(self, token):
        first = token[0]
        if first.isalpha():
            if first == "H":
                order = 0
            elif first == "P":
                order = 1
            else:
                order = 2
            return (0, order, token)
        return (1, int(token))

    def _render_rgb(self, show_block_ids=False, show_traj=True, traj_len=12):
        cell_size = 50
        img = (
            np.ones(
                (self.height * cell_size, self.width * cell_size, 3), dtype=np.uint8
            )
            * 255
        )

        block_colors = [
            COLOR_TO_RGB.get(color, [128, 128, 128]) for color in self.object_colors
        ]

        # ^ 1. draw grid
        for i in range(self.height + 1):
            img[i * cell_size : i * cell_size + 2, :] = 200
        for i in range(self.width + 1):
            img[:, i * cell_size : i * cell_size + 2] = 200

        # ^ 2. draw obstacles
        for x, y in self.blocked:
            cy = (self.height - 1 - y) * cell_size + cell_size // 2
            cx = x * cell_size + cell_size // 2
            self._draw_solid_square(img, cx, cy, cell_size // 2, [0, 0, 0])

        # ^ 2b. redraw grid over obstacles
        for i in range(self.height + 1):
            img[i * cell_size : i * cell_size + 2, :] = 200
        for i in range(self.width + 1):
            img[:, i * cell_size : i * cell_size + 2] = 200

        # ^ 3. draw trajectories
        if show_traj:
            self._draw_traj(img, self.human_traj, cell_size, traj_len)
            self._draw_traj(img, self.helper_traj, cell_size, traj_len)

        # ^ 4. draw agents
        cy_human = (self.height - 1 - self.human_pos[1]) * cell_size + cell_size // 2
        cx_human = self.human_pos[0] * cell_size + cell_size // 2
        self._draw_sprite(img, cx_human, cy_human, self.human_sprite, cell_size)

        cy_helper = (self.height - 1 - self.helper_pos[1]) * cell_size + cell_size // 2
        cx_helper = self.helper_pos[0] * cell_size + cell_size // 2
        self._draw_sprite(img, cx_helper, cy_helper, self.helper_sprite, cell_size)

        # ^ 5. draw blocks
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        for block_id in range(self.num_blocks):
            if self._who_carries(block_id) is not None:
                continue
            x, y = self.block_positions[block_id]
            cy = (self.height - 1 - y) * cell_size + cell_size // 2
            cx = x * cell_size + cell_size // 2
            self._draw_shape(
                draw,
                cx,
                cy,
                block_colors[block_id % 10],
                self.object_shapes[block_id],
                10,
            )
            if show_block_ids:
                self._draw_block_id(img, cx, cy, block_id, block_colors[block_id % 10])
        img[:] = np.array(img_pil)

        # ^ 6. draw inventory
        if self.human_holding is not None:
            self._draw_inventory(
                img, self.human_pos, self.human_holding, cell_size, block_colors, "tr"
            )
        if self.helper_holding is not None:
            self._draw_inventory(
                img, self.helper_pos, self.helper_holding, cell_size, block_colors, "br"
            )

        return img

    def _draw_circle(self, img, cx, cy, r, color):
        y_coords, x_coords = np.ogrid[: img.shape[0], : img.shape[1]]
        mask = (x_coords - cx) ** 2 + (y_coords - cy) ** 2 <= r**2
        img[mask] = color

    def _draw_shape(self, draw, cx, cy, color, shape, size):
        if shape == "circle":
            bbox = (cx - size, cy - size, cx + size, cy + size)
            draw.ellipse(bbox, fill=tuple(color))
            return
        if shape == "square":
            bbox = (cx - size, cy - size, cx + size, cy + size)
            draw.rectangle(bbox, fill=tuple(color))
            return
        if shape == "triangle":
            points = [
                (cx, cy - size),
                (cx - size, cy + size),
                (cx + size, cy + size),
            ]
            draw.polygon(points, fill=tuple(color))
            return
        if shape == "pentagon":
            points = self._regular_polygon_points(cx, cy, size, 5)
            draw.polygon(points, fill=tuple(color))
            return
        if shape == "star":
            points = self._star_points(cx, cy, size)
            draw.polygon(points, fill=tuple(color))
            return
        bbox = (cx - size, cy - size, cx + size, cy + size)
        draw.rectangle(bbox, fill=tuple(color))

    @staticmethod
    def _regular_polygon_points(cx, cy, radius, sides):
        points = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides - math.pi / 2
            points.append(
                (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            )
        return points

    @staticmethod
    def _star_points(cx, cy, radius):
        points = []
        inner = radius * 0.5
        for i in range(10):
            angle = 2 * math.pi * i / 10 - math.pi / 2
            r = radius if i % 2 == 0 else inner
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        return points

    def _draw_solid_square(self, img, cx, cy, size, color):
        x1, y1 = max(0, cx - size), max(0, cy - size)
        x2, y2 = min(img.shape[1], cx + size), min(img.shape[0], cy + size)
        img[y1:y2, x1:x2] = color

    def _draw_hollow_square(self, img, cx, cy, size, color, thickness):
        x1, y1 = cx - size, cy - size
        x2, y2 = cx + size, cy + size

        for y_start, y_end in [(y1, y1 + thickness), (y2 - thickness, y2)]:
            img[
                max(0, y_start) : min(img.shape[0], y_end),
                max(0, x1) : min(img.shape[1], x2),
            ] = color

        for x_start, x_end in [(x1, x1 + thickness), (x2 - thickness, x2)]:
            img[
                max(0, y1) : min(img.shape[0], y2),
                max(0, x_start) : min(img.shape[1], x_end),
            ] = color

    def _draw_inventory(self, img, pos, block_id, cell_size, block_colors, corner):
        cell_x = pos[0] * cell_size
        cell_y = (self.height - 1 - pos[1]) * cell_size
        if corner == "br":
            cx = cell_x + cell_size - 9
            cy = cell_y + cell_size - 9
        else:
            cx = cell_x + cell_size - 9
            cy = cell_y + 9
        color = block_colors[block_id % 10]
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        shape = self.object_shapes[block_id]
        self._draw_shape(draw, cx, cy, color, shape, 5)
        img[:] = np.array(img_pil)

    def _draw_traj(self, img, traj, cell_size, max_len):
        if len(traj) < 2:
            return
        if max_len == -1:
            points = traj
        else:
            points = traj[-max_len:]
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            if p0 is None or p1 is None:
                continue
            x0, y0 = p0
            x1, y1 = p1
            sx = x0 * cell_size + cell_size // 2
            sy = (self.height - 1 - y0) * cell_size + cell_size // 2
            ex = x1 * cell_size + cell_size // 2
            ey = (self.height - 1 - y1) * cell_size + cell_size // 2
            t = i / max(1, len(points) - 2)
            shade = int(220 - t * 140)
            color = (shade, shade, shade)
            draw.line((sx, sy, ex, ey), fill=color, width=2)

            dx = ex - sx
            dy = ey - sy
            length = max(1.0, (dx**2 + dy**2) ** 0.5)
            ux, uy = dx / length, dy / length
            left = (ex - ux * 6 - uy * 4, ey - uy * 6 + ux * 4)
            right = (ex - ux * 6 + uy * 4, ey - uy * 6 - ux * 4)
            draw.polygon([(ex, ey), left, right], fill=color)
        img[:] = np.array(img_pil)

    def _draw_block_id(self, img, cx, cy, block_id, color):
        text = str(block_id) if block_id < 10 else chr(ord("a") + block_id - 10)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luminance > 186 else (255, 255, 255)

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        bbox = draw.textbbox((0, 0), text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((cx - text_w // 2, cy - text_h // 2), text, fill=text_color)
        img[:] = np.array(img_pil)

    def _draw_sprite(self, img, cx, cy, sprite, cell_size):
        sprite_h, sprite_w = sprite.shape[:2]
        sprite_size = min(cell_size - 4, sprite_h, sprite_w)

        # * resize sprite to fit cell
        if sprite_size != sprite_h or sprite_size != sprite_w:
            sprite_pil = Image.fromarray(sprite)
            sprite_pil = sprite_pil.resize(
                (sprite_size, sprite_size), Image.Resampling.LANCZOS
            )
            sprite = np.array(sprite_pil)

        x1 = max(0, cx - sprite_size // 2)
        y1 = max(0, cy - sprite_size // 2)
        x2 = min(img.shape[1], cx + sprite_size // 2)
        y2 = min(img.shape[0], cy + sprite_size // 2)

        sprite_x1 = max(0, sprite_size // 2 - cx)
        sprite_y1 = max(0, sprite_size // 2 - cy)
        sprite_x2 = sprite_x1 + (x2 - x1)
        sprite_y2 = sprite_y1 + (y2 - y1)

        img[y1:y2, x1:x2] = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]

    def _load_sprite(self, sprite_path):
        sprite_rgba = Image.open(sprite_path).convert("RGBA")
        sprite_arr = np.array(sprite_rgba)
        bg_color = sprite_arr[0, 0, :3]
        bg_mask = np.all(sprite_arr[:, :, :3] == bg_color, axis=2)
        sprite_arr[bg_mask, 3] = 0

        white_bg = Image.new("RGBA", sprite_rgba.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(white_bg, Image.fromarray(sprite_arr))
        return np.array(composed.convert("RGB"))
