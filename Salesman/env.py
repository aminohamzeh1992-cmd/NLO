"""GridWorld and Salesman-style multi-item environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict


class GridWorld:
    """Deterministic gridworld with obstacles and one terminal goal."""

    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    def __init__(
        self,
        size: int = 10,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (9, 9),
        obstacles: Optional[Iterable[tuple[int, int]]] = None,
        doors: Optional[Iterable[tuple[int, int]]] = None,
        step_penalty: float = -1.0,
        goal_reward: float = 10.0,
        invalid_penalty: float = 0.0,
    ):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.doors = list(doors or [])
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.invalid_penalty = invalid_penalty
        self.action_dim = 4
        self.state_dim = size * size
        self.pos = start

    def reset(self) -> np.ndarray:
        self.pos = self.start
        return self.to_vec(self.pos)

    def all_states(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.obstacles
        ]

    def is_valid(self, xy: tuple[int, int]) -> bool:
        x, y = xy
        return 0 <= x < self.size and 0 <= y < self.size and xy not in self.obstacles

    def to_vec(self, xy: tuple[int, int]) -> np.ndarray:
        vec = np.zeros(self.state_dim, dtype=np.float32)
        vec[xy[0] * self.size + xy[1]] = 1.0
        return vec

    def to_xy(self, state: np.ndarray) -> tuple[int, int]:
        idx = int(np.argmax(state))
        return idx // self.size, idx % self.size

    def transition_xy(self, xy: tuple[int, int], action: int) -> tuple[tuple[int, int], bool]:
        dx, dy = self.ACTIONS[action]
        candidate = (xy[0] + dx, xy[1] + dy)
        if self.is_valid(candidate):
            return candidate, True
        return xy, False

    def simulate_step(self, state: np.ndarray, action: int):
        xy = self.to_xy(state)
        next_xy, feasible = self.transition_xy(xy, action)
        done = next_xy == self.goal
        reward = self.goal_reward if done else self.step_penalty
        if not feasible:
            reward += self.invalid_penalty
        return self.to_vec(next_xy), reward, done, {"feasible": feasible}

    def step(self, action: int):
        next_xy, feasible = self.transition_xy(self.pos, action)
        self.pos = next_xy
        done = self.pos == self.goal
        reward = self.goal_reward if done else self.step_penalty
        if not feasible:
            reward += self.invalid_penalty
        return self.to_vec(self.pos), reward, done, {"feasible": feasible}

    def bfs_distance(self) -> dict[tuple[int, int], int]:
        from collections import deque
        dist = {self.goal: 0}
        q = deque([self.goal])
        while q:
            xy = q.popleft()
            for dx, dy in self.ACTIONS.values():
                nb = (xy[0] + dx, xy[1] + dy)
                if self.is_valid(nb) and nb not in dist:
                    dist[nb] = dist[xy] + 1
                    q.append(nb)
        return dist


def _segments(size: int, walls: list[int]) -> list[tuple[int, int]]:
    segments = []
    start = 0
    for w in walls:
        segments.append((start, w - 1))
        start = w + 1
    segments.append((start, size - 1))
    return segments


def _wall_positions(size: int, n_rooms: int) -> list[int]:
    return [k * size // n_rooms for k in range(1, n_rooms)]


def make_regular_rooms(
    room_rows: int = 2,
    room_cols: int = 2,
    size: int = 15,
    goal_reward: float = 10.0,
    step_penalty: float = -1.0,
    invalid_penalty: float = 0.0,
) -> GridWorld:
    obstacles = set()
    doors = []

    horizontal_walls = _wall_positions(size, room_rows)
    vertical_walls = _wall_positions(size, room_cols)
    row_segments = _segments(size, horizontal_walls)
    col_segments = _segments(size, vertical_walls)

    for wall_row in horizontal_walls:
        for col in range(size):
            obstacles.add((wall_row, col))
        for c0, c1 in col_segments:
            door = (wall_row, (c0 + c1) // 2)
            obstacles.discard(door)
            doors.append(door)

    for wall_col in vertical_walls:
        for row in range(size):
            obstacles.add((row, wall_col))
        for r0, r1 in row_segments:
            door = ((r0 + r1) // 2, wall_col)
            obstacles.discard(door)
            doors.append(door)

    start = (0, size - 1)
    goal = (size - 1, 0)
    obstacles.discard(start)
    obstacles.discard(goal)

    return GridWorld(
        size=size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        doors=doors,
        goal_reward=goal_reward,
        step_penalty=step_penalty,
        invalid_penalty=invalid_penalty,
    )


def make_easy_grid() -> GridWorld:
    return GridWorld(size=7, goal=(6, 6), obstacles={(1, 3), (2, 3), (3, 3), (4, 3)})


def make_hard_grid() -> GridWorld:
    return GridWorld(
        size=10,
        goal=(9, 9),
        obstacles={
            (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5),
            (2, 2), (2, 3), (3, 2), (6, 8), (7, 8),
        },
    )


def make_four_rooms() -> GridWorld:
    return make_regular_rooms(room_rows=2, room_cols=2, size=15)


def make_six_rooms() -> GridWorld:
    return make_regular_rooms(room_rows=2, room_cols=3, size=15)


def make_nine_rooms() -> GridWorld:
    return make_regular_rooms(room_rows=3, room_cols=3, size=15)


def make_corridor() -> GridWorld:
    size = 20
    obstacles = set()
    for wall_col in (5, 10, 15):
        for row in range(size):
            if row != size // 2:
                obstacles.add((row, wall_col))
    doors = [(size // 2, 5), (size // 2, 10), (size // 2, 15)]
    return GridWorld(size=size, start=(0, 0), goal=(0, 19), obstacles=obstacles, doors=doors)


class MultiItemWorld:
    """Salesman-style GridWorld: collect all items before the goal counts."""

    ACTIONS = GridWorld.ACTIONS

    def __init__(
        self,
        size: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        item_positions: list[tuple[int, int]],
        obstacles=None,
        doors=None,
        step_penalty: float = -1.0,
        item_reward: float = 5.0,
        goal_reward: float = 20.0,
        missing_items_penalty: float = -2.0,
    ):
        self.size = size
        self.start = start
        self.goal = goal
        self.item_positions = list(item_positions)
        self.obstacles = set(obstacles or [])
        self.doors = list(doors or [])
        self.step_penalty = step_penalty
        self.item_reward = item_reward
        self.goal_reward = goal_reward
        self.missing_items_penalty = missing_items_penalty
        self.n_items = len(self.item_positions)
        self.all_items_mask = (1 << self.n_items) - 1
        self.action_dim = 4
        self.state_dim = size * size * (1 << self.n_items)
        self.pos = self.start
        self.collected_mask = 0

    def reset(self) -> np.ndarray:
        self.pos = self.start
        self.collected_mask = 0
        return self._to_vec(self.pos, self.collected_mask)

    def step(self, action: int):
        next_pos, next_mask, reward, done, info = self._transition(
            self.pos, self.collected_mask, action
        )
        self.pos = next_pos
        self.collected_mask = next_mask
        return self._to_vec(next_pos, next_mask), reward, done, info

    def simulate_step(self, state: np.ndarray, action: int):
        pos, mask = self._from_vec(state)
        next_pos, next_mask, reward, done, info = self._transition(pos, mask, action)
        return self._to_vec(next_pos, next_mask), reward, done, info

    def _transition(self, pos, mask, action):
        dx, dy = self.ACTIONS[action]
        candidate = (pos[0] + dx, pos[1] + dy)
        feasible = self.is_valid(candidate)
        next_pos = candidate if feasible else pos
        reward = self.step_penalty
        done = False
        next_mask = mask
        new_item = None

        for i, item_pos in enumerate(self.item_positions):
            bit = 1 << i
            if next_pos == item_pos and not (mask & bit):
                next_mask |= bit
                reward += self.item_reward
                new_item = i

        if next_pos == self.goal:
            if next_mask == self.all_items_mask:
                reward += self.goal_reward
                done = True
            else:
                reward += self.missing_items_penalty

        return next_pos, next_mask, reward, done, {
            "feasible": feasible,
            "collected_mask": next_mask,
            "collected_count": self._count_bits(next_mask),
            "all_collected": next_mask == self.all_items_mask,
            "new_item": new_item,
        }

    def is_valid(self, xy: tuple[int, int]) -> bool:
        x, y = xy
        return 0 <= x < self.size and 0 <= y < self.size and xy not in self.obstacles

    def all_states(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.obstacles
        ]

    def to_xy(self, state: np.ndarray) -> tuple[int, int]:
        pos, _ = self._from_vec(state)
        return pos

    def to_xy_full(self, state: np.ndarray) -> tuple[int, int, int]:
        pos, collected_mask = self._from_vec(state)
        return pos[0], pos[1], collected_mask

    def to_vec(self, xy: tuple[int, ...], collected_mask: int = 0) -> np.ndarray:
        if len(xy) >= 3:
            collected_mask = int(xy[2])
        return self._to_vec((int(xy[0]), int(xy[1])), collected_mask)

    def _to_vec(self, pos: tuple[int, int], collected_mask: int) -> np.ndarray:
        flat_pos = pos[0] * self.size + pos[1]
        idx = collected_mask * self.size * self.size + flat_pos
        vec = np.zeros(self.state_dim, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def _from_vec(self, state: np.ndarray) -> tuple[tuple[int, int], int]:
        idx = int(np.argmax(state))
        cells = self.size * self.size
        collected_mask = idx // cells
        flat_pos = idx % cells
        return (flat_pos // self.size, flat_pos % self.size), collected_mask

    def bfs_distance(self) -> dict[tuple[int, int], int]:
        from collections import deque
        dist = {self.goal: 0}
        q = deque([self.goal])
        while q:
            xy = q.popleft()
            for dx, dy in self.ACTIONS.values():
                nb = (xy[0] + dx, xy[1] + dy)
                if self.is_valid(nb) and nb not in dist:
                    dist[nb] = dist[xy] + 1
                    q.append(nb)
        return dist

    @staticmethod
    def _count_bits(mask: int) -> int:
        return bin(mask).count("1")


def make_salesman_four_rooms() -> MultiItemWorld:
    base = make_four_rooms()
    items = [(2, 8), (2, 2), (8, 8)]
    return MultiItemWorld(
        size=base.size,
        start=base.start,
        goal=base.goal,
        item_positions=items,
        obstacles=base.obstacles,
        doors=base.doors,
        step_penalty=-1.0,
        item_reward=50.0,
        goal_reward=100.0,
        missing_items_penalty=-2.0,
    )


def make_salesman_random(n_items: int = 3, rooms: int = 9, seed: int = 0) -> MultiItemWorld:
    if rooms == 4:
        base = make_four_rooms()
    elif rooms == 6:
        base = make_six_rooms()
    elif rooms == 9:
        base = make_nine_rooms()
    else:
        raise ValueError("rooms muss 4, 6 oder 9 sein.")

    rng = np.random.default_rng(seed)
    forbidden = set(base.obstacles)
    forbidden.add(base.start)
    forbidden.add(base.goal)
    forbidden.update(base.doors)
    candidates = sorted(xy for xy in base.all_states() if xy not in forbidden)
    if n_items > len(candidates):
        raise ValueError(
            f"Zu viele Items: n_items={n_items}, aber nur {len(candidates)} freie Kandidatenfelder."
        )

    permutation = rng.permutation(len(candidates))
    items = [candidates[int(i)] for i in permutation[:n_items]]
    env = MultiItemWorld(
        size=base.size,
        start=base.start,
        goal=base.goal,
        item_positions=items,
        obstacles=base.obstacles,
        doors=base.doors,
        step_penalty=-1.0,
        item_reward=5.0,
        goal_reward=20.0 + 5.0 * n_items,
        missing_items_penalty=-2.0,
    )
    env.rooms = rooms
    env.item_seed = seed
    return env
