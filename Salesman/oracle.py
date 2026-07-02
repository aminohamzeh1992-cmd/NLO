"""Exact oracle routes for the Salesman-style MultiItemWorld.

The oracle solves the small travelling-salesman subproblem exactly:
start -> all items in the best order -> goal. Distances are real shortest
paths in the grid, so walls and doors are respected.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from env import make_salesman_random


@dataclass(frozen=True)
class RouteResult:
    """A complete route through all items."""

    order: tuple[int, ...]
    steps: int
    path: list[tuple[int, int]]
    segment_steps: tuple[int, ...]

    @property
    def order_1based(self) -> tuple[int, ...]:
        return tuple(i + 1 for i in self.order)

    @property
    def order_label(self) -> str:
        if not self.order:
            return "-"
        return "-".join(str(i) for i in self.order_1based)


def shortest_path(env, start: tuple[int, int],
                  goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Return a real shortest grid path from start to goal, inclusive."""
    if start == goal:
        return [start]

    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q: deque[tuple[int, int]] = deque([start])

    while q:
        pos = q.popleft()
        for dx, dy in env.ACTIONS.values():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in parent or not env.is_valid(nxt):
                continue
            parent[nxt] = pos
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)

    if goal not in parent:
        raise ValueError(f"No path from {start} to {goal}.")

    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _pair_paths(env, points: list[tuple[int, int]]
                ) -> dict[tuple[int, int], list[tuple[int, int]]]:
    paths: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i != j:
                paths[(i, j)] = shortest_path(env, a, b)
    return paths


def _compose_path(
    pair_paths: dict[tuple[int, int], list[tuple[int, int]]],
    point_order: list[int],
) -> tuple[list[tuple[int, int]], tuple[int, ...]]:
    full: list[tuple[int, int]] = []
    segment_steps: list[int] = []
    for a, b in zip(point_order, point_order[1:]):
        segment = pair_paths[(a, b)]
        segment_steps.append(len(segment) - 1)
        if full:
            full.extend(segment[1:])
        else:
            full.extend(segment)
    return full, tuple(segment_steps)


def route_for_order(env, order: tuple[int, ...]) -> RouteResult:
    """Build the exact path for a fixed zero-based item order."""
    n_items = len(getattr(env, "item_positions", []))
    if sorted(order) != list(range(n_items)):
        raise ValueError(
            f"Order must contain each item index exactly once: {order}"
        )

    points = [env.start] + list(env.item_positions) + [env.goal]
    goal_idx = n_items + 1
    paths = _pair_paths(env, points)
    point_order = [0] + [i + 1 for i in order] + [goal_idx]
    path, segment_steps = _compose_path(paths, point_order)
    return RouteResult(
        order=order,
        steps=len(path) - 1,
        path=path,
        segment_steps=segment_steps,
    )


def solve_optimal_route(env) -> RouteResult:
    """Solve the item order exactly with Held-Karp dynamic programming."""
    n_items = len(getattr(env, "item_positions", []))
    if n_items == 0:
        return route_for_order(env, ())

    points = [env.start] + list(env.item_positions) + [env.goal]
    goal_idx = n_items + 1
    paths = _pair_paths(env, points)
    dist = {k: len(v) - 1 for k, v in paths.items()}

    # dp[(mask, last)] = (cost, order_tuple), where last is a zero-based item.
    dp: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
    for item in range(n_items):
        mask = 1 << item
        dp[(mask, item)] = (dist[(0, item + 1)], (item,))

    for mask in range(1, 1 << n_items):
        for last in range(n_items):
            current = dp.get((mask, last))
            if current is None:
                continue
            cost, order = current
            for nxt in range(n_items):
                bit = 1 << nxt
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_cost = cost + dist[(last + 1, nxt + 1)]
                new_order = order + (nxt,)
                old = dp.get((new_mask, nxt))
                if old is None or (new_cost, new_order) < old:
                    dp[(new_mask, nxt)] = (new_cost, new_order)

    all_mask = (1 << n_items) - 1
    best: tuple[int, tuple[int, ...]] | None = None
    for last in range(n_items):
        cost, order = dp[(all_mask, last)]
        total = cost + dist[(last + 1, goal_idx)]
        candidate = (total, order)
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    _, order = best
    return route_for_order(env, order)


def solve_naive_route(env) -> RouteResult:
    """Route that visits items in environment order."""
    return route_for_order(env, tuple(range(len(env.item_positions))))


def save_route_csv(route: RouteResult, env, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    item_by_pos = {pos: i + 1 for i, pos in enumerate(env.item_positions)}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "row", "col", "marker", "item"])
        for step, (row, col) in enumerate(route.path):
            marker = ""
            if (row, col) == env.start:
                marker = "start"
            elif (row, col) == env.goal:
                marker = "goal"
            item = item_by_pos.get((row, col), "")
            writer.writerow([step, row, col, marker, item])
    print(f"Saved: {path}")


def save_route_summary(
    optimal: RouteResult,
    naive: RouteResult,
    env,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    improvement = naive.steps - optimal.steps
    percent = 0.0 if naive.steps == 0 else 100.0 * improvement / naive.steps
    with path.open("w", encoding="utf-8") as f:
        f.write("Salesman oracle summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"size={env.size} rooms={getattr(env, 'rooms', '?')}\n")
        f.write(f"start={env.start} goal={env.goal}\n")
        f.write(f"items={list(env.item_positions)}\n")
        f.write(f"optimal_steps={optimal.steps}\n")
        f.write(f"optimal_order_items_1based={optimal.order_label}\n")
        f.write(f"optimal_segment_steps={list(optimal.segment_steps)}\n")
        f.write(f"naive_steps={naive.steps}\n")
        f.write(f"naive_order_items_1based={naive.order_label}\n")
        f.write(f"improvement_steps={improvement}\n")
        f.write(f"improvement_percent={percent:.2f}\n")
    print(f"Saved: {path}")


def plot_salesman_route(
    env,
    optimal: RouteResult,
    path: str | Path,
    naive: RouteResult | None = None,
    title: str | None = None,
) -> None:
    """Save an oracle route image in the same grid style as the project."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grid = np.zeros((env.size, env.size))
    for x, y in env.obstacles:
        grid[x, y] = 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        grid,
        cmap="Greys",
        origin="upper",
        extent=(-0.5, env.size - 0.5, env.size - 0.5, -0.5),
    )

    if naive is not None:
        nx = [c[1] for c in naive.path]
        ny = [c[0] for c in naive.path]
        ax.plot(nx, ny, "--", color="tab:gray", alpha=0.55,
                linewidth=2.0, label=f"Naive ({naive.steps})")

    ox = [c[1] for c in optimal.path]
    oy = [c[0] for c in optimal.path]
    ax.plot(ox, oy, "-o", color="tab:orange", alpha=0.86,
            linewidth=2.2, markersize=3.5,
            label=f"Optimal ({optimal.steps})")
    ax.plot(ox[0], oy[0], marker="s", markersize=11,
            color="green", markeredgecolor="black")
    ax.plot(ox[-1], oy[-1], marker="*", markersize=15,
            color="red", markeredgecolor="black")

    sx, sy = env.start
    gx, gy = env.goal
    import matplotlib.patheffects as pe
    label_effect = [pe.withStroke(linewidth=2, foreground="black")]
    ax.text(sy, sx, "S", ha="center", va="center",
            color="white", fontsize=13, fontweight="bold",
            path_effects=label_effect)
    ax.text(gy, gx, "G", ha="center", va="center",
            color="white", fontsize=13, fontweight="bold",
            path_effects=label_effect)

    for dx, dy in env.doors:
        ax.text(dy, dx, "D", ha="center", va="center",
                color="blue", fontsize=9, fontweight="bold", alpha=0.8)

    order_rank = {item_idx: rank for rank, item_idx in enumerate(optimal.order, 1)}
    for i, (ix, iy) in enumerate(env.item_positions):
        ax.text(iy, ix, f"I{i + 1}", ha="center", va="center",
                color="black", fontsize=10, fontweight="bold",
                bbox=dict(facecolor="gold", edgecolor="black",
                          boxstyle="round,pad=0.18", alpha=0.85))
        rank = order_rank.get(i)
        if rank is not None:
            ax.text(iy + 0.34, ix - 0.34, str(rank),
                    ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold",
                    bbox=dict(facecolor="tab:orange", edgecolor="none",
                              boxstyle="circle,pad=0.18", alpha=0.95))

    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    if title is None:
        title = (
            f"Salesman oracle | steps={optimal.steps} | "
            f"order={optimal.order_label}"
        )
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute exact optimal Salesman route.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rooms", type=int, default=9, choices=[4, 6, 9])
    p.add_argument("--items", type=int, default=5)
    p.add_argument("--item_seed", type=int, default=42)
    p.add_argument("--out", default="results_oracle")
    args = p.parse_args()

    env = make_salesman_random(
        n_items=args.items,
        rooms=args.rooms,
        seed=args.item_seed,
    )
    optimal = solve_optimal_route(env)
    naive = solve_naive_route(env)

    out = Path(args.out)
    stem = f"Rooms{args.rooms}_Items{args.items}_Seed{args.item_seed}"
    save_route_csv(optimal, env, out / f"Oracle_Optimal_{stem}.csv")
    save_route_csv(naive, env, out / f"Oracle_Naive_{stem}.csv")
    save_route_summary(optimal, naive, env, out / f"Oracle_{stem}.txt")
    plot_salesman_route(
        env,
        optimal,
        out / f"Oracle_Optimal_{stem}.png",
        naive=naive,
        title=(
            f"Oracle route | rooms={args.rooms} items={args.items} "
            f"seed={args.item_seed} | steps={optimal.steps}"
        ),
    )

    improvement = naive.steps - optimal.steps
    percent = 0.0 if naive.steps == 0 else 100.0 * improvement / naive.steps
    print(
        f"Optimal: {optimal.steps} steps, order {optimal.order_label}; "
        f"naive: {naive.steps} steps; saved {improvement} "
        f"steps ({percent:.2f}%)."
    )


if __name__ == "__main__":
    main()
