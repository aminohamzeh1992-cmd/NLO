"""NLO Salesman entrypoint.

Default task: 9 rooms, random items, exact optimal route.

Examples:
    python main.py
    python main.py --rooms 9 --items 5 --item_seed 42 --oracle_only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from env import make_salesman_random
from oracle import (
    plot_salesman_route,
    save_route_csv,
    save_route_summary,
    solve_naive_route,
    solve_optimal_route,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLO Salesman oracle runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rooms", type=int, default=9, choices=[4, 6, 9])
    parser.add_argument("--items", type=int, default=5)
    parser.add_argument("--item_seed", type=int, default=42)
    parser.add_argument("--out", default="results_oracle")
    parser.add_argument(
        "--oracle_only",
        action="store_true",
        help="Compatibility flag: this entrypoint currently computes the oracle route.",
    )
    args = parser.parse_args()

    env = make_salesman_random(
        n_items=args.items,
        rooms=args.rooms,
        seed=args.item_seed,
    )
    optimal = solve_optimal_route(env)
    naive = solve_naive_route(env)

    out_dir = Path(args.out)
    stem = f"Rooms{args.rooms}_Items{args.items}_Seed{args.item_seed}"
    save_route_csv(optimal, env, out_dir / f"Oracle_Optimal_{stem}.csv")
    save_route_csv(naive, env, out_dir / f"Oracle_Naive_{stem}.csv")
    save_route_summary(optimal, naive, env, out_dir / f"Oracle_{stem}.txt")
    plot_salesman_route(
        env,
        optimal,
        out_dir / f"Oracle_Optimal_{stem}.png",
        naive=naive,
        title=(
            f"Oracle route | rooms={args.rooms} items={args.items} "
            f"seed={args.item_seed} | steps={optimal.steps}"
        ),
    )

    improvement = naive.steps - optimal.steps
    percent = 0.0 if naive.steps == 0 else 100.0 * improvement / naive.steps
    print(f"Rooms: {args.rooms}")
    print(f"Items: {env.item_positions}")
    print(f"Optimal steps: {optimal.steps}")
    print(f"Optimal item order: {optimal.order_label}")
    print(f"Naive steps: {naive.steps}")
    print(f"Saved steps: {improvement} ({percent:.2f}%)")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
