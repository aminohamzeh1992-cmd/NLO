import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def is_interactive_backend():
    backend = plt.get_backend().lower()
    return not any(name in backend for name in ("agg", "pdf", "svg", "ps"))


def draw_cell(ax, row, col, color, alpha=0.5):
    """Färbt genau eine Grid-Zelle ein."""
    rect = Rectangle(
        (col - 0.5, row - 0.5),
        1,
        1,
        facecolor=color,
        edgecolor="none",
        alpha=alpha
    )
    ax.add_patch(rect)


def render_episode(env, agent, max_steps=200, delay=0.2,
                   title="Agent Rollout",
                   save_path=None,
                   save_trajectory_path=None):
    interactive = is_interactive_backend()
    state = env.reset()

    # Grid: 0 = frei, 1 = Wand
    grid = np.zeros((env.size, env.size))
    for x, y in env.obstacles:
        grid[x, y] = 1

    fig, ax = plt.subplots(figsize=(8, 8))

    # Wichtig: extent sorgt dafür, dass Zellgrenzen bei +/-0.5 liegen
    ax.imshow(
        grid,
        cmap="Greys",
        origin="upper",
        extent=(-0.5, env.size - 0.5, env.size - 0.5, -0.5)
    )

    # Start / Ziel / Türen als Zellen markieren
    sx, sy = env.start
    gx, gy = env.goal

    draw_cell(ax, sx, sy, "green", alpha=0.35)
    draw_cell(ax, gx, gy, "red", alpha=0.35)

    for dx, dy in env.doors:
        draw_cell(ax, dx, dy, "blue", alpha=0.25)

    # Text in Zellmitte
    ax.text(sy, sx, "S", ha="center", va="center",
            color="green", fontsize=14, fontweight="bold")
    ax.text(gy, gx, "G", ha="center", va="center",
            color="red", fontsize=14, fontweight="bold")

    for dx, dy in env.doors:
        ax.text(dy, dx, "D", ha="center", va="center",
                color="blue", fontsize=12, fontweight="bold")
    # Items / Gegenstände anzeigen
    item_texts = []

    if hasattr(env, "item_positions"):
        for i, (ix, iy) in enumerate(env.item_positions):
            draw_cell(ax, ix, iy, "yellow", alpha=0.45)

            txt = ax.text(
                iy, ix, f"I{i + 1}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold"
            )

            item_texts.append(txt)
    # Gitterlinien auf Zellgrenzen
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8)

    # Normale Ticks auf Zellzentren
    ax.set_xticks(np.arange(env.size))
    ax.set_yticks(np.arange(env.size))

    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)

    # Startposition Agent
    x, y = env.to_xy(state)
    # Trajektorie zusätzlich als Liste speichern
    path_cells = [(x, y)]
    # Agent als Kreis in Zellmitte
    agent_dot, = ax.plot(
        y, x,
        marker="o",
        markersize=14,
        color="orange",
        markeredgecolor="black"
    )

    # Pfad initialisieren
    path_x = [y]
    path_y = [x]

    path_line, = ax.plot(
        path_x, path_y,
        linestyle="-",
        linewidth=2,
        color="orange",
        alpha=0.8
    )

    ax.set_title(title)

    if interactive:
        plt.ion()
        plt.show()

    total_reward = 0.0

    for step in range(max_steps):
        action = agent.act(state, explore=False)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        collected_mask = info.get("collected_mask", None)

        if collected_mask is not None:
            for i, txt in enumerate(item_texts):
                item_collected = bool(collected_mask & (1 << i))
                txt.set_visible(not item_collected)
        x, y = env.to_xy(next_state)
        path_cells.append((x, y))

        # Agent verschieben
        agent_dot.set_data([y], [x])

        # Pfad erweitern
        path_x.append(y)
        path_y.append(x)
        path_line.set_data(path_x, path_y)

        if collected_mask is not None and hasattr(env, "item_positions"):
            collected_count = info.get("collected_count", 0)
            item_status = f"Items: {collected_count}/{len(env.item_positions)}"

            ax.set_title(
                f"{title} | Schritt {step + 1} | Reward: {total_reward:.1f} | {item_status}"
            )
        else:
            ax.set_title(
                f"{title} | Schritt {step + 1} | Reward: {total_reward:.1f}"
            )

        if interactive:
            plt.pause(delay)

        state = next_state

        if done:
            ax.set_title(
                f"{title} | Ziel erreicht nach {step + 1} Schritten | Reward: {total_reward:.1f}"
            )
            if interactive:
                plt.pause(1.5)
            break

    # Endbild mit komplettem Pfad speichern
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved rollout image to: {save_path}")

    # Optional: Pfad als CSV speichern
    if save_trajectory_path is not None:
        np.savetxt(
            save_trajectory_path,
            np.array(path_cells, dtype=int),
            fmt="%d",
            delimiter=",",
            header="row,col",
            comments=""
        )
        print(f"Saved rollout trajectory to: {save_trajectory_path}")

    if interactive:
        plt.pause(3.0)
    plt.close(fig)
