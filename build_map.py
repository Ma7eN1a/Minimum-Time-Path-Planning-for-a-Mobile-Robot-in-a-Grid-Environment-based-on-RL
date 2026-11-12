"""在 PyCharm 直接运行，保存地图并显示窗口；--no-show 仅保存图片。

绘图工具 Matplotlib [R07]: https://doi.org/10.1109/MCSE.2007.55
地图取自题目图1；坐标编号和配色为本项目添加。
"""

import argparse
import json

from gridworld import ROOT, GridMap


def main():
    parser = argparse.ArgumentParser(description="Reconstruct the assignment grid")
    parser.add_argument("--no-show", action="store_true", help="Save without GUI")
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    grid = GridMap.load()
    path = grid.shortest_path()
    output = ROOT / "outputs" / "maps"
    output.mkdir(parents=True, exist_ok=True)
    cells = [[int((r, c) in grid.obstacles) for c in range(grid.cols)]
             for r in range(grid.rows)]
    cells[grid.start[0]][grid.start[1]] = 2
    cells[grid.goal[0]][grid.goal[1]] = 3

    colors = ["#ffffff", "#808080", "#cdebe0", "#ffe3a3"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cells, cmap=ListedColormap(colors), vmin=0, vmax=3,
              origin="upper", interpolation="nearest")
    ax.set_xticks(range(grid.cols))
    ax.set_yticks(range(grid.rows))
    ax.set_xticks([x - 0.5 for x in range(grid.cols + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(grid.rows + 1)], minor=True)
    ax.grid(which="minor", color="#353b43", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Column (zero-based)", labelpad=12)
    ax.set_ylabel("Row (zero-based)")
    for label, (r, c) in [("S", grid.start), ("G", grid.goal)]:
        ax.text(c, r, label, ha="center", va="center", fontsize=22,
                weight="bold", color="#183044")
    ax.set_title("Assignment map | 10 x 10 | 15 obstacles", pad=50, fontsize=15)
    ax.legend(handles=[Patch(facecolor=color, edgecolor="#555", label=label)
                       for color, label in zip(colors, ["Free", "Obstacle", "Start", "Goal"])],
              loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False)
    fig.text(0.5, 0.025, "Source: assignment PDF, Fig. 1 | Rendering: Matplotlib [R07]",
             ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(top=0.80, bottom=0.13)
    fig.savefig(output / "original_map.png", dpi=180, bbox_inches="tight")
    summary = {
        "coordinate_system": "zero-based (row, column), top-left origin",
        "rows": grid.rows, "cols": grid.cols,
        "start": grid.start, "goal": grid.goal,
        "obstacles": sorted(grid.obstacles),
        "obstacle_count": len(grid.obstacles),
        "free_cell_count": grid.rows * grid.cols - len(grid.obstacles),
        "reachable": bool(path),
        "bfs_shortest_steps": len(path) - 1,
        "bfs_path": path,
        "note": "BFS validation only [R04]; not a learned policy",
    }
    (output / "map_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(grid.to_ascii())
    print(f"Start={grid.start}; Goal={grid.goal}; Obstacles={len(grid.obstacles)}")
    print(f"BFS shortest steps: {len(path) - 1} [R04]; not an RL result")
    print(f"Saved: {output / 'original_map.png'}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
