"""冻结已训练Q表，绘制首个贪心评估回合的真实轨迹，不挑选最好路径。

复用sarsa.evaluate_sarsa的评估约定；数组工具[R06]，绘图[R07]；
BFS[R04]只用于路径长度核验，不生成绘图轨迹。文献见参考文献目录。
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gridworld import ROOT, GridMap
from sarsa import evaluate_sarsa


def main(default_run=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=default_run or ROOT / "outputs/sarsa/design_validation_seed42")
    parser.add_argument("--seed", type=int, default=None)  # 评估种子：默认沿用已存评估种子，复现并列最大Q值的选择。
    parser.add_argument("--dpi", type=int, default=180)  # 图片分辨率：仅控制PNG清晰度，不影响策略。
    parser.add_argument("--no-show", action="store_true", help="仅保存图片，不弹窗")
    args = parser.parse_args()
    if args.dpi <= 0 or (args.seed is not None and args.seed < 0):
        raise ValueError("dpi须为正，seed须非负。")
    metadata = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    previous_file = args.run_dir / "evaluation.json"
    previous = json.loads(previous_file.read_text(encoding="utf-8")) if previous_file.exists() else None
    # 无历史评估时，采用与训练入口一致的独立评估种子偏移1000000。
    seed = args.seed if args.seed is not None else (
        previous["seed"] if previous else metadata["config"]["seed"] + 1000000)
    data = metadata["map"]  # 使用训练时的地图快照，避免地图后来修改造成错配。
    grid = GridMap(data["rows"], data["cols"], tuple(data["start"]), tuple(data["goal"]),
                   frozenset(tuple(p) for p in data["obstacles"]))
    q_file = args.run_dir / "q_table.npy"
    q = np.load(q_file, allow_pickle=False)
    before = q.copy()
    # 评估回合数1：展示首个回合；evaluate_sarsa内部epsilon=0，关闭探索，不更新Q。
    summary, records = evaluate_sarsa(grid, q, episodes=1, seed=seed)
    np.testing.assert_array_equal(q, before)
    path = summary["first_episode_path"]
    if len(path) != records[0]["elapsed_time"] + 1 or path[0] != grid.start:
        raise ValueError("轨迹长度或起点与评估记录不一致。")
    for state, nxt in zip(path, path[1:]):
        best = np.flatnonzero(q[state] == q[state].max())
        if not any(grid.next_position(state, int(a)) == nxt for a in best):
            raise ValueError("轨迹包含不符合贪心策略的转移。")
    historical_match = None
    if previous and seed == previous["seed"]:
        historical_match = [list(p) for p in path] == previous["first_episode_path"]
        if not historical_match:
            raise ValueError("重新执行的轨迹与历史首回合不一致，请检查Q表和代码版本。")
    shortest = len(grid.shortest_path()) - 1  # BFS独立基准[R04]，不参与选动作。
    result = {
        **records[0], "seed": seed, "epsilon": 0.0, "path": path,
        "coordinate_system": "zero-based (row, column), top-left origin",
        "blocked_steps": sum(a == b for a, b in zip(path, path[1:])),
        "bfs_shortest_steps": shortest,
        "matches_shortest_length": records[0]["success"] and len(path) - 1 == shortest,
        "matches_historical_first_path": historical_match,
        "q_table_sha256": hashlib.sha256(q_file.read_bytes()).hexdigest(),
        "source_run": str(args.run_dir.resolve()),
    }

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    cells = np.zeros((grid.rows, grid.cols), dtype=int)
    for p in grid.obstacles:
        cells[p] = 1
    cells[grid.start], cells[grid.goal] = 2, 3
    colors = ["#f8fafc", "#64748b", "#bce6ce", "#f9d58b"]
    fig, ax = plt.subplots(figsize=(8.5, 9.3))
    ax.imshow(cells, cmap=ListedColormap(colors), vmin=0, vmax=3, origin="upper")
    ax.set_xticks(range(grid.cols))
    ax.set_yticks(range(grid.rows))
    ax.set_xticks(np.arange(grid.cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid.rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#c4ced8", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("列坐标（从0开始）")
    ax.set_ylabel("行坐标（从0开始）")
    for (r, c), (nr, nc) in zip(path, path[1:]):
        if (r, c) != (nr, nc):
            ax.annotate("", xy=(nc, nr), xytext=(c, r),
                        arrowprops=dict(arrowstyle="-|>", color="#1665b5", lw=2.5,
                                        shrinkA=11, shrinkB=11, mutation_scale=15))
    visits = {}
    for step, p in enumerate(path):
        visits.setdefault(p, []).append(step)
    for (r, c), steps in visits.items():
        label = "S" if (r, c) == grid.start else "G" if (r, c) == grid.goal else ""
        # 重复访问不隐藏：完整时序在JSON中，图上标明访问次数。
        label = label or (str(steps[0]) if len(steps) == 1 else f"{steps[0]}×{len(steps)}")
        ax.text(c, r, label, ha="center", va="center", fontsize=12, weight="bold",
                color="#124775", bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.9))
    status = "成功到达" if result["success"] else "超时未到达"
    ax.set_title(f"{metadata['algorithm']} 最终策略路径：S → G\n{status} | 动作步数 {result['elapsed_time']} | "
                 f"累积奖励 {result['episode_return']}", fontsize=15, pad=15)
    ax.legend(handles=[Patch(facecolor=colors[1], label="障碍"),
                       Patch(facecolor=colors[2], label="起点 S"),
                       Patch(facecolor=colors[3], label="终点 G"),
                       Line2D([0], [0], color="#1665b5", lw=2.5, label="实际行进方向")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False)
    fig.text(0.5, 0.045, f"冻结 Q 表，ε=0；评估种子 {seed}；数字为经过该格的动作步序。",
             ha="center", fontsize=10)
    fig.text(0.5, 0.02, "地图：题目图1；轨迹：本项目策略执行结果；绘图：Matplotlib [R07]。",
             ha="center", fontsize=9, color="#526070")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.19)
    output = args.run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"final_policy_path.{suffix}", dpi=args.dpi)
    (output / "final_policy_path.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
