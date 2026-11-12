"""绘制已有训练日志，不重新训练。直接运行默认使用SARSA结果。

横轴为累计环境交互步数，纵轴为单回合累积奖励。
奖励口径遵循项目最终模型假设A08–A10；数组工具[R06]；绘图工具[R07]。
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_returns(run_dir):
    with (run_dir / "training_log.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    episodes = np.array([int(row["episode"]) for row in rows])
    returns = np.array([float(row["episode_return"]) for row in rows])
    steps = np.array([int(row["steps"]) for row in rows])
    cumulative_steps = np.array([int(row["cumulative_steps"]) for row in rows])
    if not len(rows) or not np.array_equal(episodes, np.arange(1, len(rows) + 1)):
        raise ValueError("训练日志必须非空，且回合编号从1开始连续。")
    if not np.all(np.isfinite(returns)) or np.any(steps <= 0):
        raise ValueError("回报必须有限，回合步数必须为正。")
    if not np.array_equal(cumulative_steps, np.cumsum(steps)):
        raise ValueError("累计训练步数与各回合步数求和不一致。")
    if not np.array_equal(returns, -steps):
        raise ValueError("日志不符合本项目每步奖励-1的固定设定。")
    return episodes, cumulative_steps, returns


def main(default_run=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=default_run or ROOT / "outputs/sarsa/design_validation_seed42")
    parser.add_argument("--window", type=int, default=100)  # 平滑窗口：最近多少个回合的回报取等权平均，仅影响显示。
    parser.add_argument("--dpi", type=int, default=180)  # 图片分辨率：控制PNG清晰度，不影响训练或统计。
    parser.add_argument("--no-show", action="store_true", help="只保存图片，不弹出窗口")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    episodes, cumulative_steps, returns = load_returns(args.run_dir)
    if not 1 <= args.window <= len(returns):
        raise ValueError("平滑窗口须在1和训练回合总数之间。")
    if args.dpi <= 0:
        raise ValueError("图片分辨率必须为正。")
    metadata = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    # 完整的尾随窗口：第window回合才有平滑值，不填充或借用未来回合。
    smoothed = np.convolve(returns, np.ones(args.window) / args.window, mode="valid")
    np.testing.assert_allclose(smoothed[0], returns[:args.window].mean())
    np.testing.assert_allclose(smoothed[-1], returns[-args.window:].mean())

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6.4), layout="constrained")
    fig.set_facecolor("#fafbfd")
    ax.plot(cumulative_steps, returns, color="#9bbde3", alpha=0.6, linewidth=0.65,
            label="单回合累积奖励（含超时回合）")
    ax.plot(cumulative_steps[args.window - 1:], smoothed, color="#175c9e", linewidth=2,
            label=f"最近{args.window}回合滑动平均")
    decay = config["epsilon_decay_episodes"]
    if 1 <= decay <= episodes[-1]:
        ax.axvline(cumulative_steps[decay - 1], color="#bc7a34", linestyle="--", linewidth=1.1,
                   label=f"探索率降至 {config['epsilon_end']}（第{decay}回合）")
    ax.set_title(f"{metadata['algorithm']} 训练曲线：累积奖励随训练步数的变化\n"
                 f"seed={config['seed']}  |  α={config['alpha']}  |  "
                 f"γ={metadata['environment']['gamma']}  |  共{len(returns)}回合",
                 fontsize=15, pad=15)
    ax.set_xlabel("累计训练步数（环境交互次数）", fontsize=12)
    ax.set_ylabel("单回合累积奖励（回报）", fontsize=12)
    ax.set_xlim(0, cumulative_steps[-1])
    ax.set_ylim(bottom=min(returns.min() * 1.05, -1), top=0)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    fig.supxlabel("数据：本项目训练日志；每回合结束时记录一个点，奖励越高越好。绘图 [R07]。",
                  fontsize=10, color="#526070")
    output = args.run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"training_return_curve.{suffix}", dpi=args.dpi)
    statistics = {
        "source": str((args.run_dir / "training_log.csv").resolve()),
        "metric": "episode_return", "x_axis": "cumulative_steps",
        "episodes": len(returns), "total_training_steps": int(cumulative_steps[-1]),
        "smoothing_window": args.window, "smoothed_points": len(smoothed),
        "first_window_mean": float(smoothed[0]),
        "last_window_mean": float(smoothed[-1]),
        "minimum_episode_return": float(returns.min()),
        "maximum_episode_return": float(returns.max()),
    }
    (output / "training_return_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"图片保存于：{output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
