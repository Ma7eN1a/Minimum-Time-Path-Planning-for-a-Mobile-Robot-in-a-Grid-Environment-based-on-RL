"""从论文实验原始日志生成统计与插图；NumPy[R06]、Matplotlib[R07]。

只读实验数据，统计定义为本项目设定；不覆盖历史训练日志。
"""
import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

from gridworld import ROOT

DATA = ROOT / "outputs/paper_experiments_20261104"
OUT = ROOT / "论文/强化学习作业论文/images"
WINDOW = 100  # 平滑窗口：最近100个回合等权平均，不影响训练。
DPI = 220  # 图像分辨率：论文插图清晰度。
COLORS = {"SARSA": "#2166ac", "Q-learning": "#d66022"}
SLUGS = {"SARSA": "sarsa", "Q-learning": "q_learning"}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)), min=float(a.min()), max=float(a.max()))


def grid_panel(ax, data, path=None, title=""):
    cells = np.zeros((data["rows"], data["cols"]), dtype=int)
    for r, c in data["obstacles"]:
        cells[r, c] = 1
    cells[tuple(data["start"])], cells[tuple(data["goal"])] = 2, 3
    ax.imshow(cells, cmap=ListedColormap(["#f8fafc", "#64748b", "#bce6ce", "#f9d58b"]), vmin=0, vmax=3)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticks(np.arange(11)-0.5, minor=True); ax.set_yticks(np.arange(11)-0.5, minor=True)
    ax.grid(which="minor", color="#cbd5e1", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("列"); ax.set_ylabel("行"); ax.set_title(title, fontsize=11)
    if path:
        drawn = set()
        for p, n in zip(path, path[1:]):
            pair = (tuple(p), tuple(n))
            if p != n and pair not in drawn:
                ax.annotate("", xy=(n[1], n[0]), xytext=(p[1], p[0]),
                            arrowprops=dict(arrowstyle="-|>", color="#2166ac", lw=1.7,
                                            shrinkA=3, shrinkB=3, mutation_scale=10))
                drawn.add(pair)
        if path[-1] != data["goal"]:
            ax.scatter(path[-1][1], path[-1][0], marker="x", color="#c9302c", s=70, zorder=6)
    for label, pos in [("S", data["start"]), ("G", data["goal"])]:
        ax.text(pos[1], pos[0], label, ha="center", va="center", weight="bold", fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))


def main():
    global DATA, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    DATA, OUT = args.data, args.output
    OUT.mkdir(parents=True, exist_ok=True)
    results = load(DATA / "summary.json")
    summary = {}
    for alg in SLUGS:
        runs = [r for r in results if r["algorithm"] == alg]
        summary[alg] = {}
        for scenario in ["original", "changed"]:
            block = [r[scenario] for r in runs]
            summary[alg][scenario] = {
                key: mean_sd([r[key] for r in block]) for key in
                ["training_steps", "training_timeouts", "last100_return", "mean_arrival_time", "success_rate"]}
            times = [r["confirmation_steps"] for r in block if r["confirmation_steps"] is not None]
            summary[alg][scenario]["confirmed_count"] = len(times)
            summary[alg][scenario]["confirmation_steps"] = mean_sd(times) if times else None
        summary[alg]["transfer"] = {
            "successes": sum(r["transfer"]["successes"] for r in runs),
            "episodes": sum(r["transfer"]["episodes"] for r in runs),
            "success_rate": mean_sd([r["transfer"]["success_rate"] for r in runs]),
            "mean_arrival_time": mean_sd([r["transfer"]["mean_arrival_time"] for r in runs
                                         if r["transfer"]["mean_arrival_time"] is not None])
                                 if any(r["transfer"]["successes"] for r in runs) else None,
        }
    (DATA / "aggregate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    original, changed = load(DATA / "original_map.json"), load(DATA / "changed_map.json")
    old, new = set(map(tuple, original["obstacles"])), set(map(tuple, changed["obstacles"]))
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.6), layout="constrained")
    grid_panel(axs[0], original, title="原地图")
    grid_panel(axs[1], changed, title=f"新地图：{len(old-new)}/{len(old)} 个障碍更换位置")
    for ax, points in [(axs[0], old-new), (axs[1], new-old)]:
        for r,c in points:
            ax.add_patch(Rectangle((c-.5,r-.5),1,1,fill=False,edgecolor="#c9302c",lw=1.2))
    fig.savefig(OUT / "maps.png", dpi=DPI); plt.close(fig)

    for scenario in ["original", "changed"]:
        fig, ax = plt.subplots(figsize=(8.8, 4.3), layout="constrained")
        for alg, slug in SLUGS.items():
            curves = []
            for seed in range(5):  # 种子0至4：与预先固定的论文实验一致。
                with (DATA / scenario / slug / f"seed{seed}/training_log.csv").open(encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                steps = np.array([int(r["cumulative_steps"]) for r in rows])
                rewards = np.array([float(r["episode_return"]) for r in rows])
                assert np.all(rewards == -np.array([int(r["steps"]) for r in rows]))
                smooth = np.convolve(rewards, np.ones(WINDOW)/WINDOW, mode="valid")
                curves.append((steps[WINDOW-1:], smooth))
            start, end = max(x[0] for x,y in curves), min(x[-1] for x,y in curves)
            xs = np.linspace(start, end, 400)  # 显示采样点400：仅在各运行共同范围内插值，不外推。
            ys = np.array([np.interp(xs,x,y) for x,y in curves])
            mean, sd = ys.mean(0), ys.std(0,ddof=1)
            ax.plot(xs, mean, color=COLORS[alg], lw=2, label=alg)
            ax.fill_between(xs,mean-sd,mean+sd,color=COLORS[alg],alpha=.15)
        ax.set_xlabel("累计训练步数（不含评估交互）")
        ax.set_ylabel("单回合累积奖励（100回合滑动平均）")
        ax.set_ylim(-850,0); ax.grid(alpha=.2); ax.legend(loc="lower right")
        ax.ticklabel_format(axis="x",style="plain")
        fig.savefig(OUT / f"{scenario}_learning.png",dpi=DPI); plt.close(fig)

    fig, axs = plt.subplots(1,2,figsize=(8.8,4.6),layout="constrained")
    for ax, (alg,slug) in zip(axs,SLUGS.items()):
        result = load(DATA / "original" / slug / "seed0/evaluation.json")
        grid_panel(ax,original,result["first_episode_path"],f"{alg}，种子0，16步")
    fig.savefig(OUT / "original_paths.png",dpi=DPI); plt.close(fig)

    fig, axs = plt.subplots(2,2,figsize=(8.8,8.8),layout="constrained")
    for col,(alg,slug) in enumerate(SLUGS.items()):
        for row,scenario in enumerate(["transfer","changed"]):
            result = load(DATA / scenario / slug / "seed0/evaluation.json")
            title = ("旧策略直接测试" if scenario == "transfer" else "新地图从头训练")
            title += f" | {alg}\n" + (f"首回合{len(result['first_episode_path'])-1}步到达" if result["first_episode_success"] else "首回合1000步超时；红叉为停留位置")
            grid_panel(axs[row,col],changed,result["first_episode_path"],title)
    fig.savefig(OUT / "changed_paths.png",dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    main()
