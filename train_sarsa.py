"""PyCharm运行入口：训练SARSA并保存可复现实验记录。

算法依据 R01；数组工具 R06。详见 报告/问题2_SARSA算法设计与伪代码.md。
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from gridworld import ROOT, GridMap, GridWorld
from sarsa import SarsaConfig, evaluate_sarsa, train_sarsa


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Train tabular SARSA on the assignment map")
    parser.add_argument("--episodes", type=int, default=5000)  # 训练回合数：控制训练轮数。
    parser.add_argument("--alpha", type=float, default=0.1)  # 学习率：控制单次Q值更新幅度。
    parser.add_argument("--epsilon-start", type=float, default=1.0)  # 初始探索率：初期随机动作分支概率。
    parser.add_argument("--epsilon-end", type=float, default=0.05)  # 最低探索率：后期保留的探索概率。
    parser.add_argument("--epsilon-decay-episodes", type=int, default=4000)  # 衰减回合数：控制探索率下降速度。
    parser.add_argument("--seed", type=int, default=42)  # 训练随机种子：保证同配置运行可复现。
    parser.add_argument("--eval-episodes", type=int, default=100)  # 评估回合数：冻结Q表后的测试次数。
    parser.add_argument("--map", type=Path, default=ROOT / "maps" / "original.json")
    parser.add_argument("--output", type=Path, help="New output directory; refuses to overwrite")
    args = parser.parse_args()
    config = SarsaConfig(**{key: getattr(args, key) for key in SarsaConfig.__dataclass_fields__})
    grid = GridMap.load(args.map)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or ROOT / "outputs" / "sarsa" / f"seed{config.seed}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "algorithm": "SARSA", "config": asdict(config),
        "environment": {"gamma": GridWorld.gamma, "max_steps": GridWorld.max_steps,
                        "reward_per_action": -1, "time_per_action": 1},
        "map": json.loads(args.map.read_text(encoding="utf-8")),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("sarsa.py", "train_sarsa.py", "gridworld.py")},
        "references": ["R01", "R06"],
    }
    (output / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training SARSA: {config.episodes} episodes, seed={config.seed}", flush=True)
    q, history = train_sarsa(grid, config)
    np.save(output / "q_table.npy", q)
    write_csv(output / "training_log.csv", history)
    # 评估种子=训练种子+1000000：单独初始化评估随机源，不消耗训练随机序列。
    summary, evaluation = evaluate_sarsa(grid, q, config.eval_episodes, config.seed + 1000000)
    write_csv(output / "evaluation_log.csv", evaluation)
    summary["training_episodes"] = config.episodes
    summary["training_steps"] = history[-1]["cumulative_steps"]
    summary["training_timeouts"] = sum(row["truncated"] for row in history)
    # 只在训练完成后做基准比较；BFS路径不提供给学习算法 [R04]。
    summary["bfs_shortest_steps"] = len(grid.shortest_path()) - 1
    summary["evaluation_optimal_successes"] = sum(
        row["success"] and row["arrival_time"] == summary["bfs_shortest_steps"]
        for row in evaluation)
    (output / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "first_episode_path"}, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
