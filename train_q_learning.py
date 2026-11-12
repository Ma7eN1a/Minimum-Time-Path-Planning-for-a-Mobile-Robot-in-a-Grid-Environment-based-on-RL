"""PyCharm入口：训练Q-learning[R02]、保存可复现实验并进行冻结策略评估。

数组工具[R06]；BFS仅作训练后核验[R04]。详见报告中的Q-learning设计文档。
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from gridworld import ROOT, GridMap, GridWorld
from q_learning import QLearningConfig, train_q_learning, evaluate_q_learning
from train_sarsa import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)  # 训练回合数：控制训练轮数。
    parser.add_argument("--alpha", type=float, default=0.1)  # 学习率：控制Q值更新幅度。
    parser.add_argument("--epsilon-start", type=float, default=1.0)  # 初始探索率：初期随机动作概率。
    parser.add_argument("--epsilon-end", type=float, default=0.05)  # 最低探索率：后期保留随机探索。
    parser.add_argument("--epsilon-decay-episodes", type=int, default=4000)  # 衰减回合数：控制探索率下降速度。
    parser.add_argument("--seed", type=int, default=42)  # 训练种子：保证同配置运行可复现。
    parser.add_argument("--eval-episodes", type=int, default=100)  # 评估回合数：冻结Q表后的测试次数。
    parser.add_argument("--map", type=Path, default=ROOT / "maps/original.json")
    parser.add_argument("--output", type=Path, help="新结果目录，不允许覆盖已有目录")
    args = parser.parse_args()
    config = QLearningConfig(**{k: getattr(args, k) for k in QLearningConfig.__dataclass_fields__})
    grid = GridMap.load(args.map)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or ROOT / "outputs/q_learning" / f"seed{config.seed}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "algorithm": "Q-learning", "config": asdict(config),
        "environment": {"gamma": GridWorld.gamma, "max_steps": GridWorld.max_steps,
                        "reward_per_action": -1, "time_per_action": 1},
        "map": json.loads(args.map.read_text(encoding="utf-8")),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("q_learning.py", "train_q_learning.py", "gridworld.py",
                                       "sarsa.py", "train_sarsa.py")},
        "references": ["R02", "R04", "R06"],
    }
    (output / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training Q-learning: {config.episodes} episodes, seed={config.seed}", flush=True)
    q, history = train_q_learning(grid, config)
    np.save(output / "q_table.npy", q)
    write_csv(output / "training_log.csv", history)
    # 评估种子=训练种子+1000000：创建独立随机源，不消耗训练随机序列。
    summary, records = evaluate_q_learning(grid, q, config.eval_episodes, config.seed + 1000000)
    write_csv(output / "evaluation_log.csv", records)
    summary.update(training_episodes=config.episodes,
                   training_steps=history[-1]["cumulative_steps"],
                   training_timeouts=sum(r["truncated"] for r in history),
                   bfs_shortest_steps=len(grid.shortest_path()) - 1)  # BFS仅作独立核验[R04]。
    summary["evaluation_optimal_successes"] = sum(
        r["success"] and r["arrival_time"] == summary["bfs_shortest_steps"] for r in records)
    (output / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "first_episode_path"}, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
