"""论文补充实验：复用已测试的两种学习循环，记录多种子与地图变化结果。

算法[R01,R02]、BFS核验[R04]、NumPy[R06]；具体实验协议均为本项目设定。
不覆盖旧结果，不以BFS距离指导训练。默认结果用于2026-11-04修订论文。
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np

from gridworld import ROOT, GridMap, GridWorld
from sarsa import SarsaConfig, train_episode as sarsa_episode, evaluate_sarsa
from q_learning import QLearningConfig, train_episode as q_episode
from train_sarsa import write_csv

SEEDS = (0, 1, 2, 3, 4)  # 训练种子清单：预先固定5次独立训练，不筛选最好结果。
EPISODES = 5000  # 训练回合数：与已有基线保持一致。
CHECK_INTERVAL = 250  # 检查间隔：每250回合冻结Q表做一次独立评估。
CHECK_EPISODES = 20  # 检查评估回合数：用于观察策略达到最短步数的进程。
FINAL_EPISODES = 100  # 最终评估回合数：每个已训练Q表测试100次。
STABLE_CHECKS = 3  # 达标确认长度：连续3个检查点全部成功且均达最短步数。


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train_run(grid, map_data, algorithm, seed, folder):
    config_cls, step_fn = (SarsaConfig, sarsa_episode) if algorithm == "SARSA" else (QLearningConfig, q_episode)
    config = config_cls(episodes=EPISODES, seed=seed, eval_episodes=FINAL_EPISODES)
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "config.json", dict(algorithm=algorithm, config=asdict(config), map=map_data,
              environment=dict(gamma=GridWorld.gamma, max_steps=GridWorld.max_steps,
                               reward_per_action=-1, time_per_action=1)))
    q = np.zeros((grid.rows, grid.cols, 4), dtype=np.float64)  # Q初值0：每个地图从头训练，不注入先验。
    rng = np.random.default_rng(seed)
    env = GridWorld(grid)
    history, checks = [], []
    total = 0
    shortest = len(grid.shortest_path()) - 1
    streak = 0
    confirmed = None
    for e in range(config.episodes):
        row = step_fn(env, q, rng, config, e)
        total += row["steps"]
        row["cumulative_steps"] = total
        history.append(row)
        if (e + 1) % CHECK_INTERVAL == 0:
            # 检查种子=2000000+seed：独立于训练与最终评估，每个检查点使用同一测试序列。
            summary, records = evaluate_sarsa(grid, q, episodes=CHECK_EPISODES, seed=2000000 + seed)
            optimal = sum(r["success"] and r["arrival_time"] == shortest for r in records)
            streak = streak + 1 if optimal == CHECK_EPISODES else 0
            if streak >= STABLE_CHECKS and confirmed is None:
                confirmed = total  # 以第3个检查点的累计步数记确认时刻，不回填为最早时刻。
            checks.append(dict(episode=e + 1, cumulative_steps=total, success_rate=summary["success_rate"],
                               mean_arrival_time=summary["mean_arrival_time"], optimal_rate=optimal / CHECK_EPISODES))
    # 最终种子=1000000+seed：冻结Q表，epsilon=0，不消耗训练随机源。
    summary, evaluation = evaluate_sarsa(grid, q, episodes=FINAL_EPISODES, seed=1000000 + seed)
    summary.update(training_steps=total, training_timeouts=sum(r["truncated"] for r in history),
                   training_episodes=EPISODES, bfs_shortest_steps=shortest,
                   confirmation_steps=confirmed, training_seed=seed,
                   last100_return=float(np.mean([r["episode_return"] for r in history[-100:]])),
                   evaluation_optimal_successes=sum(r["success"] and r["arrival_time"] == shortest for r in evaluation))
    np.save(folder / "q_table.npy", q)
    write_csv(folder / "training_log.csv", history)
    write_csv(folder / "evaluation_log.csv", evaluation)
    save_json(folder / "checkpoints.json", checks)
    save_json(folder / "evaluation.json", summary)
    return q, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "outputs/paper_experiments_20261104"))
    parser.add_argument("--baseline", type=Path, help="可选：复用已核验的原地图训练批次，避免重复训练")
    parser.add_argument("--changed-map", type=Path, default=ROOT / "maps/changed.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((ROOT / "maps/original.json").read_text(encoding="utf-8"))
    changed = json.loads(args.changed_map.read_text(encoding="utf-8"))
    old_obstacles = set(map(tuple, original["obstacles"]))
    new_obstacles = set(map(tuple, changed["obstacles"]))
    assert len(old_obstacles) == len(new_obstacles) == 15
    assert original["start"] == changed["start"] and original["goal"] == changed["goal"]
    if args.baseline:
        previous = json.loads((args.baseline / "protocol.json").read_text(encoding="utf-8"))
        assert json.loads((args.baseline / "original_map.json").read_text(encoding="utf-8")) == original
        for n in ["gridworld.py", "sarsa.py", "q_learning.py"]:
            assert previous["source_sha256"][n] == hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
    save_json(output / "original_map.json", original)
    save_json(output / "changed_map.json", changed)
    original_grid = GridMap.load(output / "original_map.json")
    changed_grid = GridMap.load(output / "changed_map.json")
    save_json(output / "protocol.json", dict(date="2026-11-04", seeds=SEEDS, episodes=EPISODES,
              checkpoint_interval=CHECK_INTERVAL, checkpoint_episodes=CHECK_EPISODES,
              confirmation_checks=STABLE_CHECKS, final_eval_episodes=FINAL_EPISODES,
              map_change=dict(removed=sorted(old_obstacles-new_obstacles), added=sorted(new_obstacles-old_obstacles),
                              unchanged=sorted(old_obstacles & new_obstacles),
                              relocated_fraction=len(old_obstacles-new_obstacles)/len(old_obstacles),
                              symmetric_difference=len(old_obstacles ^ new_obstacles)),
              baseline_source=str(args.baseline.resolve()) if args.baseline else None,
              original_shortest=len(original_grid.shortest_path()) - 1,
              changed_shortest=len(changed_grid.shortest_path()) - 1,
              budget="equal training episodes; report actual interaction steps; not equal step budgets",
              versions=dict(python=platform.python_version(), numpy=np.__version__),
              source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                             for n in ["gridworld.py", "sarsa.py", "q_learning.py", "run_paper_experiments.py"]}))
    all_results = []
    for algorithm, slug in [("SARSA", "sarsa"), ("Q-learning", "q_learning")]:
        for seed in SEEDS:
            original_dir = output / "original" / slug / f"seed{seed}"
            if args.baseline:
                source = args.baseline / "original" / slug / f"seed{seed}"
                config = json.loads((source / "config.json").read_text(encoding="utf-8"))
                expected = asdict((SarsaConfig if algorithm == "SARSA" else QLearningConfig)(
                    episodes=EPISODES, seed=seed, eval_episodes=FINAL_EPISODES))
                assert config["config"] == expected
                shutil.copytree(source, original_dir)
                original_q = np.load(original_dir / "q_table.npy")
                original_result = json.loads((original_dir / "evaluation.json").read_text(encoding="utf-8"))
            else:
                original_q, original_result = train_run(original_grid, original, algorithm, seed, original_dir)
            # 迁移评估不学习、不屏蔽碰撞动作，不清空原Q表；新障碍只由环境阻挡。
            before = original_q.copy()
            transferred, transfer_rows = evaluate_sarsa(changed_grid, original_q, FINAL_EPISODES, 1000000 + seed)
            np.testing.assert_array_equal(original_q, before)
            transfer_dir = output / "transfer" / slug / f"seed{seed}"
            transfer_dir.mkdir(parents=True)
            save_json(transfer_dir / "evaluation.json", transferred)
            write_csv(transfer_dir / "evaluation_log.csv", transfer_rows)
            _, changed_result = train_run(changed_grid, changed, algorithm, seed,
                                          output / "changed" / slug / f"seed{seed}")
            all_results.append(dict(algorithm=algorithm, seed=seed, original=original_result,
                                    transfer=transferred, changed=changed_result))
            print(f"{algorithm} seed={seed}: original={original_result['mean_arrival_time']} "
                  f"transfer_success={transferred['success_rate']} changed={changed_result['mean_arrival_time']}", flush=True)
    save_json(output / "summary.json", all_results)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
