import unittest
from unittest.mock import patch

import numpy as np

from gridworld import GridMap, GridWorld
from sarsa import (SarsaConfig, evaluate_sarsa, sarsa_update, select_action,
                   train_episode, train_sarsa)


class SarsaTests(unittest.TestCase):
    def test_update_uses_sampled_action_not_max(self):
        q = np.zeros((1, 2, 4))
        q[0, 0, 0] = 2
        q[0, 1] = [10, 4, 0, 0]
        error = sarsa_update(q, (0, 0), 0, -1, (0, 1), 1, False, 0.5)
        self.assertEqual(error, 1)  # -1 + 4 - 2
        self.assertEqual(q[0, 0, 0], 2.5)

    def test_terminal_has_no_bootstrap_even_if_terminal_q_nonzero(self):
        q = np.full((1, 2, 4), 10.0)
        sarsa_update(q, (0, 0), 3, -1, (0, 1), None, True, 1.0)
        self.assertEqual(q[0, 0, 3], -1.0)

    def test_next_action_is_reused_after_self_transition_update(self):
        env = GridWorld(GridMap(1, 2, (0, 0), (0, 1), frozenset()))
        q = np.zeros((1, 2, 4))
        # UP碰边界，下一动作RIGHT必须被执行，不能在更新后再抽一次。
        with patch("sarsa.select_action", side_effect=[0, 3]) as chooser:
            row = train_episode(env, q, np.random.default_rng(0), SarsaConfig(), 0)
        self.assertEqual(chooser.call_count, 2)
        self.assertEqual(row["steps"], 2)
        self.assertTrue(row["terminated"])
        self.assertEqual(row["episode_return"], -2)

    def test_truncation_keeps_bootstrap_in_training_loop(self):
        class TruncatedEnv:
            gamma = 1.0
            terminated = False
            def reset(self):
                return (0, 0)
            def step(self, action):
                return (0, 1), -1, False, True, {
                    "blocked": False, "elapsed_time": 1,
                    "total_reward": -1, "arrival_time": None}
        q = np.zeros((1, 2, 4))
        q[0, 1, 2] = 7.0
        with patch("sarsa.select_action", side_effect=[3, 2]):
            row = train_episode(TruncatedEnv(), q, None, SarsaConfig(alpha=1.0), 0)
        self.assertEqual(q[0, 0, 3], 6.0)
        self.assertTrue(row["truncated"])
        self.assertIsNone(row["arrival_time"])

    def test_epsilon_schedule_and_invalid_config(self):
        config = SarsaConfig()
        self.assertEqual(config.epsilon(0), 1.0)
        self.assertAlmostEqual(config.epsilon(3999), 0.05)
        self.assertAlmostEqual(config.epsilon(4999), 0.05)
        for kwargs in ({"episodes": 0}, {"alpha": float("nan")}, {"seed": -1},
                       {"epsilon_end": 2.0}, {"epsilon_decay_episodes": 1}):
            with self.assertRaises(ValueError):
                SarsaConfig(**kwargs)

    def test_tie_breaking_and_exploration_cover_all_actions(self):
        q = np.zeros((1, 1, 4))
        for epsilon in (0.0, 1.0):
            rng = np.random.default_rng(8)
            actions = {select_action(q, (0, 0), epsilon, rng) for _ in range(200)}
            self.assertEqual(actions, {0, 1, 2, 3})

    def test_seed_reproducibility_and_return_accounting(self):
        grid = GridMap.load()
        config = SarsaConfig(episodes=15)
        q1, log1 = train_sarsa(grid, config)
        q2, log2 = train_sarsa(grid, config)
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(log1, log2)
        self.assertTrue(all(r["episode_return"] == -r["steps"] for r in log1))
        self.assertEqual(log1[-1]["cumulative_steps"], sum(r["steps"] for r in log1))
        np.testing.assert_array_equal(q1[grid.goal], np.zeros(4))

    def test_evaluation_is_frozen_and_failures_are_explicit(self):
        grid = GridMap(1, 2, (0, 0), (0, 1), frozenset())
        q = np.zeros((1, 2, 4))
        q[0, 0, 0] = 2  # 唯一贪心动作UP持续撞边界。
        before = q.copy()
        summary, records = evaluate_sarsa(grid, q, episodes=2, seed=1)
        np.testing.assert_array_equal(q, before)
        self.assertEqual(summary["success_rate"], 0)
        self.assertIsNone(summary["mean_arrival_time"])
        self.assertTrue(all(r["elapsed_time"] == 1000 for r in records))
        self.assertTrue(all(r["arrival_time"] is None for r in records))


if __name__ == "__main__":
    unittest.main()
