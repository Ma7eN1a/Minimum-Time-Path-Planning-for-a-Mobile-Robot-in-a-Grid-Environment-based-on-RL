"""核对题图与模型规则的回归测试；使用标准库 unittest。"""

import unittest

from gridworld import Action, GridMap, GridWorld


class MapTests(unittest.TestCase):
    def test_map_matches_figure(self):
        self.assertEqual(GridMap.load().to_ascii(), "\n".join([
            "..#......G", "..#.......", "....##..##", ".....#....",
            ".....#....", "..........", ".###......", "......##..",
            ".S.....#..", ".......#..",
        ]))

    def test_obstacle_and_boundary_still_cost_time(self):
        env = GridWorld()
        env.step(Action.UP)  # (7, 1)
        pos, reward, done, truncated, info = env.step(Action.UP)
        self.assertEqual(pos, (7, 1))  # 障碍 (6,1)
        self.assertEqual((reward, done, truncated, info["elapsed_time"]), (-1, False, False, 2))
        self.assertTrue(info["blocked"])
        env.reset()
        env.step(Action.DOWN)  # (9,1)
        self.assertEqual(env.step(Action.DOWN)[0], (9, 1))
        self.assertEqual(env.elapsed, 2)

    def test_timeout_is_not_arrival(self):
        env = GridWorld()
        for _ in range(1000):
            result = env.step(Action.DOWN)
        self.assertEqual(result[2:4], (False, True))
        self.assertIsNone(result[4]["arrival_time"])
        self.assertEqual(env.total_reward, -1000)
        with self.assertRaises(RuntimeError):
            env.step(Action.UP)
        self.assertEqual(env.reset(), (8, 1))
        self.assertEqual(env.elapsed, 0)

    def test_path_executes_and_success_takes_priority_at_limit(self):
        grid = GridMap.load()
        path = grid.shortest_path()
        # 起终点的曼哈顿距离为16，若存在16步路径即达到下界。
        self.assertEqual(len(path) - 1, 16)
        env = GridWorld(grid)
        # LEFT到(8,0)，然后继续LEFT原地等待，再RIGHT返回S。
        for _ in range(983):
            env.step(Action.LEFT)
        env.step(Action.RIGHT)
        for p, nxt in zip(path, path[1:]):
            action = next(a for a in Action if grid.next_position(p, a) == nxt)
            result = env.step(action)
        self.assertEqual(result[0], grid.goal)
        self.assertEqual(result[2:4], (True, False))
        self.assertEqual(result[4]["arrival_time"], 1000)
        self.assertEqual(env.total_reward, -1000)
        with self.assertRaises(RuntimeError):
            env.step(Action.LEFT)

    def test_invalid_action_does_not_advance_time(self):
        env = GridWorld()
        with self.assertRaises(ValueError):
            env.step(4)
        self.assertEqual(env.elapsed, 0)

    def test_invalid_maps_rejected(self):
        with self.assertRaises(ValueError):
            GridMap(2, 2, (0, 0), (1, 1), frozenset({(0, 1), (1, 0)}))
        with self.assertRaises(ValueError):
            GridMap(2, 2, (0, 0), (1, 1), frozenset({(1, 1)}))


if __name__ == "__main__":
    unittest.main()
