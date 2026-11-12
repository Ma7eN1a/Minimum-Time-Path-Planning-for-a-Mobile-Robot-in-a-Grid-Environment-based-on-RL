"""
地图来源：题目/算法作业-1.pdf，第1页图1。
规则来源：模型假设/最终模型假设.md。
最短路径验证使用 BFS [R04]，仅用于独立基准，不属于学习算法。
R04: https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/BreadthFirstPaths.html
"""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Position = tuple[int, int]


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class GridMap:
    rows: int
    cols: int
    start: Position
    goal: Position
    obstacles: frozenset[Position]

    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Map dimensions must be positive")
        if any(not self.contains(p) for p in self.obstacles):
            raise ValueError("Obstacle outside map")
        if not self.is_free(self.start) or not self.is_free(self.goal):
            raise ValueError("Start and goal must be free cells inside map")
        if not self.shortest_path():
            raise ValueError("Goal is unreachable from start")

    @classmethod
    def load(cls, path: str | Path = ROOT / "maps" / "original.json"):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obstacles = [tuple(p) for p in data["obstacles"]]
        if len(set(obstacles)) != len(obstacles):
            raise ValueError("Duplicate obstacle coordinates")
        positions = [tuple(data["start"]), tuple(data["goal"]), *obstacles]
        if any(len(p) != 2 or any(type(v) is not int for v in p) for p in positions):
            raise ValueError("Coordinates must be pairs of integers")
        return cls(data["rows"], data["cols"], positions[0], positions[1],
                   frozenset(obstacles))

    def contains(self, p: Position) -> bool:
        return 0 <= p[0] < self.rows and 0 <= p[1] < self.cols

    def is_free(self, p: Position) -> bool:
        return self.contains(p) and p not in self.obstacles

    def next_position(self, position: Position, action: int) -> Position:
        if not self.is_free(position):
            raise ValueError("Current position must be a free cell")
        dr, dc = DELTAS[Action(action)]
        candidate = position[0] + dr, position[1] + dc
        return candidate if self.is_free(candidate) else position

    def shortest_path(self) -> list[Position]:
        """无权网格 BFS 基准 [R04]；返回含起终点的一条最短路径。"""
        queue = deque([self.start])
        parent = {self.start: None}
        while queue:
            p = queue.popleft()
            if p == self.goal:
                path = []
                while p is not None:
                    path.append(p)
                    p = parent[p]
                return path[::-1]
            for action in Action:
                nxt = self.next_position(p, action)
                if nxt not in parent:
                    parent[nxt] = p
                    queue.append(nxt)
        return []

    def to_ascii(self) -> str:
        cells = [["#" if (r, c) in self.obstacles else "."
                  for c in range(self.cols)] for r in range(self.rows)]
        cells[self.start[0]][self.start[1]] = "S"
        cells[self.goal[0]][self.goal[1]] = "G"
        return "\n".join("".join(row) for row in cells)


class GridWorld:
    """每步 -1、耗时 1；终点成功终止，1000 步为外部截断。

    step 返回 (位置, 奖励, terminated, truncated, info)。
    超时后 arrival_time 为 None；调用者不得把截断误当成终点。
    """

    gamma = 1.0  # 折扣因子：不折扣未来奖励，保持总回报=-耗时；已确认的固定设定。
    max_steps = 1000  # 单回合步数上限：防止无限循环，达到上限记为截断而非成功。

    def __init__(self, grid: GridMap | None = None):
        self.grid = grid if grid is not None else GridMap.load()
        self.reset()

    def reset(self) -> Position:
        self.position = self.grid.start
        self.elapsed = 0
        self.total_reward = 0
        self.terminated = self.position == self.grid.goal
        self.truncated = False
        return self.position

    def step(self, action: int):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode ended; call reset() before another step")
        previous = self.position
        self.position = self.grid.next_position(previous, action)
        self.elapsed += 1
        self.total_reward -= 1
        self.terminated = self.position == self.grid.goal
        self.truncated = not self.terminated and self.elapsed >= self.max_steps
        info = {
            "elapsed_time": self.elapsed,
            "arrival_time": self.elapsed if self.terminated else None,
            "total_reward": self.total_reward,
            "blocked": self.position == previous,
        }
        return self.position, -1, self.terminated, self.truncated, info
