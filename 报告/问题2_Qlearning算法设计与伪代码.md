# 第二问：Q-learning算法代码设计与伪代码

日期：2026-10-23。本文件与SARSA部分采用相同环境、默认参数、日志口径及冻结策略评估方式；正式多训练种子对比另行开展。

## 1. 输入、输出和模块

输入是`maps/original.json`和`QLearningConfig`。环境严格遵循最终模型假设：四方向动作，每步耗时1、奖励−1，碰撞原地停留，γ=1，首次到达终点即终止，1000步未到达则外部截断。Q表为NumPy[R06]的`float64`数组，形状(10,10,4)，动作编号为0上、1下、2左、3右；障碍和终点槽不更新。

| 文件 | 职责 |
|---|---|
| `q_learning.py` | QLearningConfig、独立的Q-learning更新和训练循环、冻结评估入口 |
| `train_q_learning.py` | PyCharm训练入口，保存Q表、配置、日志、评估和独立基准 |
| `plot_q_learning_training.py` | 累计训练步数—单回合累积奖励图 |
| `plot_q_learning_path.py` | 冻结Q表后的首个评估回合路径图 |
| `tests/test_q_learning.py` | 9项Q-learning专项回归测试 |

为保持口径一致，配置校验、线性ε调度、ε-greedy选择及纯评估复用`sarsa.py`中的对应实现；CSV写入复用`train_sarsa.write_csv`，两种图复用现有绘图实现并读取`config.json`中的算法名称。**复用的评估函数不包含学习更新，Q-learning训练不调用SARSA训练或更新函数。** 所有训练依赖源码均记录SHA-256摘要。SARSA训练代码和旧实验数据未修改。

## 2. 更新规则及与SARSA的区别

Q-learning使用下一状态的最大动作价值作为更新目标[R02，第2节，印刷页280–281]：

\[
Q(s,a)\leftarrow Q(s,a)+\alpha[y-Q(s,a)],\quad
y=\begin{cases}
r,&\text{真正到达终点},\\
r+\gamma\max_b Q(s',b),&\text{否则，包括超时截断}.
\end{cases}
\]

行为动作仍按本项目ε-greedy规则采样：概率ε均匀选四个动作，否则在严格并列最大值动作之间均匀选择。不屏蔽越界或撞障碍动作；每回合内ε固定。

SARSA用采样下一动作的Q值更新[R01，第6.4节式(6.7)]；Q-learning用最大Q值更新[R02]。本实现先算完整目标、再更新当前Q项；如果未结束，再在更新后的Q表上选择下一行为动作。即使碰撞导致s'=s，也不提前修改下一状态的目标值。

真正终止时不读取终点Q槽。外部截断仍保留最大下一状态价值，随后结束采样回合，不多执行动作；这是落实已确认的项目截断约定。起点即终点的特殊输入记录0步成功。

## 3. 项目伪代码

下列为按Q-learning更新规则[R02]、结合本项目环境和日志需求编写的伪代码，不是照录原文。

```text
输入：Env，回合数N，学习率α，探索率函数ε(e)，训练种子seed
Q ← 全零数组；创建独立训练随机源rng
累计交互步数 ← 0
对 e = 0,...,N−1：
    s ← Env.reset()
    本回合ε ← ε(e)
    若s已经为终点：记录0步成功，继续下一回合
    重复：
        a ← ε-greedy(Q[s], 本回合ε, rng)
        (s′, r, terminated, truncated, info) ← Env.step(a)
        若terminated：y ← r
        否则：y ← r + γ max_b Q[s′,b]  # 截断时仍保留
        Q[s,a] ← Q[s,a] + α(y − Q[s,a])
        累计交互步数加1；累计碰撞次数
        若terminated或truncated：
            保存回报、动作数、终止/截断、抵达耗时和累计步数
            跳出
        s ← s′                       # 下一轮才选新行为动作
冻结Q；用独立评估种子重复测试，ε=0，不更新Q
记录成功率、超时、成功耗时及首个评估回合完整轨迹
用BFS作独立最短步数核验，不把路径或距离提供给学习算法
```

## 4. 运行和产物

PyCharm选择项目`.venv`解释器，直接运行`train_q_learning.py`。不需要安装新依赖。默认每次保存到新目录`outputs/q_learning/seed42_日期时间/`，拒绝覆盖已有实验目录。

```powershell
.\.venv\Scripts\python.exe train_q_learning.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

可用`--episodes`、`--alpha`、`--epsilon-start`、`--epsilon-end`、`--epsilon-decay-episodes`、`--seed`、`--eval-episodes`覆盖算法参数；`--map`指定地图，`--output`指定尚不存在的输出目录。手动传入的相对路径按当前工作目录解释。

输出包括`config.json`（参数、环境、地图快照、版本、源码摘要）、`q_table.npy`、`training_log.csv`、`evaluation_log.csv`、`evaluation.json`。训练日志记录每回合ε、动作数、回报、终止/截断、成功抵达耗时、碰撞数、累计交互步数；不增加损失图或相关诊断统计。失败抵达耗时为空，不把超时1000步当作成功抵达。

当前已完成实验位于`outputs/q_learning/design_validation_seed42/`。绘图入口默认使用这个已验证目录；**再次训练生成的新目录需要用`--run-dir`明确指定**，避免误把旧图当作新实验结果。

```powershell
.\.venv\Scripts\python.exe plot_q_learning_training.py --no-show
.\.venv\Scripts\python.exe plot_q_learning_path.py --no-show
# 如需绘制其他实验，把下列路径替换为实际新输出目录：
# .\.venv\Scripts\python.exe plot_q_learning_training.py --run-dir outputs/q_learning/实际目录名 --no-show
```

## 5. 验证结果

2026-10-23，原有14项测试与新增9项测试共23项全部通过。新测试检查最大值目标、终点不自举、碰撞原地转移、截断仍自举且不额外抽动作、更新后选择行为动作、参数合法性和ε调度、相同种子自复现、回报计数、冻结评估与失败标记、起点即终点。

| 指标 | 当前单训练种子实际结果 |
|---|---:|
| 训练种子 / 训练回合 | 42 / 5000 |
| 累计环境交互步数 | 353078 |
| 训练超时回合 | 40 |
| 评估种子 / 评估回合 | 1000042 / 100 |
| 评估成功 / 超时 | 100 / 0 |
| 成功耗时平均 / 最少 / 最多 | 16 / 16 / 16 |
| BFS最短步数[R04] | 16 |

100次评估来自同一Q表，不是100次独立训练。本次贪心策略达到当前地图最短步数，但不能证明任意随机种子都得到相同结果，也不能据此判定Q-learning优于SARSA。固定学习率、有限训练次数以及本项目γ=1的具体设置，不直接继承原论文的渐近收敛保证。

## 实际引用

- [R02] Watkins C J C H, Dayan P. Q-learning. Machine Learning, 1992, 8: 279–292. [原论文](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)，[DOI](https://doi.org/10.1007/BF00992698)。本次核对第2节可检索正文，未宣称完整审阅证明。
- [R01] Sutton R S, Barto A G. Reinforcement Learning: An Introduction. 2nd ed. MIT Press, 2018. §6.4，式(6.7)。
- [R04] Sedgewick R, Wayne K. [BreadthFirstPaths官方文档](https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/BreadthFirstPaths.html)。
- [R06] Harris C R, et al. Array programming with NumPy. Nature, 2020, 585: 357–362. DOI: 10.1038/s41586-020-2649-2.
