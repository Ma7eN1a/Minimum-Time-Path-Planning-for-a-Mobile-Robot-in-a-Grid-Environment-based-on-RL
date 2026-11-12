# 第二问：SARSA 算法代码设计与伪代码

日期：2026-10-23。范围：本文件为第二问中的 SARSA 部分；Q-learning已另见《问题2_Qlearning算法设计与伪代码.md》，两种算法的正式多训练种子对比实验另行完成。“SASAR”按题目算法名称统一为“SARSA”。

## 1. 输入、输出及环境接口

输入为 `maps/original.json` 中的题目地图以及 `SarsaConfig` 参数。环境严格遵循 `模型假设/最终模型假设.md`：每次动作耗时1、奖励−1、碰撞停留、γ=1、到达终点终止、1000步未到达则截断。

训练输出包含Q表、逐回合日志、冻结策略评估记录及运行配置。地图最短路径只用于训练后的结果核验，不用于选择训练动作、初始化Q值或生成训练经验。

状态为 `(row, column)`，动作编号依次为0上、1下、2左、3右。Q表使用 NumPy 数组 [R06]，形状为 `(10,10,4)`、类型为 `float64`。其中15个障碍位置和终点的存储槽不参与学习；有84×4=336个非终点状态动作项。保留矩形数组是索引便利，不表示障碍也成为合法状态。

## 2. 单步更新设计

采用 SARSA 的采样下一动作更新 [R01，第6.4节，式(6.7)]：

\[
Q(s,a)\leftarrow Q(s,a)+\alpha\bigl[y-Q(s,a)\bigr],
\qquad
y=\begin{cases}
r,&\text{实际到达终点},\\
r+\gamma Q(s',a'),&\text{其他情况}.
\end{cases}
\]

下一动作 \(a'\) 由当前ε-greedy策略抽取，而非选取最大Q值作为更新目标。若回合继续，执行的正是这个已经抽取的 \(a'\)，不在更新后重新抽样。碰撞时可能有 \(s'=s\)，也保持这一顺序。

ε-greedy的项目实现：以概率ε在四个动作中均匀抽取；否则在Q值严格并列最大的动作中均匀抽取。不使用固定编号打破并列，也不屏蔽撞障碍或越界的动作。

对于1000步外部截断，当前位置不是终点：仍从截断后的状态抽取 \(a'\)，使用其Q值形成更新目标，然后停止该采样回合并重置环境。这一个 \(a'\) 仅用于估计继续执行策略的价值，截断后不额外调用环境动作、不增加时间。此处理落实本项目已确认的“截断不等于终点”约定。

## 3. 项目伪代码

以下为依据 SARSA 更新规则 [R01]、结合本项目环境接口与日志需求编写的伪代码。

```text
输入：环境 Env，回合数 N，学习率 α，探索率函数 ε(e)，训练种子 seed
Q ← 全零数组；创建种子为 seed 的训练随机源
累计环境步数 ← 0

对 e = 0, ..., N−1：
    s ← Env.reset()
    本回合ε ← ε(e)                  # 回合内保持不变
    a ← ε-greedy(Q[s], 本回合ε)
    重复：
        (s′, r, terminated, truncated, info) ← Env.step(a)
        若 terminated：
            a′ ← 空
            y ← r
        否则：
            a′ ← ε-greedy(Q[s′], 本回合ε)  # 在Q更新前抽取
            y ← r + γ Q[s′, a′]            # 截断时同样保留
        δ ← y − Q[s,a]
        Q[s,a] ← Q[s,a] + αδ
        累计环境步数 ← 累计环境步数 + 1
        记录碰撞次数及 |δ|
        若 terminated 或 truncated：
            保存回合奖励、动作数、终止原因及抵达耗时
            跳出本回合
        (s,a) ← (s′,a′)                    # 保留已选a′

冻结Q，创建与训练分离的评估随机源
重复指定次数评估：从S开始，ε=0，仅选动作，不更新Q
记录成功、超时和成功抵达步数；返回Q及全部日志
```

当前题目起点不同于终点；代码另兼容起点即终点的输入，将其记录为0步成功。

## 4. 代码组织

| 文件 / 函数 | 职责 |
|---|---|
| `sarsa.py / SarsaConfig` | 参数默认值、合法性检查及线性探索率调度 |
| `sarsa.py / select_action` | ε-greedy动作选择及随机打破并列 |
| `sarsa.py / sarsa_update` | 根据采样的下一动作完成一次更新 |
| `sarsa.py / train_episode` | 顺序执行动作、更新、终止/截断及日志 |
| `sarsa.py / train_sarsa` | 独立训练随机源、Q表初始化及全部回合 |
| `sarsa.py / evaluate_sarsa` | 冻结Q表，ε=0，使用独立评估随机源 |
| `train_sarsa.py` | PyCharm/命令行入口，保存配置、日志及结果 |
| `tests/test_sarsa.py` | 更新公式、动作复用、截断、自复现等测试 |

## 5. 运行方法和输出

在 PyCharm 中选择本项目 `.venv\Scripts\python.exe`，直接运行根目录 `train_sarsa.py`。无需新增依赖。默认参数详见 `SARSA超参数选择.md`。

在项目根目录也可执行：

```powershell
.\.venv\Scripts\python.exe train_sarsa.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

默认生成新目录 `outputs/sarsa/seed42_日期时间/`，避免覆盖已有实验。可以用 `--output` 指定不存在的目录；指定已有目录会报错。默认地图与输出位置相对脚本根目录定位；命令行手动给出的相对 `--map` / `--output` 路径相对当前运行目录。

| 输出文件 | 内容 |
|---|---|
| `config.json` | 参数、地图快照、环境规则、Python/NumPy版本、源码SHA-256摘要 |
| `q_table.npy` | 训练后Q表，可用 `numpy.load` 读取 |
| `training_log.csv` | 每回合ε、动作数、回合总奖励、终止/截断标记、抵达时间、碰撞数、平均绝对TD误差、累计交互步数 |
| `evaluation_log.csv` | 每次冻结策略评估的成功标记、截断标记、已耗时、成功抵达时间及回报 |
| `evaluation.json` | 评估汇总、首个评估回合的完整轨迹、训练步数与超时数、BFS基准比较 [R04] |

`training_log.csv` 的 `episode_return` 是单回合奖励之和，既不是跨回合累计奖励，也不是平滑曲线。`cumulative_steps` 可作为后续学习曲线的累计训练步数横轴。超时回合 `arrival_time` 留空，并保留其1000步和−1000回报。

## 6. 已执行的代码验证

2026-10-23，环境原有6项测试及SARSA新增8项测试全部通过。新增测试覆盖：

- 更新目标使用指定的下一动作Q值，而不是最大Q值。
- 真终点即使终点Q槽被人为赋非零值也不继续自举。
- 原地停留后，更新前抽取的下一动作被实际复用。
- 外部截断仍保留下一状态动作价值，抵达时间仍为空。
- 探索率首末值、无效参数拒绝、动作并列与随机探索。
- 同一配置及种子的Q表和日志精确复现，回报等于负动作数。
- 评估不修改Q表，失败回合不混作成功耗时。

默认配置功能验证记录位于 `outputs/sarsa/design_validation_seed42/`：

| 指标 | 实际结果 |
|---|---:|
| 训练种子 | 42 |
| 训练回合 | 5000 |
| 累计训练交互步数 | 347692 |
| 训练超时回合 | 37 |
| 独立评估随机源种子 | 1000042 |
| 冻结Q的评估回合 | 100 |
| 评估成功 / 超时 | 100 / 0 |
| 成功抵达平均 / 最少 / 最多步数 | 16 / 16 / 16 |
| BFS最短步数基准 | 16 |

这些数据说明本次运行所得贪心策略达到了当前地图的最短步数。它们不是多训练种子统计，不证明任意种子都得到相同结果，也不能据此宣称算法在某个回合已严格收敛或优于Q-learning。100次评估均来自同一个已训练Q表。

## 参考文献

采用项目稳定键；合并最终报告时按全文首次引用顺序重排。

- [R01] SUTTON R S, BARTO A G. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press, 2018. §6.4，印刷页129，式(6.7)。[原书](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)。
- [R06] HARRIS C R, et al. Array programming with NumPy. *Nature*, 2020, 585: 357–362. [DOI](https://doi.org/10.1038/s41586-020-2649-2)。
- [R04] SEDGEWICK R, WAYNE K. [BreadthFirstPaths 官方文档](https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/BreadthFirstPaths.html)。只用于独立最短路径基准。
