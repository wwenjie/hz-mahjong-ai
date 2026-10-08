# 杭州麻将 AI 参赛程序

公司内「杭州麻将 AI 竞技赛」参赛程序。接入官方对战平台，全程自动决策，无需人工干预。

> **参赛/评审指引**：比赛时的使用方法见 **[docs/USAGE.md](docs/USAGE.md)**（评审版使用说明）；
> 本文档为开发者版，含架构、模型与训练管线细节。

**零运行时依赖**：`src/` 只使用 Python 标准库（模型推理是纯 Python 实现）。`numpy` /
`scikit-learn` 仅用于离线数据分析与训练，不进入运行路径。

## 一、环境要求

| 项目 | 要求 |
| --- | --- |
| Python | 3.12 或以上 |
| 包管理器 | [uv](https://docs.astral.sh/uv/)（推荐）；`pip` 亦可 |
| 网络 | 需能访问比赛服务器（自签证书，代码已关闭校验） |
| 运行时依赖 | **无** |

## 二、安装

```bash
git clone <仓库地址> && cd majiang_ai
uv sync                 # 只装开发依赖（pytest 等）；运行本身不需要任何依赖
```

若用 `pip`：

```bash
pip install -e .        # 或什么都不装，直接用 PYTHONPATH=src python -m majiang
```

依赖已在 `uv.lock` 中锁定，保证干净环境可复现。

## 三、令牌配置

令牌**只经环境变量或启动参数传入**，不入仓、不入日志、不硬编码。使用环境变量可以避免
令牌出现在命令行参数和进程列表中。

```bash
export MAJIANG_TOKEN="<参赛令牌>"       # 变量名可自定，启动时用 --token-env 指定
export MAJIANG_SERVER="https://<服务器地址>:<端口>"   # 可选，覆盖默认服务器
```

优先级：`--token-env` 指定的变量 > `--env-prefix` 收集到的多个变量 > `--token` 直接传参。

> 参赛令牌与全局令牌用途不同：参赛令牌是 scoped 令牌，只能在被授权的赛事/入口使用；
> 全局令牌用于领取身份等管理操作。正式比赛用参赛令牌。

## 四、启动

### 正式比赛（2026-10-10 19:30）

参赛档位为 **`v5`**（真机 484 场一位率 10.3%、场均分最优；更新档 v6 真机显著更差已弃用）。
晋级轮用 `--mode qualifier`；若晋级决赛，用 `--mode final` 重启（决赛纯总得分制，策略自动切激进参数）。

```bash
export MAJIANG_TOKEN="<参赛令牌>"
export MAJIANG_SERVER="https://<比赛服务器地址>"
uv run python -m majiang --token-env MAJIANG_TOKEN --decider v5 --mode qualifier
```

启动后程序自动完成「确认到位 → 对局 → 阶段晋级判断 → 下一阶段重新确认」的多阶段循环，
无需人工干预；中途崩溃重赛、决赛加赛新桌均会自动发现并接入。`--duration 0`（默认）
持续运行直至赛事终态。

单身份（测试/联调）：

```bash
uv run python -m majiang --token-env MAJIANG_TOKEN
```

多身份（一个进程内各身份独立限速，适合测试房多令牌联调）：

```bash
uv run python -m majiang --env-prefix MAJIANG_TOKEN_ --duration 600
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--server` | 服务器地址，默认取 `MAJIANG_SERVER` |
| `--token-env` | 存有令牌的环境变量名（推荐） |
| `--env-prefix` | 收集该前缀下的所有环境变量作为多身份令牌 |
| `--mode` | `qualifier`（晋级轮，稳健）/ `final`（决赛，激进） |
| `--decider` | 决策器档位，见下表，默认 `heuristic` |
| `--duration` | 运行秒数，`0` 表示持续运行直到赛事终态 |
| `--log-dir` | 日志目录，默认 `logs` |
| `--rate` | 每身份每秒请求上限，默认 14（平台限制 16/s，留窗口余量） |
| `--max-workers` | 并发线程上限，默认 16 |
| `--quiet` | 不往控制台打日志 |
| `--reopen-test-room` | 测试房跑完一轮后再次到位开启下一轮（**正式赛事不要打开**） |

按 `Ctrl-C` 或发 `SIGTERM` 可优雅收尾，会打印各身份的汇总。

### 决策器档位

默认 `heuristic` 为基线启发式；**正式比赛请显式指定 `--decider v5`**。所有档位均无需模型文件即可运行。

| 名称 | 说明 |
| --- | --- |
| `v5` | **参赛档位**。精确进张比较候选面 3 张，真机验证最优 |
| `heuristic` | 基础启发式（默认）。等价于 v5 之前的基线配置 |
| `ukeire` | 同向听时改用「进张最多」做次排序（实测打平，保留备查） |
| `risk` | 用对手听牌模型驱动风险估计；模型缺失时自动回退手写启发式 |
| `value` | 用价值模型为出牌打分；模型缺失时自动回退启发式 |
| `search` / `search-deep` | 确定化前瞻搜索（实测无增益，仅实验用） |
| `route-aware` | 开启路线分叉（实测负收益，仅实验用） |
| `preserve-god` / `natural` | 实验档位：不打出财神，用于验证番型链可达性 |
| `no-chase` | 对照档位：关闭「弃胡求爆头」，用于量化该决策的期望值 |
| `first-legal` | 只选第一个合法动作，用于连通性冒烟测试 |

**模型文件缺失不会导致启动失败**——所有模型驱动的档位都会自动回退到启发式并打印警告。

## 五、日志

默认写入 `logs/`，JSONL 格式（每行一个 JSON 对象，便于检索与复盘）。日志会脱敏，
令牌不以明文出现。

```bash
tail -f logs/*.jsonl                                   # 实时跟踪
grep '"event":"decision"' logs/*.jsonl | head           # 查决策记录
grep '"event":"error"' logs/*.jsonl                     # 查异常
```

## 六、合规说明

- 推理输入**仅限公开信息**：自身手牌、场上已打出牌、各家副露、牌墙剩余数、财神状态。
- 代码中不存在读取或推断对手手牌的路径；对手手牌仅在**离线数据生成**时作为标签使用。
- 不调用官方文档未声明的外部服务；不硬编码任何凭据。
- 模型推理为进程内纯计算，无网络、无外部依赖、确定性输出。

## 七、目录结构

```
src/majiang/
  rules/       规则内核：牌、手牌、胡牌、番型、得分、向听、动作合法性
  client/      平台协议客户端：认证、状态机、长轮询、快照重建、错误分类
  runtime/     并发对局运行时：调度、限速、决策窗口、结构化日志
  strategy/    决策策略：启发式策略、风险模型、价值模型、搜索
  sim/         本地模拟器：单局推演、批量自对弈、仅测量用完全信息决策器
src/nnrl2/     Transformer 策略/价值网络（BC/PPO 训练所得，推理走 policy_v7）
tools/         离线工具：数据生成、模型训练、A/B 检验、官方番型对拍、实战采集
scripts/       运行看护：守护脚本与 systemd 单元
scripts/rl2/   nnrl2 配套训练脚本（BC 数据生成、BC/PPO 训练、A/B 评测）
research/      早期研究脚本（NN vs GBDT、序列模型探索、占用容量扫描等，冻结存档）
docs/          运维手册与评审版使用说明（USAGE.md）
tests/         测试套件
openspec/      变更提案、设计决策与任务清单
```

## 八、模型与训练

参赛程序**推理时零外部依赖**（纯 Python 标准库），所有神经网络产物均已预训练完毕并随仓提供；
以下训练代码仅用于复现与后续迭代，**比赛运行不需要执行**。

### 模型清单

| 模型 | 架构 | 训练方式 | 用途 | 状态 |
| --- | --- | --- | --- | --- |
| **v5 启发式** | 规则引擎 | 手工特征 + 参数调优 | **参赛档位**（`--decider v5`） | 现役 |
| BC v7 base | Transformer（11M 参数） | 行为克隆 105 万条真机出牌 | 实验臂（botlike 场 19.2% 天花板） | 未采用 |
| BC v7 expert_ft | 同上，微调 | 在最强 20 名对手数据上微调 | 实验臂（21.6%，孤例不可复现） | 未采用 |
| PPO v7 | 同上 | 5 万局自对弈强化学习 | 待判决（agent-e 评测中） | 待定 |
| Hybrid（expert_ft + v5 接管胡/碰） | 混合 | 响应点 v5 裁决，其余交模型 | 首个超 expert_ft（22.2%） | 待验证 |
| GBDT 对手模型 | 梯度提升树 | 真机对手行为特征 | `risk` 档位的风险估计 | 可选组件 |

### 训练代码路径

| 路径 | 内容 |
| --- | --- |
| `src/nnrl2/model_v7.py` | Transformer 架构定义（obs 编码、多头注意力、策略/价值双头） |
| `src/nnrl2/policy_v7.py` | 推理包装（`PolicyV7Decider`，对齐主仓 `sim.round._ask` 接口） |
| `src/nnrl2/obs.py` | 公开信息 → token 序列编码器（369 维，含财神/副露/牌墙状态） |
| `scripts/rl2/gen_bc_data_*.py` | BC 数据生成（从真机日志提取 (obs, action) 对） |
| `scripts/rl2/train_bc*.py` | BC 训练（v1→v3 迭代史，v7 为当前架构） |
| `scripts/rl2/train_ppo*.py` | PPO 训练（自对弈 + KL 约束 + GAE） |
| `tools/train_pfirst.py` | 一胡先制模型（GBDT，用于 `risk` 档位） |
| `tools/train_value.py` | 价值模型（MLP，用于 `value` 档位） |
| `research/` | 早期探索（NN vs GBDT 对比、序列模型、占用容量等），已冻结 |

### 训练环境（与运行环境隔离）

```bash
# 训练需要额外依赖（不进主 pyproject，不影响参赛运行）
pip install -r requirements-rl.txt   # torch + numpy + scikit-learn

# BC 训练示例
PYTHONPATH=src python scripts/rl2/train_bc_v3.py --data data/rl2/bc_v7.npz

# PPO 训练示例（需 GPU）
PYTHONPATH=src python scripts/rl2/train_ppo_oracle.py --init runs/bc_v7_base.pt
```

模型产物（`runs/*.pt`、`data/*.npz`）体积大，不入 git；参赛运行不依赖它们。

## 九、开发与验证

```bash
uv run pytest                     # 运行全部测试
uv run python tools/selfplay.py   # 本地自对弈，比较策略强弱
uv run python tools/calibrate.py  # 按路线校准胜率与番数
```

## 十、深度学习训练管线（rl2 并入）

深度策略网络的完整训练管线已并入本仓：`src/nnrl2/` 为模型与推理代码，`scripts/rl2/` 为训练与评测脚本。
**训练需要 GPU 与 `requirements-rl.txt` 中的额外依赖（torch 等）；参赛运行不经过本管线。**

### 模型

`MahjongTransformerV7`（`src/nnrl2/model_v7.py`）：128 维嵌入 / 8 注意力头 / 4 层，
**940,968 参数**，策略/价值双头。推理包装见 `src/nnrl2/policy_v7.py`（对齐主仓决策器接口）。

### 核心脚本（scripts/rl2/）

| 脚本 | 用途 |
| --- | --- |
| `gen_bc_data_v7.py` | 行为克隆数据生成（真机日志 → (obs, action) 对） |
| `build_expert_dataset.py` | 强对手子集筛选（expert 微调数据） |
| `rebuild_situations_from_logs.py` | 从日志重建对局快照（审计/回放用） |
| `convert_audit_to_bc_v7_split.py` | 审计数据转 BC 训练格式（带切分） |
| `train_bc_v7.py` | BC 训练 |
| `train_ppo_v7.py` | PPO 训练（自对弈 + KL 约束） |
| `run_ab_v7.py` / `run_ab_v7_dup.py` | A/B 评测（dup = 加倍重复协议） |
| `run_ab_v7_hybrid.py` | 混合策略评测（模型 + v5 接管胡/碰） |

### 模型产物与基线（runs/）

| 模型 | 评测胡率 | 说明 |
| --- | --- | --- |
| `bc_v7_base` | 19.2% | 全量真机数据行为克隆基线 |
| `bc_v7_expert_ft` | 21.6% | 最强对手子集上微调（单 seed 孤例，待复现） |
| `bc_v7_expertv3_ft` | 19.1% | expert v3 修复版（撞 19.2% 天花板） |
| `ppo_v7_kl02_50k` | 18.4% | PPO 5 万局自对弈（KL=0.02） |

### 评测协议

- **方案 1**：固定 seat0、双 seed 对 3×v5 基线
- **方案 2**：同牌四位置镜像（消除座位与发牌运气）
- **Hybrid**：模型决策 + v5 接管胡/碰响应点

**当前最优为 Hybrid：22.2%**（注：与 v5 对手同源，增益能否迁移到真机真人对手待验证）。
神经网络各方案均未超过启发式 v5 的真机战绩，故参赛档位仍为 v5（见「四、启动」）。

## 十一、运行看护

比赛期间程序须持续在线（平台判据：最近 90 秒内有已认证请求）。仓库提供两种守护方式：

- `scripts/supervise.sh` —— 无需 root 的守护循环，崩了 3 秒后拉起；连续快速失败会**熔断**
  并要求人工介入（避免配置错误变成无限重启、持续打平台请求）
- `scripts/majiang-ai.service` —— systemd 用户级单元，`Restart=on-failure`

两者都**只在非零退出时重启**：赛事进入终态时程序正常返回（退出码 0），此时重启只会拿到
`TOURNAMENT_CLOSED` 空转刷请求。

启停、巡检与故障处理详见 **`docs/ops.md`**。
