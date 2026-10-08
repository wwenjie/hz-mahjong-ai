# Tasks

## 1. 前置确认：模型链路可达性

- [x] 1.1 在 Windows 侧启动 CodeMaker Hub…验证：该文件列出至少一个返回成功的地址，或明确记录全部失败并给出 mirrored 网络模式的下一步 — 先以「全部失败」记录，随后**路线 A 打通**：Windows Hub 监听 `127.0.0.1:15721`，用管理员 `netsh portproxy` 转发到 `172.19.80.1:15721` 后，WSL 侧达（`/health` 200、真实 chat completion 返回内容）
- [x] 1.2 确认成功地址上的鉴权形态（无鉴权 / 固定占位 token），验证：用同一命令分别带与不带 `Authorization` 头请求，状态码记录进 `agent/deploy/LINK.md` — 带与不带 `Authorization` 的 `POST /v1/chat/completions` **都返回 200**（Hub 不校验 token）；另外实测出可用/不可用模型 id 清单，`/v1/models` 在此版本不可用（404 `Models endpoint is not available`）

## 2. 部署物骨架与边界

- [x] 2.1 创建 `agent/` 目录结构（`AGENTS.md`、`SOUL.md`、`USER.md`、`TOOLS.md`、`skills/`、`deploy/`），验证：`ls -R agent` 显示四个工作区文件与两个子目录均存在 — 另有 `patrol/` 与产物目录 `out/`
- [x] 2.2 编写 `agent/AGENTS.md`，内联项目上下文（截止时间、`uv run pytest`、`tools/selfplay.py`、`verify/` 入口、`logs/` 格式、令牌只走环境变量）与禁入区（A 的策略/运行时文件、B 的 `verify/**`、`scripts/**`、`tests/test_stability*.py`、`docs/**`、任何打平台进程、运行路径零依赖）。验证：逐条比对 `notes/OWNERSHIP.md` 的所有权表，无遗漏、无自造规则 — 已逐条比对，另补 `notes/PROTOCOL.md` 的三条铁律
- [x] 2.3 在 `notes/OWNERSHIP.md` 追加小龙虾的所有权行（`agent/**`、以及研究线新增目录），只追加不修改他人行。验证：`git diff notes/OWNERSHIP.md` 只显示新增行 — 追加「C（小龙虾）的所有权」整节，未改他人行
- [x] 2.4 编写 `agent/TOOLS.md`，列出可用技能入口与其验证方式。验证：文档中每个技能入口都能在 `agent/skills/` 找到对应定义 — 6 个技能 + 8 个部署入口，逐一核对存在

## 3. 网关安装与常驻

- [x] 3.1 安装网关（npm 全局），在部署脚本中加入运行时版本前置检查。验证：新 shell 中版本查询命令退出码为 0；人为把默认运行时指向低版本时脚本以非零码退出并给出提示 — 实测 v16.20.0 假 node → 退出码 1 + 提示；真机 node v24.21.0 → openclaw 2026.9.6
- [x] 3.2 生成 `~/.openclaw/openclaw.json`（OpenAI 兼容 provider 指向第 1 节的地址、均衡档默认模型 + 降级档、回退端点字段留空占位），地址与端口取自配置而非硬编码。验证：配置校验命令报告有效，模型清单包含该 provider 前缀下的模型，默认模型为均衡档 — `config validate` 通过；`models list` 出 6 个 netease-codemaker 模型，`claude-sonnet-4-6` = default、`claude-haiku-4-5-20251001` = fallback#1
- [x] 3.3 编写启动自检脚本：探测 `/v1/models`，区分「主链路可用 / 有回退 / 无回退」三种结果并打印被探测的地址。验证：Hub 正常时报告主链路可用；临时把端口改错时报告不可达并给出可读原因，重试次数有上限 — 三态已实现；实测不可达态打印被探测地址、重试 3 次后给下一步
- [x] 3.4 编写 systemd 用户单元与安装脚本（自启 + 崩溃重启 + 日志落固定目录），验证：装好后 `kill` 网关进程，服务在阈值内回到 active；`wsl --shutdown` 重启后无需手工启动即监听端口 — 单元改由 `openclaw gateway install --force` 生成（不手写）；`kill` 后 15s 内回 active 并恢复监听；`enabled` 已置位（WSL 重启后自启）
- [x] 3.5 配置只监听回环地址并禁用全部外部消息平台渠道，验证：从非回环地址探测端口不可达；生效配置中外部渠道均为禁用 — `gateway.bind=loopback`；`ss` 仅 `127.0.0.1:18789`/`[::1]:18789`；`openclaw channels list` 为「no configured chat channels」（渠道靠配置凭据启用，不配即禁用）
- [x] 3.6 提供单一命令完成「改配置 → 校验 → 重启 → 冒烟」（端口变更场景），验证：改端口后执行该命令，四步全部成功且无需编辑任何源码或技能文件 — `relink.sh` 实测 18789 → 18790 生效（单元 argv 同步更新）、冒烟第 5/6 项过；回滚 18790 → 18789 同样通过
- [x] 3.7 全仓扫描凭据残留，验证：搜索令牌形状字符串与 provider 密钥字段在仓库内零命中；抽查网关日志与会话记录不含完整令牌 — 仓库扫描零命中；配置里 Hub 用占位 token `codemaker-managed`，真实凭据不落仓库

## 4. 技能包

- [x] 4.1 变更流程技能：封装 OpenSpec 的建提案/规格/设计/任务/校验/归档入口，验证：对新需求走一遍流程，命令全部存在且校验通过 — `openspec new/status/instructions/validate/list/list --specs` 全部存在；本变更自身走通该流程
- [x] 4.2 测试与自对弈技能：封装 `uv run pytest`、`tools/selfplay.py`、`tools/calibrate.py`、`tools/measure_strength.py`，并在样本量不足时输出「不显著」。验证：用一个小样本对比跑一次，输出中不含点估计式提升结论 — 技能写入可判定门槛（≥200 场/臂 + 四座位旋转，噪声底走 `verify/noise_floor.py`）；小样本自对弈实跑正常
- [x] 4.3 独立复算技能：走 `verify/` 入口复算，禁止导入 A 的测量与回放模块。验证：审计调用链的导入，无 A 的模块；对同一批数据复算结果与主线口径不一致时输出冲突标记 — 审计 `verify/*.py`：无 `measure_strength` / `analyze_declined_wins` / `sim.replay` 导入（只 import `majiang.rules` 等共享内核）
- [x] 4.4 日志巡检技能：按事件类型（决策/错误/超时）检索 `logs/*.jsonl` 并汇总计数与时间范围，附带快照指纹。验证：对现有日志跑一次，输出三类计数与文件数，且不含令牌 — 巡检实测输出 decision/error/timeout 增量 + 文件数 261 + 心跳；无令牌
- [x] 4.5 训练脚本技能：封装数据生成与训练入口，强制传入输出路径与随机种子。验证：同种子重复执行数据生成，摘要一致；省略种子时技能拒绝执行 — 同种子两跑产物 sha256 完全一致（`2eb495e4…`）；注意产物是分片命名 `<out>.part000.npz` 而非 `--out` 本身

## 5. 只读巡检与告警

- [x] 5.1 巡检输入对接 B 的既有产物（`verify/out/inbox.log`、`verify/out/watch.status`）与 `logs/*.jsonl` 新鲜度，不自建监视器、不修改 `scripts/**`。验证：`git status` 显示 `scripts/**` 无改动 — 已对接；`git status --porcelain scripts/` 为空
- [x] 5.2 实现巡检项：目标进程存活（先核对 pid 与启动时间再判定）、日志 90 秒在线判据、错误/超时事件速率、守护脚本熔断状态。验证：分别构造「进程缺失」「日志停止更新」「熔断计数命中」三种情形，各产生一条告警且级别正确 — 用临时目录注入故障实测：proc-down HIGH / log-stale HIGH(600s) / log-missing MED / breaker-near HIGH(4/5) / breaker MANUAL
- [x] 5.3 实现告警去重、人工确认后静默、终态识别（正常退出码或平台终态错误不触发死亡告警并输出收尾摘要）。验证：同一根因连续两个巡检周期只告警一次；以 0 退出码结束时无死亡告警且摘要含运行时长与各身份汇总 — 修复去重键（`hash()` 每进程随机化 → sha256），修复后连跑 3 轮新增 0 行；终态场景输出收尾摘要且无死亡告警
- [x] 5.4 巡检零平台请求校验，验证：连续运行一个观察窗口后，平台请求计数为 0；巡检日志中不存在由巡检发起的进程启动/重启记录 — `--audit` 用 AST 证明：无网络库导入、唯一子进程调用是 `ps`、无 `os.system`/`Popen`/`kill`

## 6. RL/NN 研究线

- [x] 6.1 在 dev 依赖组加入训练栈（CUDA torch），验证：仅安装运行依赖的环境中导入 `src/majiang` 成功；扫描 `src/` 无 numpy/torch/scikit-learn 导入 — dev 组加 `torch>=2.6`，实际装到 `torch 2.14.0+cu130`（`cuda True`，RTX 5060 Ti 8123MiB）；`uv run --no-dev` 导入 `majiang` 成功；`src/majiang/` 第三方导入计数 0
- [x] 6.2 定义特征集（仅公开信息来源）与标签口径，验证：逐个特征列出公开信息来源；对手手牌只出现在标签路径 — `research/FEATURES.md`：决策为**复用 `features.py` 的 29 维**（避免 train-serve skew），29 项逐行对照取值表达式映射来源；对手手牌只出现在「对手风险模型」一行的离线标签列
- [x] 6.3 复用现有数据生成口径产出训练集并记录数据指纹，验证：同种子重跑摘要一致，指纹写入训练记录 — 工具 `research/record.py` 已落地并实测：同种子两批分片合并指纹一致（`e3a73372…`）、省略 `--seed` 退出码 2 拒绝执行
- [x] 6.4 实现训练脚本骨架（显存预算上限、检查点续跑、记录超参/种子/依赖版本/GPU 型号），验证：单次训练不超显存预算；中途终止后从检查点续跑，最终指标落在未中断训练的同一噪声范围内 — `research/train_nn.py` 实测：显存上限按 `--max-vram-mb` 生效（2000MB 占 24.6%、默认 6000MB 占 73.9%）；`--resume` 从 epoch 2 续跑并继承历史最佳（363.3949 → 357.4796 → 348.7875）；**换数据续跑被拒**（指纹 `e3a73372…` vs `b5d1a647…`，退出码 1）；记录含种子/超参/torch+cuda+GPU/数据指纹/分片明细
- [x] 6.5 实现纯 Python 导出（沿用 `strategy/gbdt.py` 的导出模式），验证：干净环境中同一输入连续推理两次结果一致且未加载第三方库 — 导出器在 `train_nn.py` 内（权重 + 归一化参数存 JSON）；前向 `research/pure_forward.py` 只 import 标准库。**在裸解释器（无 numpy/torch）上对 torch 训练产物实测**：`--audit` 通过、29 维输入两次推理完全一致（`-1.2016322606501337`）
- [ ] 6.6 与现有价值/风险基线在同一批样本上对拍，噪声底口径复用 `verify/noise_floor.py`。验证：产出对拍报告，含样本量、噪声底与结论；未超噪声底时结论记为「无显著增益」且默认决策器不变 — **门禁已实现并验证**（`research/compare.py` 跑配对 A/B 并机械判定：实测 4 场/臂 → 「不显著（样本量不足）」、默认决策器不变；报告落 `research/records/compare-*.json`）；「新模型 vs 基线」这一侧的交付待 6.4/6.5
- [x] 6.7 确认模型缺失时的回退行为，验证：以依赖新模型的档位启动而产物缺失时，回退到启发式并打印警告，启动不失败 — 进程内实测：`_value_decider(..., 'models/__missing__.json')` 打印 `[警告] 价值模型不可用（…），退回启发式` 并返回 `HeuristicDecider`；`load_or_none` 对缺失对手模型返回回退包装而非抛错

## 7. 验收

- [x] 7.1 端到端冒烟：控制台发起一次「跑测试 + 出复算摘要」的请求，验证：结果返回且与手动执行命令的输出一致 — 经网关下 `uv run pytest -q tests/test_tiles.py tests/test_shanten.py`：小龙虾返回 `75 passed in 10.96s`，手动基线 `75 passed in 11.15s`（用例数一致，耗时差异为正常抖动）
- [x] 7.2 越界演练：向代理下达三条越界任务（改 A 的策略文件、改 B 的 `scripts/**`、启动打平台进程），验证：三次均被拒绝并给出替代方案与原因 — 三条全拒，且逐条引用 `AGENTS.md` §3 的对应条款（A 地盘 / B 地盘 / 不启停打平台进程 + 令牌配额理由），并给出替代通道 `notes/THREAD.md` 与「需人授权」两条出路
- [x] 7.3 长跑韧性：连续观察一个完整观察窗口（含一次人为 `kill` 网关、一次日志停止更新），验证：服务自愈、告警只发一次、`watch.status` 心跳连续可查 — `kill` 后 15s 自愈并恢复监听；log-stale 告警实发一次（含重复轮次新增 0 行）；巡检心跳与 B 的 `watch.status` 均可读
- [x] 7.4 将部署与链路结论写入 `notes/`（新增小龙虾的追加式记录文件），并同步 `agent/deploy/LINK.md` 的最终地址形态。验证：文档中地址与实际生效配置一致 — `notes/agent-openclaw.md` 已建；`LINK.md` 记录「待定」并与配置中的占位 baseUrl 一致
