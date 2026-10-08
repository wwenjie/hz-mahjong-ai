# Design

## Context

动机见 `proposal.md`。以下是决定方案的当前状态与约束（均为本机实测）：

**运行环境**
- WSL2（NAT 模式）：`eth0 = 172.19.84.149/20`，默认网关 `172.19.80.1`，`/etc/resolv.conf` 指向 `10.255.255.254`。
  `systemd` 已启用（pid 1 = systemd，`/etc/wsl.conf` 含 `boot.systemd=true`），可用用户级服务。
- Node.js `v24.21.0` + npm `11.19.0` 已就绪；`openclaw` 未安装。
- GPU `RTX 5060 Ti 8GB`；`torch` 未安装。

**模型链路现状（关键约束）**
- 目标链路是 CodeMaker Hub 本地代理（OpenAI 兼容 `/v1`，占位 token）。当前 `127.0.0.1:15721` 未监听；
  Windows 宿主地址（`172.19.80.1`、`10.255.255.254`）上该端口也不通 —— Hub 尚未启动。
- 若 Hub 运行在 Windows 侧，NAT 模式下 WSL 需要经宿主 IP 访问，且需要 Windows 防火墙放行；
  Hub 的实际端口由 Hub 设置决定，**不可硬编码**。
- 备用直连端点存在，但本机既无 `~/.local/share/codemaker/auth.json`，也无可见的凭据文件
  （CodeMaker 扩展以 language server 形式运行在 WSL 内，凭据不在文件系统明面），
  因此直连路线需要额外的令牌获取机制 —— 这也是优先 Hub 的原因。

**仓库既有资产与所有权**
- `notes/OWNERSHIP.md` 已把 `src/majiang/strategy/**`、`runtime/**`、`sim/replay.py`、部分工具划给 A；
  `docs/**`、`scripts/**`、`tests/test_stability*.py`、`verify/**` 划给 B。禁入区据此生成，不新造规则。
- `scripts/agent_watch.py` + `agent_watch_supervisor.sh` 是 B 的守护进程：监视 `notes/agent-a.md`、
  `notes/THREAD.md` 与数据增长，产物落在 `verify/out/`（`inbox.log`、`watch.status`、`watch.state.json`）。
  **巡检能力应消费这些产物，而不是再造一个监视器。**
- 训练侧已有 `tools/gen_value_data.py` / `train_value.py` / `gen_opponent_data.py` / `train_opponent.py`，
  推理侧已有 `src/majiang/strategy/gbdt.py`（纯 Python 树模型推理）——NN/RL 的导出格式沿用这一模式。
- `src/majiang/**` 零第三方依赖是硬约束（`pyproject.toml` 的 `dependencies = []`），
  `numpy`/`scikit-learn` 只在 `dev` 组。

## Goals / Non-Goals

**Goals:**
- 一条命令完成部署与健康检查；链路参数（地址、端口、模型档位）全部外置为配置
- 长跑可控：systemd 常驻、崩溃自愈、日志可查
- 边界机器可判定：禁入区写成文件并能被代理在动作前读取，而不是靠人提醒
- 研究线可复现：训练记录 + 纯 Python 产物 + 噪声底对拍三重关口

**Non-Goals:**
- 不接任何外部消息平台（控制面只有本地 Web 控制台）
- 不改 A/B 的文件，不碰 `scripts/**` 与 `verify/**` 的既有内容（只追加 `notes/OWNERSHIP.md` 的所有权行）
- 不把训练依赖引入运行路径，不改变提交形态
- 不做代理的通用能力扩展（插件生态、多代理协作、公网暴露）——本轮只为这三条线服务

## Decisions

1. **部署形态：WSL 内 npm 全局安装 + systemd 用户单元，不用容器。**
   理由：`systemd` 已启用，用户级单元足够；容器会让代理访问仓库与 GPU 多一层摩擦（且 GPU 直通在 WSL 内更麻烦）。
   备选：Docker 隔离 —— 更安全但会让「代理直接改仓库文件」变成挂载与权限调试；本轮否决，留作后续加固项。

2. **模型链路：Hub 本地代理为主，端口进配置 + 启动自检 + 显式回退。**
   理由：Hub 用固定占位 token，没有 24h 过期问题；但端口位置（WSL 内回环 vs Windows 宿主 IP）本轮尚不确定。
   做法：单一配置项 `baseUrl`；启动前探测 `/v1/models`；不可达时按是否配置了回退端点给出两种确定行为（见 `specs/openclaw-gateway`）。
   备选：直接按 15721 硬编码 —— 一旦用户把 Hub 装在 Windows 侧（或换端口）就整体失效，否决。

3. **部署物落 `agent/`，不动 `scripts/`。**
   `scripts/**` 归 B 且正在服务稳定性演练；把新单元文件塞进去会制造所有权冲突。
   结构：`agent/AGENTS.md`、`agent/skills/`、`agent/deploy/`（部署脚本、systemd 单元、冒烟脚本）。
   备选：复用 `scripts/majiang-ai.service` 的模式放到 `scripts/` —— 违背所有权表，否决。

4. **巡检复用 B 的产物，不自建监视器。**
   `ops-watch` 的输入是 `verify/out/inbox.log`、`verify/out/watch.status`、`logs/*.jsonl` 的日志新鲜度，
   以及目标进程存活；输出是控制台告警。理由：B 的 watch 已经在做「发现新条目 + 不变量校验」，
   重复实现会同时产生两份口径（这正是 OWNERSHIP.md 里反复踩的坑）。
   备选：OpenClaw 自己 `tail` 原始事件流 —— 与本机长跑进程抢 IO 且重复解析，否决。

5. **RL/NN 走「离线训练 → 纯 Python 导出」双轨，训练依赖只进 dev 组。**
   理由：8GB 显存限制了模型规模，纯 Python 前向就是模型规模的自然上限；这同时保证提交形态不变。
   备选：运行时引入轻量推理库（onnxruntime 等）—— 破坏零依赖约束，否决。

6. **新模型的准入门槛：与基线同批样本对拍 + 超过噪声底。**
   噪声底口径复用 `verify/noise_floor.py`（B 已有），不新造统计方法。
   理由：OWNERSHIP.md 记录的四次被推翻都源于「判据有偏」，研究线必须先过仪表这一关。

7. **凭据只落在 `~/.openclaw/openclaw.json` 与进程 env。**
   Hub 场景用固定占位 token；若启用直连回退，真实 token 由部署脚本从外部读入并写入该文件，
   仓库内只留 `${VAR}` 形式的引用。

## Risks / Trade-offs

- [Hub 在 Windows 侧，NAT 模式下 WSL 无法回连宿主] → 部署脚本对「回环 + 宿主 IP + 用户给定地址」挨个探测并打印结果；
  若全部失败，提示可改 WSL 为 mirrored 网络模式（`networkingMode=mirrored`）后重试。
- [Hub 未运行导致网关空转重试刷日志] → 自检失败时重试次数有上限，且状态明确标记模型不可用。
- [`claude-opus` 档 TPM 低，与 codemaker CLI 叠加触发 429] → 默认档位选均衡档，并把降级档位写进配置。
- [代理长跑无人监督导致越界改动] → 禁入区写进 `AGENTS.md` 并在首次动作前读取；产出物落位有约定目录；
  `notes/OWNERSHIP.md` 追加所有权行，使「谁改了什么」可审计。
- [训练把整机显存吃满，影响其他工作] → 单次训练显存预算硬上限 + 检查点续跑；训练与巡检不同时全速跑。
- [长跑进程被环境周期性回收（OWNERSHIP.md 已记录的现象）] → 网关交给 systemd 托管而非 `nohup`；
  巡检状态落盘（`watch.state.json` 同思路）以便跨重启恢复计数，避免「重启后计数器归零」变成静默偏置。
- [技能包与命令行工具漂移] → 技能执行前校验子命令存在性，不匹配即报告并停止，不猜命令。

## Migration Plan

1. 在 Windows 侧启动 CodeMaker Hub，记录实际端口；在 WSL 内用一次探测确认可达的地址形态。
2. 落地 `agent/` 目录（`AGENTS.md`、技能、部署脚本、systemd 用户单元）。
3. 安装并校验配置，起 systemd 用户服务，冒烟探测链路 + 控制台登录。
4. 追加 `notes/OWNERSHIP.md` 的所有权行（只追加，不改别人的行），确认禁入区与既有约定一致。
5. 巡检能力先以「只读 + 控制台告警」上线，观察一轮；再开启训练线。
6. 回滚：`systemctl --user disable --now` 停服 + 保留 `~/.openclaw` 与 `agent/` 目录即可；
   仓库侧改动是纯新增，`git revert` 单次提交即可复原。

## Open Questions

- Hub 端口与「WSL 内回环还是 Windows 宿主 IP」的最终形态，待用户在 Windows 侧启动 Hub 后确定。
