---
name: mahjong-delegate-codemaker
description: 把重活委派给 CodeMaker CLI（非交互 codemaker run），用于大范围代码改动、长上下文分析或要第二意见时。需要省自己上下文、或任务超出本线能力时使用。
---

# 委派给 CodeMaker

## 何时用

- 任务**大而独立**：跨多文件改造、大范围重命名、通盘重构 —— 自己一步步做会吃掉整轮上下文。
- 要**第二意见**：对一个结论拿不准，让 CodeMaker 独立复算/复述一遍（它看不到你的推理，天然是独立信道）。
- 要**长上下文通读**：把整个模块读完再给判断。

## 何时**不要**用

- **仪表类任务**：测量、复算、对拍、噪声底 —— 这些必须是你自己从原始数据重写的代码
  （`notes/OWNERSHIP.md` 的教训：同一个 bug 会在两处犯同样的错）。
- **任何要访问比赛平台的事**：委派出去的进程仍然是你启动的，仍然算你抢 A 的令牌配额。
- **禁入区的改动**（见 `AGENTS.md` §3）：委派不能绕过边界 —— 你派什么，它就可能改什么。

## 命令

```bash
bash agent/deploy/cm.sh "把 tools/foo.py 按 <要求> 改造，改完跑 uv run pytest -q tests/test_foo.py"
bash agent/deploy/cm.sh --text "..."        # 人要读的输出（默认是机器可读的 JSON 汇总）
bash agent/deploy/cm.sh --models            # 看可用模型
bash agent/deploy/cm.sh --usage             # 查 token 消耗（不调模型）
```

包装脚本已经处理：token 走环境变量（**不进 argv、不入仓库**）、超时兜底（默认 1800s）、
`< /dev/null` 关掉 stdin、留痕到 `agent/out/codemaker-delegate.log`。

输出末尾固定有一行用量：`tokens=N cost=¥X.XXXX exit=C` —— **每次都看一眼**，这是配额纪律的执行点。

三个实测坑（踩过，别再踩）：

- **别用 `--format default`**：那会走 TUI 式输出并等 stdin，脚本里表现为「卡住不返回」。包装脚本已固定 `--format json --pure`。
- **不关 stdin 就会假死**：`codemaker run` 会等 stdin。包装脚本已加 `< /dev/null`。
- **超时先怀疑提示词**，不要只把超时调大。

## 硬规则

1. **委派提示词里必须带边界**。机器可读的写法是直接贴约束，例如：
   「只许改 `tools/` 与 `tests/` 下的文件；不许碰 `src/majiang/strategy/**`、`src/majiang/runtime/**`、
   `verify/**`、`scripts/**`、`docs/**`；不许启动任何访问比赛平台的进程。」
2. **配额共用**。你和用户 IDE 里的 CodeMaker 会话**共用同一账号**，平台按用户限 TPM ——
   外层循环用便宜的档（本仓库默认 `deepseek-flash`），重活才委派；opus 档最容易 429。
   长任务前先 `--usage` 看一眼当月消耗。
3. **不信摘要，只信 diff**。委派回来先 `git status` + `git diff` 看它到底改了什么，
   再自己跑一遍测试。它的"我改好了"不是证据。
4. **一次一件事**。别一次派三个不相关的任务，否则你不知道哪一步引入了问题。
5. **产出留痕**：把它改动的文件路径与测试结果记进 `notes/agent-openclaw.md`（追加）。
6. **超时就是信号**：任务超时说明它太大或提示词太含糊，拆小重派，不要直接调大超时。
