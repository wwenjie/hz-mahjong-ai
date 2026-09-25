# Tasks

## 1. 独立复算：结构性不变量层

- [ ] 1.1 写 `verify/invariants.py`：纯标准库解析事件流，按 `round_no` 切局，验证每局每种牌全场 ≤4、总量 136、出牌必在手；运行后输出全部 1129+ 文件的违反明细（目标 0 违反）与快照指纹（文件数、总局数）
- [ ] 1.2 处理三个已知陷阱（chi 去重、seat 0、多 block 同 round_no），用 1.1 的零违反结果反向确认解析正确

## 2. 独立复算：指标层

- [ ] 2.1 写 `verify/metrics.py`：手数/胜次数/胜率、有胡率÷4 公平份额、名次分布与名次分（rounds[] 与 round_ended 双来源对账）、均番对比、副露/局、胡牌时已出手数；输出含未完成对局的单独计数
- [ ] 2.2 与 A 的参考值逐项对比，任何不一致定位到具体文件 + round_no + seq，结论追加到 `notes/agent-b.md`

## 3. 独立复算：手算对照层

- [ ] 3.1 写 `verify/spotcheck.py`：随机抽 10 局打印手牌张数/副露/牌墙剩余/赢家，人工逐局核对原始 JSON，四项全对才算过；结果追加到 `notes/agent-b.md`

## 4. 稳定性演练（7.2）

- [ ] 4.1 实现假 transport（可控注入：连接拒绝/连续 5xx/409/pending 挂起），放 `tests/` 或 `scripts/fakes/`，验证运行时公开接口不变时假 transport 可被 pytest 使用
- [ ] 4.2 `tests/test_stability_network.py`：不可达地址 + 连续 5xx，验证退避重试、不崩溃、不丢本地状态，记录失败时表现
- [ ] 4.3 `tests/test_stability_stage.py`：`stage_crashed` 重赛 + 进程被杀后快照重建接管，验证不提交动作到作废阶段、不重复提交

## 5. 稳定性演练（7.3）

- [ ] 5.1 `tests/test_stability_phase.py`：settled 期间动作→409 的处理、长轮询 pending 不误判掉线
- [ ] 5.2 `tests/test_stability_race.py`：动作冲突竞态至多一个被接受、多场并发决策互不干扰且限速受遵守
- [ ] 5.3 每项演练的可复现运行方式 + 观察结果 + 失败表现写入 `docs/stability-report.md` 并追加 `notes/agent-b.md`

## 6. 提交物（8.3）

- [ ] 6.1 写 `scripts/verify_clean_env.sh`：干净 venv 安装 + CLI 可用 + 本地模拟 8 局 + 模型求值无 numpy/sklearn，脚本跑通退出码 0
- [ ] 6.2 写 `docs/USAGE.md` 评审版使用说明：接入方式/启动步骤/依赖环境/令牌环境变量配置；grep 提交物确认无明文令牌
- [ ] 6.3 最终验收：`openspec validate majiang-ai-agent-b` 通过，`uv run pytest -q` 全绿，全部结论已入 `notes/agent-b.md`
