# Spec Delta

## Purpose

官方杭州麻将平台的玩家 API 客户端：以正确的令牌身份完成报名、到位与每阶段出席确认，在 90 秒在线判据内维持活跃，并以长轮询增量消费对局事件流。

## ADDED Requirements

### Requirement: 令牌与认证

客户端 SHALL 支持两类令牌：绑定具体锦标赛的参赛令牌（scoped）与全局令牌。全部请求 MUST 携带 `Authorization: Bearer <令牌>` 请求头。令牌 MUST 仅通过环境变量或启动参数注入，MUST NOT 写入源码、日志或版本库。

#### Scenario: 参赛令牌查询自身信息

- **WHEN** 使用参赛令牌调用 `GET /api/me`
- **THEN** 响应中的 `tournament_id` 非空，客户端据此直达该锦标赛

#### Scenario: 全局令牌的作用域限制

- **WHEN** 使用全局令牌调用 `GET /api/tournaments/me/rules` 或 `POST /api/tournaments/me/ready`
- **THEN** 服务端返回 `400 TOKEN_NOT_SCOPED`，客户端改用显式锦标赛 id 的等价端点

### Requirement: 运行时规则配置读取

客户端 SHALL 在参赛前通过 `GET /api/tournaments/me/rules`（或 `GET /api/tournaments/{id}`）读取 `config`，并以返回值为唯一权威。客户端 MUST NOT 对 `M`、`Rounds`、`BaseScore`、`YouCaiBiKao`、`PengTimeoutSec`、`ChiTimeoutSec`、`DiscardTimeoutSec` 写死取值。

#### Scenario: 读取并应用配置

- **WHEN** 规则接口返回 `M`、`Rounds`、`BaseScore`、`YouCaiBiKao` 与三个超时秒数
- **THEN** 客户端据此配置并发场数、决策时限与计分底分，并记录配置快照用于复盘

#### Scenario: 配置变更

- **WHEN** 再次读取的 `config` 与上次不同
- **THEN** 客户端以最新值覆盖运行时参数，无需重启进程

### Requirement: 报名与到位

客户端 SHALL 在报名期以幂等方式提交 `POST /api/tournaments/{id}/register` 与 `POST /api/tournaments/{id}/ready`。当服务端返回已开赛、已关闭或已报名等冲突码时，客户端 MUST 不视为失败，并回到状态机按最新状态处理。

#### Scenario: 重复报名

- **WHEN** 客户端在同一锦标赛重复提交报名
- **THEN** 服务端幂等成功，客户端继续后续流程

#### Scenario: 恰逢开赛的竞态

- **WHEN** 提交报名的瞬间赛事已开赛并返回冲突码
- **THEN** 客户端不重试该端点，转而读取最新赛事状态并据此推进

### Requirement: 多阶段状态机

客户端 SHALL 以赛事 `status` 为唯一进度真相驱动主循环，覆盖 `registering`、`running`、`stage_done`、`stage_open`、`finished`、`closed`、`void`。客户端 MUST 区分「阶段正常结束」与「阶段中断待重赛」（`stage_crashed`），且仅在 `finished`、`closed`、`void` 时退出。

#### Scenario: 阶段结束但赛事未结束

- **WHEN** 状态为 `stage_done`
- **THEN** 客户端保持在线等待，不退出进程

#### Scenario: 赛事终态

- **WHEN** 状态为 `finished`、`closed` 或 `void`
- **THEN** 客户端停止对局循环并输出最终成绩

#### Scenario: 阶段中断待重赛

- **WHEN** 状态显示当前阶段 `stage_crashed` 为真
- **THEN** 客户端丢弃本地本阶段累计，按重新公布的开赛时间重新执行出席确认

### Requirement: 阶段出席确认

客户端 SHALL 在 `stage_open` 阶段以幂等方式提交出席确认。当 `qualified` 为假时，客户端 MUST 判定自身已被淘汰并停止参赛，且 MUST NOT 重试确认请求。

#### Scenario: 晋级者确认

- **WHEN** 状态为 `stage_open` 且 `qualified` 为真
- **THEN** 客户端提交出席确认，并记录确认结果

#### Scenario: 候补确认

- **WHEN** 状态为 `stage_open` 且 `qualify_role` 为候补
- **THEN** 客户端同样提交出席确认，以在晋级者缺席时按名次递补

#### Scenario: 已淘汰

- **WHEN** 提交确认后返回 `409 NOT_QUALIFIED`
- **THEN** 客户端判定已淘汰并退出，不再重试

### Requirement: 心跳保活

客户端 SHALL 周期性发起已认证请求以维持在线状态，两次请求间隔 MUST 显著小于 90 秒。心跳 MUST 在长轮询空闲、阶段等待与异常恢复期间持续运行。

#### Scenario: 空闲期保活

- **WHEN** 赛事处于阶段等待且无活跃对局
- **THEN** 客户端仍按配置间隔发起已认证请求，维持在线判据

#### Scenario: 心跳失败恢复

- **WHEN** 心跳请求因网络异常连续失败
- **THEN** 客户端进入重连流程并持续重试，直至恢复或赛事进入终态

### Requirement: 状态长轮询与快照重建

客户端 SHALL 通过 `GET /api/games/{id}/state?seq=N` 消费对局事件，其中 `seq=N` 语义为返回 N 之后的事件。客户端 MUST 以 `seq=0` 获取全量快照作为规范真相，并 MUST 在响应出现缺口标记或动作提交遭遇冲突时重建全量快照。

#### Scenario: 增量事件推进游标

- **WHEN** 长轮询响应携带新事件
- **THEN** 客户端以事件自身的 `seq` 推进本地游标

#### Scenario: 无新事件挂起

- **WHEN** 响应标记为挂起且不含事件
- **THEN** 客户端以同一游标继续长轮询，不重建快照

#### Scenario: 跨局落后

- **WHEN** 本地游标落后于当前局并收到全量快照
- **THEN** 客户端以快照为新局权威局面，覆盖本地手牌与局号

#### Scenario: 动作冲突后重建

- **WHEN** 动作提交返回 `409 INVALID_ACTION`
- **THEN** 客户端以 `seq=0` 重建快照后再重新决策

### Requirement: 局间停顿处理

客户端 SHALL 识别 `phase="settled"` 的局间状态，此时快照中的手牌为上一局残留且不得用于牌型计算。该阶段客户端 MUST NOT 提交任何动作。

#### Scenario: 局间停顿

- **WHEN** 快照阶段为 `settled`
- **THEN** 客户端不进行牌型计算、不提交动作，继续轮询等待新局快照

#### Scenario: 新局发牌

- **WHEN** 挂起中的轮询收到新局全量快照
- **THEN** 客户端以该快照的手牌与局号重置本局状态

### Requirement: 动作提交与错误处理

客户端 SHALL 通过 `POST /api/games/{id}/action` 提交 `discard / chi / peng / gang / hu / pass` 动作，吃牌可附带 `tiles` 指定所用两张手牌。错误处理 MUST 依据错误码字段判型，并 MUST 区分可重试与永久条件。

#### Scenario: 指定吃牌组合

- **WHEN** 同一张牌存在多种吃法且客户端选定其中一种
- **THEN** 客户端在动作体中附上 `tiles` 明确所用两张手牌

#### Scenario: 永久条件不重试

- **WHEN** 响应为 `404 NO_ROOM_AVAILABLE`、`403 FEATURE_DISABLED` 或 `403 PORTAL_BINDING_REQUIRED`
- **THEN** 客户端判定为永久条件，停止重试并按赛事状态决定后续动作

#### Scenario: 窗口重复响应

- **WHEN** 客户端对同一响应窗口重复提交 `pass` 或动作
- **THEN** 客户端依据本地已响应状态避免重复提交，若仍触发冲突则按重建快照流程恢复

### Requirement: 请求限速与退避

客户端 MUST 将状态轮询频率控制在服务端限制以内，并 MUST 限制并发挂起请求数量。收到限速响应时客户端 SHALL 退避后重试。

#### Scenario: 触发限速

- **WHEN** 状态轮询返回 `429 RATE_LIMITED`
- **THEN** 客户端退避后重试同一请求，并压低后续轮询频率

#### Scenario: 并发场次下的轮询预算

- **WHEN** 同时参赛场数为 M
- **THEN** 客户端在统一令牌桶下分配轮询额度，使总频率不超过服务端上限

### Requirement: 指南版本自检

客户端 SHALL 在启动时读取平台指南版本，并与开发时已知版本比对；发现更高版本的破坏性变更时 MUST 输出显式警告。

#### Scenario: 检测到破坏性变更

- **WHEN** 服务端指南版本高于客户端已知版本且变更记录中存在破坏性条目
- **THEN** 客户端输出该条目的摘要并标记需要人工核对
