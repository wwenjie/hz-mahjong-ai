# Spec Delta

## Purpose

让 OpenClaw 网关在本机 WSL2 内稳定常驻、只经本地 Web 控制台访问，并把模型请求接到公司 CodeMaker Hub 本地代理，链路不可达时给出可读结论而不是静默失败。

## ADDED Requirements

### Requirement: 本机安装与运行时版本约束

网关 SHALL 以 npm 全局安装方式部署在本机 WSL2 内，运行时 SHALL 满足该版本的要求；部署脚本 MUST NOT 在运行时版本不满足时静默继续。

#### Scenario: 新 shell 直接可用

- **WHEN** 新开一个 shell 执行网关命令行入口并查询版本
- **THEN** 输出可解析的版本号且退出码为 0

#### Scenario: 运行时版本不满足

- **WHEN** 默认运行时版本低于网关要求
- **THEN** 部署脚本以非零退出码结束，并提示需要切换到的版本，不执行后续安装步骤

### Requirement: 以 systemd 用户服务常驻

网关 SHALL 由 systemd 用户单元管理，具备开机自启与崩溃自动重启；日志 SHALL 落到工作区外的固定目录以便排查。

#### Scenario: 崩溃自愈

- **WHEN** 网关进程被强制终止
- **THEN** 服务在一个明确的短阈值内重新进入 active，且控制台恢复可用

#### Scenario: WSL 实例重启后自动拉起

- **WHEN** WSL 实例重启且 systemd 已启用
- **THEN** 无需人工登录或手工启动，网关即监听本机端口

### Requirement: 只监听本机且不接外部消息平台

网关 SHALL 仅绑定回环地址；所有外部消息平台渠道 SHALL 处于禁用状态，控制面只经本地 Web 控制台暴露。

#### Scenario: 非回环地址不可达

- **WHEN** 从本机非回环地址或局域网地址探测网关端口
- **THEN** 连接被拒绝或不可达

#### Scenario: 渠道白名单校验

- **WHEN** 读取生效配置中的渠道启用状态
- **THEN** WhatsApp、Telegram、Discord、iMessage 等外部渠道全部为禁用

### Requirement: 模型 provider 接驳 CodeMaker Hub

网关 SHALL 以 OpenAI 兼容 provider 指向 CodeMaker Hub 本地代理；代理地址与端口 SHALL 是配置项而非硬编码值；默认模型 SHALL 选择配额余量足够的档位，避免与同账号的 CLI 叠加触发每分钟 token 限额。

#### Scenario: 配置校验通过且模型可见

- **WHEN** 校验配置文件语法并列出可用模型
- **THEN** 校验报告有效，且模型清单包含该 provider 前缀下的模型

#### Scenario: 默认模型档位正确

- **WHEN** 查询当前模型状态
- **THEN** 默认模型为配置中指定的均衡档位，而非最高配额消耗档

#### Scenario: 命中配额降级

- **WHEN** 主模型返回 token 限额类错误
- **THEN** 网关切换到配置的降级模型继续本次会话，不中断服务

### Requirement: 启动自检与链路回退

网关 SHALL 在启动前探测模型链路可达性；不可达时 SHALL 区分「已配置回退端点」与「无回退」两种情况给出确定行为，并 MUST NOT 产生无限重试。

#### Scenario: 主链路可达

- **WHEN** 自检请求返回成功状态
- **THEN** 使用主链路，不触发回退，启动日志记为正常

#### Scenario: 主链路不可达且配置了回退端点

- **WHEN** 自检超时或连接被拒绝，且配置中存在回退端点与凭据
- **THEN** 网关仍成功启动，状态显示当前使用回退链路，控制台给出可读原因（含被探测的地址与端口）

#### Scenario: 主链路不可达且无回退

- **WHEN** 自检失败且未配置回退端点
- **THEN** 网关启动但标记模型不可用，重试次数有上限，日志不出现无界重试

### Requirement: 凭据纪律

真实凭据 SHALL 只存在于工作区外的网关配置与进程环境变量中；仓库内任何文件 MUST NOT 包含真实令牌；主链路使用 Hub 时 SHALL 使用占位 token。

#### Scenario: 仓库无凭据残留

- **WHEN** 对仓库全量搜索令牌形状字符串与 provider 密钥字段
- **THEN** 无命中

#### Scenario: 日志脱敏

- **WHEN** 检查网关日志与会话记录
- **THEN** 不出现完整令牌或可用凭据

### Requirement: 链路变更可运维

模型代理端口或地址变更 MUST NOT 需要修改代码或技能定义；SHALL 存在单一命令完成「改配置 → 校验 → 重启 → 冒烟探测」。

#### Scenario: 换端口只改配置

- **WHEN** 代理端口变更后执行该单一命令
- **THEN** 配置更新、校验通过、服务重启并且冒烟探测成功，全程无需编辑源代码
