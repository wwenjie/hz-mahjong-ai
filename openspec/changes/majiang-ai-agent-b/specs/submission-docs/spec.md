# Spec Delta

## Purpose

产出 10/8 截止要提交的《使用说明》并做干净环境验收，证明运行时代码零第三方依赖、评审方能按说明独立完成接入与启动。

## ADDED Requirements

### Requirement: 干净环境零依赖验收

验收脚本 SHALL 在全新 venv 中仅安装项目自身（无 numpy/sklearn 等离线训练依赖），验证运行时代码可导入、CLI 可用、模型求值（手写树遍历）正常工作。

#### Scenario: 干净 venv 启动
- **WHEN** 在新 venv 中 `pip install -e .` 后运行 CLI 帮助与一次本地模拟对局
- **THEN** 无 ImportError，模拟对局完整跑完 8 局

#### Scenario: 模型求值无科学计算依赖
- **WHEN** 在干净环境加载两个离线训练的小模型并求值若干局面
- **THEN** 求值结果与开发环境一致，全程未导入 numpy/sklearn

### Requirement: 提交版使用说明

《使用说明》SHALL 面向评审（区别于面向开发者的 README），包含：接入方式、启动步骤、依赖环境、令牌配置方式。令牌 MUST 以环境变量为优先配置方式，MUST NOT 出现在命令行参数、进程列表、日志或任何提交文件中。

#### Scenario: 按说明从零启动
- **WHEN** 评审按《使用说明》在干净机器上操作
- **THEN** 无需阅读源码即可完成配置与启动

#### Scenario: 无凭据泄漏
- **WHEN** 检查提交物（源码、文档、脚本、shell 历史示例）
- **THEN** 不存在任何明文令牌，文档中的示例一律使用环境变量占位符
