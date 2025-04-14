# OpenAI Agent 课程示例

本仓库包含了一系列使用 OpenAI Agent SDK 的示例代码，展示了不同的功能和最佳实践。这些示例分为三个主要部分：基础示例、Agent 模式示例和自定义 LLM 提供方示例。

## 目录结构

- `basic/`: 基础示例，展示 Agent 的核心功能
- `agent_patterns/`: Agent 模式示例，展示各种高级用法和最佳实践
- `model_providers/`: 自定义 LLM 提供方示例，展示如何集成不同的 LLM 服务

## 环境要求

所有示例都需要以下环境变量（在 `.env` 文件中配置）：
- `API_KEY`: OpenAI API 密钥
- `API_BASE`: API 基础URL（默认为 "https://api.deepseek.com"）
- `MODEL_NAME`: 使用的模型名称（默认为 "deepseek-chat"）

## 基础示例 (basic/)

基础示例展示了 Agent 的核心功能，包括：
- 流式文本输出
- 事件处理
- Agent 生命周期监控
- 使用自定义上下文来传递额外信息
- Agent 钩子函数

详细说明请参考 [basic/README.md](basic/README.md)

## Agent 模式示例 (agent_patterns/)

Agent 模式示例展示了各种高级用法和最佳实践，包括：
- 确定性流程
- 代理切换与路由
- 代理作为工具
- LLM 作为评判者
- 并行处理
- 输入/输出护栏机制
- 强制工具使用

详细说明请参考 [agent_patterns/README.md](agent_patterns/README.md)

## 自定义 LLM 提供方示例 (model_providers/)

展示了三种不同的自定义 LLM 提供方的方式：
1. Agent 级别自定义
2. 全局级别自定义
3. Runner 级别自定义

每种方式都有其特定的使用场景和优势，详细说明请参考 [model_providers/README.md](model_providers/README.md)

## 运行说明

1. 克隆仓库并安装依赖：
   ```bash
   git clone [repository_url]
   cd openai_agent_course
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   - 复制 `.env.example` 为 `.env`
   - 填写必要的环境变量

3. 运行示例：
   ```bash
   # 基础示例
   python basic/stream_text.py
   
   # Agent 模式示例
   python agent_patterns/deterministic.py
   
   # 自定义 LLM 提供方示例
   python model_providers/customer_llm_agent.py
   ```

## 注意事项

- 所有示例都使用了异步编程（asyncio）
- Windows 平台需要特殊处理事件循环策略
- 示例代码中包含了详细的注释，方便理解每个部分的功能
- 可以根据需要修改环境变量和模型参数
