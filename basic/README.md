# OpenAI Agent 基础示例

本目录包含了一系列使用 OpenAI Agent 的基础示例代码，展示了不同的功能和用法。

## 环境要求

所有示例都需要以下环境变量：
- `API_KEY`: OpenAI API 密钥
- `API_BASE`: API 基础URL（默认为 "https://api.deepseek.com"）
- `MODEL_NAME`: 使用的模型名称（默认为 "deepseek-chat"）

## 示例说明

### 1. stream_text.py
- 演示了基本的流式文本输出功能
- 创建一个简单的 Agent，用于讲笑话
- 使用 `run_streamed` 方法实现流式输出
- 展示了如何处理原始增量文本事件

### 2. stream_items.py
- 展示了更复杂的流式事件处理
- 实现了自定义工具函数 `how_many_jokes`
- 演示了如何处理不同类型的事件：
  - 原始响应事件
  - Agent 状态更新事件
  - 运行项事件（工具调用、工具输出、消息输出）

### 3. runner_hook.py
- 展示了如何使用 `RunHooks` 来监控和记录 Agent 的执行过程
- 实现了两个 Agent 之间的协作：
  - Start Agent：生成随机数
  - Multiply Agent：处理奇数（乘以2）
- 记录了 Token 使用情况和各种事件

### 4. dynamic_system_prompt.py
- 展示了如何使用自定义上下文来传递额外信息
- 通过自定义上下文类实现不同风格的回复：
  - 俳句（haiku）
  - 海盗（pirate）
  - 机器人（robot）
- 演示了如何根据上下文动态调整 Agent 的行为

### 5. agent_hook.py
- 展示了如何使用 `AgentHooks` 来监控单个 Agent 的生命周期
- 实现了与 runner_hook.py 类似的功能，但关注点更细粒度
- 演示了 Agent 之间的协作和工具调用
