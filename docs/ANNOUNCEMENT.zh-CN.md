# Conway 项目介绍稿

## 可直接发布的介绍

我在做 Conway，一个面向数字生命研究的开源 Agent 项目。

我想探索的是：当 AI 有了长期目标、可保留的记忆，以及持续接触数字环境的能力，
它能怎样行动、积累经验，并逐步变得更有能力？

Conway 的架构很简单：目标写在文件里，模型选择下一步，Harness 负责观察与执行，
结果回到记忆中，再开始下一轮。它通过工具和 GUI 接触环境，持续自主运行，
没有聊天窗口，也不需要用户逐条安排动作。

项目同时推进两条线：Conway Runtime 提供精简、可扩展的运行基础；Conway Policy
围绕开放模型、行动经验和独立评测研究策略能力。MCP 和 Agent Skills 让现有工具
与技能可以接入，模型也可以随着能力提升逐步替换。

持续学习和递归自我改进是长期方向。目前已实现运行时、文件记忆、经验导出与验收
工具；模型能力仍需要研究，还没有发布自有策略权重或实现在线学习。

我希望和关注小模型、电脑操作、Agent 工具生态与人工生命的人一起，把这个方向
做成一个可以运行、可以验证、可以持续推进的开放项目。

GitHub：https://github.com/jovial-liu/conway

Hugging Face：https://huggingface.co/jnjnkj/conway

## 简短介绍

Conway 是面向数字生命研究的开源 Agent 架构。目标驱动一个持续行动的 loop，
模型负责决策，Harness 负责观察、工具执行和记忆。项目同时研究模型与运行时，
接入 MCP 和 Agent Skills，为未来的持续学习与递归改进建立可验证的基础。
当前处于研究原型阶段。

## 一句话定位

Conway：让 AI 持续观察、行动并积累经验，面向数字生命构建开放架构。

English: Conway — an open architecture for persistent digital agents, with model and runtime research toward digital life.
