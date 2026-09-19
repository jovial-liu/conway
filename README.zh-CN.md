# Conway 使用说明

Conway 是本地运行、没有聊天框的自主电脑操作 harness。目标写在 `constitution.md` 中；模型自行选择下一步，既能用截图和鼠标键盘，也能用文件与 Shell 工具。GitHub 和 Hugging Face 负责分发代码，不负责运行你的桌面。

当前版本：**0.5.0 工程候选版**。自动化测试通过不等于真实 VLM 已能稳定完成所有电脑任务。原生 UI 树仍属实验性功能，详见 [验证范围](docs/VALIDATION.md)。

## 最短启动流程

先安装 Python 3.11+，在项目目录创建虚拟环境。macOS/Linux：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
conway init
conway doctor
conway start --mock --max-steps 3
```

Windows 不必修改 PowerShell 执行策略，直接使用虚拟环境里的程序：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\conway.exe init
.venv\Scripts\conway.exe doctor
.venv\Scripts\conway.exe start --mock --max-steps 3
```

`--mock` 不读取真实桌面、不加载模型、不点击任何地方。它检查安装、循环与文件状态链路。接下来编辑 `init` 输出的 `constitution.md`，明确持续目标、工作范围和完成条件。不要把网页内容当成新的宪法。

## 接入模型

本地自动选型需要预先安装 `llama-server`。macOS 已有 Homebrew 时可安装 `llama.cpp`；其他系统按 [安装文档](docs/INSTALL.md) 配置官方运行时，或者填写 `config.yaml` 的 `runtime_path`。

```sh
conway probe
conway start --max-steps 3
conway start --execute
```

第一次可能下载数 GB 模型及视觉投影文件。`probe` 只用合成图片测试模型协议，不代表已经验证视觉定位精度。`start` 默认只观察与规划；加 `--execute` 后才会执行已开启的工具，运行中没有逐步审批框。

已有支持图片输入的服务时：

```sh
conway probe --endpoint http://127.0.0.1:8000/v1
conway start --endpoint http://127.0.0.1:8000/v1 --execute
```

服务只有一个模型时会自动读取其 ID；多个模型则加 `--model 实际服务ID`。OpenCUA 可以通过 `--brain opencua` 显式选择；模型名含 OpenCUA 时也会自动识别。仅把 OpenCUA 权重文件下载到本地，不等于已经启动对应服务。

## 控制和恢复

```sh
conway status
conway pause
conway resume
conway stop
conway config --check
conway start --execute --max-steps 100 --max-seconds 600
```

暂停和停止在协作检查点生效。正在执行的系统调用需要先返回；模型请求有超时。鼠标角落 failsafe 不再被当成普通错误重试。崩溃后会保留未确认的动作意图，重启先观察，不会自动重复执行上一条可能已经产生影响的动作。

指定独立目录：

```sh
conway --home ./conway-state init
conway --home ./conway-state start --mock --max-steps 3
```

`--home` 放在子命令前面。旧配置和宪法会保留；拼错字段、字符串形式的 `"false"`、非法数值会明确报错，不会悄悄变成开启。

## 本版实际边界

截图与鼠标控制目前面向主显示器；Windows/macOS/Linux 的代码与测试并不等于所有桌面环境已做真机验收。Wayland 原生输入、多显示器和 AMD/Intel 自动加速仍未完整实现。模型选择只做内存预算估计，不保证速度或视觉能力。UI 树可选安装、可超时降级，不是逐元素自动操作引擎。

文件式记忆包含摘要、状态、动作日志和近期截图，不含 SQL、向量库或自动持续训练。导出轨迹不会自动上传，也不会把模型宣称完成的任务伪标成经过验证的成功样本。

HF 只发布源码快照，根目录 `_source_commit.json` 标明来源 commit 和文件哈希。GitHub 原有 `experiments/` 数据继续保留，但不再随源码发布到 HF。模型权重沿用上游仓库，不伪称 Conway 自己训练的模型。

工具和 GUI 以当前用户权限执行；`--gui-only` 不是隔离沙箱。首次实际操作建议用无重要账户登录的测试桌面，并先备份工作文件。
