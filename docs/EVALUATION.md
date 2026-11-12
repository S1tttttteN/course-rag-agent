# 评估工具

## 确定性回归

`backend/tests/` 使用临时数据库、测试向量、脚本化模型和模拟网络，检查课程范围、资料发布、对话状态、工具预算、来源保存、取消/恢复及配置。命令见[贡献指南](../CONTRIBUTING.md)。这些用例验证应用流程，不是模型回答正确率。

## 真实模型对照

`backend/evaluation/fixtures/` 和 `cases.json` 提供虚构资料与问题，涵盖单次查询、跨资料比较、分页阅读、追问、资料不足和指令边界。

配置好模型和嵌入后，可在项目根目录运行：

```powershell
backend/.venv/Scripts/python.exe backend/evaluation/run.py --output .cache/evaluation.json
backend/.venv/Scripts/python.exe backend/evaluation/summarize.py .cache/evaluation.json --output .cache/evaluation.md
```

先安装 `backend/requirements-dev.txt`。脚本使用独立临时课程库，会调用所选对话、嵌入及可选重排服务，可能产生费用。默认关闭证据核对和 Agent 联网。用 `--answer-review` 为两种模式开启证据核对，用 `--web-search` 允许 Agent 联网。

报告记录模型、参数、运行结果、来源、用量及耗时。词组覆盖和编号检查不能当作事实正确率，需要人工核对来源支持、完整性及解释质量。报告可能包含模型输出与环境信息，默认不提交，分享前自行审核。

## 本地工具调用检查

```powershell
backend/.venv/Scripts/python.exe backend/evaluation/smoke_ollama.py --model "你的已安装模型标签"
```

替换标签后执行。脚本只连接标准本机 Ollama 地址，使用虚构工具检查一次工具调用和最终回答，不读取个人课程。单个问题通过不代表完整模型能力评估。
