# 评估工具

## 确定性回归

`backend/tests/` 使用临时数据库、测试向量、脚本化模型和模拟网络，检查课程范围、资料发布、对话状态、工具预算、来源保存、取消/恢复及配置。这些用例验证应用流程，运行命令见[构建与本地检查](GETTING_STARTED.md#构建与本地检查)。

## 真实模型对照

`backend/evaluation/fixtures/` 和 `cases.json` 提供虚构资料与问题，涵盖单次查询、跨资料比较、分页阅读、追问、资料不足和指令边界。

配置好模型和嵌入后，可在项目根目录运行：

```powershell
backend/.venv/Scripts/python.exe backend/evaluation/run.py --output .cache/evaluation.json
backend/.venv/Scripts/python.exe backend/evaluation/summarize.py .cache/evaluation.json --output .cache/evaluation.md
```

先安装 `backend/requirements-dev.txt`。脚本使用独立临时课程库，会调用所选对话、嵌入及可选重排服务，可能产生费用。默认关闭证据核对和 Agent 联网。用 `--answer-review` 为两种模式开启证据核对，用 `--web-search` 允许 Agent 联网。

报告记录模型、参数、运行结果、来源、用量及耗时。词组覆盖和编号检查不能当作事实正确率，需要人工核对来源支持、完整性及解释质量。报告可能包含模型输出与环境信息，默认不提交，分享前自行审核。

## 真实课程对照

冻结课件只读取指定课程的当前已发布原件，默认把快照、题库和结果放在忽略的 `.cache/os-benchmark/`。快照记录原件哈希、版本、当前分块和嵌入配置。该目录的操作系统题库包含 30 道检索题和 24 个问答场景，可用下列命令在本机校验和复跑。

```powershell
backend/.venv/Scripts/python.exe backend/evaluation/real_course.py freeze
backend/.venv/Scripts/python.exe backend/evaluation/real_course.py validate
backend/.venv/Scripts/python.exe backend/evaluation/real_course.py run --smoke --output .cache/os-benchmark/smoke.json
backend/.venv/Scripts/python.exe backend/evaluation/real_course.py run
```

已有快照时从 `validate` 开始。换课程时指定 `--course`、`--snapshot` 和 `--cases`。题库 JSON 包含 `retrieval` 和 `qa`。检索题用 `R01` 这样的题号、题型、问题、参考要点和证据组。每个证据组列出可互换的 `document`、`section` 和 `anchor`，分页资料可附 `page`。问答题用 `Q01` 这样的题号，可通过 `from_retrieval` 复用检索题，也可直接标注 `turns`。资料不足题单独标记 `gap`。

正式运行分别度量 BM25、纯向量、融合和融合加重排，并对照固定 RAG、Agent 和固定 RAG 加重排。问答默认串行、重复 3 次。`--workers` 可并发不同场景，比较耗时时须记录并发数。真实课程评测关闭 LangSmith 自动追踪。嵌入、重排和回答仍使用项目已配置的模型服务，远程服务会接收相应输入。

```powershell
backend/.venv/Scripts/python.exe backend/evaluation/course_report.py review
backend/.venv/Scripts/python.exe backend/evaluation/course_report.py report --reviews .cache/os-benchmark/blind
backend/.venv/Scripts/python.exe backend/evaluation/course_report.py export --reviews .cache/os-benchmark/blind
backend/.venv/Scripts/python.exe backend/evaluation/course_report.py upload --reviews .cache/os-benchmark/blind
```

在 `blind/review.json` 逐题填分。完整性与引用支撑度各 0–2 分，分别代表无有效内容或证据、部分满足、满足全部主要要点或证据。资料不足题的 `gap_correct` 记 0 或 1，判断是否明确指出相应缺口且没有编造结论。可在评分前用 `report` 查看自动指标，空白分数会保留为空。

`export` 只在本地生成匿名请求。复核导出后再用 `upload` 把结果发到 LangSmith 实验，两步都复用已有运行结果。导出字段限匿名题号、题型、配置、时间、指标和用量。题目、文件名、来源和回答保留在原始报告。

## 本地工具调用检查

```powershell
backend/.venv/Scripts/python.exe backend/evaluation/smoke_ollama.py --model "你的已安装模型标签"
```

替换标签后执行。脚本只连接标准本机 Ollama 地址，使用虚构工具检查一次工具调用和最终回答，不读取个人课程。单个问题通过不代表完整模型能力评估。
