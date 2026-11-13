# 安装与排错

## 环境

使用 Python 3.13 和 Node.js 24。选择本地模型时安装并启动 Ollama；首次安装依赖和模型需要联网。本地模型速度取决于内存、显存和计算能力。

## Windows PowerShell

从项目根目录执行：

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.lock
cd frontend
npm.cmd ci --cache ../.cache/npm
cd ..
backend/.venv/Scripts/python.exe scripts/env_config.py --sync
```

根据[配置说明](CONFIGURATION.md)填写 `.env`。Ollama 模型标签以自己的 `ollama list` 为准，云端服务需对应的 Key。对话与嵌入分别配置；课程管理不依赖模型。

两个终端分别运行：

```powershell
backend/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm.cmd run dev
```

前端 http://127.0.0.1:5173，接口文档 http://127.0.0.1:8000/docs。无需激活虚拟环境；`npm.cmd` 避免 PowerShell 脚本执行策略冲突。

## macOS / Linux

```bash
python3.13 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.lock
(cd frontend && npm ci)
backend/.venv/bin/python scripts/env_config.py --sync
```

配置相同。后端使用 `backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`；另一个终端在 `frontend/` 执行 `npm run dev`。

## 存储与运行

首次启动自动执行 Alembic 迁移。默认数据在 `data/`，更换 `COURSE_AGENT_DATA_DIR` 不会自动搬迁已有数据。关闭服务后可备份整个数据目录；配置文件另行私密保存。

只运行一个后端进程，不增加 Uvicorn worker。资料处理串行；Ollama 同时回答一轮，其他对话服务最多两轮。活动任务上限六个，每段对话最多一个。

## 排错

- 提示未配置：检查 `.env` 的服务商、模型和所需 Key，修改后重启后端。系统环境变量优先于文件。
- 上传后失败：检查嵌入服务的配置和运行状态。若提示 HTTP 402，请检查嵌入服务账户的余额或可用额度，处理后重试失败资料。
- Ollama 连接失败：确认服务运行、地址正确、完整模型标签已安装。
- 切换嵌入后检索降级：用原文件替换资料生成新索引，避免混用向量空间。
- 搜索失败：检查网络，必要时关闭联网。一般知识解释不代表已核实网页。

## 构建与本地检查

在 `frontend/` 执行 `npm.cmd run build`（macOS/Linux 使用 `npm run build`）。

在项目根目录运行后端回归和配置检查：

```powershell
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements-dev.txt
backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests -v
backend/.venv/Scripts/python.exe scripts/env_config.py --check-template
backend/.venv/Scripts/python.exe scripts/check_repository.py --history --local-env
```

macOS/Linux 将 `backend/.venv/Scripts/python.exe` 换为 `backend/.venv/bin/python`。模型评估方法见[评估工具](EVALUATION.md)。
