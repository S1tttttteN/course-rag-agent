# 课程研习室

围绕课程资料学习的全栈 Agent 应用。创建课程、上传资料，在独立对话中提问，并通过引用核对课程原文或网络来源。

支持 Ollama 和云端模型服务，模型由使用者自行配置。

## 功能

- 课程与资料管理：创建、重命名、删除课程，上传及替换资料，查看处理进度和解析原文。
- 混合检索：语义检索、BM25、RRF 融合和可选重排。
- 两种问答方式：固定 RAG 先检索再回答；LangChain Agent 自主搜索课程、列目录、分页阅读，并按需联网。
- 持续对话：Markdown 回答、流式输出、中途停止、刷新恢复，以及独立保存的引用快照。
- 可选证据核对：对草稿中的结论与引用进行生成后核对，默认关闭。

支持 TXT、Markdown、HTML、DOCX 和可提取文字的 PDF，单份最多 20 MB；暂不支持扫描 PDF OCR 和 PPT。

## 快速开始

准备 Python 3.13 和 Node.js 24。使用本地模型时另行安装并运行 [Ollama](https://ollama.com/download)。以下为项目根目录下的 Windows PowerShell 命令：

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.lock
cd frontend
npm.cmd ci --cache ../.cache/npm
cd ..
backend/.venv/Scripts/python.exe scripts/env_config.py --sync
```

编辑 `.env`，分别选择对话与嵌入服务商、模型和所需密钥。Ollama 的模型标签必须与自己的 `ollama list` 一致。模板中的服务商、模型和密钥留空，不会自动选择模型。

终端一：

```powershell
backend/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

终端二：

```powershell
cd frontend
npm.cmd run dev
```

打开 http://127.0.0.1:5173；API 文档位于 http://127.0.0.1:8000/docs。只运行一个后端进程。macOS/Linux 命令见[安装说明](docs/GETTING_STARTED.md)。

首次体验可创建课程，上传 `backend/evaluation/fixtures/进程笔记.md`，等待就绪，再提问并点击引用检查原文。

## 技术栈与目录

React、TypeScript、Vite、FastAPI、SQLAlchemy、Alembic、SQLite、Chroma、LangChain。

```text
frontend/             页面、接口客户端和样式
backend/app/          API、业务服务、Agent、检索与任务处理
backend/migrations/   数据库迁移
backend/tests/        确定性回归用例
backend/evaluation/   虚构示例、评估工具和本地模型检查
docs/                 安装、架构与使用说明
scripts/              配置与仓库内容检查工具
```

## 隐私、费用与限制

原文件、索引、对话和引用默认保存到本机 `data/`。选择云端对话、嵌入或重排时，相应文本会发送给所选服务，可能产生费用。开启联网后，搜索服务会收到关键词，目标网站会收到读取请求；关键词可能包含问题或资料信息。

回答可能不准确，来源编号存在也不等于结论必然得到原文支持。Chroma 依赖仍有已知安全公告；本应用仅使用嵌入式本机索引，不启动 Chroma HTTP 服务。应用不适合直接暴露到公网。参见[安全与隐私](SECURITY.md)和[依赖风险](docs/DEPENDENCIES.md)。

## 文档

- [安装与排错](docs/GETTING_STARTED.md)
- [环境变量与模型配置](docs/CONFIGURATION.md)
- [使用演示](docs/DEMO.md)
- [架构与模块](docs/ARCHITECTURE.md)
- [并排研读设计系统](docs/DESIGN_SYSTEM.md)
- [课程资料库](docs/LIBRARY.md)
- [对话、引用与流式运行](docs/CONVERSATIONS.md)
- [Agent 工具与预算](docs/AGENT.md)
- [可选证据核对](docs/EVIDENCE_CHECK.md)
- [评估工具](docs/EVALUATION.md)
- [领域术语](GLOSSARY.md)
- [开发记录](CHANGELOG.md)

项目代码采用 [MIT](LICENSE)；第三方依赖、模型和资料的授权范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
