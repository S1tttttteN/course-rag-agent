# 配置说明

## 两个文件的职责

- `.env.example`：公共字段模板，服务商、模型选择和 API Key 留空。仅保留通用地址与产品行为默认值，按自己的运行环境填写。
- `.env`：使用者的实际配置，不提交 Git。它与模板使用相同字段、顺序、分区和注释，具体取值可以不同。

没有选择对话/嵌入服务时，后端和课程管理仍可启动，相关功能会提示尚未配置。模型服务需要显式配置。

## 创建和同步

先安装后端依赖，然后从项目根目录执行：

```powershell
backend/.venv/Scripts/python.exe scripts/env_config.py --sync
backend/.venv/Scripts/python.exe scripts/env_config.py --check
```

首次执行会用模板创建 `.env`；已有文件时保留全部已存在的配置值，只补字段和统一格式，修改前备份到忽略的 `.cache/env-backups/`。未知字段或重复字段会停止同步，避免误删自定义配置。输出只包含字段名、数量和状态，不包含值。

也可首次手动复制 `.env.example` 为 `.env`，但不要覆盖已有文件。编辑后重启后端。系统环境变量优先于 `.env`，所以修改文件后若仍与页面显示不同，应检查启动终端是否设置了同名变量。

## 选择模型

### 对话

`CHAT_PROVIDER` 可选 `deepseek / qwen / zhipu / moonshot / openai / siliconflow / ollama`。

- 选择 Ollama：先安装并运行服务，使用 `ollama list` 查看模型。将自己安装的完整标签填入 `CHAT_MODEL`；代码不默认指定某个本地模型。没有模型时先下载所选模型。
- 选择云服务：填写对应 Key 和所需 `CHAT_MODEL`。模型名留空时使用 `backend/app/core/models.py` 注册表中的该服务商默认模型。可用模型和价格以服务商为准。

Ollama 对话通过 `OLLAMA_BASE_URL` 连接，默认本机通用地址为 `http://127.0.0.1:11434`，不带 `/v1`。

### 嵌入

`EMBEDDING_PROVIDER` 可选 `siliconflow / zhipu / qwen / openai / ollama`，与对话服务独立选择。

- Ollama：`EMBEDDING_MODEL` 必填自己安装的嵌入模型标签。`OLLAMA_EMBEDDING_BASE_URL` 默认 `http://127.0.0.1:11434/v1`，使用 OpenAI 兼容接口。
- 云服务：填写对应 Key；`EMBEDDING_MODEL` 可填写所需模型，或留空使用注册表默认模型。资料片段及语义查询将发送给该服务。

更换嵌入模型需要重新处理资料、重建索引。同步格式不会替你更换模型、重建索引或迁移数据。

### 重排

`RERANK_PROVIDER=none` 默认关闭。也可选择 `siliconflow / jina / cohere`，配置相应 Key，`RERANK_MODEL` 留空则使用注册表默认模型。缺少 Key 时跳过重排，不阻止基本检索。

## 其他字段

| 字段 | 模板取值/留空含义 |
| --- | --- |
| `ANSWER_REVIEW` | `false` 自然讲解；`true` 生成后证据核对 |
| `WEB_SEARCH_ENABLED` | `true` 允许 Agent 按需联网，`false` 关闭；搜索词可能包含问题或资料信息 |
| `RUN_TIMEOUT` | 留空按对话服务计算：Ollama 600 秒，其余 180 秒 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 100；修改后已有资料需重新处理才应用新分块 |
| `TOP_K` | 默认 4，页面可调整，Agent 上限 4 |
| `COURSE_AGENT_DATA_DIR` | 留空使用项目 `data/`，可填写绝对目录；改变路径不会自动搬迁旧数据 |

API Key 仅填写所选服务商的字段，其余留空；Ollama 不需要 API Key。不要把 Key 写入前端、文档、Issue 或截图。

有关数据流向和风险，参见 [安全说明](../SECURITY.md)。
