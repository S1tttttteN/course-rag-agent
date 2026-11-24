# 架构与模块

## 运行形态

一个源码仓库、独立 React 前端和单进程 FastAPI 后端。Agent、检索、索引任务与业务服务均在后端中协作。

```mermaid
flowchart TD
    UI[React 页面] -->|REST / SSE| API[FastAPI]
    API --> S[课程 / 资料 / 对话 / 运行服务]
    S --> A[LangChain Agent]
    A --> T[绑定课程范围的工具]
    T --> K[资料检索与阅读]
    T --> W[公开搜索与网页摘录]
    K --> V[Chroma / BM25 / RRF / 可选重排]
    S --> DB[SQLite]
    S --> F[本地上传文件]
    A --> M[配置的模型服务]
```

## 从上传资料到多轮问答

```mermaid
flowchart TD
    subgraph INDEX[资料入库]
        Upload[上传资料，创建待处理版本] --> BuildIndex[后台解析、切片并写入向量]
        BuildIndex --> Ready{索引是否完整？}
        Ready -->|成功| Publish[保存片段并发布可检索版本]
        Ready -->|失败| Keep[记录失败，已有版本继续可查]
    end

    subgraph QA[问答与工具循环]
        Ask[提交问题和本轮联网许可] --> Run[创建 Run 和消息，启动后台任务]
        Run --> History[加载已完成历史，按需摘要]
        History --> Agent[create_agent 装配模型、工具和中间件]
        Agent --> Decide{模型下一步做什么？}
        Decide -->|继续查阅| Budget{工具预算是否充足？}
        Budget -->|充足| Tools{调用哪个工具？}
        Tools --> Search[课程搜索：BM25、向量召回、RRF 和可选重排]
        Tools --> Catalog[列出课程资料]
        Tools --> Read[按资料版本分页阅读]
        Tools --> Web[搜索或阅读公开网页，仅本轮允许联网时提供]
        Publish -.->|提供可用片段| Search
        Publish -.->|提供可用版本| Read
        Search --> Sources{新来源是否在引用预算内？}
        Read --> Sources
        Web --> Sources
        Sources -->|是| Snapshot[保存引用快照，重复来源复用编号]
        Snapshot --> Result[将工具结果交回模型]
        Catalog --> Result
        Result --> Decide
        Sources -->|否| Fallback[用已登记的来源做无工具总结]
        Budget -->|耗尽| Fallback
        Decide -->|开始回答| Answer[流式生成回答]
        Fallback --> Answer
    end

    subgraph OUTPUT[流式显示、结束与续聊]
        Events[保存工具进度、引用、正文增量和结束事件]
        SSE[浏览器通过 SSE 更新，断线后按事件 ID 续接]
        Complete[提交成功终态及可用检查点 ID]
        Stop[提交取消或失败终态，保留已发布内容]
        Next[用户继续追问]

        Tools -.->|查阅进度| Events
        Snapshot -.->|引用更新| Events
        Answer -.->|正文增量| Events
        Events --> SSE
        Answer -->|成功| Complete
        Run -.->|取消或执行失败| Stop
        Complete -.->|结束事件| Events
        Stop -.->|结束事件| Events
        Complete --> Next
        Stop --> Next
        Next --> Ask
    end
```

默认 Top K 为 4，限制单次课程搜索返回的片段数。一次问答可多次搜索和阅读，引用面板展示本轮累计登记的来源。

下一轮从最近一次成功提交的检查点继续，并从业务库补入此后已完成的问答。缺少可用检查点时，从业务库加载最近的已完成问答。取消或失败的回答可在页面回看，后续模型上下文只使用已完成的问答。参见[对话、引用与流式运行](CONVERSATIONS.md)。

## 职责

| 目录 | 职责 |
| --- | --- |
| `frontend/src/features/` | 课程、资料和学习对话页面 |
| `frontend/src/api/` | HTTP 和 SSE 客户端 |
| `backend/app/api/` | 路由、输入验证与 HTTP 错误 |
| `backend/app/services/` | 业务事务、状态变更与运行事件 |
| `backend/app/repositories/` | 数据访问与历史查询 |
| `backend/app/qa/` | Agent、工具中间件、上下文、运行调度和用量记录 |
| `backend/app/retrieval/` | 解析、索引、混合召回、重排和网页工具 |
| `backend/app/jobs/` | 自动建库、清理与恢复 |
| `backend/app/db/` | SQLAlchemy 数据模型、连接 |
| `backend/app/core/` | 配置、模型工厂与并发锁 |
| `backend/migrations/` | Alembic 数据库基线和后续迁移 |
| `backend/evaluation/` | 离线固定 RAG 基线和对照评测 |

## 数据与一致性

业务表的核心字段、外键和应用引用见[数据库关系图](DATABASE.md)。

SQLite 保存课程、资料版本、原文片段、对话、运行事件和引用。Chroma 保存向量，课程与嵌入配置共同决定集合。资料当前可用版本由 SQLite 决定，未发布向量不参与检索。

替换资料时先处理新版本，完整成功后切换可用版本；失败则继续使用当前版本。删除业务记录时同时保存清理任务，后台回收原件和向量。

引用独立保存回答当时的内容，删除或替换原资料不会改写既有回答依据。只有已完成问答进入后续上下文；中止和失败内容保留展示。

LangGraph 用独立 SQLite 文件保存 Agent 图状态。业务库只在回答完成时提交可继续使用的检查点 ID。下一轮以该检查点为起点，补入之后完成的问答。可续接的检查点保存摘要与已发布问答。

## 运行控制

后端绑定课程范围，模型不能传入任意存储路径或选择其他课程。每段对话最多一个活动运行，工具次数、文本量与时间受限。SSE 读取持久化事件，断线重连不会重复调用模型。重启将未完成任务标为中断。

课程隔离属于业务范围控制，应用目前没有用户身份和多用户授权。使用限制见 [SECURITY.md](../SECURITY.md)。
