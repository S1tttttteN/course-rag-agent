# 参与开发

请先阅读[安装说明](docs/GETTING_STARTED.md)和[架构](docs/ARCHITECTURE.md)。开发使用 Python 3.13、Node.js 24，以及自己选择的模型服务。

提交 Issue 时提供复现步骤、预期与实际结果，移除密钥、真实课程原文、对话记录和个人信息。安全问题按 [SECURITY.md](SECURITY.md) 报告。

提交代码前可执行：

```powershell
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements-dev.txt
backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests -v
backend/.venv/Scripts/python.exe scripts/env_config.py --check-template
backend/.venv/Scripts/python.exe scripts/check_repository.py --history --local-env
cd frontend
npm.cmd ci
npm.cmd run build
```

回归使用临时存储、脚本化模型和模拟网络，不读取个人课程，不代表真实模型质量评估。GitHub Actions 在 Windows/Linux 执行后端检查，并构建前端。

新增环境变量时更新 `.env.example` 与配置文档，服务商、模型选择和密钥留空。本机格式同步使用 `scripts/env_config.py --sync`，修改前备份并保留仍受支持的配置值。

请勿提交 `.env`、`data/`、`.cache/`、生成的报告、下载模型或未授权资料。贡献应为原创或具有兼容授权，代码采用 [MIT](LICENSE)，第三方内容见 [授权说明](THIRD_PARTY_NOTICES.md)。
