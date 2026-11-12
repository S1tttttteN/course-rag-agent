# 已知依赖风险

项目的依赖版本见 `backend/requirements.lock` 与 `frontend/package-lock.json`。漏洞信息会变化，应在安装和部署前复查。

## Chroma

锁定的 `chromadb==1.5.9` 仍涉及以下公告；核对时没有可用修复版本，不能称为零漏洞：

| 公告 | 涉及入口 |
| --- | --- |
| [CVE-2026-45829](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) | HTTP 集合创建时处理不受信任的模型配置 |
| [CVE-2026-45833](https://github.com/advisories/GHSA-36p7-vc44-83pf) | HTTP 集合更新的模型配置 |
| [CVE-2026-45831](https://github.com/advisories/GHSA-xph7-9rjv-w5fr) | SimpleRBAC 资源范围检查 |
| [CVE-2026-45830](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr) | Python HTTP 服务端跨租户访问 |

应用固定使用本机 `RustBindingsAPI`，显式禁用集合内置嵌入函数，自行计算向量，不启动 Chroma HTTP 服务，不接受任意模型仓库配置。这是对当前调用路径的范围约束，不是对整个依赖的安全保证。

不要额外启动 `chroma run`、连接不受信任的远端 Chroma 或导入不受信任的数据目录。公网或多人服务需要先解决相关问题、采用已修复版本或更换存储，并重新验收。

复查可使用 `npm audit` 和 PyPA `pip-audit`；不要通过静默忽略公告把运行流程检查表述为安全审计通过。
