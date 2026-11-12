"""提供协调资料目录和问答状态读写的共享锁。"""

from threading import RLock

# 串行化索引发布、删除和上传等短事务，模型请求在锁外执行。
catalog_lock = RLock()
