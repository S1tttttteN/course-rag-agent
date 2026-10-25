from threading import RLock

# Serialize short publication, deletion and upload transactions; never hold during API calls.
catalog_lock = RLock()
