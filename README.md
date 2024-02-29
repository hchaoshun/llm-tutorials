# 目录
## LLM算法
* LLM架构
  * [bthread基础](docs/bthread_basis.md)
* LLM数据集
  * [无异常状态下的一次完整RPC请求过程](docs/client_rpc_normal.md)
  * [RPC请求可能遇到的多种异常及应对策略](docs/client_rpc_exception.md)
  * 重试&Backup Request
  * [同一RPC过程中各个bthread间的互斥](docs/client_bthread_sync.md)
* LLM训练
  * 处理一次RPC请求的完整过程
  * 服务器自动限流
  * 防雪崩
* LLM评估
  * protobuf编程模式
  * [多线程向同一TCP连接写入数据](docs/io_write.md)
  * [从TCP连接读取数据的并发处理](docs/io_read.md)


## LLM工程
* LLM运行
  * [bthread基础](docs/bthread_basis.md)
* LLM Prompt 工程
  * [无异常状态下的一次完整RPC请求过程](docs/client_rpc_normal.md)
* LLM RAG
  * [ResourcePool：多线程下高效的内存分配与回收](docs/resource_pool.md)
* LLM推理优化
  * [无异常状态下的一次完整RPC请求过程](docs/client_rpc_normal.md)
* LLM部署
  * 处理一次RPC请求的完整过程