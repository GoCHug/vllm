"""lib — kv_visualizer 后端业务包.

模块职责:
    tensor_reader     .pt 归档读取与基础张量/统计(exporter/verify 共用)
    utils             路径校验 / 目录签名 / JSON 落盘
    dataset_loader    kv_datasets_root 数据集层(LOCK / ensure_fresh / api_verify)
    exporter          五场景导出(kvc / p1d / matrix / kvcFlow / qwen35)
    verify            独立核验器(A~D 60 项断言)
    server_stdlib     纯标准库 HTTP 回退装配(fastapi 缺失时)
    pack              standalone 单文件打包器
"""