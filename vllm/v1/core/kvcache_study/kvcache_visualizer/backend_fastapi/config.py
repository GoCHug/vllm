"""config — kv_visualizer 后端全局配置(路径 / 常量 / 源清单).

路径按本文件位置解析: backend_fastapi/ 上两级为 kvcache_docs_v2/.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent          # backend_fastapi/
ROOT = HERE.parent.parent / "kv_experiment"    # 四个实验源目录(kvc/kvc_1p1d/kvc_1p1d_prefix/qwen35)所在
PAGE_DIR = HERE.parent / "frontend_vue"         # 前端工程(页面构建产物 dist/)
DATA_ROOT = HERE.parent / "kv_datasets_root"   # DATA_ROOT: vLLM 导出 KV 数据, 后端读取这里

HEAT_N = 32          # 热图边长: 块张量头0通道 128×128 做 4×4 平均池化 -> 32×32
POOLS = ("K", "V")    # KV 两池

SOURCE_GLOBS = [      # 实验源节点清单(新鲜度 mtime 签名遍历范围)
    "kvc/tensors/req*/kv_*.pt",
    "kvc/logs/patchs/kvc*.log",
    "kvc_1p1d/tensors/req*/[PD]/kv_*.pt",
    "kvc_1p1d_prefix/logs/q*/*",
    "qwen35/tensors/*/req*/kv_*.pt",
    "qwen35/logs/analysis/*.out",
]
