"""config — kv_visualizer 后端全局配置(路径 / 常量 / 源清单).

路径按本文件位置解析:
    backend_fastapi/          <- HERE
        ../frontend_vue       <- PAGE_DIR   (Vue 工程, 页面产物 dist/)
        ../kv_datasets_root   <- DATA_ROOT  (后端读取的导出数据)
        ../../kvcache_experiment  <- ROOT   (四个实验源目录所在)

实验源目录名与场景 key 的映射见 SRC(历史版本曾叫 kv_experiment / kvc, 已改名)。
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent            # backend_fastapi/
_STUDY = HERE.parent.parent                       # kvcache_study/
ROOT = _STUDY / "kvcache_experiment"              # 实验源根(只读)
PAGE_DIR = HERE.parent / "frontend_vue"           # 前端工程(构建产物 dist/)
DATA_ROOT = HERE.parent / "kv_datasets_root"      # DATA_ROOT: 导出数据, 后端读取这里

HEAT_N = 32          # 热图边长: 块张量头0通道 128×128 做 4×4 平均池化 -> 32×32
POOLS = ("K", "V")   # KV 两池

# ---- msprobe tensor dump 数据源（物理页视图 / kvms 场景）----
# 扫描该目录下形如 *_dump_tensor/step{N}/rank0/dump_tensor_data 的数据源
KVMS_ROOT = ROOT / "qwen35" / "scripts" / "msprobe" / "tensor"
PHYS_BLOCK = 128     # attn cache 的物理块宽（token），对应 FIA 的 block_size
# 混合模型一张共享张量内按页切成的三段池（字节）；实现里优先从数据形状反推
POOL_PAGE_BYTES = {
    "conv": 3 * 6144 * 2,        # conv_state: 3 × 6144 bf16 = 36 KiB
    "k_ssm": 16 * 128 * 128 * 4,  # SSM 段: 16 × 128 × 128 fp32 = 1 MiB（K 段与其共享）
    "v": 1 << 20,
}

# 场景 key -> 实验源子目录名(2026-10 目录改名: kvc->kvcache_mix 等)
SRC = {
    "kvc": "kvcache_mix",
    "p1d": "kvcache_1p1d",
    "matrix": "kvcache_1p1d_prefix",
    "qwen35": "qwen35",
}

# 实验源节点清单(新鲜度 mtime 签名遍历范围), 相对 ROOT; 不含 qwen35 的大 .pt(单独入口)
SOURCE_GLOBS = [
    "kvcache_mix/tensors/req*/kv_*.pt",
    "kvcache_mix/logs/patchs/kvc*.log",
    "kvcache_1p1d/tensors/req*/[PD]/kv_*.pt",
    "kvcache_1p1d_prefix/logs/q*/*",
    "qwen35/tensors/*/req*/kv_*.pt",
    "qwen35/logs/analysis/*.out",
    "qwen35/qwen35_step*.log",
]