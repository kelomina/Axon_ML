# 神枢 (Axon V2) - 恶意软件检测与家族分类系统

神枢 (Axon V2) 是一个基于机器学习的高性能恶意软件检测与家族分类系统。它结合了 LightGBM 梯度提升决策树、路由门控专家系统以及基于 HDBSCAN 的家族聚类分析，旨在提供快速、准确且可扩展的恶意样本识别能力。

## 核心特性

- **双层检测机制**：
    - **二分类检测**：使用 LightGBM 判断文件是否为恶意样本。
    - **路由门控与专家系统 (Gating & Expert Models)**：动态路由机制自动将样本分发至 **Normal Expert** (处理常规样本) 或 **Packed Expert** (处理加壳/高熵样本)，显著提升对复杂加壳样本的检测精度。
- **家族识别与归属**：
    - 基于 **fast-hdbscan** 的无监督聚类分析，自动发现恶意软件家族。
    - 训练家族分类器，支持对新样本进行家族归属预测。
- **混合动力引擎**：
    - **C++ 扫描内核**：底层 `kvd_core` 提供高性能的文件解析与特征提取能力。
    - **Python 智能调度**：上层 Python 框架负责模型训练、评估及业务逻辑调度。
- **多维评估与可视化**：
    - 提供混淆矩阵、ROC-AUC 曲线、PCA 聚类图等多种评估报表。
    - 支持 AutoML (Optuna) 超参数自动调优。
- **灵活的部署方式**：
    - 支持单机命令行批量扫描。
    - 提供基于 IPC 的扫描服务接口，便于系统集成。

## 项目结构

```text
/ (项目根目录)
├── src/                      # 源代码根目录
│   ├── cpp/                  # C++ 核心扫描引擎 (kvd_core)
│   └── python/               # Python 检测框架 (kvd_detector)
├── resources/                # 资源管理 (统一存放权重、聚类结果与报表)
│   └── weights_cluster_eval/
│       ├── weights/          # 模型权重 (*.txt, *.pkl)
│       ├── cluster/          # 聚类结果与特征缓存
│       └── eval/             # 评估报告与扫描结果
├── data/                     # 预处理中间数据 (processed_lightgbm)
├── benign_samples/           # 良性样本存放区 (不入库)
├── malicious_samples/        # 恶意样本存放区 (不入库)
├── PROJECT_STRUCTURE.md      # 详细的目录规范说明
└── README.md                 # 项目主文档
```

## 环境要求

- **操作系统**：Windows 10/11 (推荐) 或 Linux (x86_64)
- **Python**：3.10 及以上
- **C++ 编译**：CMake 3.24+，支持 C++17 的编译器 (如 MSVC, GCC, Clang)

## 安装指南

1. **克隆仓库**
2. **配置 Python 环境**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r src/python/requirements.txt
   ```
3. **编译 C++ 内核 (可选)**
   ```powershell
   cmake -S src/cpp -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```

## 快速使用

所有核心操作均通过 `src/python/kvd_detector/main.py` 入口脚本完成：

### 1. 数据准备与特征提取
从 `benign_samples/` 和 `malicious_samples/` 目录提取特征并生成处理后的数据：
```bash
python src/python/kvd_detector/main.py extract --output-dir data/processed_lightgbm --label-inference directory
```

### 2. 模型训练 (全流程)
一键执行特征提取、LightGBM 训练、路由系统训练、家族聚类及评估：
```bash
python src/python/kvd_detector/main.py train-all
```

### 3. 独立模块训练
- **AutoML 调优**：`python main.py auto-tune --trials 50`
- **训练路由系统**：`python main.py train-routing`
- **家族聚类训练**：`python main.py finetune --plot-pca`

### 4. 样本扫描
- **扫描单文件**：`python src/python/kvd_detector/main.py scan --file-path path/to/sample.exe`
- **递归扫描目录**：`python src/python/kvd_detector/main.py scan --dir-path path/to/dir --recursive`

### 5. 启动扫描服务 (IPC)
```bash
python src/python/kvd_detector/main.py serve
```

## 配置说明

核心配置位于 `src/python/kvd_detector/main.py` 的 `config.config` 模块中，包括：
- **路径配置**：`PROJECT_ROOT`, `RESOURCES_DIR`, `PROCESSED_DATA_DIR` 等。
- **训练参数**：`DEFAULT_NUM_BOOST_ROUND`, `DEFAULT_MAX_FILE_SIZE` (默认 64KB)。
- **聚类参数**：`DEFAULT_MIN_CLUSTER_SIZE`, `DEFAULT_MIN_SAMPLES`。

## DLL 扫描引擎调用文档

### 1. DLL 与导出接口

- 编译后会生成两个 DLL：`axon_engine.dll`、`signature_engine.dll`。
- 导出函数定义见 `src/cpp/kvd_core/include/kvd/api.h`，导出表见 `src/cpp/kvd_core/src/kvd.def`。
- 核心导出函数：
  - `kvd_create`
  - `kvd_destroy`
  - `kvd_scan_path`
  - `kvd_scan_bytes`
  - `kvd_scan_paths`
  - `kvd_train_path`
  - `kvd_train_paths`
  - `kvd_train_from_path`
  - `kvd_signature_flush`
  - `kvd_free`
  - `kvd_validate_models`
  - `kvd_extract_pe_features`
  - `kvd_extract_pe_features_batch`

### 2. 通用调用约定

- 调用约定：Windows 下为 `__cdecl`。
- `kvd_create` 成功返回非空句柄，失败返回 `nullptr`。
- 所有 `out_json` / `out_error` 输出缓冲区由 DLL 内部分配，调用方必须用 `kvd_free` 释放。
- 扫描类接口返回值：`0` 为成功，`!=0` 为失败。
- `kvd_extract_pe_features` 相关接口中，特征维度固定为 `1500`。

`kvd_config` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| model_path | const char* | 主模型路径 |
| model_normal_path | const char* | normal 路由模型路径 |
| model_packed_path | const char* | packed 路由模型路径 |
| family_classifier_json_path | const char* | 家族分类器 JSON 路径 |
| allowed_scan_root | const char* | 允许扫描的根目录 |
| max_file_size | unsigned int | 最大读取文件大小 |
| prediction_threshold | float | 恶意判定阈值 |

`kvd_validate_models` 主要返回码：

| 返回码 | 含义 |
|---|---|
| 0 | 模型检查通过 |
| -1 | 参数错误 |
| -10 | 主模型缺失 |
| -11 | 主模型无效 |
| -12 | 路由模型配置不完整 |
| -13 | normal 模型缺失 |
| -14 | normal 模型无效 |
| -15 | packed 模型缺失 |
| -16 | packed 模型无效 |
| -17 | 家族分类器缺失 |
| -18 | 家族分类器无效 |
| -100 | 内存分配失败 |

扫描结果 JSON 典型字段：

```json
{
  "is_malware": true,
  "confidence": 0.98,
  "axon_malware": true,
  "axon_score": 0.96,
  "signature_hit": false,
  "signature_score": 0.0,
  "signature_reason": "",
  "error": "",
  "malware_family": {
    "family_name": "example_family",
    "cluster_id": 12,
    "is_new_family": false
  }
}
```

### 3. C++ 调用示例

```cpp
#include "kvd/api.h"
#include <windows.h>
#include <iostream>
#include <string>

int main() {
  HMODULE mod = LoadLibraryA("axon_engine.dll");
  if (!mod) return 1;

  using kvd_create_fn = kvd_handle* (KVD_CALL*)(const kvd_config*);
  using kvd_destroy_fn = void (KVD_CALL*)(kvd_handle*);
  using kvd_scan_path_fn = int (KVD_CALL*)(kvd_handle*, const char*, char**, size_t*);
  using kvd_free_fn = void (KVD_CALL*)(char*);

  auto kvd_create_p = reinterpret_cast<kvd_create_fn>(GetProcAddress(mod, "kvd_create"));
  auto kvd_destroy_p = reinterpret_cast<kvd_destroy_fn>(GetProcAddress(mod, "kvd_destroy"));
  auto kvd_scan_path_p = reinterpret_cast<kvd_scan_path_fn>(GetProcAddress(mod, "kvd_scan_path"));
  auto kvd_free_p = reinterpret_cast<kvd_free_fn>(GetProcAddress(mod, "kvd_free"));
  if (!kvd_create_p || !kvd_destroy_p || !kvd_scan_path_p || !kvd_free_p) return 2;

  kvd_config cfg{};
  cfg.model_path = "resources/weights_cluster_eval/weights/lightgbm_model.txt";
  cfg.model_normal_path = "resources/weights_cluster_eval/weights/lightgbm_model_normal.txt";
  cfg.model_packed_path = "resources/weights_cluster_eval/weights/lightgbm_model_packed.txt";
  cfg.family_classifier_json_path = "hdbscan_cluster_results/family_classifier.json";
  cfg.prediction_threshold = 0.5f;
  cfg.max_file_size = 65536;

  kvd_handle* h = kvd_create_p(&cfg);
  if (!h) return 3;

  char* out_json = nullptr;
  size_t out_len = 0;
  int rc = kvd_scan_path_p(h, "D:/samples/test.exe", &out_json, &out_len);
  if (rc == 0 && out_json) {
    std::cout.write(out_json, static_cast<std::streamsize>(out_len));
    std::cout << std::endl;
    kvd_free_p(out_json);
  }

  kvd_destroy_p(h);
  return rc == 0 ? 0 : 4;
}
```

### 4. Python 调用示例

```python
import ctypes
import json

class KvdConfig(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("model_normal_path", ctypes.c_char_p),
        ("model_packed_path", ctypes.c_char_p),
        ("family_classifier_json_path", ctypes.c_char_p),
        ("allowed_scan_root", ctypes.c_char_p),
        ("max_file_size", ctypes.c_uint),
        ("prediction_threshold", ctypes.c_float),
    ]

dll = ctypes.CDLL("axon_engine.dll")
dll.kvd_create.argtypes = [ctypes.POINTER(KvdConfig)]
dll.kvd_create.restype = ctypes.c_void_p
dll.kvd_destroy.argtypes = [ctypes.c_void_p]
dll.kvd_scan_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
dll.kvd_scan_path.restype = ctypes.c_int
dll.kvd_free.argtypes = [ctypes.c_char_p]

cfg = KvdConfig(
    model_path=b"resources/weights_cluster_eval/weights/lightgbm_model.txt",
    model_normal_path=b"resources/weights_cluster_eval/weights/lightgbm_model_normal.txt",
    model_packed_path=b"resources/weights_cluster_eval/weights/lightgbm_model_packed.txt",
    family_classifier_json_path=b"hdbscan_cluster_results/family_classifier.json",
    allowed_scan_root=None,
    max_file_size=65536,
    prediction_threshold=0.5,
)

h = dll.kvd_create(ctypes.byref(cfg))
if not h:
    raise RuntimeError("kvd_create failed")

out_json = ctypes.c_char_p()
out_len = ctypes.c_size_t()
rc = dll.kvd_scan_path(h, b"D:/samples/test.exe", ctypes.byref(out_json), ctypes.byref(out_len))
if rc != 0:
    dll.kvd_destroy(h)
    raise RuntimeError(f"kvd_scan_path failed: {rc}")

raw = ctypes.string_at(out_json, out_len.value)
result = json.loads(raw.decode("utf-8"))
print(result)
dll.kvd_free(out_json)
dll.kvd_destroy(h)
```

### 5. JavaScript 调用示例

依赖安装：

```bash
npm i ffi-napi ref-napi ref-struct-di
```

```javascript
const ffi = require('ffi-napi')
const ref = require('ref-napi')
const StructDi = require('ref-struct-di')
const Struct = StructDi(ref)

const charPtr = ref.refType(ref.types.char)
const charPtrPtr = ref.refType(charPtr)
const sizeTPtr = ref.refType(ref.types.size_t)
const voidPtr = ref.refType(ref.types.void)

const KvdConfig = Struct({
  model_path: charPtr,
  model_normal_path: charPtr,
  model_packed_path: charPtr,
  family_classifier_json_path: charPtr,
  allowed_scan_root: charPtr,
  max_file_size: ref.types.uint,
  prediction_threshold: ref.types.float
})

const kvd = ffi.Library('axon_engine', {
  kvd_create: [voidPtr, [ref.refType(KvdConfig)]],
  kvd_destroy: ['void', [voidPtr]],
  kvd_scan_path: ['int', [voidPtr, 'string', charPtrPtr, sizeTPtr]],
  kvd_free: ['void', [charPtr]]
})

const cfg = new KvdConfig({
  model_path: Buffer.from('resources/weights_cluster_eval/weights/lightgbm_model.txt\0'),
  model_normal_path: Buffer.from('resources/weights_cluster_eval/weights/lightgbm_model_normal.txt\0'),
  model_packed_path: Buffer.from('resources/weights_cluster_eval/weights/lightgbm_model_packed.txt\0'),
  family_classifier_json_path: Buffer.from('hdbscan_cluster_results/family_classifier.json\0'),
  allowed_scan_root: ref.NULL,
  max_file_size: 65536,
  prediction_threshold: 0.5
})

const handle = kvd.kvd_create(cfg.ref())
if (ref.isNull(handle)) {
  throw new Error('kvd_create failed')
}

const outJsonPtr = ref.alloc(charPtr)
const outLenPtr = ref.alloc(ref.types.size_t)
const rc = kvd.kvd_scan_path(handle, 'D:/samples/test.exe', outJsonPtr, outLenPtr)
if (rc !== 0) {
  kvd.kvd_destroy(handle)
  throw new Error(`kvd_scan_path failed: ${rc}`)
}

const jsonPtr = outJsonPtr.deref()
const jsonText = ref.readCString(jsonPtr, 0)
console.log(JSON.parse(jsonText))
kvd.kvd_free(jsonPtr)
kvd.kvd_destroy(handle)
```

### 6. 建议调用顺序

1. 先调用 `kvd_validate_models` 检查模型文件可用性。
2. 再调用 `kvd_create` 创建句柄。
3. 扫描阶段调用 `kvd_scan_path` / `kvd_scan_bytes` / `kvd_scan_paths`。
4. 每次读取完 JSON 输出后调用 `kvd_free`。
5. 结束时调用 `kvd_destroy` 释放句柄。

## 许可证

本项目遵循 [LICENSE](LICENSE) 中的规定。
