# 神枢 (Axon V2) - 恶意软件检测与家族分类系统

神枢 (Axon V2) 是一个基于机器学习的高性能恶意软件检测与家族分类系统。该系统采用了混合架构，包含用于高性能特征提取与推理的 C++ 扫描内核 (`kvd_core`)，以及用于模型训练、评估、调优和复杂业务逻辑的 Python 智能调度框架 (`kvd_detector`)。

Axon V2 旨在提供快速、准确且可扩展的恶意样本识别能力，不仅能够进行基础的恶意/良性二分类，还具备处理加壳样本、攻坚困难样本以及进行家族聚类归属的深度分析能力。

## 🌟 核心特性

- **双层检测机制与路由专家系统 (Gating & Expert Models)**：
  - 动态路由门控网络，能够自动判断样本是否加壳/高熵。
  - 将样本分发至对应的 **常规专家 (Normal Expert)** 或 **加壳专家 (Packed Expert)** 进行深度检测，显著提升了对复杂加壳样本的识别精度。
- **困难样本与误报/漏报攻坚 (Hardcase Deep Learning)**：
  - 针对传统 GBDT 难以区分的困难样本 (Hard Samples)、假阳性 (False Positives) 和假阴性 (False Negatives)，引入了专用的深度学习级联模型 (`HardCaseNet`) 进行二次研判。
  - 支持多模型方案对比试验 (A/B/C Trials) 以选择最佳攻坚策略。
- **无监督家族识别与归属**：
  - 基于 **fast-hdbscan** 对恶意样本进行无监督聚类分析，自动发现未知的恶意软件家族。
  - 训练专门的家族分类器，实现对新未知样本的快速家族归属预测。
- **全方位特征工程引擎**：
  - C++ 内核底层支持超过 1500 维的细粒度特征提取，涵盖文件级统计、PE 结构、Section 属性、导入/导出表、安全标志、轻量级哈希 (API/DLL 名) 以及组合交互特征等。
- **多引擎混合架构与统一 ONNX 部署**：
  - **C++ 扫描内核**：底层 `axon_engine.dll` 和 `signature_engine.dll` 集成了 LightGBM C API 与 ONNX Runtime，提供微秒级的本地特征解析与推理性能。
  - **ONNX 优先**：提供一键权重转换工具 (`convert-weights-onnx`)，将所有 LightGBM 模型、路由模型、深度学习模型和缩放器统一转换为 `.onnx` 格式，实现跨语言与环境的无缝部署。

## 📂 项目结构

```text
/ (项目根目录)
├── src/                      # 源代码根目录
│   ├── cpp/                  # C++ 核心扫描引擎 (kvd_core)
│   │   ├── kvd_core/         # 引擎实现，包含 LightGBM/ONNX 推理及特征提取 (LIEF)
│   │   └── CMakeLists.txt
│   └── python/               # Python 检测与训练框架 (kvd_detector)
│       ├── kvd_detector/     # 包含主程序、特征提取、聚类及各模型训练逻辑
│       ├── requirements.txt  # Python 依赖清单
│       └── pyproject.toml    # 项目配置 (提供 kvd-scan 命令行)
├── resources/                # 资源管理 (权重、聚类结果与报表)
│   └── weights_cluster_eval/
│       ├── weights/          # 训练好的模型权重 (*.txt, *.pkl, *.onnx)
│       ├── cluster/          # 家族分类器与聚类结果 (*.json)
│       └── eval/             # 评估报告与混淆矩阵
├── data/                     # 预处理特征中间数据目录
├── benign_samples/           # 良性样本存放区 (供特征提取使用)
├── malicious_samples/        # 恶意样本存放区 (供特征提取使用)
├── .gitignore                # 顶层 Git 忽略文件
├── LICENSE                   # 项目许可证
├── PROJECT_STANDARDS.md      # 项目开发规范指南 (代码、文件、测试规范)
├── PROJECT_STRUCTURE.md      # 目录结构职责与规范
├── feature-dictionary.md     # 详细的 PE 病毒检测特征字典说明
└── README.md                 # 项目主文档 (本文档)
```

## 🛠 环境要求

### Python 训练与调度环境
- **Python**：3.8 及以上 (推荐 3.10)
- **依赖库**：见 `src/python/requirements.txt` (包括 `lightgbm`, `fast-hdbscan`, `torch`, `optuna`, `onnxruntime` 等)

### C++ 编译环境 (扫描内核)
- **操作系统**：Windows 10/11 (当前 CMake 配置主要针对 Windows DLL 构建) 或 Linux (需适配 CMake)
- **C++ 编译**：CMake 3.20+，支持 C++20 的编译器 (如 MSVC, GCC, Clang)
- **第三方库依赖**：
  - [LIEF](https://lief.re/) (PE 解析)
  - [nlohmann_json](https://json.nlohmann.me/)
  - **LightGBM C API** 与 **ONNX Runtime**
  *(注意：在编译 C++ 内核前，需根据本机环境修改 `src/cpp/kvd_core/src/CMakeLists.txt` 中硬编码的 `ONNXRUNTIME_ROOT` 和 `LIGHTGBM_INCLUDE_DIR` 路径)*

## 🚀 安装指南

1. **配置 Python 环境**
   ```powershell
   python -m venv .venv
   # Windows
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   
   pip install -r src/python/requirements.txt
   # 可选：安装本地包以启用 `kvd-scan` 命令
   pip install -e src/python/
   ```

2. **配置并编译 C++ 内核 (按需)**
   修改 `CMakeLists.txt` 中的依赖路径后：
   ```powershell
   cmake -S src/cpp -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```
   编译产物 `axon_engine.dll` 将生成于 `build/bin` 目录下。

## 💡 快速使用

项目的核心入口是 `src/python/kvd_detector/main.py`。你可以通过命令行参数执行全生命周期的任务：

### 1. 数据准备与特征提取
从指定的恶意和良性样本目录提取特征并生成用于训练的中间数据：
```bash
python src/python/kvd_detector/main.py extract --output-dir data/processed_lightgbm --label-inference directory
```

### 2. 自动化训练流程
**一键训练全流程** (依次执行特征提取、LightGBM 训练、路由系统训练、家族聚类及评估)：
```bash
python src/python/kvd_detector/main.py train-all
```

**分步独立训练**：
- **基础模型预训练**：`python main.py pretrain`
- **家族聚类与分类器发现**：`python main.py finetune --plot-pca`
- **训练路由与专家系统**：`python main.py train-routing`
- **训练困难样本攻坚模型**：`python main.py train-hardcase-dl`
- **AutoML 超参调优**：`python main.py auto-tune`
- **困难样本模型对比试验**：`python main.py hardcase-model-trials`

### 3. 模型转换与部署
生产环境推荐将所有权重统一转换为 ONNX 格式：
```bash
python src/python/kvd_detector/main.py convert-weights-onnx
```
转换成功后将在 `resources/weights_cluster_eval/weights/` 目录下生成 `*.onnx` 模型以及转换清单 `weights_onnx_manifest.json`。

### 4. 样本扫描
支持扫描单个文件或递归扫描目录：
```bash
# 扫描单文件
python src/python/kvd_detector/main.py scan --file-path path/to/sample.exe

# 递归扫描目录
python src/python/kvd_detector/main.py scan --dir-path path/to/dir --recursive
```

### 5. 启动 IPC 扫描服务
启动基于 IPC 的扫描服务端，便于与其他系统或进程集成：
```bash
python src/python/kvd_detector/main.py serve
```

## 📖 C++ 扫描引擎 DLL 调用

编译生成的 `axon_engine.dll` 提供了高效的 C ABI 导出接口，支持通过 C/C++、Python (ctypes)、JavaScript (ffi-napi) 等语言进行跨平台调用。

### 核心接口
- `kvd_create` / `kvd_destroy`：句柄的创建与销毁。
- `kvd_validate_models`：校验模型及配置文件的完整性与有效性。
- `kvd_scan_path` / `kvd_scan_bytes`：执行恶意软件扫描、特征提取及推理，返回 JSON 格式结果。
- `kvd_free`：释放引擎内部分配的字符串内存。

详细导出定义请参考 `src/cpp/kvd_core/include/kvd/api.h`。

## 📜 许可证

本项目遵循 [LICENSE](LICENSE) 中的规定。

## 🙏 致谢

本项目的开发离不开众多优秀的开源项目的支持（包括但不限于 LightGBM, ONNX Runtime, LIEF, scikit-learn, fast-hdbscan, PyTorch 等），在此特别致谢所有采用 permissive 许可证（BSD、MIT、Apache 2.0 等）并无私奉献的开源社区。
