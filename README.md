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

## 许可证

本项目遵循 [LICENSE](LICENSE) 中的规定。
