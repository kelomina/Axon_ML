# 项目结构规范 (PROJECT_STRUCTURE.md)

## 1. 目录结构树
```text
/ (项目根目录)
├── .gitignore                # 顶层 Git 忽略文件
├── LICENSE                   # 项目许可证
├── README.md                 # 项目总体说明
├── PROJECT_STANDARDS.md      # 项目开发规范指南 (代码、文件、测试规范)
├── PROJECT_STRUCTURE.md      # 目录结构职责与规范 (本文档)
├── feature-dictionary.md     # 详细的特征字典说明
├── build/                    # C++ 中间编译产物 (CMake 自动生成)
├── bin/                      # C++ 编译产物输出目录 (完全隔离，支持 Debug/Release)
│   ├── Debug/                # Debug 配置输出
│   └── Release/              # Release 配置输出
├── scripts/                  # 辅助构建、部署、分析等自动化脚本
│   └── analysis_hard_samples.py # 困难样本分析脚本
├── src/                      # 源代码根目录
│   ├── cpp/                  # C++ 源代码
│   │   ├── kvd_core/         # 核心扫描引擎模块
│   │   │   ├── include/      # 公共头文件
│   │   │   ├── src/          # 源代码实现
│   │   │   ├── examples/     # 示例代码
│   │   │   ├── tests/        # C++ 单元测试代码
│   │   │   └── CMakeLists.txt
│   │   └── CMakeLists.txt    # 顶层 C++ 构建配置
│   └── python/               # Python 源代码
│       ├── kvd_detector/     # 主功能包
│       │   ├── __init__.py
│       │   ├── main.py       # 入口脚本
│       │   └── ...           # 其他子包
│       ├── requirements.txt  # Python 依赖清单
│       ├── pyproject.toml    # 符合 PEP 518/621 的 Python 项目配置
│       └── axon_ml.spec      # PyInstaller 打包配置文件
├── tests/                    # Python 单元测试和集成测试代码 (与 src/ 结构保持一致)
│   └── python/               # 对应 src/python/
├── resources/                # 统一资源目录
│   └── weights_cluster_eval/ # 权重、聚类结果与测评报告
│       ├── weights/          # 模型权重 (*.pth, *.pt, *.onnx, *.bin)
│       ├── cluster/          # 聚类结果 (*.pkl, *.json, *.npy)
│       ├── eval/             # 测评报告 (*.md, *.pdf, *.html, *.csv)
│       └── README.md         # 资源来源与用途说明
├── data/                     # 原始数据或预处理中间数据 (非代码资源)
├── benign_samples/           # 良性样本 (受保护，不进入代码仓库)
└── malicious_samples/        # 恶意样本 (受保护，不进入代码仓库)
```

## 2. 目录职责说明
- **src/cpp**: 存放所有 C++ 模块。每个模块应有独立的 `CMakeLists.txt`。
- **src/python**: 存放所有 Python 代码。应以包的形式组织，方便安装与导入。
- **bin**: 存放所有二进制产物，严禁将产物提交至 Git。
- **resources**: 存放训练好的模型与运行结果，按功能细分。

## 3. 命名规范
- **C++**: 文件名使用 `snake_case.cpp/h`，类名使用 `PascalCase`。
- **Python**: 包名与文件名使用 `snake_case.py`，类名使用 `PascalCase`。
- **资源文件**: 使用 `[模块名]_[版本]_[日期].[后缀]` 命名，如 `lightgbm_v1_20260303.txt`。

## 4. 新增文件流程
1. **C++ 代码**: 在 `src/cpp` 下创建子目录，编写 `CMakeLists.txt` 并在顶层 `src/cpp/CMakeLists.txt` 中添加 `add_subdirectory`。
2. **Python 代码**: 在 `src/python/kvd_detector` 下按功能划分子包。
3. **资源文件**: 运行结果必须输出至 `resources/weights_cluster_eval` 下的对应子目录。

## 5. CI/CD 验证示例 (GitHub Actions)
```yaml
name: Build and Verify

on: [push, pull_request]

jobs:
  build-cpp:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure CMake
        run: cmake -S src/cpp -B build -DCMAKE_BUILD_TYPE=Release
      - name: Build
        run: cmake --build build --config Release
      - name: Verify Output
        run: |
          if (!(Test-Path bin/Release/axon_engine.dll)) { exit 1 }
          if (Get-ChildItem src/cpp -Include *.dll,*.exe -Recurse) { exit 1 } # 检查源码目录无泄漏

  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install package
        run: pip install src/python/
      - name: Run Tests
        run: python -m pytest
```
