# 项目开发规范指南 (Project Development Standards)

本文档旨在统一大型现代项目的开发规范，包括**代码规范**、**文件规范**和**目录规范**。遵循这些规范能够提升代码的整洁度、可读性和可维护性，降低团队协作的沟通成本。

## 1. 目录规范 (Directory Standards)

现代大型项目应当遵循清晰、模块化、高内聚的目录结构设计。

- **命名规则**：目录名一律使用**全小写字母**，多个单词之间使用下划线 `_` 或中划线 `-` 分隔（本项目推荐使用下划线 `_`，如 `kvd_core`）。
- **层级扁平化**：避免过深的目录层级，尽量保持扁平化设计，避免“迷宫式”目录。
- **职责单一**：每个目录应当只包含相关性最高的文件。
- **标准顶层结构**：
  - `src/`：存放所有源代码。按照语言或微服务进一步划分子目录（如 `src/cpp/`, `src/python/`）。
  - `tests/`：存放单元测试和集成测试代码，目录结构应与 `src/` 保持一致，便于对应查找。
  - `docs/`：存放项目相关文档（架构设计、API 接口、规范说明等）。
  - `resources/` 或 `assets/`：存放非代码的静态资源、模型权重、配置文件等。
  - `scripts/` 或 `tools/`：存放辅助构建、部署、自动化运维的脚本。

## 2. 文件规范 (File Standards)

- **命名规则**：
  - **Python 文件**：使用下划线命名法（Snake Case），如 `malware_scanner.py`。
  - **C/C++ 文件**：头文件使用 `.h` 或 `.hpp`，源文件使用 `.cpp`。使用下划线命名法，如 `lightgbm_infer.cpp`。
  - **文档文件**：Markdown 文档推荐使用大写字母（如 `README.md`, `PROJECT_STANDARDS.md`）或小写中划线（如 `feature-dictionary.md`）。
- **编码格式**：所有文本文件必须使用 **UTF-8 (无 BOM)** 编码。
- **换行符**：统一使用 **LF (`\n`)** 作为换行符（即使在 Windows 平台下开发，也请配置 Git 的 `core.autocrlf = input` 或 `false`，并由 `.gitattributes` 强制约束）。
- **文件末尾**：所有代码文件和文档文件末尾必须保留一个**空行 (EOF newline)**。
- **文件头部**：核心代码文件顶部可包含版权声明或简短的用途说明。
- **文件大小限制**：单一代码文件不宜过大（一般建议不超过 1000 行），超长文件应根据功能模块进行拆分。

## 3. 代码规范 (Code Standards)

### 3.1 Python 开发规范
本项目基于 Python 3.8+ 进行数据处理和模型训练。

- **代码风格**：严格遵循 [PEP 8](https://peps.python.org/pep-0008/) 规范。
- **格式化工具**：推荐使用 `Black`（行宽 88 或 120）进行代码格式化，使用 `Ruff` 或 `Flake8` 进行静态代码检查。
- **类型提示 (Type Hints)**：强制要求在函数签名、类成员中添加类型注解，提高代码可读性和静态检查能力。
  ```python
  def extract_features(file_path: str, max_size: int = 65536) -> dict:
      pass
  ```
- **文档注释 (Docstrings)**：使用 Google Style 或 Sphinx Style 编写 Docstring。公共模块、类和函数必须包含文档说明。
- **包管理**：使用虚拟环境（`venv` 或 `conda`），所有依赖必须在 `requirements.txt` 或 `pyproject.toml` 中明确声明及锁定版本。

### 3.2 C++ 开发规范
本项目 C++ 内核（如 `kvd_core`）基于 C++17/20 构建。

- **代码风格**：推荐遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)。
- **格式化工具**：使用 `.clang-format` 统一格式化（推荐设置缩进为 4 个空格）。
- **内存管理**：
  - 严禁在现代 C++ 中直接使用 `new`/`delete`。
  - 必须使用智能指针（`std::unique_ptr`, `std::shared_ptr`）或标准容器（`std::vector`, `std::string`）管理内存。
  - **特例**：C ABI 导出接口（如 DLL 导出 API）中允许使用裸指针，但必须提供明确的配套释放函数（如 `kvd_free`）。
- **命名规范**：
  - 类名、结构体名：大驼峰命名法（PascalCase），如 `MalwareScanner`。
  - 变量名、函数名：下划线命名法（snake_case），如 `extract_pe_features`。
  - 宏定义、常量：全大写加下划线，如 `MAX_BUFFER_SIZE`。
- **异常处理**：
  - 内部逻辑可使用 C++ 异常。
  - **边界安全**：在导出 C 接口边界处（`extern "C"`），必须捕获所有异常并转换为错误码（Error Codes），防止跨语言/跨编译器边界抛出异常导致程序崩溃。

## 4. Git 与提交规范 (Git & Commit Standards)

- **分支命名**：
  - 功能开发：`feat/xxx` 或 `feature/xxx`
  - Bug 修复：`fix/xxx` 或 `bugfix/xxx`
  - 文档更新：`docs/xxx`
  - 重构优化：`refactor/xxx`
- **提交信息 (Commit Message)**：遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。
  - 格式：`<type>(<scope>): <subject>`
  - 示例：`feat(cpp): add ONNX runtime support for inference`
  - 常用类型：`feat` (新特性), `fix` (修复 bug), `docs` (文档), `style` (格式化), `refactor` (重构), `test` (测试), `chore` (构建或辅助工具)。