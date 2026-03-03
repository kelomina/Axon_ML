import os
import argparse
import asyncio
import sys
import importlib.abc
import importlib.util

_EMBEDDED_SOURCES = {}
_EMBEDDED_PACKAGES = set()

def _register_embedded(name, source, is_package=False):
    _EMBEDDED_SOURCES[name] = source
    if is_package:
        _EMBEDDED_PACKAGES.add(name)

class _EmbeddedImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _EMBEDDED_SOURCES:
            return importlib.util.spec_from_loader(fullname, self, is_package=fullname in _EMBEDDED_PACKAGES)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = _EMBEDDED_SOURCES.get(module.__name__)
        if source is None:
            return
        module.__file__ = __file__
        if module.__name__ in _EMBEDDED_PACKAGES:
            module.__path__ = []
            module.__package__ = module.__name__
        else:
            module.__package__ = module.__name__.rpartition('.')[0]
        exec(source, module.__dict__)

def _install_embedded_importer():
    for item in sys.meta_path:
        if isinstance(item, _EmbeddedImporter):
            return
    sys.meta_path.insert(0, _EmbeddedImporter())

_register_embedded("config", r'''pass
''', is_package=True)

_register_embedded("config.config", r'''import os

# 路径参数：统一管理数据与模型的存储位置
# BASE_DIR：配置文件所在目录；用途：定位项目根；推荐值：自动计算
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT：项目根目录；用途：拼接各模块路径；推荐值：BASE_DIR 的上级目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# RESOURCES_DIR：资源目录；用途：存放权重、聚类结果与报告
RESOURCES_DIR = os.path.join(PROJECT_ROOT, 'resources', 'weights_cluster_eval')
# PROCESSED_DATA_DIR：预处理输出目录；用途：存放 .npz 与 metadata.json；推荐值：data/processed_lightgbm
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_lightgbm')
# METADATA_FILE：数据元信息文件；用途：提供文件名与标签映射；推荐值：PROCESSED_DATA_DIR/metadata.json
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'metadata.json')
# SAVED_MODEL_DIR：模型保存目录；用途：保存训练生成的模型文件；推荐值：resources/weights_cluster_eval/weights
SAVED_MODEL_DIR = os.path.join(RESOURCES_DIR, 'weights')
# MODEL_PATH：LightGBM 模型文件路径；用途：扫描与评估加载模型；推荐值：resources/weights_cluster_eval/weights/lightgbm_model.txt
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'lightgbm_model.txt')
# HDBSCAN_SAVE_DIR：聚类结果目录；用途：保存标签与可视化；推荐值：resources/weights_cluster_eval/cluster
HDBSCAN_SAVE_DIR = os.path.join(RESOURCES_DIR, 'cluster')
# FEATURES_PKL_PATH：特征持久化文件；用途：跳过重复特征提取；推荐值：resources/weights_cluster_eval/cluster/extracted_features.pkl
FEATURES_PKL_PATH = os.path.join(HDBSCAN_SAVE_DIR, 'extracted_features.pkl')
# FAMILY_CLASSIFIER_PATH：家族分类器模型路径；用途：家族预测；推荐值：resources/weights_cluster_eval/cluster/family_classifier.pkl
FAMILY_CLASSIFIER_PATH = os.path.join(HDBSCAN_SAVE_DIR, 'family_classifier.pkl')
# BENIGN_SAMPLES_DIR：良性样本目录；用途：训练/评估数据来源；推荐值：benign_samples
BENIGN_SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'benign_samples')
# BENIGN_WHITELIST_PENDING_DIR：待白名单样本目录；用途：存放待审核的良性样本；推荐值：benign_samples/待加入白名单
BENIGN_WHITELIST_PENDING_DIR = os.path.join(BENIGN_SAMPLES_DIR, '待加入白名单')
# MALICIOUS_SAMPLES_DIR：恶意样本目录；用途：训练/评估数据来源；推荐值：malicious_samples
MALICIOUS_SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'malicious_samples')

# 可视化输出路径：统一保存训练与聚类图表
# MODEL_EVAL_FIG_DIR：图表输出目录；用途：集中管理报告文件；推荐值：resources/weights_cluster_eval/eval
MODEL_EVAL_FIG_DIR = os.path.join(RESOURCES_DIR, 'eval')
# SCAN_CACHE_PATH：扫描缓存路径；用途：避免重复计算；推荐值：resources/weights_cluster_eval/eval/scan_cache.json
SCAN_CACHE_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'scan_cache.json')
# SCAN_OUTPUT_DIR：扫描结果输出目录；用途：保存 JSON/CSV 结果；推荐值：resources/weights_cluster_eval/eval/scan_results
SCAN_OUTPUT_DIR = os.path.join(MODEL_EVAL_FIG_DIR, 'scan_results')
# PE_DIM_SUMMARY_DATASET：数据集 PE 维度摘要
PE_DIM_SUMMARY_DATASET = os.path.join(SCAN_OUTPUT_DIR, 'pe_dim_summary_dataset.json')
# PE_DIM_SUMMARY_INCREMENTAL：增量数据 PE 维度摘要
PE_DIM_SUMMARY_INCREMENTAL = os.path.join(SCAN_OUTPUT_DIR, 'pe_dim_summary_incremental.json')
# PE_DIM_SUMMARY_RAW：原始数据 PE 维度摘要
PE_DIM_SUMMARY_RAW = os.path.join(SCAN_OUTPUT_DIR, 'pe_dim_summary_raw.json')


# 训练参数：控制特征长度与 LightGBM 超参数
# DEFAULT_MAX_FILE_SIZE：每个文件字节序列长度；用途：截断/填充字节序列；推荐值：64KB-256KB（训练与存储需一致）
DEFAULT_MAX_FILE_SIZE = 64 * 1024
# DEFAULT_NUM_BOOST_ROUND：总迭代轮数；用途：控制训练步数与拟合能力；推荐值：2000-5000
DEFAULT_NUM_BOOST_ROUND = 5000
# DEFAULT_INCREMENTAL_ROUNDS：增量训练轮数；用途：在已有模型基础上追加训练；推荐值：50-200
DEFAULT_INCREMENTAL_ROUNDS = 200
# DEFAULT_EARLY_STOPPING_ROUNDS：训练早停轮数；用途：防止过拟合；推荐值：100-300
DEFAULT_EARLY_STOPPING_ROUNDS = 200
# DEFAULT_INCREMENTAL_EARLY_STOPPING：增量训练早停轮数；用途：增量阶段防止过拟合；推荐值：100-300
DEFAULT_INCREMENTAL_EARLY_STOPPING = 200
# DEFAULT_MAX_FINETUNE_ITERATIONS：强化训练最大迭代次数；用途：循环微调以降低误报；推荐值：5-15
DEFAULT_MAX_FINETUNE_ITERATIONS = 15
# STAT_CHUNK_COUNT：统计分块数量；用途：分段计算均值/方差/熵；推荐值：10（5-20）
STAT_CHUNK_COUNT = 10
# BYTE_HISTOGRAM_BINS：字节直方图分箱；用途：构建熵等统计；推荐值：256（64-256）
BYTE_HISTOGRAM_BINS = 256
# ENTROPY_BLOCK_SIZE：熵块大小；用途：计算分块熵；推荐值：1024（512-4096）
ENTROPY_BLOCK_SIZE = 2048
# ENTROPY_SAMPLE_SIZE：熵采样大小；用途：快速估计全局熵；推荐值：10240（8192-16384）
ENTROPY_SAMPLE_SIZE = 10240
# LIGHTWEIGHT_FEATURE_DIM：轻量特征维度；用途：前 256 维统计特征长度；推荐值：256
LIGHTWEIGHT_FEATURE_DIM = 256
# LIGHTWEIGHT_FEATURE_SCALE：轻量特征缩放系数；用途：融合时权重调整；推荐值：1.5（1.0-2.0）
LIGHTWEIGHT_FEATURE_SCALE = 1.5
# PE_FEATURE_VECTOR_DIM：综合特征向量总维度；用途：模型输入维度；推荐值：1500
PE_FEATURE_VECTOR_DIM = 1500
# SIZE_NORM_MAX：文件大小归一化上限；用途：避免尺度过大；推荐值：100MB
SIZE_NORM_MAX = 128 * 1024 * 1024
# TIMESTAMP_MAX/TIMESTAMP_YEAR_*：时间戳归一化参数；用途：规范时间特征；推荐值：MAX=2147483647，范围 1970-2038
TIMESTAMP_MAX = 2147483647
TIMESTAMP_YEAR_BASE = 1970
TIMESTAMP_YEAR_MAX = 2038
# LIGHTGBM_FEATURE_FRACTION：特征采样比例；用途：提升泛化、降过拟合；推荐值：0.7-0.9
LIGHTGBM_FEATURE_FRACTION = 0.6734990233925464
# LIGHTGBM_BAGGING_FRACTION：样本采样比例；用途：随机采样增强稳健性；推荐值：0.7-0.9
LIGHTGBM_BAGGING_FRACTION = 0.8951208064215719
# LIGHTGBM_BAGGING_FREQ：Bagging 频率；用途：每 N 轮进行一次样本采样；推荐值：5（3-10）
LIGHTGBM_BAGGING_FREQ = 13
# LIGHTGBM_MIN_GAIN_TO_SPLIT：最小分裂增益；用途：控制树的复杂度；推荐值：0.01（0.0-0.1）
LIGHTGBM_MIN_GAIN_TO_SPLIT = 0.002330096559042368
# LIGHTGBM_MIN_DATA_IN_LEAF：叶子最小样本数；用途：避免过拟合；推荐值：20（10-50）
LIGHTGBM_MIN_DATA_IN_LEAF = 71
# LIGHTGBM_NUM_THREADS_MAX：最大线程数；用途：并行训练；推荐值：8（按 CPU 调整）
LIGHTGBM_NUM_THREADS_MAX = 16
# DEFAULT_LIGHTGBM_NUM_LEAVES/LEARNING_RATE：默认叶子数与学习率；用途：基础复杂度与步长；推荐值：30/0.07
DEFAULT_LIGHTGBM_NUM_LEAVES = 281
DEFAULT_LIGHTGBM_LEARNING_RATE = 0.0054273608259950085


# 帮助文本（训练 CLI）：用途：命令行参数说明文字；推荐值：按需维护
HELP_MAX_FILE_SIZE = 'Maximum file size in bytes to process'
HELP_FAST_DEV_RUN = 'Use a small portion of data for quick development testing'
HELP_SAVE_FEATURES = 'Save extracted features to file'
HELP_FINETUNE_ON_FALSE_POSITIVES = 'Perform reinforcement training when false positive samples are detected'
HELP_INCREMENTAL_TRAINING = 'Enable incremental training (continue training based on existing model)'
HELP_INCREMENTAL_DATA_DIR = 'Incremental training data directory (.npz files)'
HELP_INCREMENTAL_RAW_DATA_DIR = 'Incremental training raw data directory (for feature extraction)'
HELP_FILE_EXTENSIONS = 'File extensions to process, e.g. .exe .dll'
HELP_LABEL_INFERENCE = 'Label inference method: filename (based on file name) or directory (based on directory name)'
HELP_NUM_BOOST_ROUND = 'Number of boosting rounds for training'
HELP_INCREMENTAL_ROUNDS = 'Number of rounds for incremental training'
HELP_INCREMENTAL_EARLY_STOPPING = 'Early stopping rounds for incremental training'
HELP_MAX_FINETUNE_ITERATIONS = 'Maximum reinforcement training iterations'
HELP_USE_EXISTING_FEATURES = 'Use existing extracted_features.pkl file, skip feature extraction'
HELP_AUTOML_METHOD = 'AutoML method: optuna or hyperopt'
HELP_AUTOML_TRIALS = 'AutoML tuning trials count'
HELP_AUTOML_CV = 'Cross-validation folds for AutoML comparison'
HELP_AUTOML_METRIC = 'Evaluation metric: roc_auc or accuracy'
HELP_AUTOML_FAST_DEV_RUN = 'Use small subset for AutoML quick run'
HELP_SKIP_TUNING = 'Skip AutoML hyperparameter tuning phase'


# 聚类与服务参数：控制 HDBSCAN 与服务端行为
# DEFAULT_MIN_CLUSTER_SIZE：最小簇大小；用途：过滤小簇噪声；推荐值：2（每一个样本独立成家时设为 2）
DEFAULT_MIN_CLUSTER_SIZE = 2
# DEFAULT_MIN_SAMPLES：核心点最小样本数；用途：影响簇密度判定；推荐值：1（每一个样本独立成家时设为 1）
DEFAULT_MIN_SAMPLES = 1
# DEFAULT_MIN_FAMILY_SIZE：家族保留阈值；用途：过小家族视为噪声；推荐值：1（每一个样本独立成家时设为 1）
DEFAULT_MIN_FAMILY_SIZE = 1
# DEFAULT_TREAT_NOISE_AS_FAMILY：将噪声视为独立家族；用途：开启后每个噪声点将分配独立家族 ID；推荐值：True
DEFAULT_TREAT_NOISE_AS_FAMILY = True
# SCAN_CACHE_PATH：扫描缓存路径；用途：避免重复计算；推荐值：项目根/scan_cache.json
SCAN_CACHE_PATH = os.path.join(PROJECT_ROOT, 'scan_cache.json')
# SCAN_OUTPUT_DIR：扫描结果输出目录；用途：保存 JSON/CSV 结果；推荐值：scan_results
SCAN_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'scan_results')
# PE_DIM_SUMMARY_DATASET：数据集 PE 维度摘要；用途：记录数据集特征分布；推荐值：scan_results/pe_dim_summary_dataset.json
PE_DIM_SUMMARY_DATASET = os.path.join(SCAN_OUTPUT_DIR, 'pe_dim_summary_dataset.json')
# PE_DIM_SUMMARY_INCREMENTAL：增量数据 PE 维度摘要；用途：记录增量特征分布；推荐值：scan_results/pe_dim_summary_incremental.json
PE_DIM_SUMMARY_INCREMENTAL = os.path.join(SCAN_OUTPUT_DIR, 'pe_dim_summary_incremental.json')
# PE_DIM_SUMMARY_RAW：原始数据 PE 维度摘要；用途：记录原始特征分布；推荐值：scan_results/pe_dim_summary_raw.json
PE_DIM_SUMMARY_RAW = os.path.join(SCAN_OUTPUT_DIR, 'pe_dim_summary_raw.json')
# HDBSCAN_SAVE_DIR：聚类结果目录；用途：保存标签与可视化；推荐值：hdbscan_cluster_results
HDBSCAN_SAVE_DIR = os.path.join(PROJECT_ROOT, 'hdbscan_cluster_results')


# 帮助文本（聚类/服务 CLI）：用途：命令行参数说明文字；推荐值：按需维护
HELP_DATA_DIR = 'Directory of processed dataset (.npz & metadata)'
HELP_FEATURES_PATH = 'Path to extracted features pickle'
HELP_SAVE_DIR = 'Directory to save HDBSCAN results'
HELP_MIN_CLUSTER_SIZE = 'Minimum cluster size for HDBSCAN'
HELP_MIN_SAMPLES = 'Minimum samples for HDBSCAN core points'
HELP_MIN_FAMILY_SIZE = 'Minimum family size to keep'
HELP_PLOT_PCA = 'Plot PCA for clusters'
HELP_EXPLAIN_DISCREPANCY = 'Explain discrepancies between cluster and ground truth'
HELP_TREAT_NOISE_AS_FAMILY = 'Treat HDBSCAN noise as a separate family'
HELP_LIGHTGBM_MODEL_PATH = 'Path to LightGBM model file'
HELP_FAMILY_CLASSIFIER_PATH = 'Path to family classifier pickle'
HELP_CACHE_FILE = 'Path to scan cache file'
HELP_FILE_PATH = 'Single file path to scan'
HELP_DIR_PATH = 'Directory path to scan'
HELP_RECURSIVE = 'Recursively scan directories'
HELP_OUTPUT_PATH = 'Directory to save scan results'


# 可视化输出路径：统一保存训练与聚类图表
# MODEL_EVAL_FIG_DIR：图表输出目录；用途：集中管理报告文件；推荐值：reports
MODEL_EVAL_FIG_DIR = os.path.join(PROJECT_ROOT, 'reports')
# MODEL_EVAL_FIG_PATH：模型评估图路径；用途：保存准确率/混淆矩阵等；推荐值：reports/model_evaluation.png
MODEL_EVAL_FIG_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'model_evaluation.png')
# MODEL_EVAL_AUC_PATH：ROC-AUC 曲线路径；用途：评估分类器性能；推荐值：reports/model_auc_curve.png
MODEL_EVAL_AUC_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'model_auc_curve.png')
# ROUTING_EVAL_REPORT_PATH：路由评估报告路径；用途：保存路由系统评估文本；推荐值：reports/routing_evaluation_report.txt
ROUTING_EVAL_REPORT_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'routing_evaluation_report.txt')
# ROUTING_CONFUSION_MATRIX_PATH：路由混淆矩阵路径；用途：保存路由系统混淆矩阵图；推荐值：reports/routing_confusion_matrix.png
ROUTING_CONFUSION_MATRIX_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'routing_confusion_matrix.png')
# ROUTING_ROC_AUC_PATH：路由 ROC-AUC 曲线路径；用途：评估路由门控性能；推荐值：reports/routing_roc_auc.png
ROUTING_ROC_AUC_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'routing_roc_auc.png')
# AUTOML_RESULTS_PATH：AutoML 结果路径；用途：保存调优实验结果；推荐值：reports/automl_comparison.json
AUTOML_RESULTS_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'automl_comparison.json')
# DETECTED_MALICIOUS_PATHS_REPORT_PATH：恶意路径报告路径；用途：记录扫描发现的威胁；推荐值：reports/detected_malicious_paths.txt
DETECTED_MALICIOUS_PATHS_REPORT_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'detected_malicious_paths.txt')
# SCAN_PRINT_ONLY_MALICIOUS：仅打印恶意样本；用途：简化扫描输出；推荐值：True
SCAN_PRINT_ONLY_MALICIOUS = True
# SERVICE_CONCURRENCY_LIMIT：服务并发限制；用途：控制服务端最大连接数；推荐值：256
SERVICE_CONCURRENCY_LIMIT = 256
# SERVICE_PRINT_MALICIOUS_PATHS：服务打印恶意路径；用途：服务端实时输出威胁；推荐值：False
SERVICE_PRINT_MALICIOUS_PATHS = False
# SERVICE_EXIT_COMMAND：服务退出指令；用途：远程关闭服务的口令；推荐值：'exit'
SERVICE_EXIT_COMMAND = 'exit'
# SERVICE_ADMIN_TOKEN：服务管理令牌；用途：身份验证；推荐值：自定义字符串
SERVICE_ADMIN_TOKEN = ''
# SERVICE_CONTROL_LOCALHOSTS：允许控制的本地地址；用途：安全访问控制；推荐值：['127.0.0.1', '::1']
SERVICE_CONTROL_LOCALHOSTS = ['127.0.0.1', '::1']
# SERVICE_MAX_BATCH_SIZE：服务最大批处理大小；用途：限制单次扫描请求的文件数；推荐值：64
SERVICE_MAX_BATCH_SIZE = 64
# SERVICE_IPC_HOST：IPC 服务主机；用途：进程间通信监听地址；推荐值：'127.0.0.1'
SERVICE_IPC_HOST = '127.0.0.1'
# SERVICE_IPC_PORT：IPC 服务端口；用途：进程间通信监听端口；推荐值：8765
SERVICE_IPC_PORT = 8765
# SERVICE_IPC_MAX_MESSAGE_BYTES：IPC 最大消息字节数；用途：限制通信包大小；推荐值：1MB
SERVICE_IPC_MAX_MESSAGE_BYTES = 1024 * 1024
# SERVICE_IPC_READ_TIMEOUT_SEC：IPC 读取超时；用途：防止读取阻塞；推荐值：5
SERVICE_IPC_READ_TIMEOUT_SEC = 5
# SERVICE_IPC_WRITE_TIMEOUT_SEC：IPC 写入超时；用途：防止写入阻塞；推荐值：5
SERVICE_IPC_WRITE_TIMEOUT_SEC = 5
# SERVICE_IPC_REQUEST_TIMEOUT_SEC：IPC 请求总超时；用途：控制单次扫描最长时间；推荐值：120
SERVICE_IPC_REQUEST_TIMEOUT_SEC = 120
# SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION：IPC 单连接最大请求数；用途：防止连接长久占用；推荐值：128
SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION = 128

# 路由系统训练参数
# PACKED_SECTIONS_RATIO_THRESHOLD：加壳节比例阈值；用途：判定是否加壳；推荐值：0.4
PACKED_SECTIONS_RATIO_THRESHOLD = 0.4
# PACKER_KEYWORD_HITS_THRESHOLD：加壳关键词命中阈值；用途：判定是否加壳；推荐值：0
PACKER_KEYWORD_HITS_THRESHOLD = 0

# HDBSCAN_CLUSTER_FIG_PATH：聚类可视化路径；用途：保存聚类热图等；推荐值：reports/hdbscan_clustering_visualization.png
HDBSCAN_CLUSTER_FIG_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'hdbscan_clustering_visualization.png')
# HDBSCAN_PCA_FIG_PATH：聚类 PCA 图路径；用途：PCA 降维可视化；推荐值：reports/hdbscan_clustering_visualization_pca.png
HDBSCAN_PCA_FIG_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'hdbscan_clustering_visualization_pca.png')
# 环境变量键：用于在运行时覆盖默认配置
# ENV_LIGHTGBM_MODEL_PATH：覆盖二分类模型路径；用途：部署时灵活配置；推荐值：SCANNER_LIGHTGBM_MODEL_PATH
ENV_LIGHTGBM_MODEL_PATH = 'SCANNER_LIGHTGBM_MODEL_PATH'
# ENV_FAMILY_CLASSIFIER_PATH：覆盖家族分类器路径；用途：部署时灵活配置；推荐值：SCANNER_FAMILY_CLASSIFIER_PATH
ENV_FAMILY_CLASSIFIER_PATH = 'SCANNER_FAMILY_CLASSIFIER_PATH'
# ENV_CACHE_PATH：覆盖扫描缓存路径；用途：持久化缓存位置；推荐值：SCANNER_CACHE_PATH
ENV_CACHE_PATH = 'SCANNER_CACHE_PATH'
# ENV_MAX_FILE_SIZE：覆盖最大字节序列长度；用途：动态调整扫描性能；推荐值：SCANNER_MAX_FILE_SIZE
ENV_MAX_FILE_SIZE = 'SCANNER_MAX_FILE_SIZE'
# ENV_ALLOWED_SCAN_ROOT：限制扫描根路径；用途：安全访问控制；推荐值：SCANNER_ALLOWED_SCAN_ROOT
ENV_ALLOWED_SCAN_ROOT = 'SCANNER_ALLOWED_SCAN_ROOT'
# ENV_SERVICE_ADMIN_TOKEN：环境变量-服务管理令牌；用途：运行时覆盖服务令牌
ENV_SERVICE_ADMIN_TOKEN = 'SCANNER_SERVICE_ADMIN_TOKEN'
# ENV_SERVICE_EXIT_COMMAND：环境变量-服务退出指令；用途：运行时覆盖退出指令
ENV_SERVICE_EXIT_COMMAND = 'SCANNER_SERVICE_EXIT_COMMAND'
# ENV_SERVICE_IPC_HOST：环境变量-IPC 主机；用途：运行时覆盖 IPC 监听地址
ENV_SERVICE_IPC_HOST = 'SCANNER_SERVICE_IPC_HOST'
# ENV_SERVICE_IPC_PORT：环境变量-IPC 端口；用途：运行时覆盖 IPC 监听端口
ENV_SERVICE_IPC_PORT = 'SCANNER_SERVICE_IPC_PORT'
# ENV_SERVICE_IPC_MAX_MESSAGE_BYTES：环境变量-IPC 最大消息；用途：运行时覆盖最大消息大小
ENV_SERVICE_IPC_MAX_MESSAGE_BYTES = 'SCANNER_SERVICE_IPC_MAX_MESSAGE_BYTES'
# ENV_SERVICE_IPC_READ_TIMEOUT_SEC：环境变量-IPC 读取超时；用途：运行时覆盖读取超时
ENV_SERVICE_IPC_READ_TIMEOUT_SEC = 'SCANNER_SERVICE_IPC_READ_TIMEOUT_SEC'
# ENV_SERVICE_IPC_WRITE_TIMEOUT_SEC：环境变量-IPC 写入超时；用途：运行时覆盖写入超时
ENV_SERVICE_IPC_WRITE_TIMEOUT_SEC = 'SCANNER_SERVICE_IPC_WRITE_TIMEOUT_SEC'
# ENV_SERVICE_IPC_REQUEST_TIMEOUT_SEC：环境变量-IPC 请求超时；用途：运行时覆盖请求总超时
ENV_SERVICE_IPC_REQUEST_TIMEOUT_SEC = 'SCANNER_SERVICE_IPC_REQUEST_TIMEOUT_SEC'
# ENV_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION：环境变量-IPC 最大请求数；用途：运行时覆盖单连接最大请求
ENV_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION = 'SCANNER_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION'

# COLLECT_SOURCE_ROOT：采集源码根目录；用途：样本自动化采集的起始路径；推荐值：'C:\\'
COLLECT_SOURCE_ROOT = os.getenv(ENV_ALLOWED_SCAN_ROOT) or 'C:\\'
# COLLECT_ALLOWED_EXTENSIONS：采集允许的后缀；用途：过滤采集的文件类型；推荐值：['.exe', '.dll']
COLLECT_ALLOWED_EXTENSIONS = ['.exe', '.dll']
# COLLECT_MAX_FILE_SIZE：采集最大文件大小；用途：限制采集样本的大小；推荐值：SIZE_NORM_MAX
COLLECT_MAX_FILE_SIZE = SIZE_NORM_MAX



# 评估与训练细节参数：控制可视化与学习率策略
# PREDICTION_THRESHOLD：恶意预测阈值；用途：判定样本为恶意的概率下限；推荐值：0.95-0.99
PREDICTION_THRESHOLD = 0.98
# VIS_SAMPLE_SIZE：可视化采样数；用途：绘图子样本大小；推荐值：5000-20000（默认 10000）
VIS_SAMPLE_SIZE = 20000
# VIS_TSNE_PERPLEXITY：t-SNE 困惑度；用途：嵌入稳定性与结构；推荐值：30（10-50）
VIS_TSNE_PERPLEXITY = 30
# PCA_DIMENSION_FOR_CLUSTERING：聚类前 PCA 维度；用途：降噪与提速；推荐值：50（30-100）
PCA_DIMENSION_FOR_CLUSTERING = 50
FAST_HDBSCAN_PCA_DIMENSION = 20
HDBSCAN_FLOAT32_FOR_CLUSTERING = True
FAMILY_CLUSTERING_BACKEND = 'auto'
FAST_HDBSCAN_MAX_SAMPLES = 50000
KMEANS_N_CLUSTERS = 512
KMEANS_BATCH_SIZE = 4096
KMEANS_MAX_ITER = 100
KMEANS_N_INIT = 1
# EVAL_HIST_BINS：评估直方图分箱；用途：概率分布可视化精度；推荐值：50（30-100）
EVAL_HIST_BINS = 100
# EVAL_TOP_FEATURE_COUNT：Top 特征数量；用途：训练后输出前 N 个重要特征；推荐值：20（10-50）
EVAL_TOP_FEATURE_COUNT = 50
# EVAL_FONT_FAMILY：评估图中文字体；用途：确保中文标签正常显示；推荐值：['SimHei','Microsoft YaHei']
EVAL_FONT_FAMILY = ['SimHei', 'Microsoft YaHei']
# DEFAULT_TEST_SIZE：测试集比例；用途：数据集划分的测试集占比；推荐值：0.2
DEFAULT_TEST_SIZE = 0.2
# DEFAULT_VAL_SIZE：验证集比例（占训练+验证集的比例）；用途：配合 DEFAULT_TEST_SIZE 实现 6:2:2 划分；推荐值：0.25
DEFAULT_VAL_SIZE = 0.25
# DEFAULT_RANDOM_STATE：随机种子；用途：保证训练/可视化结果可复现；推荐值：42
DEFAULT_RANDOM_STATE = 42
# COMMON_SECTIONS：常见节名列表；用途：节存在布尔特征与结构判断；推荐值：['.text','.data','.rdata','.reloc','.rsrc']
COMMON_SECTIONS = ['.text', '.data', '.rdata', '.reloc', '.rsrc']
# PACKER_SECTION_KEYWORDS：打包器关键词；用途：基于节名识别常见打包器；推荐值：按需扩展
PACKER_SECTION_KEYWORDS = ['upx', 'mpress', 'aspack', 'themida', 'petite', 'pecompact', 'fsg']
# SYSTEM_DLLS：系统 DLL 集合；用途：统计导入系统 DLL 的数量/占比；推荐值：常见基础系统库集合
SYSTEM_DLLS = {'kernel32', 'user32', 'gdi32', 'advapi32', 'shell32', 'ole32', 'comctl32'}
# ENTROPY_HIGH_THRESHOLD：高熵阈值；用途：计算高熵块占比；推荐值：0.8（0.7-0.9）
ENTROPY_HIGH_THRESHOLD = 0.8
# ENTROPY_LOW_THRESHOLD：低熵阈值；用途：计算低熵块占比；推荐值：0.2（0.1-0.3）
ENTROPY_LOW_THRESHOLD = 0.2
# LARGE_TRAILING_DATA_SIZE：大尾部数据大小阈值（字节）；用途：识别异常附加数据；推荐值：1MB（512KB-4MB）
LARGE_TRAILING_DATA_SIZE = 1024 * 1024
# SECTION_ENTROPY_MIN_SIZE：节熵计算的最小字节数；用途：避免小样本噪声
SECTION_ENTROPY_MIN_SIZE = 256
# OVERLAY_ENTROPY_MIN_SIZE：叠加区熵计算的最小字节数；用途：稳定估计
OVERLAY_ENTROPY_MIN_SIZE = 1024
# FP_WEIGHT_BASE：误报权重基数；用途：强化训练中错误分类良性样本的惩罚倍数；推荐值：5.0
FP_WEIGHT_BASE = 5.0
# FP_WEIGHT_GROWTH_PER_ITER：误报权重增长步长；用途：随迭代次数增加惩罚力度；推荐值：3.0
FP_WEIGHT_GROWTH_PER_ITER = 3.0
# FP_WEIGHT_MAX：最大误报权重；用途：限制惩罚上限；推荐值：100.0
FP_WEIGHT_MAX = 100.0
# FAMILY_THRESHOLD_PERCENTILE：家族阈值百分位；用途：确定家族置信阈；推荐值：95（90-99）
FAMILY_THRESHOLD_PERCENTILE = 90
# FAMILY_THRESHOLD_MULTIPLIER：家族阈值放大倍数；用途：放宽/收紧家族判定；推荐值：1.2（1.0-1.5）
FAMILY_THRESHOLD_MULTIPLIER = 1.0
# WARMUP_ROUNDS：学习率暖启动轮数；用途：前期小步长稳定训练；推荐值：100（50-200）
WARMUP_ROUNDS = 200
# WARMUP_START_LR：暖启动起始学习率；用途：初始学习率；推荐值：0.001（0.0005-0.005）
WARMUP_START_LR = 0.001
# WARMUP_TARGET_LR：暖启动目标学习率；用途：结束时学习率
WARMUP_TARGET_LR = 0.07

# API 分类关键词：用于特征提取中的语义分析
# API_CATEGORY_NETWORK：网络相关 API；用途：识别网络通信行为
API_CATEGORY_NETWORK = ['ws2_32', 'wininet', 'winhttp', 'internet', 'socket', 'connect', 'send', 'recv', 'http', 'url']
# API_CATEGORY_PROCESS：进程相关 API；用途：识别进程注入与控制行为
API_CATEGORY_PROCESS = ['createprocess', 'openprocess', 'terminateprocess', 'getprocaddress', 'loadlibrary', 'virtualallocex', 'writeprocessmemory']
# API_CATEGORY_FILESYSTEM：文件系统相关 API；用途：识别文件读写与遍历行为
API_CATEGORY_FILESYSTEM = ['createfile', 'readfile', 'writefile', 'deletefile', 'movefile', 'copyfile', 'findfirstfile', 'findnextfile', 'setfileattributes', 'getfileattributes', 'getfilesize']
# API_CATEGORY_REGISTRY：注册表相关 API；用途：识别持久化与配置修改行为
API_CATEGORY_REGISTRY = ['regopenkey', 'regsetvalue', 'regcreatekey', 'regdeletekey', 'regqueryvalue', 'regenumkey', 'regclosekey']

# 路由门控与专家模型配置
# GATING_ENABLED：启用路由门控；用途：开启混合专家模型 (MoE) 架构；推荐值：True
GATING_ENABLED = True
# GATING_MODE：路由模式；用途：'rule' (基于规则) 或 'model' (基于神经网络)；推荐值：'rule'
GATING_MODE = 'rule'
# GATING_MODEL_PATH：门控模型路径；用途：加载训练好的门控神经网络；推荐值：saved_models/gating_model.pth
GATING_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'gating_model.pth')
# GATING_INPUT_DIM corresponds to the total feature dimension (Statistical + PE features)
# Statistical features: 49 (based on current logic with STAT_CHUNK_COUNT=10)
# PE features: 1500 (PE_FEATURE_VECTOR_DIM)
# Total: 1549
# GATING_INPUT_DIM：门控输入维度；用途：统计特征 + PE 特征总和；推荐值：1549
GATING_INPUT_DIM = 1549 
# GATING_HIDDEN_DIM：门控隐藏层维度；用途：控制门控网络复杂度；推荐值：256
GATING_HIDDEN_DIM = 256
# GATING_OUTPUT_DIM：门控输出维度；用途：0 为普通样本，1 为加壳样本；推荐值：2
GATING_OUTPUT_DIM = 2  # 0: Normal, 1: Packed
# GATING_THRESHOLD：门控判定阈值；用途：加壳判定的概率阈值；推荐值：0.5
GATING_THRESHOLD = 0.5
# GATING_LEARNING_RATE：门控训练学习率；用途：优化门控网络的步长；推荐值：0.001
GATING_LEARNING_RATE = 0.001
# GATING_EPOCHS：门控训练轮数；用途：训练门控网络的最大轮数；推荐值：20
GATING_EPOCHS = 20
# GATING_BATCH_SIZE：门控训练批大小；用途：梯度下降的样本块大小；推荐值：64
GATING_BATCH_SIZE = 64
# GATE_HIGH_ENTROPY_RATIO：高熵占比门限；用途：规则路由判定加壳的依据；推荐值：0.8
GATE_HIGH_ENTROPY_RATIO = 0.8
# GATE_PACKED_SECTIONS_RATIO：加壳节占比门限；用途：规则路由判定加壳的依据；推荐值：0.3
GATE_PACKED_SECTIONS_RATIO = 0.3
# GATE_PACKER_RATIO：打包器特征门限；用途：规则路由判定加壳的依据；推荐值：0.1
GATE_PACKER_RATIO = 0.1

# EXPERT_NORMAL_MODEL_PATH：普通样本专家模型；用途：处理未加壳样本的 LightGBM 模型
EXPERT_NORMAL_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'lightgbm_model_normal.txt')
# EXPERT_PACKED_MODEL_PATH：加壳样本专家模型；用途：处理加壳样本的 LightGBM 模型
EXPERT_PACKED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'lightgbm_model_packed.txt')

# FEATURE_GATING_TOP_K：特征选择 K 值；用途：特征重要性实验中的保留数量；推荐值：1150
FEATURE_GATING_TOP_K = 1150
# FEATURE_GATING_REPORT_PATH：特征选择报告路径；用途：保存特征消融实验结果
FEATURE_GATING_REPORT_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'feature_gating_experiment.json')
# FEATURE_GATING_K_START：特征实验起始 K 值；用途：实验步进的起点
FEATURE_GATING_K_START = 50
# FEATURE_GATING_K_STEP：特征实验步进值；用途：每次增加的特征数量
FEATURE_GATING_K_STEP = 50

# AutoML 配置
# AUTOML_ENABLED：启用 AutoML；用途：开启自动超参数优化
AUTOML_ENABLED = True
# AUTOML_METHOD_DEFAULT：默认调优方法；用途：'optuna' 或 'hyperopt'
AUTOML_METHOD_DEFAULT = 'optuna'
# AUTOML_TRIALS_DEFAULT：默认试验次数；用途：超参数搜索的迭代次数
AUTOML_TRIALS_DEFAULT = 50
# AUTOML_CV_FOLDS_DEFAULT：默认交叉验证折数；用途：评估参数时的 CV 次数
AUTOML_CV_FOLDS_DEFAULT = 5
# AUTOML_METRIC_DEFAULT：默认评估指标；用途：优化目标 (roc_auc/accuracy/f1/precision/recall)
AUTOML_METRIC_DEFAULT = 'f1'
# AUTOML_ADDITIONAL_METRICS：额外监控指标；用途：在调优报告中显示的辅助指标
AUTOML_ADDITIONAL_METRICS = ['precision', 'recall', 'f1']
# AUTOML_TIMEOUT：调优超时时间；用途：限制搜索的最长时间（秒）
AUTOML_TIMEOUT = None
# AUTOML_LGBM_SCALE_POS_WEIGHT_MIN/MAX：类别权重搜索范围；用途：平衡正负样本比例
AUTOML_LGBM_SCALE_POS_WEIGHT_MIN = 0.3
AUTOML_LGBM_SCALE_POS_WEIGHT_MAX = 1.0
# AUTOML_LGBM_NUM_LEAVES_MIN/MAX：叶子数搜索范围
AUTOML_LGBM_NUM_LEAVES_MIN = 16
AUTOML_LGBM_NUM_LEAVES_MAX = 512
# AUTOML_LGBM_LEARNING_RATE_MIN/MAX：学习率搜索范围
AUTOML_LGBM_LEARNING_RATE_MIN = 0.005
AUTOML_LGBM_LEARNING_RATE_MAX = 0.2
# AUTOML_LGBM_FEATURE_FRACTION_MIN/MAX：特征采样比例搜索范围
AUTOML_LGBM_FEATURE_FRACTION_MIN = 0.6
AUTOML_LGBM_FEATURE_FRACTION_MAX = 1.0
# AUTOML_LGBM_BAGGING_FRACTION_MIN/MAX：样本采样比例搜索范围
AUTOML_LGBM_BAGGING_FRACTION_MIN = 0.6
AUTOML_LGBM_BAGGING_FRACTION_MAX = 1.0
# AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN/MAX：叶子最小样本数搜索范围
AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN = 10
AUTOML_LGBM_MIN_DATA_IN_LEAF_MAX = 100
# AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN/MAX：最小分裂增益搜索范围
AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN = 0.0
AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MAX = 0.2
# AUTOML_LGBM_BAGGING_FREQ_MIN/MAX：Bagging 频率搜索范围
AUTOML_LGBM_BAGGING_FREQ_MIN = 1
AUTOML_LGBM_BAGGING_FREQ_MAX = 20
''')

_register_embedded("data", r'''pass
''', is_package=True)

_register_embedded("data.dataset", r'''import os
import numpy as np

class MalwareDataset:
    def __init__(self, data_dir, file_list, label_list, max_length=256*1024):
        self.data_dir = data_dir
        self.file_list = file_list
        self.label_list = label_list
        self.max_length = max_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.label_list[idx]
        if filename.endswith('.npz'):
            file_path = os.path.join(self.data_dir, filename)
        else:
            file_path = os.path.join(self.data_dir, f"{filename}.npz")
        try:
            with np.load(file_path) as data:
                byte_sequence = data['byte_sequence']
                if 'pe_features' in data:
                    pe_features = data['pe_features']
                    if pe_features.ndim > 1:
                        pe_features = pe_features.flatten()
                else:
                    from config.config import PE_FEATURE_VECTOR_DIM
                    pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                orig_length = int(data['orig_length']) if 'orig_length' in data else self.max_length
        except FileNotFoundError:
            print(f"[Warning] File not found: {file_path}, using zero padding.")
            byte_sequence = np.zeros(self.max_length, dtype=np.uint8)
            from config.config import PE_FEATURE_VECTOR_DIM
            pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
            orig_length = 0
        except Exception as e:
            print(f"[Warning] Error reading file {file_path}: {e}, using zero padding.")
            byte_sequence = np.zeros(self.max_length, dtype=np.uint8)
            from config.config import PE_FEATURE_VECTOR_DIM
            pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
            orig_length = 0
        if len(byte_sequence) > self.max_length:
            byte_sequence = byte_sequence[:self.max_length]
        else:
            byte_sequence = np.pad(byte_sequence, (0, self.max_length - len(byte_sequence)), 'constant')
        
        return byte_sequence, pe_features, label, orig_length
''')

_register_embedded("features", r'''pass
''', is_package=True)

_register_embedded("features.statistics", r'''import numpy as np
from config.config import BYTE_HISTOGRAM_BINS, STAT_CHUNK_COUNT

def extract_statistical_features(byte_sequence, pe_features, orig_length=None):
    if orig_length is not None and orig_length >= 0:
        byte_array = np.array(byte_sequence[:orig_length], dtype=np.uint8)
    else:
        byte_array = np.array(byte_sequence, dtype=np.uint8)
    length = len(byte_array)
    features = []
    if length > 0:
        mean_val = float(np.mean(byte_array))
        std_val = float(np.std(byte_array))
        min_val = float(np.min(byte_array))
        max_val = float(np.max(byte_array))
        median_val = float(np.median(byte_array))
        q25 = float(np.percentile(byte_array, 25))
        q75 = float(np.percentile(byte_array, 75))
    else:
        mean_val = 0.0
        std_val = 0.0
        min_val = 0.0
        max_val = 0.0
        median_val = 0.0
        q25 = 0.0
        q75 = 0.0
    features.extend([mean_val, std_val, min_val, max_val, median_val, q25, q75])
    features.extend([
        int(np.sum(byte_array == 0)),
        int(np.sum(byte_array == 255)),
        int(np.sum(byte_array == 0x90)),
        int(np.sum((byte_array >= 32) & (byte_array <= 126))),
    ])
    counts = np.bincount(byte_array, minlength=256) if length > 0 else np.zeros(256, dtype=np.int64)
    p = counts.astype(np.float64) / float(length) if length > 0 else np.zeros_like(counts, dtype=np.float64)
    p = p[p > 0]
    entropy = float((-np.sum(p * np.log2(p)) / 8.0) if p.size > 0 else 0.0)
    features.append(entropy)
    if length >= 3:
        one_third = length // 3
        segments = [
            byte_array[:one_third],
            byte_array[one_third:2 * one_third],
            byte_array[2 * one_third:],
        ]
    else:
        segments = [byte_array, byte_array, byte_array]
    for seg in segments:
        if len(seg) == 0:
            seg_mean = 0.0
            seg_std = 0.0
            seg_entropy = 0.0
        else:
            seg_mean = float(np.mean(seg))
            seg_std = float(np.std(seg))
            seg_counts = np.bincount(seg, minlength=256)
            seg_p = seg_counts.astype(np.float64) / float(len(seg))
            seg_p = seg_p[seg_p > 0]
            seg_entropy = float((-np.sum(seg_p * np.log2(seg_p)) / 8.0) if seg_p.size > 0 else 0.0)
        features.extend([seg_mean, seg_std, seg_entropy])
    chunk_size = max(1, length // STAT_CHUNK_COUNT)
    chunk_means = []
    chunk_stds = []
    for i in range(STAT_CHUNK_COUNT):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < STAT_CHUNK_COUNT - 1 else length
        chunk = byte_array[start_idx:end_idx]
        if len(chunk) > 0:
            chunk_means.append(float(np.mean(chunk)))
            chunk_stds.append(float(np.std(chunk)))
        else:
            chunk_means.append(0.0)
            chunk_stds.append(0.0)
    features.extend(chunk_means)
    features.extend(chunk_stds)
    chunk_means = np.array(chunk_means, dtype=np.float32)
    chunk_stds = np.array(chunk_stds, dtype=np.float32)
    if len(chunk_means) > 1:
        mean_diffs = np.diff(chunk_means)
        std_diffs = np.diff(chunk_stds)
        features.extend([
            float(np.mean(np.abs(mean_diffs))),
            float(np.std(mean_diffs)),
            float(np.max(mean_diffs)),
            float(np.min(mean_diffs)),
        ])
        features.extend([
            float(np.mean(np.abs(std_diffs))),
            float(np.std(std_diffs)),
            float(np.max(std_diffs)),
            float(np.min(std_diffs)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
        features.extend([0.0, 0.0, 0.0, 0.0])
    features.extend(pe_features.tolist())
    return np.array(features, dtype=np.float32)
''')

_register_embedded("features.extractor_save", r'''import numpy as np
from features.extractor_in_memory import extract_features_in_memory
from config.config import DEFAULT_MAX_FILE_SIZE

def process_file_directory(input_file_path, output_file_path, max_file_size=DEFAULT_MAX_FILE_SIZE):
    byte_sequence, pe_features, orig_length = extract_features_in_memory(input_file_path, max_file_size)
    if byte_sequence is None or pe_features is None:
        raise Exception(f"Failed to process file {input_file_path}")
    np.savez_compressed(output_file_path, byte_sequence=byte_sequence, pe_features=pe_features, orig_length=orig_length)
''')

_register_embedded("features.extractor_in_memory", r'''import os
import ctypes
import threading
import numpy as np
import pefile
import hashlib
from utils.path_utils import validate_path
from config.config import PROJECT_ROOT, DEFAULT_MAX_FILE_SIZE, ENTROPY_BLOCK_SIZE, ENTROPY_SAMPLE_SIZE, PE_FEATURE_VECTOR_DIM, LIGHTWEIGHT_FEATURE_DIM, LIGHTWEIGHT_FEATURE_SCALE, SIZE_NORM_MAX, TIMESTAMP_MAX, TIMESTAMP_YEAR_BASE, TIMESTAMP_YEAR_MAX, COMMON_SECTIONS, SYSTEM_DLLS, LARGE_TRAILING_DATA_SIZE, ENTROPY_HIGH_THRESHOLD, SECTION_ENTROPY_MIN_SIZE, OVERLAY_ENTROPY_MIN_SIZE, PACKER_SECTION_KEYWORDS, API_CATEGORY_NETWORK, API_CATEGORY_PROCESS, API_CATEGORY_FILESYSTEM, API_CATEGORY_REGISTRY

_NATIVE_DLL = None
_NATIVE_DLL_READY = False
_NATIVE_DLL_LOCK = threading.Lock()

def _native_dll_candidates():
    env_path = os.getenv('KVD_FEATURE_DLL')
    project_root = PROJECT_ROOT if PROJECT_ROOT else os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return [
        env_path,
        os.path.join(project_root, 'src', 'cpp', 'build', 'bin', 'Release', 'axon_engine.dll'),
        os.path.join(project_root, 'src', 'cpp', 'build', 'bin', 'Debug', 'axon_engine.dll'),
        os.path.join(project_root, 'src', 'cpp', 'build', 'src', 'Release', 'axon_engine.dll'),
        os.path.join(project_root, 'src', 'cpp', 'build', 'src', 'Debug', 'axon_engine.dll'),
        os.path.join(project_root, 'src', 'cpp', 'kvd_core', 'build', 'bin', 'Release', 'axon_engine.dll'),
        os.path.join(project_root, 'src', 'cpp', 'kvd_core', 'build', 'bin', 'Debug', 'axon_engine.dll'),
        os.path.join(project_root, 'src', 'cpp', 'kvd_core', 'build', 'src', 'Release', 'axon_engine.dll'),
        os.path.join(project_root, 'src', 'cpp', 'kvd_core', 'build', 'src', 'Debug', 'axon_engine.dll'),
        os.path.join(os.path.dirname(__file__), 'axon_engine.dll')
    ]

def _load_native_dll():
    global _NATIVE_DLL, _NATIVE_DLL_READY
    if _NATIVE_DLL_READY:
        return _NATIVE_DLL
    with _NATIVE_DLL_LOCK:
        if _NATIVE_DLL_READY:
            return _NATIVE_DLL
        for candidate in _native_dll_candidates():
            if not candidate:
                continue
            if not os.path.isfile(candidate):
                continue
            try:
                dll = ctypes.CDLL(candidate)
                dll.kvd_extract_pe_features.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
                dll.kvd_extract_pe_features.restype = ctypes.c_int
                _NATIVE_DLL = dll
                _NATIVE_DLL_READY = True
                return _NATIVE_DLL
            except Exception:
                continue
        _NATIVE_DLL_READY = True
        return None

def _extract_combined_pe_features_native(file_path):
    valid_path = validate_path(file_path)
    if not valid_path:
        return None
    dll = _load_native_dll()
    if dll is None:
        return None
    try:
        out_buf = (ctypes.c_float * PE_FEATURE_VECTOR_DIM)()
        rc = dll.kvd_extract_pe_features(valid_path.encode('utf-8'), out_buf, PE_FEATURE_VECTOR_DIM)
        if rc != 0:
            return None
        return np.ctypeslib.as_array(out_buf).astype(np.float32, copy=True)
    except Exception:
        return None

def calculate_byte_entropy(byte_sequence, block_size=ENTROPY_BLOCK_SIZE):
    if byte_sequence is None or len(byte_sequence) == 0:
        return 0, 0, 0, [], 0
    hist = np.bincount(byte_sequence, minlength=256)
    prob = hist / len(byte_sequence)
    prob = prob[prob > 0]
    overall_entropy = -np.sum(prob * np.log2(prob)) / 8
    block_entropies = []
    num_blocks = min(10, max(1, len(byte_sequence) // block_size))
    if num_blocks > 1:
        block_size = len(byte_sequence) // num_blocks
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size if i < num_blocks - 1 else len(byte_sequence)
            block = byte_sequence[start_idx:end_idx]
            if len(block) > 0:
                block_hist = np.bincount(block, minlength=256)
                block_prob = block_hist / len(block)
                block_prob = block_prob[block_prob > 0]
                if len(block_prob) > 0:
                    block_entropy = -np.sum(block_prob * np.log2(block_prob)) / 8
                    block_entropies.append(block_entropy)
    else:
        block = byte_sequence
        if len(block) > 0:
            block_hist = np.bincount(block, minlength=256)
            block_prob = block_hist / len(block)
            block_prob = block_prob[block_prob > 0]
            if len(block_prob) > 0:
                block_entropy = -np.sum(block_prob * np.log2(block_prob)) / 8
                block_entropies.append(block_entropy)
    if block_entropies:
        return overall_entropy, np.min(block_entropies), np.max(block_entropies), block_entropies, np.std(block_entropies)
    else:
        return overall_entropy, overall_entropy, overall_entropy, [], 0

def extract_byte_sequence(file_path, max_file_size):
    valid_path = validate_path(file_path)
    if not valid_path:
        return None, 0
    try:
        with open(valid_path, 'rb') as f:
            f.seek(8)
            raw_bytes = np.fromfile(f, dtype=np.uint8, count=max_file_size - 8)
        orig_len = len(raw_bytes)
        if orig_len < max_file_size - 8:
            padded_sequence = np.zeros(max_file_size, dtype=np.uint8)
            padded_sequence[:orig_len] = raw_bytes
            return padded_sequence, orig_len
        full_sequence = np.zeros(max_file_size, dtype=np.uint8)
        full_sequence[:orig_len] = raw_bytes
        return full_sequence, orig_len
    except Exception:
        return None, 0

def extract_file_attributes(file_path):
    features = {}
    missing_flags = {}
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            raise ValueError
        stat = os.stat(valid_path)
        features['size'] = stat.st_size
        features['log_size'] = np.log(stat.st_size + 1)
        with open(valid_path, 'rb') as f:
            sample_data = np.fromfile(f, dtype=np.uint8, count=ENTROPY_SAMPLE_SIZE)
        avg_entropy, min_entropy, max_entropy, block_entropies, entropy_std = calculate_byte_entropy(sample_data)
        features['file_entropy_avg'] = avg_entropy
        features['file_entropy_min'] = min_entropy
        features['file_entropy_max'] = max_entropy
        features['file_entropy_range'] = max_entropy - min_entropy
        features['file_entropy_std'] = entropy_std
        if block_entropies:
            features['file_entropy_q25'] = np.percentile(block_entropies, 25)
            features['file_entropy_q75'] = np.percentile(block_entropies, 75)
            features['file_entropy_median'] = np.median(block_entropies)
            high_entropy_count = sum(1 for e in block_entropies if e > 0.8)
            features['high_entropy_ratio'] = high_entropy_count / len(block_entropies)
            low_entropy_count = sum(1 for e in block_entropies if e < 0.2)
            features['low_entropy_ratio'] = low_entropy_count / len(block_entropies)
            if len(block_entropies) > 1:
                entropy_changes = np.diff(block_entropies)
                features['entropy_change_rate'] = np.mean(np.abs(entropy_changes))
                features['entropy_change_std'] = np.std(entropy_changes)
            else:
                features['entropy_change_rate'] = 0
                features['entropy_change_std'] = 0
        else:
            features['file_entropy_q25'] = 0
            features['file_entropy_q75'] = 0
            features['file_entropy_median'] = 0
            features['high_entropy_ratio'] = 0
            features['low_entropy_ratio'] = 0
            features['entropy_change_rate'] = 0
            features['entropy_change_std'] = 0
        if len(sample_data) > 0:
            zero_ratio = np.sum(sample_data == 0) / len(sample_data)
            printable_ratio = np.sum((sample_data >= 32) & (sample_data <= 126)) / len(sample_data)
            features['zero_byte_ratio'] = zero_ratio
            features['printable_byte_ratio'] = printable_ratio
        else:
            features['zero_byte_ratio'] = 0
            features['printable_byte_ratio'] = 0
    except Exception:
        for name in ['size','log_size','file_entropy_avg','file_entropy_min','file_entropy_max','file_entropy_range','file_entropy_std','file_entropy_q25','file_entropy_q75','file_entropy_median','high_entropy_ratio','low_entropy_ratio','entropy_change_rate','entropy_change_std','zero_byte_ratio','printable_byte_ratio']:
            features[name] = 0
    return features

def extract_enhanced_pe_features(file_path):
    features = {}
    missing_flags = {}
    file_size = 0
    pe = None
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            raise ValueError
        pe = pefile.PE(valid_path, fast_load=True)
        try:
            with open(valid_path, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
        except Exception:
            file_size = 0
        features['sections_count'] = len(pe.sections) if hasattr(pe, 'sections') else 0
        features['symbols_count'] = len(pe.SYMBOL_TABLE) if hasattr(pe, 'SYMBOL_TABLE') else 0
        features['imports_count'] = 0
        features['exports_count'] = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            imports = []
            api_names = []
            dll_names = []
            features['imports_count'] = len(pe.DIRECTORY_ENTRY_IMPORT)
            features['import_ordinal_only_count'] = 0
            total_import_functions = 0
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8').lower() if entry.dll else ''
                dll_names.append(dll_name)
                for imp in entry.imports:
                    if imp.name:
                        func_name = imp.name.decode('utf-8')
                        imports.append((dll_name, func_name))
                        api_names.append(func_name)
                        total_import_functions += 1
                    else:
                        try:
                            if hasattr(imp, 'ordinal') and imp.ordinal is not None:
                                features['import_ordinal_only_count'] = features.get('import_ordinal_only_count', 0) + 1
                                total_import_functions += 1
                        except Exception:
                            pass
            features['unique_imports'] = len(set(imports))
            features['unique_dlls'] = len(set(dll_names))
            features['unique_apis'] = len(set(api_names))
            try:
                features['import_ordinal_only_ratio'] = features['import_ordinal_only_count'] / (total_import_functions + 1)
            except Exception:
                features['import_ordinal_only_ratio'] = 0.0
            try:
                features['avg_imports_per_dll'] = total_import_functions / (features['unique_dlls'] + 1)
            except Exception:
                features['avg_imports_per_dll'] = 0.0
            if dll_names:
                dll_name_lengths = [len(name) for name in dll_names if name]
                features['dll_name_avg_length'] = np.mean(dll_name_lengths)
                features['dll_name_max_length'] = np.max(dll_name_lengths)
                features['dll_name_min_length'] = np.min(dll_name_lengths)
            imported_system_dlls = set(dll.split('.')[0].lower() for dll in dll_names if dll) & set(SYSTEM_DLLS)
            features['imported_system_dlls_count'] = len(imported_system_dlls)
            try:
                if dll_names:
                    dll_counts = {}
                    for n in dll_names:
                        if n:
                            dll_counts[n] = dll_counts.get(n, 0) + 1
                    total = sum(dll_counts.values())
                    p = np.array(list(dll_counts.values()), dtype=np.float64) / float(total) if total > 0 else np.array([], dtype=np.float64)
                    p = p[p > 0]
                    features['dll_imports_entropy'] = float((-np.sum(p * np.log2(p)) / 8.0) if p.size > 0 else 0.0)
                else:
                    features['dll_imports_entropy'] = 0.0
                if api_names:
                    api_counts = {}
                    for n in api_names:
                        if n:
                            api_counts[n] = api_counts.get(n, 0) + 1
                    total = sum(api_counts.values())
                    p = np.array(list(api_counts.values()), dtype=np.float64) / float(total) if total > 0 else np.array([], dtype=np.float64)
                    p = p[p > 0]
                    features['api_imports_entropy'] = float((-np.sum(p * np.log2(p)) / 8.0) if p.size > 0 else 0.0)
                else:
                    features['api_imports_entropy'] = 0.0
            except Exception:
                features['dll_imports_entropy'] = 0.0
                features['api_imports_entropy'] = 0.0
            features['imported_system_dlls_ratio'] = features['imported_system_dlls_count'] / (features['unique_dlls'] + 1)
            try:
                if api_names:
                    syscall_like = [n for n in api_names if n.lower().startswith('nt') or n.lower().startswith('zw')]
                    features['syscall_api_ratio'] = len(syscall_like) / (len(api_names) + 1)
                else:
                    features['syscall_api_ratio'] = 0.0
            except Exception:
                features['syscall_api_ratio'] = 0.0
            try:
                if imports:
                    net_cnt = 0
                    proc_cnt = 0
                    fs_cnt = 0
                    reg_cnt = 0
                    for dll, func in imports:
                        dl = (dll or '').lower()
                        fn = (func or '').lower()
                        if any(k in dl or k in fn for k in API_CATEGORY_NETWORK):
                            net_cnt += 1
                        if any(k in dl or k in fn for k in API_CATEGORY_PROCESS):
                            proc_cnt += 1
                        if any(k in dl or k in fn for k in API_CATEGORY_FILESYSTEM):
                            fs_cnt += 1
                        if any(k in dl or k in fn for k in API_CATEGORY_REGISTRY):
                            reg_cnt += 1
                    denom = total_import_functions + 1
                    features['api_network_ratio'] = net_cnt / denom
                    features['api_process_ratio'] = proc_cnt / denom
                    features['api_filesystem_ratio'] = fs_cnt / denom
                    features['api_registry_ratio'] = reg_cnt / denom
                else:
                    features['api_network_ratio'] = 0.0
                    features['api_process_ratio'] = 0.0
                    features['api_filesystem_ratio'] = 0.0
                    features['api_registry_ratio'] = 0.0
            except Exception:
                features['api_network_ratio'] = 0.0
                features['api_process_ratio'] = 0.0
                features['api_filesystem_ratio'] = 0.0
                features['api_registry_ratio'] = 0.0
        else:
            features['unique_imports'] = 0
            features['unique_dlls'] = 0
            features['unique_apis'] = 0
            features['dll_name_avg_length'] = 0
            features['dll_name_max_length'] = 0
            features['dll_name_min_length'] = 0
            features['imported_system_dlls_count'] = 0
            features['dll_imports_entropy'] = 0.0
            features['api_imports_entropy'] = 0.0
            features['imported_system_dlls_ratio'] = 0.0
            features['syscall_api_ratio'] = 0.0
            features['import_ordinal_only_count'] = 0
            features['import_ordinal_only_ratio'] = 0.0
            features['avg_imports_per_dll'] = 0.0
            features['api_network_ratio'] = 0.0
            features['api_process_ratio'] = 0.0
            features['api_filesystem_ratio'] = 0.0
            features['api_registry_ratio'] = 0.0
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            features['exports_count'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
            export_names = []
            for symbol in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if symbol.name:
                    export_names.append(symbol.name.decode('utf-8'))
            if export_names:
                export_name_lengths = [len(name) for name in export_names]
                features['export_name_avg_length'] = np.mean(export_name_lengths)
                features['export_name_max_length'] = np.max(export_name_lengths)
                features['export_name_min_length'] = np.min(export_name_lengths)
                features['exports_density'] = len(export_names) / (file_size + 1)
                try:
                    features['exports_name_ratio'] = len(export_names) / (features['exports_count'] + 1)
                except Exception:
                    features['exports_name_ratio'] = 0.0
            else:
                features['export_name_avg_length'] = 0
                features['export_name_max_length'] = 0
                features['export_name_min_length'] = 0
                features['exports_density'] = 0
                features['exports_name_ratio'] = 0.0
        else:
            features['exports_count'] = 0
            features['export_name_avg_length'] = 0
            features['export_name_max_length'] = 0
            features['export_name_min_length'] = 0
            features['exports_density'] = 0
            features['exports_name_ratio'] = 0.0
        if hasattr(pe, 'sections'):
            section_names = []
            section_sizes = []
            section_vsizes = []
            section_entropies = []
            code_section_size = 0
            data_section_size = 0
            code_section_vsize = 0
            data_section_vsize = 0
            executable_sections_count = 0
            writable_sections_count = 0
            readable_sections_count = 0
            rwx_sections_count = 0
            non_standard_executable_sections_count = 0
            executable_writable_sections = 0
            alignment_mismatch_count = 0
            common_executable_section_names = {'.text','text','.code'}
            file_align = getattr(pe.OPTIONAL_HEADER, 'FileAlignment', 0) if hasattr(pe, 'OPTIONAL_HEADER') else 0
            sect_align = getattr(pe.OPTIONAL_HEADER, 'SectionAlignment', 0) if hasattr(pe, 'OPTIONAL_HEADER') else 0
            for section in pe.sections:
                try:
                    name = section.Name.decode('utf-8', 'ignore').strip('\x00')
                    section_names.append(name)
                    section_sizes.append(section.SizeOfRawData)
                    section_vsizes.append(section.VirtualSize)
                    try:
                        raw = section.get_data()
                        arr = np.frombuffer(raw, dtype=np.uint8) if raw else np.array([], dtype=np.uint8)
                        if arr.size >= SECTION_ENTROPY_MIN_SIZE:
                            hist = np.bincount(arr, minlength=256)
                            prob = hist / arr.size
                            prob = prob[prob > 0]
                            se = float((-np.sum(prob * np.log2(prob)) / 8.0) if prob.size > 0 else 0.0)
                        else:
                            se = 0.0
                    except Exception:
                        se = 0.0
                    section_entropies.append(se)
                    if section.Characteristics & 0x20000000:
                        executable_sections_count += 1
                        code_section_size += section.SizeOfRawData
                        code_section_vsize += section.VirtualSize
                        if name.lower() not in common_executable_section_names:
                            non_standard_executable_sections_count += 1
                    if section.Characteristics & 0x80000000:
                        writable_sections_count += 1
                    if section.Characteristics & 0x40000000:
                        readable_sections_count += 1
                        data_section_size += section.SizeOfRawData
                        data_section_vsize += section.VirtualSize
                    if (section.Characteristics & 0x20000000) and (section.Characteristics & 0x80000000):
                        rwx_sections_count += 1
                        executable_writable_sections += 1
                    try:
                        if file_align and (section.SizeOfRawData % file_align != 0):
                            alignment_mismatch_count += 1
                        if sect_align and (section.VirtualSize % sect_align != 0):
                            alignment_mismatch_count += 1
                    except Exception:
                        pass
                except Exception:
                    pass
            features['section_names_count'] = len(section_names)
            features['section_total_size'] = sum(section_sizes)
            features['section_total_vsize'] = sum(section_vsizes)
            features['avg_section_size'] = np.mean(section_sizes) if section_sizes else 0
            features['avg_section_vsize'] = np.mean(section_vsizes) if section_vsizes else 0
            features['max_section_size'] = np.max(section_sizes) if section_sizes else 0
            features['min_section_size'] = np.min(section_sizes) if section_sizes else 0
            if section_entropies:
                features['section_entropy_avg'] = float(np.mean(section_entropies))
                features['section_entropy_min'] = float(np.min(section_entropies))
                features['section_entropy_max'] = float(np.max(section_entropies))
                features['section_entropy_std'] = float(np.std(section_entropies))
                high_entropy_secs = sum(1 for e in section_entropies if e > ENTROPY_HIGH_THRESHOLD)
                features['packed_sections_ratio'] = high_entropy_secs / (len(section_entropies) + 1)
            else:
                features['section_entropy_avg'] = 0.0
                features['section_entropy_min'] = 0.0
                features['section_entropy_max'] = 0.0
                features['section_entropy_std'] = 0.0
                features['packed_sections_ratio'] = 0.0
            features['code_section_ratio'] = code_section_size / (features['section_total_size'] + 1)
            features['data_section_ratio'] = data_section_size / (features['section_total_size'] + 1)
            features['code_vsize_ratio'] = code_section_vsize / (features['section_total_vsize'] + 1)
            features['data_vsize_ratio'] = data_section_vsize / (features['section_total_vsize'] + 1)
            features['executable_sections_count'] = executable_sections_count
            features['writable_sections_count'] = writable_sections_count
            features['readable_sections_count'] = readable_sections_count
            features['executable_sections_ratio'] = executable_sections_count / (len(section_names) + 1)
            features['writable_sections_ratio'] = writable_sections_count / (len(section_names) + 1)
            features['readable_sections_ratio'] = readable_sections_count / (len(section_names) + 1)
            features['rwx_sections_count'] = rwx_sections_count
            features['rwx_sections_ratio'] = rwx_sections_count / (len(section_names) + 1)
            features['non_standard_executable_sections_count'] = non_standard_executable_sections_count
            features['executable_writable_sections'] = executable_writable_sections
            features['executable_code_density'] = code_section_size / (features['section_total_size'] + 1)
            features['alignment_mismatch_count'] = alignment_mismatch_count
            features['alignment_mismatch_ratio'] = alignment_mismatch_count / ((len(section_names) * 2) + 1)
            try:
                max_end = max((getattr(s, 'PointerToRawData', 0) + getattr(s, 'SizeOfRawData', 0)) for s in pe.sections) if hasattr(pe, 'sections') else 0
            except Exception:
                max_end = 0
            trailing_size = max(0, file_size - max_end)
            features['trailing_data_size'] = trailing_size
            features['trailing_data_ratio'] = trailing_size / (file_size + 1)
            features['has_large_trailing_data'] = 1 if trailing_size >= LARGE_TRAILING_DATA_SIZE else 0
            try:
                if trailing_size >= OVERLAY_ENTROPY_MIN_SIZE:
                    with open(valid_path, 'rb') as f:
                        f.seek(max_end)
                        overlay = np.fromfile(f, dtype=np.uint8, count=trailing_size)
                    hist = np.bincount(overlay, minlength=256)
                    prob = hist / overlay.size if overlay.size > 0 else np.array([], dtype=np.float64)
                    prob = prob[prob > 0]
                    oe = float((-np.sum(prob * np.log2(prob)) / 8.0) if prob.size > 0 else 0.0)
                    features['overlay_entropy'] = oe
                    features['overlay_high_entropy_flag'] = 1 if oe > ENTROPY_HIGH_THRESHOLD else 0
                else:
                    features['overlay_entropy'] = 0.0
                    features['overlay_high_entropy_flag'] = 0
            except Exception:
                features['overlay_entropy'] = 0.0
                features['overlay_high_entropy_flag'] = 0
            for sec in COMMON_SECTIONS:
                features[f'has_{sec}_section'] = 1 if any(sec.lower() == n.lower() for n in section_names) else 0
            if section_sizes:
                features['section_size_std'] = np.std(section_sizes)
                features['section_size_cv'] = np.std(section_sizes) / (np.mean(section_sizes) + 1e-8)
            else:
                features['section_size_std'] = 0
                features['section_size_cv'] = 0
            if section_names:
                section_name_lengths = [len(name) for name in section_names]
                features['section_name_avg_length'] = np.mean(section_name_lengths)
                features['section_name_max_length'] = np.max(section_name_lengths)
                features['section_name_min_length'] = np.min(section_name_lengths)
                lower_names = [n.lower() for n in section_names]
                features['has_upx_section'] = 1 if any('upx' in n for n in lower_names) else 0
                features['has_mpress_section'] = 1 if any('mpress' in n for n in lower_names) else 0
                features['has_aspack_section'] = 1 if any('aspack' in n for n in lower_names) else 0
                features['has_themida_section'] = 1 if any('themida' in n for n in lower_names) else 0
                try:
                    hits = 0
                    for kw in PACKER_SECTION_KEYWORDS:
                        if any(kw in n for n in lower_names):
                            hits += 1
                    features['packer_keyword_hits_count'] = float(hits)
                    features['packer_keyword_hits_ratio'] = hits / (len(section_names) + 1)
                except Exception:
                    features['packer_keyword_hits_count'] = 0.0
                    features['packer_keyword_hits_ratio'] = 0.0
                special_char_count = 0
                total_chars = 0
                for name in section_names:
                    total_chars += len(name)
                    for c in name:
                        if not (c.isalnum() or c in '_.'):
                            special_char_count += 1
                features['special_char_ratio'] = special_char_count / (total_chars + 1)
                long_sections = [name for name in section_names if len(name) > 6]
                short_sections = [name for name in section_names if len(name) < 3]
                features['long_sections_count'] = len(long_sections)
                features['short_sections_count'] = len(short_sections)
                features['long_sections_ratio'] = len(long_sections) / (len(section_names) + 1)
                features['short_sections_ratio'] = len(short_sections) / (len(section_names) + 1)
            else:
                features['section_name_avg_length'] = 0
                features['section_name_max_length'] = 0
                features['section_name_min_length'] = 0
                features['has_upx_section'] = 0
                features['has_mpress_section'] = 0
                features['has_aspack_section'] = 0
                features['has_themida_section'] = 0
                features['special_char_ratio'] = 0
                features['long_sections_count'] = 0
                features['short_sections_count'] = 0
                features['long_sections_ratio'] = 0
                features['short_sections_ratio'] = 0
        else:
            features['section_name_avg_length'] = 0
            features['section_name_max_length'] = 0
            features['section_name_min_length'] = 0
            features['trailing_data_size'] = 0
            features['trailing_data_ratio'] = 0
            features['has_large_trailing_data'] = 0
            features['max_section_size'] = 0
            features['min_section_size'] = 0
            features['code_section_ratio'] = 0
            features['data_section_ratio'] = 0
            features['code_vsize_ratio'] = 0
            features['data_vsize_ratio'] = 0
            features['section_size_std'] = 0
            features['section_size_cv'] = 0
            features['executable_sections_count'] = 0
            features['writable_sections_count'] = 0
            features['readable_sections_count'] = 0
            features['executable_sections_ratio'] = 0
            features['writable_sections_ratio'] = 0
            features['readable_sections_ratio'] = 0
            features['rwx_sections_count'] = 0
            features['rwx_sections_ratio'] = 0.0
            features['non_standard_executable_sections_count'] = 0
            features['executable_writable_sections'] = 0
            features['executable_code_density'] = 0
            for sec in COMMON_SECTIONS:
                features[f'has_{sec}_section'] = 0
            features['special_char_ratio'] = 0
            features['long_sections_count'] = 0
            features['short_sections_count'] = 0
            features['long_sections_ratio'] = 0
            features['short_sections_ratio'] = 0
        if hasattr(pe.OPTIONAL_HEADER, 'Subsystem'):
            features['subsystem'] = pe.OPTIONAL_HEADER.Subsystem
        else:
            features['subsystem'] = 0
        if hasattr(pe.OPTIONAL_HEADER, 'DllCharacteristics'):
            features['dll_characteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
            features['has_nx_compat'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x100 else 0
            features['has_aslr'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x40 else 0
            features['has_seh'] = 1 if not (pe.OPTIONAL_HEADER.DllCharacteristics & 0x400) else 0
            features['has_guard_cf'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x4000 else 0
        else:
            features['dll_characteristics'] = 0
            features['has_nx_compat'] = 0
            features['has_aslr'] = 0
            features['has_seh'] = 0
            features['has_guard_cf'] = 0
        features['has_resources'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') else 0
        try:
            if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') and hasattr(pe.DIRECTORY_ENTRY_RESOURCE, 'entries'):
                def count_resource_entries(entries):
                    cnt = 0
                    for e in entries:
                        try:
                            if hasattr(e, 'directory') and hasattr(e.directory, 'entries'):
                                cnt += count_resource_entries(e.directory.entries)
                            elif hasattr(e, 'data'):
                                cnt += 1
                        except Exception:
                            pass
                    return cnt
                features['resources_count'] = count_resource_entries(pe.DIRECTORY_ENTRY_RESOURCE.entries)
            else:
                features['resources_count'] = 0
        except Exception:
            features['resources_count'] = 0
        features['has_debug_info'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') else 0
        features['has_tls'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_TLS') else 0
        try:
            tls_cnt = 0
            if hasattr(pe, 'DIRECTORY_ENTRY_TLS') and hasattr(pe.DIRECTORY_ENTRY_TLS, 'struct'):
                addr = getattr(pe.DIRECTORY_ENTRY_TLS.struct, 'AddressOfCallBacks', 0)
                tls_cnt = 1 if addr else 0
            features['tls_callbacks_count'] = tls_cnt
        except Exception:
            features['tls_callbacks_count'] = 0
        features['has_relocs'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') else 0
        try:
            if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC'):
                blocks = pe.DIRECTORY_ENTRY_BASERELOC
                features['reloc_blocks_count'] = len(blocks)
                total_entries = 0
                for b in blocks:
                    try:
                        total_entries += len(getattr(b, 'entries', []))
                    except Exception:
                        pass
                features['reloc_entries_count'] = total_entries
            else:
                features['reloc_blocks_count'] = 0
                features['reloc_entries_count'] = 0
        except Exception:
            features['reloc_blocks_count'] = 0
            features['reloc_entries_count'] = 0
        features['has_exceptions'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_EXCEPTION') else 0
        try:
            tds = getattr(pe.FILE_HEADER, 'TimeDateStamp', 0)
            features['timestamp'] = int(tds) if tds else 0
            from datetime import datetime
            features['timestamp_year'] = datetime.utcfromtimestamp(int(tds)).year if tds else 0
        except Exception:
            features['timestamp'] = 0
            features['timestamp_year'] = 0
        try:
            ep = getattr(pe.OPTIONAL_HEADER, 'AddressOfEntryPoint', 0)
            features['entry_point_ratio'] = float(ep) / float(file_size + 1)
            try:
                ep_flag = 0
                if hasattr(pe, 'sections'):
                    for section in pe.sections:
                        va = getattr(section, 'VirtualAddress', 0)
                        vs = getattr(section, 'Misc_VirtualSize', getattr(section, 'VirtualSize', 0))
                        if ep >= va and ep < va + vs:
                            name = section.Name.decode('utf-8', 'ignore').strip('\x00').lower()
                            common = {'.text','text','.code'}
                            ep_flag = 0 if name in common else 1
                            break
                features['entry_in_nonstandard_section_flag'] = ep_flag
            except Exception:
                features['entry_in_nonstandard_section_flag'] = 0
        except Exception:
            features['entry_point_ratio'] = 0.0
            features['entry_in_nonstandard_section_flag'] = 0
        try:
            sec_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_SECURITY']]
            sig_size = getattr(sec_dir, 'Size', 0)
            features['has_signature'] = 1 if sig_size and sig_size > 0 else 0
            features['signature_size'] = sig_size if sig_size else 0
            try:
                va = getattr(sec_dir, 'VirtualAddress', 0)
                sz = getattr(sec_dir, 'Size', 0)
                if va and sz:
                    with open(valid_path, 'rb') as f:
                        f.seek(va)
                        blob = f.read(sz)
                    has_st = (b'signingTime' in blob) or (b'1.2.840.113549.1.9.5' in blob)
                    features['signature_has_signing_time'] = 1 if has_st else 0
                else:
                    features['signature_has_signing_time'] = 0
            except Exception:
                features['signature_has_signing_time'] = 0
        except Exception:
            features['has_signature'] = 0
            features['signature_size'] = 0
            features['signature_has_signing_time'] = 0
        version_info_present = 0
        company_name_len = 0
        product_name_len = 0
        file_version_len = 0
        original_filename_len = 0
        try:
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']])
            if hasattr(pe, 'FileInfo'):
                for fi in pe.FileInfo:
                    if hasattr(fi, 'StringTable'):
                        for st in fi.StringTable:
                            if hasattr(st, 'entries'):
                                version_info_present = 1
                                for k, v in st.entries.items():
                                    key = k.strip().lower()
                                    val = v.strip() if isinstance(v, str) else ''
                                    if key == 'companyname':
                                        company_name_len = max(company_name_len, len(val))
                                    elif key == 'productname':
                                        product_name_len = max(product_name_len, len(val))
                                    elif key == 'fileversion':
                                        file_version_len = max(file_version_len, len(val))
                                    elif key == 'originalfilename':
                                        original_filename_len = max(original_filename_len, len(val))
        except Exception:
            pass
        features['version_info_present'] = version_info_present
        features['company_name_len'] = company_name_len
        features['product_name_len'] = product_name_len
        features['file_version_len'] = file_version_len
        features['original_filename_len'] = original_filename_len
        try:
            pe_header_size = pe.OPTIONAL_HEADER.SizeOfHeaders
            features['pe_header_size'] = pe_header_size
            features['header_size_ratio'] = pe_header_size / (file_size + 1)
        except Exception:
            features['pe_header_size'] = 0
            features['header_size_ratio'] = 0
        try:
            ck = getattr(pe.OPTIONAL_HEADER, 'CheckSum', 0)
            features['checksum_zero_flag'] = 1 if not ck else 0
        except Exception:
            features['checksum_zero_flag'] = 0
    except Exception:
        default_keys = [
            'sections_count','symbols_count','imports_count','exports_count','unique_imports','unique_dlls','unique_apis',
            'section_names_count','section_total_size','section_total_vsize','avg_section_size','avg_section_vsize',
            'section_entropy_avg','section_entropy_min','section_entropy_max','section_entropy_std','packed_sections_ratio','overlay_entropy','overlay_high_entropy_flag','packer_keyword_hits_count','packer_keyword_hits_ratio',
            'subsystem','dll_characteristics','code_section_ratio','data_section_ratio','code_vsize_ratio','data_vsize_ratio',
            'has_nx_compat','has_aslr','has_seh','has_guard_cf','has_resources','has_debug_info','has_tls','has_relocs',
            'has_exceptions','dll_name_avg_length','dll_name_max_length','dll_name_min_length','section_name_avg_length',
            'section_name_max_length','section_name_min_length','export_name_avg_length','export_name_max_length',
            'export_name_min_length','max_section_size','min_section_size','long_sections_count','short_sections_count',
            'section_size_std','section_size_cv','executable_writable_sections','file_entropy_avg','file_entropy_min',
            'file_entropy_max','file_entropy_range','zero_byte_ratio','printable_byte_ratio','trailing_data_size',
            'trailing_data_ratio','imported_system_dlls_count','exports_density','has_large_trailing_data','pe_header_size',
            'header_size_ratio','file_entropy_std','file_entropy_q25','file_entropy_q75','file_entropy_median',
            'high_entropy_ratio','low_entropy_ratio','entropy_change_rate','entropy_change_std','executable_sections_count',
            'writable_sections_count','readable_sections_count','executable_sections_ratio','writable_sections_ratio',
            'readable_sections_ratio','executable_code_density','non_standard_executable_sections_count','rwx_sections_count',
            'rwx_sections_ratio','special_char_ratio','long_sections_ratio','short_sections_ratio','dll_imports_entropy',
            'api_imports_entropy','imported_system_dlls_ratio','resources_count','alignment_mismatch_count','alignment_mismatch_ratio','entry_point_ratio',
            'syscall_api_ratio','import_ordinal_only_count','import_ordinal_only_ratio','avg_imports_per_dll','exports_name_ratio','entry_in_nonstandard_section_flag','tls_callbacks_count','reloc_blocks_count','reloc_entries_count','checksum_zero_flag','api_network_ratio','api_process_ratio','api_filesystem_ratio','api_registry_ratio'
        ]
        for key in default_keys:
            features[key] = 0
    finally:
        if pe is not None:
            try:
                pe.close()
            except Exception:
                pass
    return features

def extract_lightweight_pe_features(file_path):
    feature_vector = np.zeros(256, dtype=np.float32)
    pe = None
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            return feature_vector
        pe = pefile.PE(valid_path, fast_load=True)
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                if entry.dll:
                    dll_name = entry.dll.decode('utf-8').lower()
                    dll_hash = int(hashlib.sha256(dll_name.encode('utf-8')).hexdigest(), 16)
                    feature_vector[dll_hash % 128] = 1
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        api_name = imp.name.decode('utf-8')
                        api_hash = int(hashlib.sha256(api_name.encode('utf-8')).hexdigest(), 16)
                        feature_vector[128 + (api_hash % 128)] = 1
        if hasattr(pe, 'sections'):
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', 'ignore').strip('\x00')
                section_hash = int(hashlib.sha256(section_name.encode('utf-8')).hexdigest(), 16)
                feature_vector[(section_hash % 32) + 224] = 1
        norm = np.linalg.norm(feature_vector)
        if norm > 0 and not np.isnan(norm):
            feature_vector /= norm
        if pe is not None:
            try:
                pe.close()
            except Exception:
                pass
        return feature_vector
    except Exception:
        if pe is not None:
            try:
                pe.close()
            except Exception:
                pass
        return feature_vector

PE_FEATURE_ORDER = [
    'size','log_size','sections_count','symbols_count','imports_count','exports_count',
    'unique_imports','unique_dlls','unique_apis','section_names_count','section_total_size',
    'section_total_vsize','avg_section_size','avg_section_vsize','section_entropy_avg','section_entropy_min','section_entropy_max','section_entropy_std','packed_sections_ratio','subsystem','dll_characteristics',
    'code_section_ratio','data_section_ratio','code_vsize_ratio','data_vsize_ratio',
    'has_nx_compat','has_aslr','has_seh','has_guard_cf','has_resources','has_debug_info',
    'has_tls','has_relocs','has_exceptions','dll_name_avg_length','dll_name_max_length',
    'dll_name_min_length','section_name_avg_length','section_name_max_length','section_name_min_length',
    'export_name_avg_length','export_name_max_length','export_name_min_length','max_section_size',
    'min_section_size','long_sections_count','short_sections_count','section_size_std','section_size_cv',
    'executable_writable_sections','file_entropy_avg','file_entropy_min','file_entropy_max','file_entropy_range',
    'zero_byte_ratio','printable_byte_ratio','trailing_data_size','trailing_data_ratio','imported_system_dlls_count',
    'exports_density','has_large_trailing_data','pe_header_size','header_size_ratio','file_entropy_std',
    'file_entropy_q25','file_entropy_q75','file_entropy_median','high_entropy_ratio','low_entropy_ratio',
    'entropy_change_rate','entropy_change_std','executable_sections_count','writable_sections_count',
    'readable_sections_count','executable_sections_ratio','writable_sections_ratio','readable_sections_ratio',
    'executable_code_density','non_standard_executable_sections_count','rwx_sections_count','rwx_sections_ratio',
    'special_char_ratio','long_sections_ratio','short_sections_ratio','dll_imports_entropy','api_imports_entropy',
    'imported_system_dlls_ratio','resources_count','alignment_mismatch_count','alignment_mismatch_ratio','entry_point_ratio',
    'syscall_api_ratio','import_ordinal_only_count','import_ordinal_only_ratio','avg_imports_per_dll','exports_name_ratio','entry_in_nonstandard_section_flag','tls_callbacks_count','reloc_blocks_count','reloc_entries_count','checksum_zero_flag','api_network_ratio','api_process_ratio','api_filesystem_ratio','api_registry_ratio','overlay_entropy','overlay_high_entropy_flag','packer_keyword_hits_count','packer_keyword_hits_ratio'
]

def extract_combined_pe_features(file_path):
    native_vector = _extract_combined_pe_features_native(file_path)
    if native_vector is not None and native_vector.shape[0] == PE_FEATURE_VECTOR_DIM:
        return native_vector
    lightweight_features = extract_lightweight_pe_features(file_path)
    enhanced_features = extract_enhanced_pe_features(file_path)
    file_attrs = extract_file_attributes(file_path)
    all_features = {}
    all_features.update(enhanced_features)
    all_features.update(file_attrs)
    combined_vector = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
    combined_vector[:LIGHTWEIGHT_FEATURE_DIM] = lightweight_features * LIGHTWEIGHT_FEATURE_SCALE
    feature_order = list(PE_FEATURE_ORDER)
    for sec in COMMON_SECTIONS:
        feature_order.append(f'has_{sec}_section')
    feature_order.extend([
        'has_signature','signature_size','signature_has_signing_time','version_info_present','company_name_len','product_name_len','file_version_len','original_filename_len',
        'has_upx_section','has_mpress_section','has_aspack_section','has_themida_section','timestamp','timestamp_year'
    ])
    for i, key in enumerate(feature_order):
        if 256 + i >= PE_FEATURE_VECTOR_DIM:
            break
        if key in all_features:
            val = all_features[key]
            if key == 'log_size' and isinstance(val, (int, float)):
                val = val / np.log(SIZE_NORM_MAX)
            elif 'size' in key and isinstance(val, (int, float)):
                val = val / SIZE_NORM_MAX
            elif key == 'timestamp' and isinstance(val, (int, float)):
                val = val / TIMESTAMP_MAX
            elif key == 'timestamp_year' and isinstance(val, (int, float)):
                val = (val - TIMESTAMP_YEAR_BASE) / (TIMESTAMP_YEAR_MAX - TIMESTAMP_YEAR_BASE)
            elif key.startswith('has_') and isinstance(val, (int, float)):
                val = float(val)
            combined_vector[256 + i] = val * 0.8 if isinstance(val, (int, float)) else 0
    norm = np.linalg.norm(combined_vector)
    if norm > 0 and not np.isnan(norm):
        combined_vector /= norm
    else:
        combined_vector = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
    try:
        mapped_keys = min(len(feature_order), PE_FEATURE_VECTOR_DIM - LIGHTWEIGHT_FEATURE_DIM)
        truncated_flag = 1 if len(feature_order) > (PE_FEATURE_VECTOR_DIM - LIGHTWEIGHT_FEATURE_DIM) else 0
        padded_flag = 1 if len(feature_order) < (PE_FEATURE_VECTOR_DIM - LIGHTWEIGHT_FEATURE_DIM) else 0
        #print(f"[+] PE特征维度={PE_FEATURE_VECTOR_DIM}，已映射键数={mapped_keys}，截断={truncated_flag}，填充={padded_flag}")
    except Exception:
        pass
    return combined_vector

def extract_features_in_memory(input_file_path, max_file_size=DEFAULT_MAX_FILE_SIZE):
    byte_sequence, orig_len = extract_byte_sequence(input_file_path, max_file_size)
    if byte_sequence is None:
        return None, None, 0
    pe_features = extract_combined_pe_features(input_file_path)
    return byte_sequence, pe_features, orig_len
''')

_register_embedded("utils", r'''pass
''', is_package=True)

_register_embedded("utils.logging_utils", r'''import logging

def get_logger(name='kolo'):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
''')

_register_embedded("utils.path_utils", r'''import os

def validate_path(path):
    if not path:
        return None
    normalized_path = os.path.normpath(path)
    if '\0' in normalized_path:
        return None
    if not os.path.exists(normalized_path):
        return None
    return normalized_path
''')

_register_embedded("models", r'''pass
''', is_package=True)

_register_embedded("models.family_classifier", r'''import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.config import FAMILY_THRESHOLD_PERCENTILE, FAMILY_THRESHOLD_MULTIPLIER

class FamilyClassifier:
    def __init__(self):
        self.centroids = {}
        self.thresholds = {}
        self.family_names = {}
        self.scaler = None

    def fit(self, features, labels, family_names_map):
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue
            mask = labels == label
            cluster_features = features_scaled[mask]
            centroid = np.mean(cluster_features, axis=0)
            self.centroids[int(label)] = centroid
            dists = np.linalg.norm(cluster_features - centroid, axis=1)
            limit_dist = np.percentile(dists, FAMILY_THRESHOLD_PERCENTILE) if len(dists) > 0 else 0
            self.thresholds[int(label)] = limit_dist * FAMILY_THRESHOLD_MULTIPLIER if limit_dist > 0 else 1.0
            self.family_names[int(label)] = family_names_map.get(int(label), f"Family_{label}")

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'centroids': self.centroids,
                'thresholds': self.thresholds,
                'family_names': self.family_names,
                'scaler': self.scaler
            }, f)

    def load(self, path):
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.centroids = data['centroids']
                self.thresholds = data['thresholds']
                self.family_names = data['family_names']
                self.scaler = data.get('scaler')
            return True
        except Exception:
            return False

    def predict(self, feature_vector):
        if not self.centroids:
            return None, "Model_Not_Loaded", True
        if self.scaler:
            feature_vector = self.scaler.transform([feature_vector])[0]
        min_dist = float('inf')
        best_label = None
        for label, centroid in self.centroids.items():
            dist = np.linalg.norm(feature_vector - centroid)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        if best_label is not None:
            threshold = self.thresholds[best_label]
            if min_dist <= threshold:
                return best_label, self.family_names[best_label], False
        return None, "New_Unknown_Family", True
''')

_register_embedded("models.gating", r'''import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatingMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        # return logits
        return x

class GatingTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead=4, num_layers=2):
        super(GatingTransformer, self).__init__()
        # Project input to hidden_dim for transformer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Transformer expects sequence, so we might need to reshape or project
        # Here we treat the feature vector as a single token embedding if we just project it, 
        # but to use attention we usually need a sequence. 
        # For simplicity, let's just project and pass through encoder as (batch, 1, hidden)
        
        x = self.embedding(x) # (batch, hidden)
        x = x.unsqueeze(1)    # (batch, 1, hidden)
        x = self.transformer_encoder(x) # (batch, 1, hidden)
        x = x.squeeze(1)      # (batch, hidden)
        x = self.fc(x)        # (batch, output)
        return x

def create_gating_model(model_type, input_dim, hidden_dim, output_dim):
    if model_type == 'mlp':
        return GatingMLP(input_dim, hidden_dim, output_dim)
    elif model_type == 'transformer':
        return GatingTransformer(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown gating model type: {model_type}")
''')

_register_embedded("models.routing_model", r'''import os
import numpy as np
import lightgbm as lgb
from features.extractor_in_memory import PE_FEATURE_ORDER
from config.config import (
    GATING_MODE, EXPERT_NORMAL_MODEL_PATH, EXPERT_PACKED_MODEL_PATH, PE_FEATURE_VECTOR_DIM,
    PACKED_SECTIONS_RATIO_THRESHOLD, PACKER_KEYWORD_HITS_THRESHOLD
)

class RoutingModel:
    def __init__(self):
        self.expert_normal = None
        self.expert_packed = None
        self._idx_packed_sections = self._feature_index('packed_sections_ratio')
        self._idx_packer_hits = self._feature_index('packer_keyword_hits_count')
        self.load_models()

    def load_models(self):
        if GATING_MODE != 'rule':
            print(f"[!] GATING_MODE={GATING_MODE} 非 'rule'，将使用规则门控以避免依赖 torch")
        if os.path.exists(EXPERT_NORMAL_MODEL_PATH):
            print(f"[*] Loading Normal Expert from {EXPERT_NORMAL_MODEL_PATH}")
            self.expert_normal = lgb.Booster(model_file=EXPERT_NORMAL_MODEL_PATH)
        else:
            print(f"[!] Normal expert model not found at {EXPERT_NORMAL_MODEL_PATH}")
        if os.path.exists(EXPERT_PACKED_MODEL_PATH):
            print(f"[*] Loading Packed Expert from {EXPERT_PACKED_MODEL_PATH}")
            self.expert_packed = lgb.Booster(model_file=EXPERT_PACKED_MODEL_PATH)
        else:
            print(f"[!] Packed expert model not found at {EXPERT_PACKED_MODEL_PATH}")

    def predict(self, features):
        x = np.asarray(features)
        routing_decisions = self._rule_gating(x)
        predictions = np.zeros(len(x))
        normal_indices = np.where(routing_decisions == 0)[0]
        packed_indices = np.where(routing_decisions == 1)[0]
        if len(normal_indices) > 0:
            if self.expert_normal:
                X_normal = x[normal_indices]
                pred_normal = self.expert_normal.predict(X_normal)
                predictions[normal_indices] = pred_normal
            else:
                print("[!] Expert Normal not loaded, skipping predictions for normal samples.")
        if len(packed_indices) > 0:
            if self.expert_packed:
                X_packed = x[packed_indices]
                pred_packed = self.expert_packed.predict(X_packed)
                predictions[packed_indices] = pred_packed
            else:
                print("[!] Expert Packed not loaded, skipping predictions for packed samples.")
        return predictions, routing_decisions

    def _feature_index(self, key):
        try:
            return 256 + PE_FEATURE_ORDER.index(key)
        except ValueError:
            return None

    def _rule_gating(self, x):
        start = x.shape[1] - PE_FEATURE_VECTOR_DIM
        p = self._idx_packed_sections
        k = self._idx_packer_hits
        ps = x[:, start + p] if p is not None else np.zeros(len(x))
        kh = x[:, start + k] if k is not None else np.zeros(len(x))
        return np.logical_or(
            ps > PACKED_SECTIONS_RATIO_THRESHOLD,
            kh > PACKER_KEYWORD_HITS_THRESHOLD
        ).astype(int)

    def get_routing_stats(self, routing_decisions):
        total = len(routing_decisions)
        packed_count = np.sum(routing_decisions)
        normal_count = total - packed_count
        return {
            'total': total,
            'normal': normal_count,
            'packed': packed_count,
            'packed_ratio': packed_count / total if total > 0 else 0
        }
''')

_register_embedded("training", r'''pass
''', is_package=True)

_register_embedded("training.feature_io", r'''import os
import numpy as np
import csv
import pandas as pd

def save_features(X, y, files, save_dir):
    print("[*] Saving features to file...")
    os.makedirs(save_dir, exist_ok=True)
    features_path = os.path.join(save_dir, 'features.npz')
    np.savez_compressed(features_path, X=X, y=y, files=files)
    csv_path = os.path.join(save_dir, 'features.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['filename', 'label'] + [f'feature_{i}' for i in range(X.shape[1])]
        writer.writerow(header)
        for i in range(X.shape[0]):
            row = [files[i], y[i]] + X[i].tolist()
            writer.writerow(row)
    print(f"[+] Features saved to: {features_path}")
    print(f"[+] CSV format features saved to: {csv_path}")

def save_features_to_csv(X, y, files, output_path):
    print(f"[*] Saving features to {output_path}...")
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df_data = { 'filename': files, 'label': y }
    for i, feature_name in enumerate(feature_names):
        df_data[feature_name] = X[:, i]
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)
    print(f"[+] Features saved to: {output_path}")

def save_features_to_pickle(X, y, files, output_path):
    print(f"[*] Saving features to {output_path}...")
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df_data = { 'filename': files, 'label': y }
    for i, feature_name in enumerate(feature_names):
        df_data[feature_name] = X[:, i]
    df = pd.DataFrame(df_data)
    df.to_pickle(output_path)
    print(f"[+] Features saved to: {output_path}")
''')

_register_embedded("training.evaluate", r'''import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import MODEL_EVAL_FIG_DIR, MODEL_EVAL_FIG_PATH, EVAL_HIST_BINS, PREDICTION_THRESHOLD, EVAL_FONT_FAMILY, MODEL_EVAL_AUC_PATH

def evaluate_model(model, X_test, y_test, files_test=None):
    print("[*] Evaluating model...")
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > PREDICTION_THRESHOLD).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[+] Accuracy: {accuracy:.4f}")
    print("\n[*] Classification report:")
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = []
    if 0 in unique_labels:
        target_names.append('Benign')
    if 1 in unique_labels:
        target_names.append('Malicious')
    if len(unique_labels) > 1:
        print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels))
    else:
        label_name = 'Benign' if unique_labels[0] == 0 else 'Malicious'
        print(f"All samples in test set belong to '{label_name}' category")
        precision = precision_score(y_test, y_pred, zero_division=0)
        print(f"Precision: {precision:.4f}")
    false_positives = []
    if files_test is not None:
        fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
        false_positives = [files_test[i] for i in fp_indices]
        print(f"\n[*] Detected {len(false_positives)} false positive samples:")
        for fp_file in false_positives[:10]:
            print(f"    - {fp_file}")
        if len(false_positives) > 10:
            print(f"    ... and {len(false_positives) - 10} more false positive samples")
    plt.rcParams['font.sans-serif'] = EVAL_FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False
    if len(unique_labels) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=target_names, yticklabels=target_names)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        if 0 in unique_labels:
            ax2.hist(y_pred_proba[y_test == 0], bins=EVAL_HIST_BINS, alpha=0.7, label='Benign', color='blue')
        if 1 in unique_labels:
            ax2.hist(y_pred_proba[y_test == 1], bins=EVAL_HIST_BINS, alpha=0.7, label='Malicious', color='red')
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Sample Count')
        ax2.set_title('Prediction Probability Distribution')
        ax2.legend()
        plt.tight_layout()
        os.makedirs(MODEL_EVAL_FIG_DIR, exist_ok=True)
        plt.savefig(MODEL_EVAL_FIG_PATH, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[+] Evaluation charts saved to: {MODEL_EVAL_FIG_PATH}")
        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_value = roc_auc_score(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC={auc_value:.4f}', color='darkorange')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            os.makedirs(MODEL_EVAL_FIG_DIR, exist_ok=True)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(MODEL_EVAL_AUC_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[+] ROC AUC curve saved to: {MODEL_EVAL_AUC_PATH}")
        except Exception:
            pass
    else:
        print("[*] Skipping visualization chart generation as test set contains only one category")
    return accuracy, false_positives
''')

_register_embedded("training.incremental", r'''import numpy as np
import lightgbm as lgb
import multiprocessing
from config.config import LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ, LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF, LIGHTGBM_NUM_THREADS_MAX, DEFAULT_LIGHTGBM_NUM_LEAVES, DEFAULT_LIGHTGBM_LEARNING_RATE, DEFAULT_INCREMENTAL_EARLY_STOPPING

def incremental_train_lightgbm_model(existing_model, X_train, y_train, X_val, y_val, false_positive_files=None, files_train=None, num_boost_round=100, early_stopping_rounds=DEFAULT_INCREMENTAL_EARLY_STOPPING):
    print("[*] Performing incremental reinforcement training...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    if false_positive_files is not None and files_train is not None:
        print(f"[*] Detected {len(false_positive_files)} false positive samples, increasing their training weights")
        weights = np.ones(len(X_train), dtype=np.float32)
        false_positive_count = 0
        for i, file in enumerate(files_train):
            if file in false_positive_files:
                weights[i] = 10.0
                false_positive_count += 1
        print(f"[+] Identified {false_positive_count} false positive samples, adjusted weights")
        train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = existing_model.params if existing_model else {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': DEFAULT_LIGHTGBM_NUM_LEAVES,
        'learning_rate': DEFAULT_LIGHTGBM_LEARNING_RATE,
        'feature_fraction': LIGHTGBM_FEATURE_FRACTION,
        'bagging_fraction': LIGHTGBM_BAGGING_FRACTION,
        'bagging_freq': LIGHTGBM_BAGGING_FREQ,
        'min_gain_to_split': LIGHTGBM_MIN_GAIN_TO_SPLIT,
        'min_data_in_leaf': LIGHTGBM_MIN_DATA_IN_LEAF,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), LIGHTGBM_NUM_THREADS_MAX)
    }
    if existing_model:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['validation'],
            num_boost_round=num_boost_round,
            init_model=existing_model,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(10)]
        )
    else:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['validation'],
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(10)]
        )
    print("[+] Incremental reinforcement training completed")
    return model
''')

_register_embedded("training.model_io", r'''import os
import lightgbm as lgb

def save_model(model, model_path):
    model.save_model(model_path)
    print(f"[+] Model saved to: {model_path}")

def load_existing_model(model_path):
    if os.path.exists(model_path):
        print(f"[*] Loading existing model: {model_path}")
        try:
            model = lgb.Booster(model_file=model_path)
            print("[+] Existing model loaded successfully")
            return model
        except Exception as e:
            print(f"[!] Model loading failed: {e}")
            return None
    else:
        print(f"[-] Existing model not found: {model_path}")
        return None
''')

_register_embedded("training.train_lightgbm", r'''import numpy as np
import lightgbm as lgb
import multiprocessing
from config.config import WARMUP_ROUNDS, WARMUP_START_LR, LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ, LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF, LIGHTGBM_NUM_THREADS_MAX, FP_WEIGHT_BASE, FP_WEIGHT_GROWTH_PER_ITER, FP_WEIGHT_MAX, DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_LIGHTGBM_LEARNING_RATE, DEFAULT_LIGHTGBM_NUM_LEAVES

def warmup_scheduler(warmup_rounds=WARMUP_ROUNDS, start_lr=WARMUP_START_LR, target_lr=0.05):
    def callback(env):
        if env.iteration < warmup_rounds:
            lr = start_lr + (target_lr - start_lr) * (env.iteration / warmup_rounds)
            env.model.params['learning_rate'] = lr
            if env.iteration % 20 == 0:
                 print(f"[*] Warmup: Iteration {env.iteration}, LR: {lr:.6f}")
    return callback

def train_lightgbm_model(X_train, y_train, X_val, y_val, false_positive_files=None, files_train=None, iteration=1, num_boost_round=5000, init_model=None, params_override=None):
    print(f"[*] Training LightGBM model (Round {iteration})...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    if false_positive_files is not None and files_train is not None:
        print(f"[*] Detected {len(false_positive_files)} false positive samples, increasing their training weights")
        weights = np.ones(len(X_train), dtype=np.float32)
        false_positive_count = 0
        weight_factor = min(FP_WEIGHT_BASE + iteration * FP_WEIGHT_GROWTH_PER_ITER, FP_WEIGHT_MAX)
        print(f"[*] Current false positive weight factor: {weight_factor}")
        for i, file in enumerate(files_train):
            if file in false_positive_files:
                weights[i] = weight_factor
                false_positive_count += 1
        print(f"[+] Identified {false_positive_count} false positive samples, adjusted weights")
        train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    learning_rate = DEFAULT_LIGHTGBM_LEARNING_RATE
    num_leaves = DEFAULT_LIGHTGBM_NUM_LEAVES
    feature_fraction = LIGHTGBM_FEATURE_FRACTION
    bagging_fraction = LIGHTGBM_BAGGING_FRACTION
    bagging_freq = LIGHTGBM_BAGGING_FREQ
    min_gain_to_split = LIGHTGBM_MIN_GAIN_TO_SPLIT
    min_data_in_leaf = LIGHTGBM_MIN_DATA_IN_LEAF
    if params_override:
        learning_rate = params_override.get('learning_rate', learning_rate)
        num_leaves = params_override.get('num_leaves', num_leaves)
        feature_fraction = params_override.get('feature_fraction', feature_fraction)
        bagging_fraction = params_override.get('bagging_fraction', bagging_fraction)
        bagging_freq = params_override.get('bagging_freq', bagging_freq)
        min_gain_to_split = params_override.get('min_gain_to_split', min_gain_to_split)
        min_data_in_leaf = params_override.get('min_data_in_leaf', min_data_in_leaf)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'min_gain_to_split': min_gain_to_split,
        'min_data_in_leaf': min_data_in_leaf,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), LIGHTGBM_NUM_THREADS_MAX)
    }
    print(f"[*] Current training parameters - Learning rate: {learning_rate:.4f}, Number of leaves: {num_leaves}")
    callbacks = [lgb.early_stopping(DEFAULT_EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50)]
    if iteration == 1 and init_model is None:
        print("[*] Applying Warm Start scheduler...")
        callbacks.append(warmup_scheduler(warmup_rounds=WARMUP_ROUNDS, start_lr=WARMUP_START_LR, target_lr=learning_rate))
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['validation'],
        num_boost_round=num_boost_round,
        init_model=init_model,
        callbacks=callbacks
    )
    return model
''')

_register_embedded("training.export_family_classifier_json", r'''import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def export_family_classifier(pkl_path: Path, out_path: Path) -> None:
    obj = pickle.loads(pkl_path.read_bytes())
    centroids = obj["centroids"]
    thresholds = obj["thresholds"]
    family_names = obj["family_names"]
    scaler = obj.get("scaler", None)

    cluster_ids = sorted(set(centroids.keys()) & set(thresholds.keys()) & set(family_names.keys()))
    centroids_list = [np.asarray(centroids[cid], dtype=np.float32).tolist() for cid in cluster_ids]

    thresholds_list = [float(thresholds[cid]) for cid in cluster_ids]
    family_names_list = [str(family_names[cid]) for cid in cluster_ids]

    scaler_mean = []
    scaler_scale = []
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        scaler_mean = np.asarray(scaler.mean_, dtype=np.float32).tolist()
        scaler_scale = np.asarray(scaler.scale_, dtype=np.float32).tolist()

    out = {
        "cluster_ids": [int(x) for x in cluster_ids],
        "centroids": centroids_list,
        "thresholds": thresholds_list,
        "family_names": family_names_list,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="hdbscan_cluster_results/family_classifier.pkl")
    parser.add_argument("--output", default="hdbscan_cluster_results/family_classifier.json")
    args = parser.parse_args()

    export_family_classifier(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
''')

_register_embedded("training.data_loader", r'''import os
import json
import numpy as np
from tqdm import tqdm

from data.dataset import MalwareDataset
from features.statistics import extract_statistical_features
from config.config import DEFAULT_MAX_FILE_SIZE

def load_dataset(data_dir, metadata_file, max_file_size=DEFAULT_MAX_FILE_SIZE, fast_dev_run=False):
    print("[*] Loading dataset...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    label_map = {}
    for file, label in metadata.items():
        fname_lower = file.lower()
        if ('待加入白名单' in file) or ('whitelist' in fname_lower) or ('benign' in fname_lower) or ('good' in fname_lower) or ('clean' in fname_lower):
            label_map[file] = 0
        elif ('malicious' in fname_lower) or ('virus' in fname_lower) or ('trojan' in fname_lower):
            label_map[file] = 1
        elif label == 'benign' or label == 0:
            label_map[file] = 0
        elif label == 'malicious' or label == 1:
            label_map[file] = 1
        elif label == '待加入白名单':
            label_map[file] = 0
        else:
            label_map[file] = 1
    benign_count = sum(1 for v in label_map.values() if v == 0)
    malicious_count = sum(1 for v in label_map.values() if v == 1)
    print(f"[+] Label distribution: Benign={benign_count}, Malicious={malicious_count}")
    all_files = list(metadata.keys())
    all_labels = [label_map[fname] for fname in all_files]
    if fast_dev_run:
        print("[!] Fast development mode enabled, balancing benign and malicious samples.")
        benign_files = [f for f, label in zip(all_files, all_labels) if label == 0]
        malicious_files = [f for f, label in zip(all_files, all_labels) if label == 1]
        n_samples_per_class = 5000
        selected_benign_files = benign_files[:min(n_samples_per_class, len(benign_files))]
        selected_malicious_files = malicious_files[:min(n_samples_per_class, len(malicious_files))]
        all_files = selected_benign_files + selected_malicious_files
        all_labels = [0] * len(selected_benign_files) + [1] * len(selected_malicious_files)
        print(f"    Benign samples: {len(selected_benign_files)}")
        print(f"    Malicious samples: {len(selected_malicious_files)}")
    print(f"[+] Loaded {len(all_files)} files")
    features_list = []
    labels_list = []
    valid_files = []
    dataset = MalwareDataset(data_dir, all_files, all_labels, max_file_size)
    total_samples = len(dataset)
    progress_desc = "Extracting features"
    from config.config import PE_FEATURE_VECTOR_DIM
    count_ok = 0
    count_padded = 0
    count_truncated = 0
    for i in tqdm(range(total_samples), desc=progress_desc):
        try:
            byte_sequence, pe_features, label, orig_length = dataset[i]
            orig_pe_len = len(pe_features)
            status = 'ok'
            if orig_pe_len != PE_FEATURE_VECTOR_DIM:
                fixed_pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                copy_len = min(orig_pe_len, PE_FEATURE_VECTOR_DIM)
                fixed_pe_features[:copy_len] = pe_features[:copy_len]
                pe_features = fixed_pe_features
                status = 'padded' if orig_pe_len < PE_FEATURE_VECTOR_DIM else 'truncated'
            if status == 'ok':
                count_ok += 1
            elif status == 'padded':
                count_padded += 1
            else:
                count_truncated += 1
            features = extract_statistical_features(byte_sequence, pe_features, orig_length)
            features_list.append(features)
            labels_list.append(label)
            valid_files.append(all_files[i])
        except Exception as e:
            print(f"[!] Error processing file {all_files[i]}: {e}")
            continue
    try:
        X = np.array(features_list, dtype=np.float32)
    except ValueError as e:
        print(f"[!] Feature array shape inconsistency: {e}")
        print("[*] Attempting to manually align feature dimensions...")
        max_features = max(len(f) for f in features_list)
        aligned_features = []
        for f in features_list:
            if len(f) < max_features:
                padded_f = np.zeros(max_features, dtype=np.float32)
                padded_f[:len(f)] = f
                aligned_features.append(padded_f)
            else:
                aligned_features.append(f)
        X = np.array(aligned_features, dtype=np.float32)
    y = np.array(labels_list)
    print(f"[+] Feature extraction completed, feature dimension: {X.shape[1]}")
    print(f"[+] Valid samples: {X.shape[0]}")
    try:
        total = count_ok + count_padded + count_truncated
        if total > 0:
            print(f"[+] PE维度汇总：total={total}，ok={count_ok}，padded={count_padded}，truncated={count_truncated}")
            from config.config import SCAN_OUTPUT_DIR, PE_DIM_SUMMARY_DATASET
            os.makedirs(SCAN_OUTPUT_DIR, exist_ok=True)
            with open(PE_DIM_SUMMARY_DATASET, 'w', encoding='utf-8') as f:
                json.dump({
                    'total': int(total),
                    'ok': int(count_ok),
                    'padded': int(count_padded),
                    'truncated': int(count_truncated),
                    'feature_dim': int(X.shape[1]),
                    'pe_dim': int(PE_FEATURE_VECTOR_DIM)
                }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return X, y, valid_files

def extract_features_from_raw_files(data_dir, output_dir, max_file_size=DEFAULT_MAX_FILE_SIZE, file_extensions=None, label_inference='filename'):
    print(f"[*] Extracting features from raw files: {data_dir}")
    print(f"[*] Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_extensions:
                _, ext = os.path.splitext(file)
                if ext.lower() not in file_extensions:
                    continue
            all_files.append(file_path)
    if not all_files:
        print(f"[!] No files found in raw file directory: {data_dir}")
        return [], []
    print(f"[+] Found {len(all_files)} files in raw file directory")
    try:
        from features.extractor_save import process_file_directory
        print("[+] Successfully imported feature extraction module")
    except ImportError as e:
        print(f"[!] Failed to import feature extraction module: {e}")
        return [], []
    labels = []
    output_files = []
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, data_dir)
        output_file = os.path.join(output_dir, rel_path + '.npz')
        output_files.append(output_file)
        output_subdir = os.path.dirname(output_file)
        os.makedirs(output_subdir, exist_ok=True)
        if label_inference == 'filename':
            file_name = os.path.basename(file_path)
            if 'benign' in file_name.lower() or 'good' in file_name.lower() or 'clean' in file_name.lower():
                labels.append(0)
            else:
                labels.append(1)
        elif label_inference == 'directory':
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if 'benign' in parent_dir.lower() or 'good' in parent_dir.lower() or 'clean' in parent_dir.lower():
                labels.append(0)
            else:
                labels.append(1)
        else:
            labels.append(1)
    print("[*] Starting feature extraction...")
    success_count = 0
    from tqdm import tqdm
    for i, (input_file, output_file) in enumerate(tqdm(zip(all_files, output_files), total=len(all_files), desc="Feature extraction")):
        try:
            process_file_directory(input_file, output_file, max_file_size)
            success_count += 1
        except Exception as e:
            print(f"[!] Error processing file {input_file}: {e}")
            if output_file in output_files:
                idx = output_files.index(output_file)
                output_files.pop(idx)
                labels.pop(idx)
    print(f"[+] Feature extraction completed: {success_count}/{len(all_files)} files processed successfully")
    try:
        from config.config import PE_FEATURE_VECTOR_DIM
        count_ok = 0
        count_padded = 0
        count_truncated = 0
        for of in output_files:
            try:
                with np.load(of) as data:
                    if 'pe_features' in data:
                        pe = data['pe_features']
                        orig_len = pe.shape[0] if hasattr(pe, 'shape') else len(pe)
                        if orig_len == PE_FEATURE_VECTOR_DIM:
                            count_ok += 1
                        elif orig_len < PE_FEATURE_VECTOR_DIM:
                            count_padded += 1
                        else:
                            count_truncated += 1
            except Exception:
                pass
        total = count_ok + count_padded + count_truncated
        if total > 0:
            print(f"[+] 原始批处理PE维度汇总：total={total}，ok={count_ok}，padded={count_padded}，truncated={count_truncated}")
            from config.config import SCAN_OUTPUT_DIR, PE_DIM_SUMMARY_RAW
            os.makedirs(SCAN_OUTPUT_DIR, exist_ok=True)
            with open(PE_DIM_SUMMARY_RAW, 'w', encoding='utf-8') as f:
                json.dump({
                    'total': int(total),
                    'ok': int(count_ok),
                    'padded': int(count_padded),
                    'truncated': int(count_truncated),
                    'pe_dim': int(PE_FEATURE_VECTOR_DIM)
                }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    file_names = [os.path.relpath(f, output_dir) for f in output_files]
    return file_names, labels

def load_incremental_dataset(data_dir, max_file_size=DEFAULT_MAX_FILE_SIZE):
    print(f"[*] Loading dataset from incremental directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"[!] Incremental training directory does not exist: {data_dir}")
        return None, None, None
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                all_files.append(os.path.join(root, file))
    if not all_files:
        print(f"[!] No .npz files found in incremental training directory: {data_dir}")
        return None, None, None
    print(f"[+] Found {len(all_files)} files in incremental directory")
    labels = []
    valid_files = []
    file_names = []
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        if 'benign' in file_name.lower() or 'good' in file_name.lower() or 'clean' in file_name.lower():
            labels.append(0)
        else:
            labels.append(1)
        valid_files.append(file_path)
        file_names.append(file_name)
    features_list = []
    valid_file_names = []
    from config.config import PE_FEATURE_VECTOR_DIM
    count_ok = 0
    count_padded = 0
    count_truncated = 0
    for i, file_path in enumerate(tqdm(valid_files, desc="Extracting incremental features")):
        try:
            with np.load(file_path) as data:
                byte_sequence = data['byte_sequence']
                if 'pe_features' in data:
                    pe_features = data['pe_features']
                    if pe_features.ndim > 1:
                        pe_features = pe_features.flatten()
                else:
                    pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                orig_length = data['orig_length'] if 'orig_length' in data else max_file_size
            if len(byte_sequence) > max_file_size:
                byte_sequence = byte_sequence[:max_file_size]
            else:
                byte_sequence = np.pad(byte_sequence, (0, max_file_size - len(byte_sequence)), 'constant')
            orig_pe_len = len(pe_features)
            status = 'ok'
            if orig_pe_len != PE_FEATURE_VECTOR_DIM:
                fixed_pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                copy_len = min(orig_pe_len, PE_FEATURE_VECTOR_DIM)
                fixed_pe_features[:copy_len] = pe_features[:copy_len]
                pe_features = fixed_pe_features
                status = 'padded' if orig_pe_len < PE_FEATURE_VECTOR_DIM else 'truncated'
            if status == 'ok':
                count_ok += 1
            elif status == 'padded':
                count_padded += 1
            else:
                count_truncated += 1
            features = extract_statistical_features(byte_sequence, pe_features, int(orig_length))
            features_list.append(features)
            valid_file_names.append(file_names[i])
        except Exception as e:
            print(f"[!] Error processing file {file_path}: {e}")
            continue
    if not features_list:
        print("[!] Failed to extract any features from incremental data")
        return None, None, None
    try:
        X = np.array(features_list, dtype=np.float32)
    except ValueError as e:
        print(f"[!] Incremental feature array shape inconsistency: {e}")
        return None, None, None
    y = np.array(labels[:len(features_list)])
    print(f"[+] Incremental feature extraction completed, feature dimension: {X.shape[1]}")
    print(f"[+] Valid samples: {X.shape[0]}")
    try:
        total = count_ok + count_padded + count_truncated
        if total > 0:
            print(f"[+] PE维度汇总：total={total}，ok={count_ok}，padded={count_padded}，truncated={count_truncated}")
            from config.config import SCAN_OUTPUT_DIR, PE_DIM_SUMMARY_INCREMENTAL
            os.makedirs(SCAN_OUTPUT_DIR, exist_ok=True)
            with open(PE_DIM_SUMMARY_INCREMENTAL, 'w', encoding='utf-8') as f:
                json.dump({
                    'total': int(total),
                    'ok': int(count_ok),
                    'padded': int(count_padded),
                    'truncated': int(count_truncated),
                    'feature_dim': int(X.shape[1]),
                    'pe_dim': int(PE_FEATURE_VECTOR_DIM)
                }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return X, y, valid_file_names
''')

_register_embedded("training.automl", r'''import os
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from config.config import (
    FEATURES_PKL_PATH, PROCESSED_DATA_DIR, METADATA_FILE, DEFAULT_MAX_FILE_SIZE,
    DEFAULT_RANDOM_STATE, DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_LIGHTGBM_NUM_LEAVES, DEFAULT_LIGHTGBM_LEARNING_RATE,
    LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ,
    LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF,
    AUTOML_RESULTS_PATH, AUTOML_TRIALS_DEFAULT, AUTOML_CV_FOLDS_DEFAULT, AUTOML_METHOD_DEFAULT,
    AUTOML_METRIC_DEFAULT, AUTOML_ADDITIONAL_METRICS,
    AUTOML_LGBM_NUM_LEAVES_MIN, AUTOML_LGBM_NUM_LEAVES_MAX,
    AUTOML_LGBM_LEARNING_RATE_MIN, AUTOML_LGBM_LEARNING_RATE_MAX,
    AUTOML_LGBM_FEATURE_FRACTION_MIN, AUTOML_LGBM_FEATURE_FRACTION_MAX,
    AUTOML_LGBM_BAGGING_FRACTION_MIN, AUTOML_LGBM_BAGGING_FRACTION_MAX,
    AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN, AUTOML_LGBM_MIN_DATA_IN_LEAF_MAX,
    AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN, AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MAX,
    AUTOML_LGBM_BAGGING_FREQ_MIN, AUTOML_LGBM_BAGGING_FREQ_MAX,
    AUTOML_LGBM_SCALE_POS_WEIGHT_MIN, AUTOML_LGBM_SCALE_POS_WEIGHT_MAX
)
from training.data_loader import load_dataset

def _load_data(use_existing_features=False, max_file_size=DEFAULT_MAX_FILE_SIZE, fast_dev_run=False):
    import pandas as pd
    X = None
    y = None
    if use_existing_features and os.path.exists(FEATURES_PKL_PATH):
        df = pd.read_pickle(FEATURES_PKL_PATH)
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        try:
            feature_cols = sorted(feature_cols, key=lambda c: int(c.split('_')[1]))
        except Exception:
            pass
        X = df[feature_cols]
        y = df['label'].astype(int)
    else:
        X_np, y_np, _ = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE, max_file_size, fast_dev_run=fast_dev_run)
        feature_cols = [f'feature_{i}' for i in range(X_np.shape[1])]
        X = pd.DataFrame(X_np, columns=feature_cols)
        y = pd.Series(y_np)
    return X, y

def _make_baseline_model():
    return lgb.LGBMClassifier(
        objective='binary',
        n_estimators=DEFAULT_NUM_BOOST_ROUND,
        num_leaves=DEFAULT_LIGHTGBM_NUM_LEAVES,
        learning_rate=DEFAULT_LIGHTGBM_LEARNING_RATE,
        feature_fraction=LIGHTGBM_FEATURE_FRACTION,
        bagging_fraction=LIGHTGBM_BAGGING_FRACTION,
        bagging_freq=LIGHTGBM_BAGGING_FREQ,
        min_gain_to_split=LIGHTGBM_MIN_GAIN_TO_SPLIT,
        min_data_in_leaf=LIGHTGBM_MIN_DATA_IN_LEAF,
        subsample=LIGHTGBM_BAGGING_FRACTION,
        subsample_freq=LIGHTGBM_BAGGING_FREQ,
        verbosity=-1,
        random_state=DEFAULT_RANDOM_STATE
    )

def _cv_score(model, X, y, cv_folds, metric):
    if len(np.unique(y)) < 2:
        return 0.0
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    # 支持更多指标
    valid_metrics = {
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    }
    scoring = valid_metrics.get(metric, 'roc_auc')
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return float(np.mean(scores))

def _optuna_tune_lgbm(X, y, cv_folds, trials, metric):
    import optuna
    def objective(trial):
        params = {
            'objective': 'binary',
            'n_estimators': DEFAULT_NUM_BOOST_ROUND,
            'num_leaves': trial.suggest_int('num_leaves', AUTOML_LGBM_NUM_LEAVES_MIN, AUTOML_LGBM_NUM_LEAVES_MAX),
            'learning_rate': trial.suggest_float('learning_rate', AUTOML_LGBM_LEARNING_RATE_MIN, AUTOML_LGBM_LEARNING_RATE_MAX, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', AUTOML_LGBM_FEATURE_FRACTION_MIN, AUTOML_LGBM_FEATURE_FRACTION_MAX),
            'bagging_fraction': trial.suggest_float('bagging_fraction', AUTOML_LGBM_BAGGING_FRACTION_MIN, AUTOML_LGBM_BAGGING_FRACTION_MAX),
            'bagging_freq': trial.suggest_int('bagging_freq', AUTOML_LGBM_BAGGING_FREQ_MIN, AUTOML_LGBM_BAGGING_FREQ_MAX),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN, AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MAX),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN, AUTOML_LGBM_MIN_DATA_IN_LEAF_MAX),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', AUTOML_LGBM_SCALE_POS_WEIGHT_MIN, AUTOML_LGBM_SCALE_POS_WEIGHT_MAX),
            'verbosity': -1,
            'random_state': DEFAULT_RANDOM_STATE
        }
        model = lgb.LGBMClassifier(**params)
        score = _cv_score(model, X, y, cv_folds, metric)
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    best_params = study.best_params
    best_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=DEFAULT_NUM_BOOST_ROUND,
        verbosity=-1,
        random_state=DEFAULT_RANDOM_STATE,
        **best_params
    )
    best_score = _cv_score(best_model, X, y, cv_folds, metric)
    return best_score, best_params

def _hyperopt_tune_lgbm(X, y, cv_folds, trials, metric):
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    space = {
        'num_leaves': hp.quniform('num_leaves', AUTOML_LGBM_NUM_LEAVES_MIN, AUTOML_LGBM_NUM_LEAVES_MAX, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(AUTOML_LGBM_LEARNING_RATE_MIN), np.log(AUTOML_LGBM_LEARNING_RATE_MAX)),
        'feature_fraction': hp.uniform('feature_fraction', AUTOML_LGBM_FEATURE_FRACTION_MIN, AUTOML_LGBM_FEATURE_FRACTION_MAX),
        'bagging_fraction': hp.uniform('bagging_fraction', AUTOML_LGBM_BAGGING_FRACTION_MIN, AUTOML_LGBM_BAGGING_FRACTION_MAX),
        'bagging_freq': hp.quniform('bagging_freq', AUTOML_LGBM_BAGGING_FREQ_MIN, AUTOML_LGBM_BAGGING_FREQ_MAX, 1),
        'min_gain_to_split': hp.uniform('min_gain_to_split', AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN, AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MAX),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN, AUTOML_LGBM_MIN_DATA_IN_LEAF_MAX, 1),
        'scale_pos_weight': hp.uniform('scale_pos_weight', AUTOML_LGBM_SCALE_POS_WEIGHT_MIN, AUTOML_LGBM_SCALE_POS_WEIGHT_MAX)
    }
    def objective(params):
        params_cast = {
            'objective': 'binary',
            'n_estimators': DEFAULT_NUM_BOOST_ROUND,
            'num_leaves': int(params['num_leaves']),
            'learning_rate': float(params['learning_rate']),
            'feature_fraction': float(params['feature_fraction']),
            'bagging_fraction': float(params['bagging_fraction']),
            'bagging_freq': int(params['bagging_freq']),
            'min_gain_to_split': float(params['min_gain_to_split']),
            'min_data_in_leaf': int(params['min_data_in_leaf']),
            'scale_pos_weight': float(params['scale_pos_weight']),
            'verbosity': -1,
            'random_state': DEFAULT_RANDOM_STATE
        }
        model = lgb.LGBMClassifier(**params_cast)
        score = _cv_score(model, X, y, cv_folds, metric)
        loss = -score
        return {'loss': loss, 'status': STATUS_OK}
    trials_obj = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=trials, trials=trials_obj)
    best_params = {
        'num_leaves': int(best['num_leaves']),
        'learning_rate': float(best['learning_rate']),
        'feature_fraction': float(best['feature_fraction']),
        'bagging_fraction': float(best['bagging_fraction']),
        'bagging_freq': int(best['bagging_freq']),
        'min_gain_to_split': float(best['min_gain_to_split']),
        'min_data_in_leaf': int(best['min_data_in_leaf']),
        'scale_pos_weight': float(best['scale_pos_weight'])
    }
    best_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=DEFAULT_NUM_BOOST_ROUND,
        verbosity=-1,
        random_state=DEFAULT_RANDOM_STATE,
        **best_params
    )
    best_score = _cv_score(best_model, X, y, cv_folds, metric)
    return best_score, best_params

def run_cross_test(method=AUTOML_METHOD_DEFAULT, trials=AUTOML_TRIALS_DEFAULT, cv_folds=AUTOML_CV_FOLDS_DEFAULT, metric=AUTOML_METRIC_DEFAULT, use_existing_features=True, max_file_size=DEFAULT_MAX_FILE_SIZE, fast_dev_run=False):
    X, y = _load_data(use_existing_features=use_existing_features, max_file_size=max_file_size, fast_dev_run=fast_dev_run)
    baseline_model = _make_baseline_model()
    baseline_score = _cv_score(baseline_model, X, y, cv_folds, metric)
    
    # 计算基线的额外指标
    baseline_additional = {}
    for m in AUTOML_ADDITIONAL_METRICS:
        baseline_additional[m] = _cv_score(baseline_model, X, y, cv_folds, m)

    tuned_score = baseline_score
    best_params = {}
    if method == 'optuna':
        tuned_score, best_params = _optuna_tune_lgbm(X, y, cv_folds, trials, metric)
    elif method == 'hyperopt':
        tuned_score, best_params = _hyperopt_tune_lgbm(X, y, cv_folds, trials, metric)
    
    # 计算调优后的额外指标
    best_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=DEFAULT_NUM_BOOST_ROUND,
        verbosity=-1,
        random_state=DEFAULT_RANDOM_STATE,
        **best_params
    )
    tuned_additional = {}
    for m in AUTOML_ADDITIONAL_METRICS:
        tuned_additional[m] = _cv_score(best_model, X, y, cv_folds, m)

    os.makedirs(os.path.dirname(AUTOML_RESULTS_PATH), exist_ok=True)
    result = {
        'method': method,
        'metric': metric,
        'cv_folds': cv_folds,
        'trials': trials,
        'baseline_score': baseline_score,
        'baseline_additional_metrics': baseline_additional,
        'tuned_score': tuned_score,
        'tuned_additional_metrics': tuned_additional,
        'best_params': best_params
    }
    with open(AUTOML_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result

def main(args):
    method = getattr(args, 'method', AUTOML_METHOD_DEFAULT)
    trials = getattr(args, 'trials', AUTOML_TRIALS_DEFAULT)
    cv = getattr(args, 'cv', AUTOML_CV_FOLDS_DEFAULT)
    metric = getattr(args, 'metric', AUTOML_METRIC_DEFAULT)
    use_existing = getattr(args, 'use_existing_features', True)
    fast_dev_run = getattr(args, 'fast_dev_run', False)
    max_file_size = getattr(args, 'max_file_size', DEFAULT_MAX_FILE_SIZE)
    return run_cross_test(method=method, trials=trials, cv_folds=cv, metric=metric, use_existing_features=use_existing, max_file_size=max_file_size, fast_dev_run=fast_dev_run)
''')

_register_embedded("training.train_routing", r'''import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score

from config.config import (
    GATING_MODE, GATING_INPUT_DIM, GATING_HIDDEN_DIM, GATING_OUTPUT_DIM,
    GATING_LEARNING_RATE, GATING_EPOCHS, GATING_BATCH_SIZE, GATING_MODEL_PATH,
    EXPERT_NORMAL_MODEL_PATH, EXPERT_PACKED_MODEL_PATH,
    FEATURES_PKL_PATH, PROCESSED_DATA_DIR, METADATA_FILE, DEFAULT_MAX_FILE_SIZE,
    DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, DEFAULT_RANDOM_STATE, DEFAULT_NUM_BOOST_ROUND, DEFAULT_INCREMENTAL_ROUNDS,
    ROUTING_EVAL_REPORT_PATH, ROUTING_CONFUSION_MATRIX_PATH, MODEL_EVAL_FIG_DIR,
    EVAL_FONT_FAMILY, PREDICTION_THRESHOLD,
    EVAL_TOP_FEATURE_COUNT,
    PACKED_SECTIONS_RATIO_THRESHOLD, PACKER_KEYWORD_HITS_THRESHOLD,
    DEFAULT_MAX_FINETUNE_ITERATIONS
)
from models.gating import create_gating_model
from training.train_lightgbm import train_lightgbm_model
from training.data_loader import load_dataset, extract_features_from_raw_files, load_incremental_dataset
from training.feature_io import save_features_to_pickle
from training.model_io import load_existing_model, save_model
from training.evaluate import evaluate_model
from features.extractor_in_memory import PE_FEATURE_ORDER

from models.routing_model import RoutingModel

# Feature indices based on analysis
STAT_FEATURE_DIM = 49 
LIGHTWEIGHT_PE_DIM = 256
IDX_PACKED_SECTIONS_RATIO = STAT_FEATURE_DIM + LIGHTWEIGHT_PE_DIM + PE_FEATURE_ORDER.index('packed_sections_ratio')
IDX_PACKER_KEYWORD_HITS_COUNT = STAT_FEATURE_DIM + LIGHTWEIGHT_PE_DIM + PE_FEATURE_ORDER.index('packer_keyword_hits_count')

def get_feature_semantics(index):
    n_stat = 49
    if index < n_stat:
        if index == 0: return '字节均值'
        elif index == 1: return '字节标准差'
        elif index == 2: return '字节最小值'
        elif index == 3: return '字节最大值'
        elif index == 4: return '字节中位数'
        elif index == 5: return '字节25分位'
        elif index == 6: return '字节75分位'
        elif index == 7: return '零字节计数'
        elif index == 8: return '0xFF字节计数'
        elif index == 9: return '0x90字节计数'
        elif index == 10: return '可打印字节计数'
        elif index == 11: return '全局熵'
        elif 12 <= index <= 20:
            pos = (index - 12) // 3
            mod = (index - 12) % 3
            seg = ['前段','中段','后段'][pos]
            name = ['均值','标准差','熵'][mod]
            return seg + name
        elif 21 <= index <= 30: return f'分块均值_{index-21}'
        elif 31 <= index <= 40: return f'分块标准差_{index-31}'
        elif 41 <= index <= 44: return ['分块均值差绝对均值','分块均值差标准差','分块均值差最大值','分块均值差最小值'][index-41]
        elif 45 <= index <= 48: return ['分块标准差差绝对均值','分块标准差差标准差','分块标准差差最大值','分块标准差差最小值'][index-45]
        else: return '统计特征'
    j = index - n_stat
    if j < 256:
        if j < 128: return '轻量哈希位:导入DLL'
        elif j < 224: return '轻量哈希位:导入API'
        else: return '轻量哈希位:节名'
    k = j - 256
    if k < len(PE_FEATURE_ORDER):
        return PE_FEATURE_ORDER[k]
    return 'PE特征'

def evaluate_routing_system(X_test, y_test, files_test=None):
    print("\n[*] Evaluating Routing System on Test Set...")
    
    # Reload the full system
    try:
        routing_model = RoutingModel(device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"[!] Failed to load Routing System for evaluation: {e}")
        return

    # Predictions
    print("    Running predictions...")
    predictions, routing_decisions = routing_model.predict(X_test)
    
    # Binary Classification Metrics
    y_pred_binary = (predictions > PREDICTION_THRESHOLD).astype(int)
    
    acc = accuracy_score(y_test, y_pred_binary)
    print(f"[+] System Accuracy: {acc:.4f}")
    
    report = classification_report(y_test, y_pred_binary, target_names=['Benign', 'Malicious'])
    print("\n[*] Classification Report:")
    print(report)
    
    # Routing Stats
    stats = routing_model.get_routing_stats(routing_decisions)
    print("\n[*] Routing Statistics on Test Set:")
    print(f"    Total: {stats['total']}")
    print(f"    Routed to Normal Expert: {stats['normal']} ({stats['normal']/stats['total']:.1%})")
    print(f"    Routed to Packed Expert: {stats['packed']} ({stats['packed_ratio']:.1%})")

    # Threshold Sensitivity Analysis
    print("\n[*] Threshold sensitivity (0.90–0.99):")
    thresholds = np.arange(0.90, 1.00, 0.01)
    for t in thresholds:
        y_pred_t = (predictions > t).astype(int)
        cm_t = confusion_matrix(y_test, y_pred_t)
        if cm_t.shape == (2, 2):
            tn, fp, fn, tp = cm_t.ravel()
        else:
            tn = fp = fn = tp = 0 # Handle edge cases
            if len(np.unique(y_test)) == 1:
                if y_test[0] == 0: tn = len(y_test)
                else: tp = len(y_test) # Rough approx if prediction matches

        acc_t = accuracy_score(y_test, y_pred_t)
        pre_t = precision_score(y_test, y_pred_t, zero_division=0)
        rec_t = recall_score(y_test, y_pred_t, zero_division=0)
        fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"    t={t:.2f} acc={acc_t:.4f} pre={pre_t:.4f} rec={rec_t:.4f} FPR={fpr_t:.4f} TPR={tpr_t:.4f} FP={int(fp)}")

    if routing_model.expert_normal:
        print(f"\n[*] Top {EVAL_TOP_FEATURE_COUNT} important features (Expert Normal):")
        feature_importance = routing_model.expert_normal.feature_importance(importance_type='gain')
        indices_sorted = np.argsort(feature_importance)[::-1]
        for rank, idx in enumerate(indices_sorted[:EVAL_TOP_FEATURE_COUNT], 1):
            semantics = get_feature_semantics(idx)
            print(f"    {rank:2d}. feature_{idx}: {feature_importance[idx]:.2f} ({semantics})")

    # --- Reporting & Visualization ---
    os.makedirs(MODEL_EVAL_FIG_DIR, exist_ok=True)

    # 1. Save Text Report
    with open(ROUTING_EVAL_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("Routing System Evaluation Report\n")
        f.write("================================\n\n")
        f.write(f"System Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nRouting Statistics:\n")
        f.write(f"    Total: {stats['total']}\n")
        f.write(f"    Routed to Normal Expert: {stats['normal']} ({stats['normal']/stats['total']:.1%})\n")
        f.write(f"    Routed to Packed Expert: {stats['packed']} ({stats['packed_ratio']:.1%})\n")
    print(f"[+] Evaluation report saved to {ROUTING_EVAL_REPORT_PATH}")

    # 2. Plot Confusion Matrix
    try:
        cm = confusion_matrix(y_test, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Malicious'], 
                    yticklabels=['Benign', 'Malicious'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Routing System Confusion Matrix')
        plt.tight_layout()
        plt.savefig(ROUTING_CONFUSION_MATRIX_PATH)
        plt.close()
        print(f"[+] Confusion matrix plot saved to {ROUTING_CONFUSION_MATRIX_PATH}")
    except Exception as e:
        print(f"[!] Failed to generate confusion matrix plot: {e}")
    try:
        fpr, tpr, _ = roc_curve(y_test, predictions)
        auc_value = roc_auc_score(y_test, predictions)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC={auc_value:.4f}', color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Routing System ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        from config.config import ROUTING_ROC_AUC_PATH
        plt.savefig(ROUTING_ROC_AUC_PATH)
        plt.close()
        print(f"[+] ROC AUC curve saved to {ROUTING_ROC_AUC_PATH}")
    except Exception as e:
        print(f"[!] Failed to generate ROC AUC curve: {e}")

def generate_routing_labels(X):
    """
    Generate routing labels based on heuristics.
    Label 1 (Packed): High packed section ratio or packer keyword hits.
    Label 0 (Normal): Otherwise.
    """
    print("[*] Generating routing labels based on heuristics...")
    
    # Check feature dimension
    if X.shape[1] <= max(IDX_PACKED_SECTIONS_RATIO, IDX_PACKER_KEYWORD_HITS_COUNT):
        print(f"[!] Warning: Feature dimension {X.shape[1]} is smaller than expected indices.")
        return np.zeros(len(X), dtype=int)

    packed_ratio = X[:, IDX_PACKED_SECTIONS_RATIO]
    packer_hits = X[:, IDX_PACKER_KEYWORD_HITS_COUNT]
    
    # Heuristic: Packed if packed_sections_ratio > PACKED_SECTIONS_RATIO_THRESHOLD OR packer_keyword_hits_count > PACKER_KEYWORD_HITS_THRESHOLD
    is_packed = (packed_ratio > PACKED_SECTIONS_RATIO_THRESHOLD) | (packer_hits > PACKER_KEYWORD_HITS_THRESHOLD)
    
    labels = is_packed.astype(int)
    print(f"    Total samples: {len(labels)}")
    print(f"    Normal samples: {np.sum(labels == 0)}")
    print(f"    Packed samples: {np.sum(labels == 1)}")
    return labels

def train_gating_model_process(X_train, y_train, X_val, y_val):
    print(f"[*] Training Gating Model ({GATING_MODE})...")
    if GATING_MODE == 'rule':
        print("    Using heuristic rule gating, no training performed")
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    model = create_gating_model(GATING_MODE, GATING_INPUT_DIM, GATING_HIDDEN_DIM, GATING_OUTPUT_DIM)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=GATING_LEARNING_RATE)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=GATING_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=GATING_BATCH_SIZE)
    best_val_acc = 0.0
    for epoch in range(GATING_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"    Epoch {epoch+1}/{GATING_EPOCHS} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), GATING_MODEL_PATH)
    print(f"[+] Gating Model saved to {GATING_MODEL_PATH} (Best Acc: {best_val_acc:.4f})")
    return model

def train_expert_model_with_finetuning(X_train, y_train, X_val, y_val, files_train, files_val, 
                                     model_path, args, expert_name="Expert"):
    """
    Train an expert model with optional incremental training and False Positive finetuning.
    """
    print(f"\n[*] Training {expert_name}...")
    
    existing_model = None
    inc_training = getattr(args, 'incremental_training', False)
    inc_rounds = getattr(args, 'incremental_rounds', DEFAULT_INCREMENTAL_ROUNDS)
    num_rounds = getattr(args, 'num_boost_round', DEFAULT_NUM_BOOST_ROUND)
    fp_finetune = getattr(args, 'finetune_on_false_positives', False)
    if inc_training and os.path.exists(model_path):
        print(f"    Loading existing model for incremental training: {model_path}")
        existing_model = load_existing_model(model_path)

    model = None
    if existing_model:
        model = train_lightgbm_model(
            X_train, y_train, X_val, y_val, 
            files_train=files_train,
            num_boost_round=inc_rounds,
            init_model=existing_model,
            params_override=getattr(args, 'override_params', None)
        )
    else:
        model = train_lightgbm_model(
            X_train, y_train, X_val, y_val,
            files_train=files_train,
            num_boost_round=num_rounds,
            params_override=getattr(args, 'override_params', None)
        )

    if fp_finetune:
        print(f"[*] Starting False Positive Finetuning for {expert_name}...")
        
        # We need to evaluate on a hold-out set to find FPs. 
        # Here we use X_val as a proxy if we don't have a separate test set passed in.
        # But wait, we should really use the X_val FPs to improve training? 
        # Standard practice: Use Validation FPs to hard-mine.
        
        current_X_train = X_train
        current_y_train = y_train
        current_files_train = files_train

        max_targeted_iterations = DEFAULT_MAX_FINETUNE_ITERATIONS
        for i in range(max_targeted_iterations):
            # Evaluate on Validation Set
            # Note: We use X_val to find FPs, then add them to Train.
            # This "leaks" Val into Train, but for FP mining it's often accepted or we need a 3rd split.
            # Here we follow the aggressive approach.
            
            y_pred_proba = model.predict(X_val)
            y_pred = (y_pred_proba > PREDICTION_THRESHOLD).astype(int)
            
            # Find FPs
            fp_indices = np.where((y_val == 0) & (y_pred == 1))[0]
            if len(fp_indices) == 0:
                print(f"    [Round {i+1}] No False Positives found in validation set.")
                break
                
            print(f"    [Round {i+1}] Found {len(fp_indices)} False Positives. Retraining...")
            
            # Extract FP samples
            X_fps = X_val[fp_indices]
            y_fps = y_val[fp_indices]
            files_fps = [files_val[idx] for idx in fp_indices]
            
            # Add to Training Data (Augmentation)
            # We assume these are "hard" negatives.
            current_X_train = np.vstack([current_X_train, X_fps])
            current_y_train = np.concatenate([current_y_train, y_fps])
            current_files_train = current_files_train + files_fps
            
            # Retrain (Incremental/Continued)
            model = train_lightgbm_model(
                current_X_train, current_y_train, X_val, y_val,
                files_train=current_files_train,
                false_positive_files=files_fps,
                num_boost_round=num_rounds,
                init_model=model,
                iteration=i+2,
                params_override=getattr(args, 'override_params', None)
            )
            
    model.save_model(model_path)
    print(f"[+] {expert_name} saved to {model_path}")
    return model

def main(args=None):
    # Default Args Handling if None (for direct script execution compatibility)
    if args is None:
        parser = argparse.ArgumentParser()
        # Add minimal defaults or just rely on config
        # But really this function expects args object.
        pass

    use_existing = args.use_existing_features if args else False
    save_features_flag = args.save_features if args else False
    fast_dev_run = getattr(args, 'fast_dev_run', False)
    incremental_training = getattr(args, 'incremental_training', False)
    incremental_data_dir = getattr(args, 'incremental_data_dir', None)
    incremental_raw_data_dir = getattr(args, 'incremental_raw_data_dir', None)
    max_file_size = getattr(args, 'max_file_size', DEFAULT_MAX_FILE_SIZE)
    file_extensions = getattr(args, 'file_extensions', None)
    label_inference = getattr(args, 'label_inference', 'filename')
    
    X, y, files = None, None, None

    if incremental_training and incremental_data_dir:
        if incremental_raw_data_dir:
            print("[*] Extracting features from raw files (Incremental)...")
            extract_features_from_raw_files(
                incremental_raw_data_dir,
                incremental_data_dir,
                max_file_size,
                file_extensions,
                label_inference
            )
        print("[*] Loading incremental dataset...")
        X, y, files = load_incremental_dataset(incremental_data_dir, max_file_size)
        if X is None:
            print("[!] Failed to load incremental data.")
            return
    
    # Standard Data Loading
    elif use_existing and os.path.exists(FEATURES_PKL_PATH):
        print(f"[*] Loading features from {FEATURES_PKL_PATH}...")
        try:
            import pandas as pd
            df = pd.read_pickle(FEATURES_PKL_PATH)
            X = df.drop(['filename', 'label'], axis=1).values
            y = df['label'].values
            files = df['filename'].tolist()
        except Exception as e:
            print(f"[!] Failed to load existing features: {e}")
            print("    Falling back to feature extraction...")
            
    if X is None:
        print(f"[*] Extracting features (this may take a while)...")
        X, y, files = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE, max_file_size, fast_dev_run=fast_dev_run)
        
        if save_features_flag:
            save_features_to_pickle(X, y, files, FEATURES_PKL_PATH)

    print(f"[*] Total samples: {len(X)}")
    print(f"[*] Feature dimension: {X.shape[1]}")
    
    # 2. Generate Routing Labels
    routing_labels = generate_routing_labels(X)
    
    # 3. Train Gating Model
    # Split for Gating Model Training
    X_train_g, X_val_g, y_train_g, y_val_g = train_test_split(
        X, routing_labels, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=routing_labels
    )
    
    # Only train gating model if NOT in incremental mode or if explicitly requested?
    # For now, we always retrain gating model to ensure it adapts to new data distribution if any.
    train_gating_model_process(X_train_g, y_train_g, X_val_g, y_val_g)
    
    # 4. Train Expert Models
    print("\n[*] Training Expert Models...")
    
    # Split Data by Routing Label
    # We split the entire dataset into Train/Test first to have a global evaluation set.
    
    X_train_main, X_test_main, y_train_main, y_test_main, r_train_main, r_test_main, files_train_main, files_test_main = train_test_split(
        X, y, routing_labels, files, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y
    )
    
    # --- Expert Normal (Routing Label 0) ---
    mask_normal = (r_train_main == 0)
    X_normal = X_train_main[mask_normal]
    y_normal = y_train_main[mask_normal]
    files_normal = [files_train_main[i] for i in range(len(files_train_main)) if mask_normal[i]]
    
    if len(X_normal) > 10:
        X_t_norm, X_v_norm, y_t_norm, y_v_norm, f_t_norm, f_v_norm = train_test_split(
            X_normal, y_normal, files_normal, test_size=DEFAULT_VAL_SIZE, random_state=DEFAULT_RANDOM_STATE
        )
        print(f"[*] Expert Normal - Train: {len(X_t_norm)}, Val: {len(X_v_norm)}")
        
        train_expert_model_with_finetuning(
            X_t_norm, y_t_norm, X_v_norm, y_v_norm, f_t_norm, f_v_norm,
            EXPERT_NORMAL_MODEL_PATH, args, expert_name="Expert Normal"
        )
    else:
        print("[!] Not enough samples for Expert Normal training.")

    # --- Expert Packed (Routing Label 1) ---
    mask_packed = (r_train_main == 1)
    X_packed = X_train_main[mask_packed]
    y_packed = y_train_main[mask_packed]
    files_packed = [files_train_main[i] for i in range(len(files_train_main)) if mask_packed[i]]
    
    if len(X_packed) > 10:
        X_t_pack, X_v_pack, y_t_pack, y_v_pack, f_t_pack, f_v_pack = train_test_split(
            X_packed, y_packed, files_packed, test_size=DEFAULT_VAL_SIZE, random_state=DEFAULT_RANDOM_STATE
        )
        print(f"[*] Expert Packed - Train: {len(X_t_pack)}, Val: {len(X_v_pack)}")
        
        train_expert_model_with_finetuning(
            X_t_pack, y_t_pack, X_v_pack, y_v_pack, f_t_pack, f_v_pack,
            EXPERT_PACKED_MODEL_PATH, args, expert_name="Expert Packed"
        )
    else:
        print("[!] Not enough samples for Expert Packed training.")

    print("\n[*] Training pipeline completed.")

    # 5. Final System Evaluation
    if len(X_test_main) > 0:
        evaluate_routing_system(X_test_main, y_test_main, files_test_main)
    else:
        print("[!] No test samples available for evaluation.")

if __name__ == '__main__':
    # Argument Parsing if run directly
    from config.config import (
        DEFAULT_NUM_BOOST_ROUND, DEFAULT_INCREMENTAL_ROUNDS, DEFAULT_INCREMENTAL_EARLY_STOPPING, 
        DEFAULT_MAX_FINETUNE_ITERATIONS
    )
    
    parser = argparse.ArgumentParser(description="KoloVirusDetector Routing System Training")
    parser.add_argument('--use-existing-features', action='store_true')
    parser.add_argument('--save-features', action='store_true')
    parser.add_argument('--fast-dev-run', action='store_true')
    parser.add_argument('--finetune-on-false-positives', action='store_true')
    parser.add_argument('--incremental-training', action='store_true')
    parser.add_argument('--incremental-data-dir', type=str)
    parser.add_argument('--incremental-raw-data-dir', type=str)
    parser.add_argument('--file-extensions', type=str, nargs='+')
    parser.add_argument('--label-inference', type=str, default='filename')
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument('--incremental-rounds', type=int, default=DEFAULT_INCREMENTAL_ROUNDS)
    parser.add_argument('--incremental-early-stopping', type=int, default=DEFAULT_INCREMENTAL_EARLY_STOPPING)
    parser.add_argument('--max-finetune-iterations', type=int, default=DEFAULT_MAX_FINETUNE_ITERATIONS)
    parser.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE)

    args = parser.parse_args()
    main(args)
''')

_register_embedded("validation", r'''pass
''', is_package=True)

_register_embedded("validation.feature_gating_experiment", r'''import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from training.data_loader import load_dataset
from training.train_lightgbm import train_lightgbm_model
from training.evaluate import evaluate_model
from config.config import PROCESSED_DATA_DIR, METADATA_FILE, FEATURES_PKL_PATH, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, DEFAULT_RANDOM_STATE, FEATURE_GATING_TOP_K, FEATURE_GATING_REPORT_PATH, PE_FEATURE_VECTOR_DIM, DEFAULT_NUM_BOOST_ROUND, FEATURE_GATING_K_START, FEATURE_GATING_K_STEP

def load_features(use_existing_features: bool):
    if use_existing_features and os.path.exists(FEATURES_PKL_PATH):
        df = pd.read_pickle(FEATURES_PKL_PATH)
        files = df['filename'].tolist()
        y = df['label'].values
        X = df.drop(['filename', 'label'], axis=1).values.astype(np.float32)
        return X, y, files
    X, y, files = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE)
    return X, y, files

def split_sets(X, y, files):
    if len(X) > 10:
        X_temp, X_test, y_temp, y_test, files_temp, files_test = train_test_split(
            X, y, files, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y if len(np.unique(y)) > 1 else None
        )
        if len(X_temp) > 5:
            X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
                X_temp, y_temp, files_temp, test_size=DEFAULT_VAL_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
            )
        else:
            X_train, X_val = X_temp, X_temp
            y_train, y_val = y_temp, y_temp
            X_test, y_test = X_temp, y_temp
            files_train, files_val, files_test = files_temp, files_temp, files_temp
    else:
        X_train, X_val, X_test = X, X, X
        y_train, y_val, y_test = y, y, y
        files_train, files_val, files_test = files, files, files
    return X_train, y_train, files_train, X_val, y_val, files_val, X_test, y_test, files_test

def mask_top_k(importances: np.ndarray, k: int, n_features: int) -> np.ndarray:
    k = max(1, min(k, n_features))
    idx = np.argsort(importances)[::-1][:k]
    mask = np.zeros(n_features, dtype=bool)
    mask[idx] = True
    return mask

def mask_random(k: int, n_features: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    k = max(1, min(k, n_features))
    idx = rng.choice(n_features, size=k, replace=False)
    mask = np.zeros(n_features, dtype=bool)
    mask[idx] = True
    return mask

def mask_pre_pe_only(n_features: int, pe_dim: int) -> np.ndarray:
    pre_len = max(0, n_features - pe_dim)
    mask = np.zeros(n_features, dtype=bool)
    mask[:pre_len] = True
    return mask

def mask_pe_only(n_features: int, pe_dim: int) -> np.ndarray:
    pre_len = max(0, n_features - pe_dim)
    mask = np.zeros(n_features, dtype=bool)
    mask[pre_len:] = True
    return mask

def apply_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return X[:, mask]

def get_k_values(n_features: int, k_start: int, k_step: int) -> list:
    values = []
    start = max(1, min(k_start, n_features))
    step = max(1, k_step)
    cur = start
    while cur <= n_features:
        values.append(cur)
        cur += step
    if values[-1] != n_features:
        values.append(n_features)
    return values

def run_experiments(use_existing_features: bool, k_start: int, k_step: int, num_boost_round: int):
    X, y, files = load_features(use_existing_features)
    X_train, y_train, files_train, X_val, y_val, files_val, X_test, y_test, files_test = split_sets(X, y, files)
    base_model = train_lightgbm_model(X_train, y_train, X_val, y_val, iteration=1, num_boost_round=num_boost_round)
    base_acc, _ = evaluate_model(base_model, X_test, y_test, files_test)
    importances = base_model.feature_importance(importance_type='gain')
    n_features = X_train.shape[1]
    k_values = get_k_values(n_features, k_start, k_step)
    series = []
    for k in k_values:
        mk = mask_top_k(importances, k, n_features)
        X_train_k = apply_mask(X_train, mk)
        X_val_k = apply_mask(X_val, mk)
        X_test_k = apply_mask(X_test, mk)
        model_k = train_lightgbm_model(X_train_k, y_train, X_val_k, y_val, iteration=1, num_boost_round=num_boost_round)
        acc_k, _ = evaluate_model(model_k, X_test_k, y_test, files_test)
        series.append({'k': int(k), 'accuracy': float(acc_k)})
    result = {
        'n_features': int(n_features),
        'k_start': int(k_values[0] if k_values else 0),
        'k_step': int(k_step),
        'k_values': [int(v) for v in k_values],
        'accuracy_baseline': float(base_acc),
        'series_topk': series
    }
    os.makedirs(os.path.dirname(FEATURE_GATING_REPORT_PATH), exist_ok=True)
    with open(FEATURE_GATING_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Feature gating cross experiments')
    parser.add_argument('--use-existing-features', action='store_true')
    parser.add_argument('--k-start', type=int, default=FEATURE_GATING_K_START)
    parser.add_argument('--k-step', type=int, default=FEATURE_GATING_K_STEP)
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND)
    args = parser.parse_args()
    run_experiments(args.use_existing_features, args.k_start, args.k_step, args.num_boost_round)

if __name__ == '__main__':
    main()
''')

_register_embedded("validation.gating_validator", r'''import os
import json
import argparse
from pathlib import Path
from features.extractor_in_memory import extract_enhanced_pe_features
from config.config import PACKED_SECTIONS_RATIO_THRESHOLD, PACKER_KEYWORD_HITS_THRESHOLD

def collect_signals(file_path: str) -> dict:
    pe = extract_enhanced_pe_features(file_path)
    signals = {
        'packed_sections_ratio': float(pe.get('packed_sections_ratio', 0.0)),
        'packer_keyword_hits_count': float(pe.get('packer_keyword_hits_count', 0.0)),
    }
    return signals

def decide(signals: dict) -> str:
    packed_sections_ratio = float(signals.get('packed_sections_ratio', 0.0))
    packer_keyword_hits_count = float(signals.get('packer_keyword_hits_count', 0.0))
    is_packed = (packed_sections_ratio > float(PACKED_SECTIONS_RATIO_THRESHOLD)) or (
        packer_keyword_hits_count > float(PACKER_KEYWORD_HITS_THRESHOLD)
    )
    return 'packed' if is_packed else 'normal'

def evaluate_directory(directory_path: str, recursive: bool = False) -> dict:
    files = Path(directory_path).rglob('*') if recursive else Path(directory_path).glob('*')
    files = [f for f in files if f.is_file()]
    packed = 0
    normal = 0
    details = []
    for f in files:
        s = collect_signals(str(f))
        d = decide(s)
        if d == 'packed':
            packed += 1
        else:
            normal += 1
        if len(details) < 50:
            details.append({'file_path': str(f), 'decision': d, 'signals': s})
    return {'total': len(files), 'packed': packed, 'normal': normal, 'details_sample': details}

def main():
    parser = argparse.ArgumentParser(description='Gating validation')
    parser.add_argument('--dir-path', type=str)
    parser.add_argument('--file-path', type=str)
    parser.add_argument('--recursive', '-r', action='store_true')
    args = parser.parse_args()
    if args.file_path:
        s = collect_signals(args.file_path)
        d = decide(s)
        print(json.dumps({'file_path': args.file_path, 'decision': d, 'signals': s}, ensure_ascii=False, indent=2))
        return
    if args.dir_path:
        stats = evaluate_directory(args.dir_path, args.recursive)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    parser.print_help()

if __name__ == '__main__':
    main()
''')

_register_embedded("settings", r'''import os
from config import config as base_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

DATA_DIR = base_config.PROCESSED_DATA_DIR
METADATA_FILE = base_config.METADATA_FILE
MAX_FILE_SIZE = base_config.DEFAULT_MAX_FILE_SIZE

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
HARD_NEGATIVE_POOL_PATH = os.path.join(BASE_DIR, "hard_negative_pool.json")
ONNX_MODEL_DIR = os.path.join(MODEL_DIR, "onnx")
ONNX_ENABLED = False
ONNX_PROVIDERS = ["CPUExecutionProvider"]

NGRAM_SPECS = [(2, 512), (3, 512)]
NGRAM_SCALE = 1.0

STACKING_FOLDS = 3
ENSEMBLE_SEEDS = [17, 31, 47]

CALIBRATION_METHOD = "isotonic"
THRESHOLD_MAX_FPR = None
THRESHOLD_FP_WEIGHT = 5.0
THRESHOLD_FN_WEIGHT = 1.0

COST_POS_WEIGHT = 1.0
COST_NEG_WEIGHT = 3.0
POS_WEIGHT_SCALE = 0.01
HARD_NEGATIVE_WEIGHT = 5.0
HARD_NEGATIVE_MAX = 5000

GATING_HIDDEN_DIM = 256
GATING_EPOCHS = 10
GATING_BATCH_SIZE = 128
GATING_LEARNING_RATE = 0.001
GATING_THRESHOLD = 0.5

TIME_SPLIT_TEST_RATIO = base_config.DEFAULT_TEST_SIZE
TIME_SPLIT_VAL_RATIO = base_config.DEFAULT_VAL_SIZE
RANDOM_STATE = base_config.DEFAULT_RANDOM_STATE

LIGHTGBM_NUM_ROUNDS = base_config.DEFAULT_NUM_BOOST_ROUND
LIGHTGBM_EARLY_STOPPING = base_config.DEFAULT_EARLY_STOPPING_ROUNDS
''')

_register_embedded("threshold", r'''import numpy as np

def choose_threshold(y_true, y_prob, fp_weight=1.0, fn_weight=1.0, max_fpr=None):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float32)
    thresholds = np.unique(y_prob)
    if thresholds.size == 0:
        return 0.5, {}
    best_t = 0.5
    best_cost = float("inf")
    best_stats = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fpr = fp / max(1, fp + tn)
        if max_fpr is not None and fpr > max_fpr:
            continue
        cost = fp_weight * fp + fn_weight * fn
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
            best_stats = {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "fpr": float(fpr)
            }
    return best_t, best_stats
''')

_register_embedded("calibration", r'''import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class ProbabilityCalibrator:
    def __init__(self, method="isotonic"):
        self.method = method
        self.model = None

    def fit(self, y_true, y_prob):
        y_true = np.asarray(y_true).astype(np.int64)
        y_prob = np.asarray(y_prob).astype(np.float32)
        y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(y_prob, y_true)
        else:
            logit = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(logit, y_true)
            self.model = lr
        return self

    def predict(self, y_prob):
        y_prob = np.asarray(y_prob).astype(np.float32)
        y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
        if isinstance(self.model, IsotonicRegression):
            return self.model.predict(y_prob)
        logit = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
        return self.model.predict_proba(logit)[:, 1]
''')

_register_embedded("hard_negative", r'''import os
import json
import random
from settings import HARD_NEGATIVE_POOL_PATH, HARD_NEGATIVE_MAX

def load_pool(path=HARD_NEGATIVE_POOL_PATH):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(dict.fromkeys(data))
    except Exception:
        return []

def save_pool(pool, path=HARD_NEGATIVE_POOL_PATH, limit=HARD_NEGATIVE_MAX * 10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    uniq = list(dict.fromkeys(pool))
    if len(uniq) > limit:
        uniq = uniq[-limit:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(uniq, f, ensure_ascii=False, indent=2)
    return uniq

def update_pool(false_positive_files, path=HARD_NEGATIVE_POOL_PATH):
    pool = load_pool(path)
    for f in false_positive_files:
        if f not in pool:
            pool.append(f)
    return save_pool(pool, path)

def sample_pool(pool, max_count=HARD_NEGATIVE_MAX, seed=42):
    if not pool:
        return []
    if len(pool) <= max_count:
        return list(pool)
    rng = random.Random(seed)
    return rng.sample(pool, max_count)
''')

_register_embedded("feature_enhancer", r'''import numpy as np
from features.statistics import extract_statistical_features
from features.extractor_in_memory import PE_FEATURE_ORDER
from config.config import PE_FEATURE_VECTOR_DIM
from settings import NGRAM_SPECS, NGRAM_SCALE

STAT_FEATURE_DIM = 49
STAT_NAMES = [
    "byte_mean","byte_std","byte_min","byte_max","byte_median","byte_q25","byte_q75",
    "count_0","count_255","count_0x90","count_printable","entropy",
    "seg0_mean","seg0_std","seg0_entropy",
    "seg1_mean","seg1_std","seg1_entropy",
    "seg2_mean","seg2_std","seg2_entropy",
    "chunk_mean_0","chunk_mean_1","chunk_mean_2","chunk_mean_3","chunk_mean_4",
    "chunk_mean_5","chunk_mean_6","chunk_mean_7","chunk_mean_8","chunk_mean_9",
    "chunk_std_0","chunk_std_1","chunk_std_2","chunk_std_3","chunk_std_4",
    "chunk_std_5","chunk_std_6","chunk_std_7","chunk_std_8","chunk_std_9",
    "chunk_mean_diff_mean_abs","chunk_mean_diff_std","chunk_mean_diff_max","chunk_mean_diff_min",
    "chunk_std_diff_mean_abs","chunk_std_diff_std","chunk_std_diff_max","chunk_std_diff_min"
]

def _hashed_ngram(byte_sequence, n, dim):
    if byte_sequence is None or len(byte_sequence) < n:
        return np.zeros(dim, dtype=np.float32)
    seq = np.asarray(byte_sequence, dtype=np.uint8)
    if n == 2:
        idx = (seq[:-1].astype(np.int64) * 257 + seq[1:].astype(np.int64)) % dim
    else:
        idx = (seq[:-2].astype(np.int64) * 65537 + seq[1:-1].astype(np.int64) * 257 + seq[2:].astype(np.int64)) % dim
    counts = np.bincount(idx, minlength=dim).astype(np.float32)
    denom = max(1, idx.shape[0])
    return counts / float(denom)

def build_ngram_features(byte_sequence, ngram_specs=None):
    specs = ngram_specs or NGRAM_SPECS
    features = []
    for n, dim in specs:
        f = _hashed_ngram(byte_sequence, n, dim)
        features.append(f)
    if not features:
        return np.zeros(0, dtype=np.float32)
    out = np.concatenate(features, axis=0)
    if NGRAM_SCALE != 1.0:
        out = out * float(NGRAM_SCALE)
    return out.astype(np.float32)

def build_feature_vector(byte_sequence, pe_features, orig_length=None, ngram_specs=None):
    base = extract_statistical_features(byte_sequence, pe_features, orig_length)
    ngram = build_ngram_features(byte_sequence, ngram_specs=ngram_specs)
    if ngram.size == 0:
        return base
    return np.concatenate([base, ngram]).astype(np.float32)

def get_pe_feature_index(feature_key):
    try:
        return PE_FEATURE_ORDER.index(feature_key)
    except ValueError:
        return None

def get_packed_feature_indices():
    packed_idx = get_pe_feature_index("packed_sections_ratio")
    packer_idx = get_pe_feature_index("packer_keyword_hits_count")
    if packed_idx is None or packer_idx is None:
        return None, None
    return STAT_FEATURE_DIM + packed_idx, STAT_FEATURE_DIM + packer_idx

def split_feature_slices(total_dim, base_dim):
    base_slice = slice(0, base_dim)
    ngram_slice = slice(base_dim, total_dim)
    return base_slice, ngram_slice

def get_base_dim():
    return STAT_FEATURE_DIM + PE_FEATURE_VECTOR_DIM

def get_feature_names(ngram_specs=None):
    names = list(STAT_NAMES)
    lw = [f"lw_{i}" for i in range(256)]
    names.extend(lw)
    names.extend(list(PE_FEATURE_ORDER))
    if len(names) < STAT_FEATURE_DIM + PE_FEATURE_VECTOR_DIM:
        for i in range(len(names), STAT_FEATURE_DIM + PE_FEATURE_VECTOR_DIM):
            names.append(f"pe_extra_{i - (STAT_FEATURE_DIM + 256)}")
    specs = ngram_specs or NGRAM_SPECS
    for n, dim in specs:
        for i in range(dim):
            names.append(f"ngram_{n}_{i}")
    return names
''')

_register_embedded("dataset_v2", r'''import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.dataset import MalwareDataset
from feature_enhancer import build_feature_vector, get_base_dim
from settings import DATA_DIR, METADATA_FILE, MAX_FILE_SIZE, CACHE_DIR

def _build_label_map(metadata):
    label_map = {}
    for file, label in metadata.items():
        fname_lower = file.lower()
        if ('待加入白名单' in file) or ('whitelist' in fname_lower) or ('benign' in fname_lower) or ('good' in fname_lower) or ('clean' in fname_lower):
            label_map[file] = 0
        elif ('malicious' in fname_lower) or ('virus' in fname_lower) or ('trojan' in fname_lower):
            label_map[file] = 1
        elif label == 'benign' or label == 0:
            label_map[file] = 0
        elif label == 'malicious' or label == 1:
            label_map[file] = 1
        elif label == '待加入白名单':
            label_map[file] = 0
        else:
            label_map[file] = 1
    return label_map

def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)

def load_dataset_v2(use_cache=True, fast_dev_run=False):
    cache_path = _cache_path("features_v2.pkl")
    if use_cache and os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        y = df["label"].values.astype(np.int64)
        files = df["filename"].tolist()
        X = df.drop(["filename", "label"], axis=1).values.astype(np.float32)
        base_dim = get_base_dim()
        return X, y, files, base_dim
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    label_map = _build_label_map(metadata)
    all_files = list(metadata.keys())
    all_labels = [label_map[fname] for fname in all_files]
    if fast_dev_run:
        benign_files = [f for f, label in zip(all_files, all_labels) if label == 0]
        malicious_files = [f for f, label in zip(all_files, all_labels) if label == 1]
        n_samples_per_class = 2000
        selected_benign_files = benign_files[:min(n_samples_per_class, len(benign_files))]
        selected_malicious_files = malicious_files[:min(n_samples_per_class, len(malicious_files))]
        all_files = selected_benign_files + selected_malicious_files
        all_labels = [0] * len(selected_benign_files) + [1] * len(selected_malicious_files)
    dataset = MalwareDataset(DATA_DIR, all_files, all_labels, MAX_FILE_SIZE)
    features_list = []
    labels_list = []
    valid_files = []
    for i in tqdm(range(len(dataset)), desc="Extracting enhanced features"):
        byte_sequence, pe_features, label, orig_length = dataset[i]
        feat = build_feature_vector(byte_sequence, pe_features, orig_length)
        features_list.append(feat)
        labels_list.append(label)
        valid_files.append(all_files[i])
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    df = pd.DataFrame(X)
    df["filename"] = valid_files
    df["label"] = y
    if use_cache:
        df.to_pickle(cache_path)
    base_dim = get_base_dim()
    return X, y, valid_files, base_dim
''')

_register_embedded("ensemble", r'''import os
import json
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from settings import STACKING_FOLDS, ENSEMBLE_SEEDS, LIGHTGBM_NUM_ROUNDS, LIGHTGBM_EARLY_STOPPING, COST_POS_WEIGHT, COST_NEG_WEIGHT, POS_WEIGHT_SCALE, HARD_NEGATIVE_WEIGHT, MODEL_DIR

def _slice_array(X, slc):
    if slc is None:
        return X
    return X[:, slc]

def _train_lgb(X_train, y_train, X_val, y_val, seed, sample_weights=None, params_override=None, num_boost_round=None):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 40,
        "min_gain_to_split": 0.0,
        "seed": seed,
        "verbose": -1
    }
    if params_override:
        params.update(params_override)
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=LIGHTGBM_NUM_ROUNDS if num_boost_round is None else num_boost_round,
        callbacks=[lgb.early_stopping(LIGHTGBM_EARLY_STOPPING, verbose=False)]
    )
    return model

def _train_lr(X_train, y_train):
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    base = (n_neg / max(1, n_pos)) * POS_WEIGHT_SCALE
    pos_weight = max(0.01, base)
    class_weight = {0: COST_NEG_WEIGHT, 1: COST_POS_WEIGHT * pos_weight}
    model = LogisticRegression(max_iter=200, solver="liblinear", class_weight=class_weight)
    model.fit(X_train, y_train)
    return model

def _predict_model(model, X, model_type):
    if model_type == "lgb":
        return model.predict(X)
    return model.predict_proba(X)[:, 1]

class EnsembleModel:
    def __init__(self, base_models, meta_model, base_dim, total_dim):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_dim = base_dim
        self.total_dim = total_dim

    def _base_predictions(self, X):
        preds = []
        for item in self.base_models:
            slc = item["slice"]
            X_sub = _slice_array(X, slc)
            preds.append(_predict_model(item["model"], X_sub, item["type"]))
        return np.vstack(preds).T

    def predict_proba(self, X):
        meta_X = self._base_predictions(X)
        return self.meta_model.predict_proba(meta_X)[:, 1]

    def save(self, model_dir, prefix):
        os.makedirs(model_dir, exist_ok=True)
        meta_path = os.path.join(model_dir, f"{prefix}_meta.joblib")
        joblib.dump(self.meta_model, meta_path)
        meta = {
            "base_dim": int(self.base_dim),
            "total_dim": int(self.total_dim),
            "models": []
        }
        for idx, item in enumerate(self.base_models):
            name = item["name"]
            mtype = item["type"]
            slc = item["slice"]
            if mtype == "lgb":
                path = os.path.join(model_dir, f"{prefix}_{name}_{idx}.txt")
                item["model"].save_model(path)
            else:
                path = os.path.join(model_dir, f"{prefix}_{name}_{idx}.joblib")
                joblib.dump(item["model"], path)
            meta["models"].append({
                "name": name,
                "type": mtype,
                "path": path,
                "slice": [slc.start if slc else None, slc.stop if slc else None]
            })
        with open(os.path.join(model_dir, f"{prefix}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(model_dir, prefix):
        meta_path = os.path.join(model_dir, f"{prefix}_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta_model = joblib.load(os.path.join(model_dir, f"{prefix}_meta.joblib"))
        base_models = []
        for item in meta["models"]:
            slc = None
            if item["slice"][0] is not None:
                slc = slice(item["slice"][0], item["slice"][1])
            if item["type"] == "lgb":
                model = lgb.Booster(model_file=item["path"])
            else:
                model = joblib.load(item["path"])
            base_models.append({
                "name": item["name"],
                "type": item["type"],
                "slice": slc,
                "model": model
            })
        return EnsembleModel(base_models, meta_model, meta["base_dim"], meta["total_dim"])

def _compute_sample_weights(y, files=None, hard_negative_set=None):
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    base = (n_neg / max(1, n_pos)) * POS_WEIGHT_SCALE
    pos_weight = max(0.01, base)
    weights = np.where(y == 1, COST_POS_WEIGHT * pos_weight, COST_NEG_WEIGHT).astype(np.float32)
    if files is not None and hard_negative_set:
        for i, f in enumerate(files):
            if y[i] == 0 and f in hard_negative_set:
                weights[i] = weights[i] * HARD_NEGATIVE_WEIGHT
    return weights

def train_ensemble(X_train, y_train, X_val, y_val, base_dim, total_dim, prefix, params_override=None, files_train=None, hard_negative_set=None, num_boost_round=None):
    base_slice = slice(0, base_dim)
    ngram_slice = slice(base_dim, total_dim) if total_dim > base_dim else None
    model_specs = []
    for seed in ENSEMBLE_SEEDS:
        model_specs.append(("lgb_full_" + str(seed), "lgb", None, seed))
        model_specs.append(("lgb_base_" + str(seed), "lgb", base_slice, seed))
    if ngram_slice is not None:
        model_specs.append(("lr_ngram", "lr", ngram_slice, None))
    kfold = StratifiedKFold(n_splits=STACKING_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X_train), len(model_specs)), dtype=np.float32)
    for m_idx, (name, mtype, slc, seed) in enumerate(model_specs):
        for train_idx, val_idx in kfold.split(X_train, y_train):
            X_tr = _slice_array(X_train[train_idx], slc)
            y_tr = y_train[train_idx]
            X_va = _slice_array(X_train[val_idx], slc)
            y_va = y_train[val_idx]
            if mtype == "lgb":
                files_tr = [files_train[i] for i in train_idx] if files_train is not None else None
                weights = _compute_sample_weights(y_tr, files=files_tr, hard_negative_set=hard_negative_set)
                model = _train_lgb(X_tr, y_tr, X_va, y_va, seed, sample_weights=weights, params_override=params_override, num_boost_round=num_boost_round)
                pred = _predict_model(model, X_va, "lgb")
            else:
                model = _train_lr(X_tr, y_tr)
                pred = _predict_model(model, X_va, "lr")
            oof[val_idx, m_idx] = pred
    meta_model = LogisticRegression(max_iter=200, solver="liblinear")
    meta_model.fit(oof, y_train)
    base_models = []
    for name, mtype, slc, seed in model_specs:
        X_tr = _slice_array(X_train, slc)
        if mtype == "lgb":
            weights = _compute_sample_weights(y_train, files=files_train, hard_negative_set=hard_negative_set)
            model = _train_lgb(X_tr, y_train, _slice_array(X_val, slc), y_val, seed, sample_weights=weights, params_override=params_override, num_boost_round=num_boost_round)
        else:
            model = _train_lr(X_tr, y_train)
        base_models.append({
            "name": name,
            "type": mtype,
            "slice": slc,
            "model": model
        })
    ensemble = EnsembleModel(base_models, meta_model, base_dim, total_dim)
    ensemble.save(MODEL_DIR, prefix)
    return ensemble
''')

_register_embedded("gating_v2", r'''import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from settings import GATING_HIDDEN_DIM, GATING_EPOCHS, GATING_BATCH_SIZE, GATING_LEARNING_RATE, MODEL_DIR

class GatingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_gating_model(X_train, y_train, X_val, y_val, model_path, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatingMLP(input_dim, GATING_HIDDEN_DIM, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=GATING_LEARNING_RATE)
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=GATING_BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=GATING_BATCH_SIZE)
    best_acc = 0.0
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for _ in range(GATING_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
    return best_acc

def load_gating_model(model_path, input_dim):
    model = GatingMLP(input_dim, GATING_HIDDEN_DIM, 2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict_gating(model, X):
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs
''')

_register_embedded("onnx_backend", r'''import os
import json
import numpy as np

def _try_import_onnxruntime():
    try:
        import onnxruntime as ort
        return ort
    except Exception:
        return None

class OnnxPredictor:
    def __init__(self, model_path, feature_names, providers=None):
        self.model_path = model_path
        self.feature_names = feature_names
        self.providers = providers
        self.session = None
        self.input_name = None
        self.index_map = None
        ort = _try_import_onnxruntime()
        if ort is None:
            return
        if not os.path.exists(model_path):
            return
        self.session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        feat_path = os.path.join(os.path.dirname(model_path), "features.json")
        if os.path.exists(feat_path):
            with open(feat_path, "r", encoding="utf-8") as f:
                model_features = json.load(f)
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            self.index_map = [name_to_idx.get(n) for n in model_features]

    def available(self):
        return self.session is not None and self.input_name is not None

    def _reorder(self, X):
        if self.index_map is None:
            return X
        out = np.zeros((X.shape[0], len(self.index_map)), dtype=np.float32)
        for i, idx in enumerate(self.index_map):
            if idx is None:
                continue
            out[:, i] = X[:, idx]
        return out

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = self._reorder(X)
        outputs = self.session.run(None, {self.input_name: X})
        if isinstance(outputs, list) and len(outputs) > 1:
            result = outputs[1]
        else:
            result = outputs[0]
        if isinstance(result, list) and len(result) > 0 and hasattr(result[0], "get"):
            return np.array([float(r.get(1, r.get("1", 0.0))) for r in result], dtype=np.float32)
        if isinstance(result, np.ndarray):
            if result.ndim == 2 and result.shape[1] > 1:
                return result[:, 1].astype(np.float32)
            return result.reshape(-1).astype(np.float32)
        return np.zeros(X.shape[0], dtype=np.float32)
''')

_register_embedded("pipeline", r'''import os
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from config.config import PACKED_SECTIONS_RATIO_THRESHOLD, PACKER_KEYWORD_HITS_THRESHOLD
from dataset_v2 import load_dataset_v2
from feature_enhancer import get_packed_feature_indices, get_feature_names
from ensemble import train_ensemble, EnsembleModel
from calibration import ProbabilityCalibrator
from threshold import choose_threshold
from hard_negative import update_pool, load_pool, sample_pool
from gating_v2 import train_gating_model, load_gating_model, predict_gating
from settings import MODEL_DIR, REPORT_DIR, CALIBRATION_METHOD, THRESHOLD_FP_WEIGHT, THRESHOLD_FN_WEIGHT, THRESHOLD_MAX_FPR, GATING_THRESHOLD, ONNX_ENABLED, ONNX_MODEL_DIR, ONNX_PROVIDERS, DATA_DIR, TIME_SPLIT_TEST_RATIO, TIME_SPLIT_VAL_RATIO, RANDOM_STATE

def _index_map(files):
    return {f: i for i, f in enumerate(files)}

def _slice_by_files(X, y, files, target_files):
    idx_map = _index_map(files)
    indices = [idx_map[f] for f in target_files if f in idx_map]
    return X[indices], y[indices]

def _file_path(name):
    if name.endswith(".npz"):
        return os.path.join(DATA_DIR, name)
    return os.path.join(DATA_DIR, f"{name}.npz")

def _get_mtime(name):
    path = _file_path(name)
    if os.path.exists(path):
        return os.path.getmtime(path)
    return None

def _split_by_indices(files, labels, train_idx, val_idx, test_idx):
    X_train = [files[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_val = [files[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    X_test = [files[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test

def random_split(files, labels, test_size=TIME_SPLIT_TEST_RATIO, val_size=TIME_SPLIT_VAL_RATIO, random_state=RANDOM_STATE):
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        files, labels, list(range(len(files))), test_size=test_size, random_state=random_state, stratify=labels if len(np.unique(labels)) > 1 else None
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp, test_size=val_size, random_state=random_state, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def time_based_split(files, labels, test_size=TIME_SPLIT_TEST_RATIO, val_size=TIME_SPLIT_VAL_RATIO, random_state=RANDOM_STATE):
    times = [(_get_mtime(f), i) for i, f in enumerate(files)]
    valid = [(t, i) for t, i in times if t is not None]
    if len(valid) < max(10, int(0.3 * len(files))):
        return random_split(files, labels, test_size, val_size, random_state)
    valid.sort(key=lambda x: x[0])
    indices = [i for _, i in valid]
    n_total = len(indices)
    n_test = max(1, int(n_total * test_size))
    n_train_val = n_total - n_test
    n_val = max(1, int(n_train_val * val_size))
    train_indices = indices[:n_train_val - n_val]
    val_indices = indices[n_train_val - n_val:n_train_val]
    test_indices = indices[n_train_val:]
    return _split_by_indices(files, labels, train_indices, val_indices, test_indices)

def _make_gating_labels(X_base):
    packed_idx, packer_idx = get_packed_feature_indices()
    if packed_idx is None or packer_idx is None:
        return np.zeros(X_base.shape[0], dtype=np.int64)
    packed_ratio = X_base[:, packed_idx]
    packer_hits = X_base[:, packer_idx]
    labels = ((packed_ratio > PACKED_SECTIONS_RATIO_THRESHOLD) | (packer_hits > PACKER_KEYWORD_HITS_THRESHOLD)).astype(np.int64)
    return labels

def _train_one_route(X_train, y_train, X_val, y_val, base_dim, total_dim, prefix, files_train=None, hard_negative_set=None):
    if len(np.unique(y_train)) < 2:
        y_train = np.asarray(y_train)
        y_train[0] = 1 - y_train[0]
    ensemble = train_ensemble(X_train, y_train, X_val, y_val, base_dim, total_dim, prefix, files_train=files_train, hard_negative_set=hard_negative_set)
    y_val_prob = ensemble.predict_proba(X_val)
    calibrator = ProbabilityCalibrator(CALIBRATION_METHOD).fit(y_val, y_val_prob)
    y_cal = calibrator.predict(y_val_prob)
    threshold, stats = choose_threshold(y_val, y_cal, fp_weight=THRESHOLD_FP_WEIGHT, fn_weight=THRESHOLD_FN_WEIGHT, max_fpr=THRESHOLD_MAX_FPR)
    with open(os.path.join(MODEL_DIR, f"{prefix}_threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "stats": stats}, f, ensure_ascii=False, indent=2)
    import joblib
    joblib.dump(calibrator, os.path.join(MODEL_DIR, f"{prefix}_calibrator.joblib"))
    return ensemble, calibrator, threshold

def _evaluate_system(X, y, gating_model, base_dim, normal_artifacts, packed_artifacts):
    normal_model, normal_cal, normal_th = normal_artifacts
    packed_model, packed_cal, packed_th = packed_artifacts
    X_base = X[:, :base_dim]
    gating_probs = predict_gating(gating_model, X_base)
    routing = (gating_probs >= GATING_THRESHOLD).astype(np.int64)
    probs = np.zeros(len(X), dtype=np.float32)
    normal_idx = np.where(routing == 0)[0]
    packed_idx = np.where(routing == 1)[0]
    if normal_idx.size > 0:
        p = normal_model.predict_proba(X[normal_idx])
        p = normal_cal.predict(p)
        probs[normal_idx] = p
    if packed_idx.size > 0:
        p = packed_model.predict_proba(X[packed_idx])
        p = packed_cal.predict(p)
        probs[packed_idx] = p
    thresholds = np.where(routing == 0, normal_th, packed_th)
    preds = (probs >= thresholds).astype(np.int64)
    acc = accuracy_score(y, preds)
    pre = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    fp = int(np.sum((y == 0) & (preds == 1)))
    tn = int(np.sum((y == 0) & (preds == 0)))
    fpr = fp / max(1, fp + tn)
    return {"accuracy": float(acc), "precision": float(pre), "recall": float(rec), "fpr": float(fpr)}

def train_pipeline(fast_dev_run=False, use_cache=True):
    X, y, files, base_dim = load_dataset_v2(use_cache=use_cache, fast_dev_run=fast_dev_run)
    os.makedirs(MODEL_DIR, exist_ok=True)
    feature_names = get_feature_names()
    if len(feature_names) != X.shape[1]:
        feature_names = feature_names[:X.shape[1]]
        while len(feature_names) < X.shape[1]:
            feature_names.append(f"feature_{len(feature_names)}")
    with open(os.path.join(MODEL_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    train_files, y_train_list, val_files, y_val_list, test_files, y_test_list = time_based_split(files, y)
    X_train, y_train = _slice_by_files(X, y, files, train_files)
    X_val, y_val = _slice_by_files(X, y, files, val_files)
    X_test, y_test = _slice_by_files(X, y, files, test_files)
    X_base_train = X_train[:, :base_dim]
    X_base_val = X_val[:, :base_dim]
    gating_labels_train = _make_gating_labels(X_base_train)
    gating_labels_val = _make_gating_labels(X_base_val)
    gating_path = os.path.join(MODEL_DIR, "gating_model.pt")
    train_gating_model(X_base_train, gating_labels_train, X_base_val, gating_labels_val, gating_path, base_dim)
    gating_model = load_gating_model(gating_path, base_dim)
    routing_train = (predict_gating(gating_model, X_base_train) >= GATING_THRESHOLD).astype(np.int64)
    routing_val = (predict_gating(gating_model, X_base_val) >= GATING_THRESHOLD).astype(np.int64)
    normal_train_idx = np.where(routing_train == 0)[0]
    packed_train_idx = np.where(routing_train == 1)[0]
    normal_val_idx = np.where(routing_val == 0)[0]
    packed_val_idx = np.where(routing_val == 1)[0]
    if normal_train_idx.size == 0:
        normal_train_idx = np.arange(len(X_train))
        normal_val_idx = np.arange(len(X_val))
    if packed_train_idx.size == 0:
        packed_train_idx = np.arange(len(X_train))
        packed_val_idx = np.arange(len(X_val))
    pool = load_pool()
    hard_negative_set = set(sample_pool(pool))
    normal_files_train = [train_files[i] for i in normal_train_idx]
    packed_files_train = [train_files[i] for i in packed_train_idx]
    normal_artifacts = _train_one_route(X_train[normal_train_idx], y_train[normal_train_idx], X_val[normal_val_idx], y_val[normal_val_idx], base_dim, X.shape[1], "normal", files_train=normal_files_train, hard_negative_set=hard_negative_set)
    packed_artifacts = _train_one_route(X_train[packed_train_idx], y_train[packed_train_idx], X_val[packed_val_idx], y_val[packed_val_idx], base_dim, X.shape[1], "packed", files_train=packed_files_train, hard_negative_set=hard_negative_set)
    normal_pred = normal_artifacts[0].predict_proba(X_val[normal_val_idx])
    normal_fp = [val_files[i] for i, p in zip(normal_val_idx, normal_pred) if y_val[i] == 0 and p >= normal_artifacts[2]]
    packed_pred = packed_artifacts[0].predict_proba(X_val[packed_val_idx])
    packed_fp = [val_files[i] for i, p in zip(packed_val_idx, packed_pred) if y_val[i] == 0 and p >= packed_artifacts[2]]
    update_pool(normal_fp + packed_fp)
    metrics = _evaluate_system(X_test, y_test, gating_model, base_dim, normal_artifacts, packed_artifacts)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

def load_system():
    normal_model = EnsembleModel.load(MODEL_DIR, "normal")
    packed_model = EnsembleModel.load(MODEL_DIR, "packed")
    import joblib
    normal_cal = joblib.load(os.path.join(MODEL_DIR, "normal_calibrator.joblib"))
    packed_cal = joblib.load(os.path.join(MODEL_DIR, "packed_calibrator.joblib"))
    with open(os.path.join(MODEL_DIR, "normal_threshold.json"), "r", encoding="utf-8") as f:
        normal_th = json.load(f)["threshold"]
    with open(os.path.join(MODEL_DIR, "packed_threshold.json"), "r", encoding="utf-8") as f:
        packed_th = json.load(f)["threshold"]
    gating_path = os.path.join(MODEL_DIR, "gating_model.pt")
    gating_model = load_gating_model(gating_path, normal_model.base_dim)
    if ONNX_ENABLED:
        try:
            from onnx_backend import OnnxPredictor
            features_path = os.path.join(MODEL_DIR, "features.json")
            with open(features_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
            normal_onnx = os.path.join(ONNX_MODEL_DIR, "normal.onnx")
            packed_onnx = os.path.join(ONNX_MODEL_DIR, "packed.onnx")
            n_pred = OnnxPredictor(normal_onnx, feature_names, providers=ONNX_PROVIDERS)
            p_pred = OnnxPredictor(packed_onnx, feature_names, providers=ONNX_PROVIDERS)
            if n_pred.available():
                normal_model = n_pred
            if p_pred.available():
                packed_model = p_pred
        except Exception:
            pass
    return gating_model, (normal_model, normal_cal, normal_th), (packed_model, packed_cal, packed_th)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    if args.train:
        metrics = train_pipeline(fast_dev_run=args.fast_dev_run, use_cache=args.use_cache)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
''')

_register_embedded("scanner_v2", r'''import os
import json
import numpy as np
from features.extractor_in_memory import extract_features_in_memory
from feature_enhancer import build_feature_vector, get_base_dim
from pipeline import load_system
from settings import MAX_FILE_SIZE

class MalwareScannerV2:
    def __init__(self):
        self.gating_model, self.normal_artifacts, self.packed_artifacts = load_system()
        self.base_dim = get_base_dim()

    def _predict(self, features):
        gating_model, normal_artifacts, packed_artifacts = self.gating_model, self.normal_artifacts, self.packed_artifacts
        normal_model, normal_cal, normal_th = normal_artifacts
        packed_model, packed_cal, packed_th = packed_artifacts
        x = features.reshape(1, -1)
        x_base = x[:, :self.base_dim]
        from gating_v2 import predict_gating
        g = predict_gating(gating_model, x_base)[0]
        if g >= 0.5:
            prob = packed_model.predict_proba(x)[0]
            prob = packed_cal.predict(np.array([prob]))[0]
            is_malware = prob >= packed_th
        else:
            prob = normal_model.predict_proba(x)[0]
            prob = normal_cal.predict(np.array([prob]))[0]
            is_malware = prob >= normal_th
        confidence = float(prob if is_malware else (1 - prob))
        return bool(is_malware), confidence

    def scan_file(self, file_path):
        byte_sequence, pe_features, orig_length = extract_features_in_memory(file_path, MAX_FILE_SIZE)
        if byte_sequence is None or pe_features is None:
            return None
        features = build_feature_vector(byte_sequence, pe_features, orig_length)
        is_malware, confidence = self._predict(features)
        return {
            "file_path": os.path.abspath(file_path),
            "is_malware": is_malware,
            "confidence": float(confidence)
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    scanner = MalwareScannerV2()
    result = scanner.scan_file(args.file)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
''')

_register_embedded("pretrain", r'''import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from training.data_loader import load_dataset, extract_features_from_raw_files, load_incremental_dataset
from training.feature_io import save_features_to_pickle
from training.train_lightgbm import train_lightgbm_model
from training.evaluate import evaluate_model
from training.model_io import save_model, load_existing_model
from training.incremental import incremental_train_lightgbm_model
from config.config import PROCESSED_DATA_DIR, METADATA_FILE, SAVED_MODEL_DIR, MODEL_PATH, FEATURES_PKL_PATH, DEFAULT_MAX_FILE_SIZE, DEFAULT_NUM_BOOST_ROUND, DEFAULT_INCREMENTAL_ROUNDS, DEFAULT_INCREMENTAL_EARLY_STOPPING, DEFAULT_MAX_FINETUNE_ITERATIONS, HELP_MAX_FILE_SIZE, HELP_FAST_DEV_RUN, HELP_SAVE_FEATURES, HELP_FINETUNE_ON_FALSE_POSITIVES, HELP_INCREMENTAL_TRAINING, HELP_INCREMENTAL_DATA_DIR, HELP_INCREMENTAL_RAW_DATA_DIR, HELP_FILE_EXTENSIONS, HELP_LABEL_INFERENCE, HELP_NUM_BOOST_ROUND, HELP_INCREMENTAL_ROUNDS, HELP_INCREMENTAL_EARLY_STOPPING, HELP_MAX_FINETUNE_ITERATIONS, HELP_USE_EXISTING_FEATURES


def main(args):
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

    if args.use_existing_features and os.path.exists(FEATURES_PKL_PATH):

        print("[*] Loading existing feature file...")
        try:
            df = pd.read_pickle(FEATURES_PKL_PATH)
            files = df['filename'].tolist()
            y = df['label'].values
            X = df.drop(['filename', 'label'], axis=1).values

            print(f"[+] Successfully loaded feature file, total {len(files)} samples, feature dimension: {X.shape[1]}")
        except Exception as e:

            print(f"[!] Failed to load feature file: {e}")

            print("[-] Exiting training")
            return
    else:
        if args.incremental_training and args.incremental_data_dir:
            if args.incremental_raw_data_dir:

                print("[*] Extracting features from raw files...")
                output_features_dir = args.incremental_data_dir
                file_names, labels = extract_features_from_raw_files(
                    args.incremental_raw_data_dir,
                    output_features_dir,
                    args.max_file_size,
                    args.file_extensions,
                    args.label_inference
                )

                if not file_names:

                    print("[!] Failed to extract features from raw files, exiting training")
                    return

            X, y, files = load_incremental_dataset(args.incremental_data_dir, args.max_file_size)
            if X is None:

                print("[!] Failed to load incremental data, exiting training")
                return
        else:
            X, y, files = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE, args.max_file_size, args.fast_dev_run)

        save_features_to_pickle(X, y, files, FEATURES_PKL_PATH)

    if len(X) > 10:
        from config.config import DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, DEFAULT_RANDOM_STATE
        X_temp, X_test, y_temp, y_test, files_temp, files_test = train_test_split(
            X, y, files, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y if len(np.unique(y)) > 1 else None
        )
        if len(X_temp) > 5:
            X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
                X_temp, y_temp, files_temp, test_size=DEFAULT_VAL_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
            )
        else:
            X_train, X_val = X_temp, X_temp
            y_train, y_val = y_temp, y_temp
            X_test, y_test = X_temp, y_temp
            files_train, files_val, files_test = files_temp, files_temp, files_temp
    else:
        X_train, X_val, X_test = X, X, X
        y_train, y_val, y_test = y, y, y
        files_train, files_val, files_test = files, files, files

    print(f"[*] Dataset split completed:")
    print(f"    Training set: {len(X_train)} samples")
    print(f"    Validation set: {len(X_val)} samples")
    print(f"    Test set: {len(X_test)} samples")
    print(f"    Class distribution - Train: Benign={np.sum(y_train==0)}, Malicious={np.sum(y_train==1)}")
    print(f"    Class distribution - Val: Benign={np.sum(y_val==0)}, Malicious={np.sum(y_val==1)}")
    print(f"    Class distribution - Test: Benign={np.sum(y_test==0)}, Malicious={np.sum(y_test==1)}")

    existing_model = None
    if args.incremental_training:
        existing_model = load_existing_model(MODEL_PATH)

    model = None

    if args.incremental_training and existing_model:

        print("\n[*] Performing incremental training...")
        model = incremental_train_lightgbm_model(
            existing_model, X_train, y_train, X_val, y_val,
            num_boost_round=args.incremental_rounds,
            early_stopping_rounds=args.incremental_early_stopping
        )
    else:
        model = train_lightgbm_model(X_train, y_train, X_val, y_val, iteration=1, num_boost_round=args.num_boost_round, params_override=getattr(args, 'override_params', None))

    max_finetune_iterations = args.max_finetune_iterations
    finetune_iteration = 0
    false_positives = []

    while finetune_iteration < max_finetune_iterations:
        if args.finetune_on_false_positives:
            finetune_iteration += 1

            print(f"\n[*] Performing round {finetune_iteration} reinforcement training...")
            model = train_lightgbm_model(X_train, y_train, X_val, y_val, 
                                       iteration=finetune_iteration+1, 
                                       num_boost_round=args.num_boost_round,
                                       init_model=model,
                                       params_override=getattr(args, 'override_params', None))

            if finetune_iteration >= max_finetune_iterations:

                print("[*] Reached maximum reinforcement training rounds")
                break
        else:

            print("[*] Reinforcement training not enabled, skipping reinforcement training phase")
            break

    print("\n[*] Reinforcement training completed, performing final evaluation...")
    if len(X_test) > 0:
        test_accuracy, false_positives = evaluate_model(model, X_test, y_test, files_test)

        if false_positives and args.finetune_on_false_positives:

            print(f"\n[*] Detected {len(false_positives)} false positive samples, performing targeted reinforcement training...")

            targeted_iteration = 0
            max_targeted_iterations = 5
            previous_fp_count = len(false_positives)

            while len(false_positives) > 0 and targeted_iteration < max_targeted_iterations:
                targeted_iteration += 1

                print(f"\n[*] Performing round {targeted_iteration} targeted reinforcement training...")
                model = train_lightgbm_model(X_train, y_train, X_val, y_val,
                                           false_positives, files_train,
                                           finetune_iteration + targeted_iteration,
                                           num_boost_round=args.num_boost_round,
                                           init_model=model,
                                           params_override=getattr(args, 'override_params', None))

                print(f"\n[*] Evaluating after round {targeted_iteration} targeted reinforcement training...")
                test_accuracy, false_positives = evaluate_model(model, X_test, y_test, files_test)

                if len(false_positives) >= previous_fp_count:

                    print("[*] Targeted reinforcement training failed to reduce false positives, stopping training")
                    break
                previous_fp_count = len(false_positives)

            if len(false_positives) == 0:

                print("[*] Successfully eliminated all false positive samples")
            else:

                print(f"[*] Targeted reinforcement training completed, remaining {len(false_positives)} false positive samples")
        elif false_positives:

            print(f"\n[*] Detected {len(false_positives)} false positive samples, but reinforcement training is not enabled")

            print("    To enable reinforcement training, use the --finetune-on-false-positives parameter")
        try:
            y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
            thresholds = np.arange(0.90, 0.99, 0.01)
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
            print("\n[*] Threshold sensitivity (0.90–0.98):")
            for t in thresholds:
                y_pred_t = (y_pred_proba > t).astype(int)
                cm = confusion_matrix(y_test, y_pred_t)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0
                acc = accuracy_score(y_test, y_pred_t)
                pre = precision_score(y_test, y_pred_t, zero_division=0)
                rec = recall_score(y_test, y_pred_t, zero_division=0)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fp_count = int(fp)
                print(f"    t={t:.2f} acc={acc:.4f} pre={pre:.4f} rec={rec:.4f} FPR={fpr:.4f} TPR={tpr:.4f} FP={fp_count}")
        except Exception:
            pass
    else:

        print("[*] Test set is empty, skipping model evaluation")

    save_model(model, MODEL_PATH)

    from config.config import EVAL_TOP_FEATURE_COUNT
    print(f"\n[*] Top {EVAL_TOP_FEATURE_COUNT} important features:")
    feature_importance = model.feature_importance(importance_type='gain')
    indices_sorted = np.argsort(feature_importance)[::-1]
    def get_feature_semantics(index):
        n_stat = 49
        if index < n_stat:
            if index == 0:
                return '字节均值'
            elif index == 1:
                return '字节标准差'
            elif index == 2:
                return '字节最小值'
            elif index == 3:
                return '字节最大值'
            elif index == 4:
                return '字节中位数'
            elif index == 5:
                return '字节25分位'
            elif index == 6:
                return '字节75分位'
            elif index == 7:
                return '零字节计数'
            elif index == 8:
                return '0xFF字节计数'
            elif index == 9:
                return '0x90字节计数'
            elif index == 10:
                return '可打印字节计数'
            elif index == 11:
                return '全局熵'
            elif 12 <= index <= 20:
                pos = (index - 12) // 3
                mod = (index - 12) % 3
                seg = ['前段','中段','后段'][pos]
                name = ['均值','标准差','熵'][mod]
                return seg + name
            elif 21 <= index <= 30:
                return f'分块均值_{index-21}'
            elif 31 <= index <= 40:
                return f'分块标准差_{index-31}'
            elif 41 <= index <= 44:
                return ['分块均值差绝对均值','分块均值差标准差','分块均值差最大值','分块均值差最小值'][index-41]
            elif 45 <= index <= 48:
                return ['分块标准差差绝对均值','分块标准差差标准差','分块标准差差最大值','分块标准差差最小值'][index-45]
            else:
                return '统计特征'
        j = index - n_stat
        if j < 256:
            if j < 128:
                return '轻量哈希位:导入DLL'
            elif j < 224:
                return '轻量哈希位:导入API'
            else:
                return '轻量哈希位:节名'
        k = j - 256
        order = [
            'size','log_size','sections_count','symbols_count','imports_count','exports_count',
            'unique_imports','unique_dlls','unique_apis','section_names_count','section_total_size',
            'section_total_vsize','avg_section_size','avg_section_vsize','section_entropy_avg','section_entropy_min','section_entropy_max','section_entropy_std','packed_sections_ratio','subsystem','dll_characteristics',
            'code_section_ratio','data_section_ratio','code_vsize_ratio','data_vsize_ratio',
            'has_nx_compat','has_aslr','has_seh','has_guard_cf','has_resources','has_debug_info',
            'has_tls','has_relocs','has_exceptions','dll_name_avg_length','dll_name_max_length',
            'dll_name_min_length','section_name_avg_length','section_name_max_length','section_name_min_length',
            'export_name_avg_length','export_name_max_length','export_name_min_length','max_section_size',
            'min_section_size','long_sections_count','short_sections_count','section_size_std','section_size_cv',
            'executable_writable_sections','file_entropy_avg','file_entropy_min','file_entropy_max','file_entropy_range',
            'zero_byte_ratio','printable_byte_ratio','trailing_data_size','trailing_data_ratio','imported_system_dlls_count',
            'exports_density','has_large_trailing_data','pe_header_size','header_size_ratio','file_entropy_std',
            'file_entropy_q25','file_entropy_q75','file_entropy_median','high_entropy_ratio','low_entropy_ratio',
            'entropy_change_rate','entropy_change_std','executable_sections_count','writable_sections_count',
            'readable_sections_count','executable_sections_ratio','writable_sections_ratio','readable_sections_ratio',
            'executable_code_density','non_standard_executable_sections_count','rwx_sections_count','rwx_sections_ratio',
            'special_char_ratio','long_sections_ratio','short_sections_ratio','has_.text_section','has_.data_section','has_.rdata_section','has_.reloc_section','has_.rsrc_section',
            'has_signature','signature_size','signature_has_signing_time','version_info_present','company_name_len','product_name_len','file_version_len','original_filename_len',
            'has_upx_section','has_mpress_section','has_aspack_section','has_themida_section','api_network_ratio','api_process_ratio','api_filesystem_ratio','api_registry_ratio','overlay_entropy','overlay_high_entropy_flag','packer_keyword_hits_count','packer_keyword_hits_ratio','timestamp','timestamp_year'
        ]
        if k < len(order):
            m = order[k]
            return m
        return 'PE特征'
    for rank, idx in enumerate(indices_sorted[:EVAL_TOP_FEATURE_COUNT], 1):
        semantics = get_feature_semantics(idx)
        print(f"    {rank:2d}. feature_{idx}: {feature_importance[idx]:.2f} ({semantics})")

    print(f"\n[+] LightGBM pre-training completed! Model saved to: {MODEL_PATH}")

    print(f"[+] Extracted features saved to: {FEATURES_PKL_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LightGBM-based malware detection pre-training script")
    parser.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)
    parser.add_argument('--fast-dev-run', action='store_true', help=HELP_FAST_DEV_RUN)
    parser.add_argument('--save-features', action='store_true', help=HELP_SAVE_FEATURES)
    parser.add_argument('--finetune-on-false-positives', action='store_true', help=HELP_FINETUNE_ON_FALSE_POSITIVES)
    parser.add_argument('--incremental-training', action='store_true', help=HELP_INCREMENTAL_TRAINING)
    parser.add_argument('--incremental-data-dir', type=str, help=HELP_INCREMENTAL_DATA_DIR)
    parser.add_argument('--incremental-raw-data-dir', type=str, help=HELP_INCREMENTAL_RAW_DATA_DIR)
    parser.add_argument('--file-extensions', type=str, nargs='+', help=HELP_FILE_EXTENSIONS)
    parser.add_argument('--label-inference', type=str, default='filename', choices=['filename', 'directory'], help=HELP_LABEL_INFERENCE)
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND, help=HELP_NUM_BOOST_ROUND)
    parser.add_argument('--incremental-rounds', type=int, default=DEFAULT_INCREMENTAL_ROUNDS, help=HELP_INCREMENTAL_ROUNDS)
    parser.add_argument('--incremental-early-stopping', type=int, default=DEFAULT_INCREMENTAL_EARLY_STOPPING, help=HELP_INCREMENTAL_EARLY_STOPPING)
    parser.add_argument('--max-finetune-iterations', type=int, default=DEFAULT_MAX_FINETUNE_ITERATIONS, help=HELP_MAX_FINETUNE_ITERATIONS)
    parser.add_argument('--use-existing-features', action='store_true', help=HELP_USE_EXISTING_FEATURES)

    args = parser.parse_args()

    if args.incremental_training and not args.incremental_data_dir:

        print("[!] --incremental-data-dir parameter must be specified when enabling incremental training")
        exit(1)

    if args.incremental_raw_data_dir and not args.incremental_data_dir:

        print("[!] --incremental-data-dir parameter must be specified when specifying --incremental-raw-data-dir")
        exit(1)

    main(args)
''')

_register_embedded("finetune", r'''import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

try:
    import fast_hdbscan
    FAST_HDBSCAN_AVAILABLE = True
    print("[+] fast_hdbscan available")
except ImportError:
    FAST_HDBSCAN_AVAILABLE = False
    print("[-] fast_hdbscan not available")

from training.data_loader import load_dataset
from models.family_classifier import FamilyClassifier
from config.config import (
    PROCESSED_DATA_DIR, FEATURES_PKL_PATH, DEFAULT_MAX_FILE_SIZE,
    DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_MIN_SAMPLES, DEFAULT_MIN_FAMILY_SIZE,
    DEFAULT_TREAT_NOISE_AS_FAMILY,
    HDBSCAN_SAVE_DIR, HDBSCAN_CLUSTER_FIG_PATH, HDBSCAN_PCA_FIG_PATH,
    VIS_SAMPLE_SIZE, VIS_TSNE_PERPLEXITY, PCA_DIMENSION_FOR_CLUSTERING, DEFAULT_RANDOM_STATE,
    FAST_HDBSCAN_PCA_DIMENSION, HDBSCAN_FLOAT32_FOR_CLUSTERING,
    FAMILY_CLUSTERING_BACKEND, FAST_HDBSCAN_MAX_SAMPLES,
    KMEANS_N_CLUSTERS, KMEANS_BATCH_SIZE, KMEANS_MAX_ITER, KMEANS_N_INIT
)


def load_features_from_pickle(pickle_path):

    print(f"[*] Loading features from {pickle_path}...")

    df = pd.read_pickle(pickle_path)

    files = df['filename'].tolist()
    labels = df['label'].values

    feature_columns = [col for col in df.columns if col.startswith('feature_')]
    features = df[feature_columns].values

    print(f"[+] Successfully loaded features for {len(files)} samples")
    print(f"    Feature dimension: {features.shape[1]}")

    return features, labels, files

def extract_features_with_labels(data_dir, metadata_file, max_file_size=DEFAULT_MAX_FILE_SIZE):

    X, y, files = load_dataset(data_dir, metadata_file, max_file_size)
    return X, y, files

def filter_malicious_samples(features, labels, files):

    print("[*] Filtering malware samples for family clustering...")

    malicious_indices = np.where(labels == 1)[0]

    malicious_features = features[malicious_indices]
    malicious_labels = labels[malicious_indices]
    malicious_files = [files[i] for i in malicious_indices]

    print(f"[+] Filtering completed:")
    print(f"    Malware samples: {len(malicious_files)}")
    print(f"    Benign samples: {len(files) - len(malicious_files)}")

    return malicious_features, malicious_labels, malicious_files

def perform_hdbscan_clustering(features, min_cluster_size=50, min_samples=10):

    print("[*] Performing clustering analysis using HDBSCAN...")
    print(f"    [*] Original feature dimension: {features.shape[1]}")

    print("    [*] Standardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca_dim = PCA_DIMENSION_FOR_CLUSTERING
    try:
        if int(FAST_HDBSCAN_PCA_DIMENSION) > 0:
            pca_dim = min(pca_dim, int(FAST_HDBSCAN_PCA_DIMENSION))
    except Exception:
        pass

    if features_scaled.shape[1] > pca_dim:
        print(f"    [*] High feature dimension, reducing to {pca_dim} dimensions using PCA for better clustering...")
        pca = PCA(n_components=pca_dim, random_state=DEFAULT_RANDOM_STATE)
        features_for_clustering = pca.fit_transform(features_scaled)
        print(f"    [*] Reduced feature dimension: {features_for_clustering.shape[1]}")
    else:
        features_for_clustering = features_scaled

    if HDBSCAN_FLOAT32_FOR_CLUSTERING:
        try:
            features_for_clustering = features_for_clustering.astype(np.float32, copy=False)
        except Exception:
            pass

    backend = FAMILY_CLUSTERING_BACKEND
    if backend == 'auto':
        use_fast = FAST_HDBSCAN_AVAILABLE
        try:
            max_samples = int(FAST_HDBSCAN_MAX_SAMPLES)
        except Exception:
            max_samples = 0
        if (not use_fast) or (max_samples > 0 and features_for_clustering.shape[0] > max_samples):
            backend = 'minibatch_kmeans'
        else:
            backend = 'fast_hdbscan'

    if backend == 'fast_hdbscan':
        if not FAST_HDBSCAN_AVAILABLE:
            raise RuntimeError('fast_hdbscan not available')
        print("    [*] Using fast_hdbscan multicore optimized version")
        clusterer = fast_hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom'
        )

        try:
            labels = clusterer.fit_predict(features_for_clustering)
        except MemoryError as e:
            print(f"[!] HDBSCAN MemoryError: {e}")
            print(f"    features_for_clustering.shape={getattr(features_for_clustering, 'shape', None)} dtype={getattr(features_for_clustering, 'dtype', None)}")
            raise
        except Exception as e:
            print(f"[!] HDBSCAN failed: {e}")
            print(f"    features_for_clustering.shape={getattr(features_for_clustering, 'shape', None)} dtype={getattr(features_for_clustering, 'dtype', None)}")
            raise
    elif backend == 'minibatch_kmeans':
        print("    [*] Using MiniBatchKMeans for clustering")
        clusterer = MiniBatchKMeans(
            n_clusters=int(KMEANS_N_CLUSTERS),
            batch_size=int(KMEANS_BATCH_SIZE),
            max_iter=int(KMEANS_MAX_ITER),
            n_init=int(KMEANS_N_INIT),
            random_state=DEFAULT_RANDOM_STATE
        )
        labels = clusterer.fit_predict(features_for_clustering)
    else:
        raise ValueError(f'Unsupported FAMILY_CLUSTERING_BACKEND: {backend}')

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

    print(f"[+] Clustering completed:")
    print(f"    Total clusters: {n_clusters}")
    print(f"    Noise points: {np.sum(labels == -1) if -1 in labels else 0}")
    print(f"    Total samples: {len(labels)}")

    return labels, clusterer

def analyze_clusters(files, labels, min_family_size=20, treat_noise_as_family=False):

    print("[*] Analyzing clustering results to discover new families...")

    unique_labels, counts = np.unique(labels, return_counts=True)

    family_clusters = {}
    noise_count = 0
    small_cluster_count = 0
    noise_families = 0

    for label, count in zip(unique_labels, counts):
        if label == -1:
            noise_count = count
            if treat_noise_as_family and count >= min_family_size:
                family_clusters[int(label)] = int(count)
                noise_families += 1
        elif count >= min_family_size:
            family_clusters[int(label)] = int(count)
        else:
            small_cluster_count += 1
            if treat_noise_as_family:
                family_clusters[int(label)] = int(count)
                noise_families += 1 if label == -1 else 0

    noise_as_family_text = f" (of which {noise_families} are noise point families)" if treat_noise_as_family else ""
    print(f"[+] Family analysis completed:")
    print(f"    Identified {len(family_clusters)} potential malware families{noise_as_family_text}")
    print(f"    Noise samples: {noise_count}")
    print(f"    Small clusters (less than {min_family_size} samples): {small_cluster_count}")

    return {
        'families': family_clusters,
        'noise_count': noise_count,
        'small_clusters': small_cluster_count
    }

def visualize_clusters(features, labels, save_path, plot_pca=False):

    print("[*] Generating clustering visualization...")
    print(f"    [*] Feature dimension: {features.shape[1]}")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    original_features = features.copy()

    if features_scaled.shape[1] > PCA_DIMENSION_FOR_CLUSTERING:
        print(f"    [*] High feature dimension, reducing to {PCA_DIMENSION_FOR_CLUSTERING} dimensions using PCA")
        pca = PCA(n_components=PCA_DIMENSION_FOR_CLUSTERING, random_state=42)
        features_vis = pca.fit_transform(features_scaled)
    else:
        features_vis = features_scaled

    if features_vis.shape[0] > VIS_SAMPLE_SIZE:
        print(f"    [*] Large number of samples, randomly sampling {VIS_SAMPLE_SIZE} points for visualization")
        indices = np.random.choice(features_vis.shape[0], VIS_SAMPLE_SIZE, replace=False)
        features_vis = features_vis[indices]
        features_scaled = features_scaled[indices]
        labels = labels[indices]

    print("    [*] Using t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=DEFAULT_RANDOM_STATE, perplexity=min(VIS_TSNE_PERPLEXITY, len(features_vis)-1))
    features_2d = tsne.fit_transform(features_vis)

    plt.figure(figsize=(12, 10))
    colors = ['gray' if label == -1 else None for label in labels]
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Malware Clustering Results Visualization (HDBSCAN + t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[+] Clustering visualization saved to: {save_path}")

    if plot_pca:
        pca_save_path = HDBSCAN_PCA_FIG_PATH
        print("    [*] Generating additional PCA dimensionality reduction visualization...")
        pca_2d = PCA(n_components=2, random_state=DEFAULT_RANDOM_STATE)
        features_pca_2d = pca_2d.fit_transform(features_scaled)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_pca_2d[:, 0], features_pca_2d[:, 1], c=labels, cmap='tab20', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Malware Clustering Results Visualization (HDBSCAN + PCA)')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.grid(True)
        plt.savefig(pca_save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[+] PCA visualization saved to: {pca_save_path}")

def explain_clustering_discrepancy(features, labels, sample_indices=None, num_samples=5):

    print("[*] Explaining visualization discrepancies in clustering results...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if sample_indices is None:
        non_noise_indices = np.where(labels != -1)[0]
        if len(non_noise_indices) > 0:
            sample_indices = np.random.choice(non_noise_indices,
                                            min(num_samples, len(non_noise_indices)),
                                            replace=False)
        else:
            print("    [!] No samples found in non-noise clusters")
            return

    print(f"    [*] Analyzing {len(sample_indices)} samples in different dimensional spaces")

    print("    [*] Euclidean distances between samples in high-dimensional space (Scaled):")
    for i, idx in enumerate(sample_indices):
        same_cluster = np.where(labels == labels[idx])[0]
        if len(same_cluster) > 1:
            same_cluster_distances = [np.linalg.norm(features_scaled[idx] - features_scaled[j])
                                   for j in same_cluster if j != idx]
            avg_same_cluster_dist = np.mean(same_cluster_distances)
            print(f"        Sample {idx} average distance to same cluster samples: {avg_same_cluster_dist:.4f}")

        diff_cluster = np.where(labels != labels[idx])[0]
        diff_cluster = diff_cluster[diff_cluster != idx]
        if len(diff_cluster) > 0:
            diff_cluster_distances = [np.linalg.norm(features_scaled[idx] - features_scaled[j])
                                   for j in diff_cluster[:100]]
            avg_diff_cluster_dist = np.mean(diff_cluster_distances)
            print(f"        Sample {idx} average distance to different cluster samples: {avg_diff_cluster_dist:.4f}")

    print("    [*] Notes:")
    print("        1. t-SNE is a nonlinear dimensionality reduction method that prioritizes preserving local structure over global distances")
    print("        2. Points that are far apart in high-dimensional space may appear close in t-SNE")
    print("        3. Points that are close in high-dimensional space may be mapped far apart in t-SNE")
    print("        4. HDBSCAN performs clustering based on density in high-dimensional space, not distances in reduced space")

def save_clustering_results(files, labels, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    file_cluster_mapping = {}
    for file, label in zip(files, labels):
        file_cluster_mapping[file] = int(label)

    mapping_path = os.path.join(save_dir, 'file_cluster_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(file_cluster_mapping, f, indent=2, ensure_ascii=False)

    print(f"[+] File-cluster mapping saved to: {mapping_path}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_stats = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    stats_path = os.path.join(save_dir, 'cluster_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_stats, f, indent=2, ensure_ascii=False)

    print(f"[+] Cluster statistics saved to: {stats_path}")

def identify_new_families(files, labels, save_dir, min_family_size=20, treat_noise_as_family=False):

    print("[*] Identifying newly discovered malware families...")

    cluster_analysis = analyze_clusters(files, labels, min_family_size, treat_noise_as_family)

    family_names = {}
    for cluster_id, count in cluster_analysis['families'].items():
        family_names[cluster_id] = f"Malware_Family_{cluster_id}_Size{count}"

    print(f"[+] Identified {len(family_names)} new families")

    family_mapping_path = os.path.join(save_dir, 'family_names_mapping.json')
    with open(family_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(family_names, f, indent=2, ensure_ascii=False)

    print(f"[+] Family name mapping saved to: {family_mapping_path}")

    return family_names

def main(args):

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    print("\n[*] Step 1/3: Preparing dataset...")

    features_pkl_path = args.features_path if args.features_path else FEATURES_PKL_PATH

    if not os.path.exists(features_pkl_path):
        print(f"[!] Error: Feature file not found {features_pkl_path}")
        print("    Please run pretrain.py first to generate feature file")
        metadata_file = os.path.join(args.data_dir, 'metadata.json')
        if os.path.exists(args.data_dir) and os.path.exists(metadata_file):
            print(f"    [*] Attempting to generate features from data directory: {args.data_dir}")
            features, labels, files = extract_features_with_labels(
                args.data_dir, metadata_file, args.max_file_size
            )
            feature_df = pd.DataFrame(features)
            feature_df.columns = [f'feature_{i}' for i in range(features.shape[1])]
            feature_df['label'] = labels
            feature_df['filename'] = files
            feature_df.to_pickle(features_pkl_path)
            print(f"    [+] Features saved to {features_pkl_path}")
        else:
            return
    else:
        print("\n[*] Step 2/3: Loading features...")
        features, labels, files = load_features_from_pickle(features_pkl_path)

    if features.size == 0:
        print("[!] Failed to load any features, cannot perform clustering. Please check feature file.")
        return

    print("\n[*] Step 3/5: Filtering malware samples...")
    malicious_features, malicious_labels, malicious_files = filter_malicious_samples(features, labels, files)

    if malicious_features.size == 0:
        print("[!] No malware samples in dataset, cannot perform family clustering analysis.")
        return

    print("\n[*] Step 4/5: Performing HDBSCAN clustering analysis...")
    cluster_labels, clusterer = perform_hdbscan_clustering(
        malicious_features,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )

    if args.treat_noise_as_family and args.min_family_size == 1:
        print("    [*] Treating each noise point as a unique family...")
        noise_indices = np.where(cluster_labels == -1)[0]
        if len(noise_indices) > 0:
            max_label = cluster_labels.max()
            for i, idx in enumerate(noise_indices):
                cluster_labels[idx] = max_label + 1 + i
            print(f"    [+] Assigned unique family IDs to {len(noise_indices)} noise points")

    print("\n[*] Step 5/6: Analyzing clusters and identifying new families...")
    family_analysis = analyze_clusters(malicious_files, cluster_labels, args.min_family_size, args.treat_noise_as_family)
    family_names = identify_new_families(malicious_files, cluster_labels, args.save_dir, args.min_family_size, args.treat_noise_as_family)

    print("\n[*] Step 6/6: Saving clustering results and training classifier...")
    save_clustering_results(malicious_files, cluster_labels, args.save_dir)

    classifier = FamilyClassifier()
    classifier.fit(malicious_features, cluster_labels, family_names)
    classifier.save(os.path.join(args.save_dir, 'family_classifier.pkl'))
    print(f"[+] Family classifier saved to: {os.path.join(args.save_dir, 'family_classifier.pkl')}")

    os.makedirs(HDBSCAN_SAVE_DIR, exist_ok=True)
    visualize_clusters(malicious_features, cluster_labels,
                      HDBSCAN_CLUSTER_FIG_PATH,
                      plot_pca=args.plot_pca)

    if args.explain_discrepancy:
        explain_clustering_discrepancy(malicious_features, cluster_labels)

    try:
        if len(np.unique(cluster_labels)) > 1 and -1 not in cluster_labels:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            sil_score = silhouette_score(malicious_features, cluster_labels)
            ch_score = calinski_harabasz_score(malicious_features, cluster_labels)
            print(f"\n[*] Clustering quality assessment:")
            print(f"    Silhouette Score: {sil_score:.4f}")
            print(f"    Calinski-Harabasz Index: {ch_score:.4f}")
    except Exception as e:
        print(f"[!] Unable to calculate clustering quality metrics: {e}")

    print("\n[+] HDBSCAN clustering fine-tuning completed!")
    print(f"[*] Processed {len(malicious_files)} malware files")
    print(f"[*] Discovered {len(family_analysis['families'])} new families")
    print(f"[*] Identified {family_analysis['noise_count']} noise points")
    print(f"[*] Results saved to: {args.save_dir}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="HDBSCAN-based malware family discovery script")

    parser.add_argument('--data-dir', type=str, default=PROCESSED_DATA_DIR,
                       help='Directory of processed dataset (.npz & metadata)')
    parser.add_argument('--features-path', type=str, default=FEATURES_PKL_PATH,
                       help='Path to extracted features pickle')
    parser.add_argument('--save-dir', type=str, default=HDBSCAN_SAVE_DIR,
                       help='Directory to save HDBSCAN results')

    parser.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE,
                       help='Maximum input file size in bytes')
    parser.add_argument('--min-cluster-size', type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
                       help='HDBSCAN minimum cluster size')
    parser.add_argument('--min-samples', type=int, default=DEFAULT_MIN_SAMPLES,
                       help='HDBSCAN minimum samples for core points')
    parser.add_argument('--min-family-size', type=int, default=DEFAULT_MIN_FAMILY_SIZE,
                       help='Minimum samples to define a family')
    parser.add_argument('--plot-pca', action='store_true',
                       help='Generate additional PCA dimensionality reduction visualization')
    parser.add_argument('--explain-discrepancy', action='store_true',
                       help='Explain why nearby points belong to different clusters')
    parser.add_argument('--treat-noise-as-family', action='store_true', default=DEFAULT_TREAT_NOISE_AS_FAMILY,
                       help='Treat noise points as separate families (if min-family-size is 1, each noise point is unique)')

    args = parser.parse_args()
    main(args)
''')

_register_embedded("scanner", r'''import os
import sys
import json
import numpy as np
import lightgbm as lgb
from pathlib import Path
import csv
import hashlib
import gzip
import tempfile
import shutil
import argparse
from models.family_classifier import FamilyClassifier
from features.extractor_in_memory import extract_features_in_memory
from config.config import (
    MODEL_PATH, FAMILY_CLASSIFIER_PATH, DEFAULT_MAX_FILE_SIZE, SCAN_CACHE_PATH, 
    SCAN_OUTPUT_DIR, HELP_LIGHTGBM_MODEL_PATH, HELP_FAMILY_CLASSIFIER_PATH, 
    HELP_CACHE_FILE, HELP_FILE_PATH, HELP_DIR_PATH, HELP_RECURSIVE, 
    HELP_OUTPUT_PATH, HELP_MAX_FILE_SIZE, ENV_ALLOWED_SCAN_ROOT, PREDICTION_THRESHOLD,
    GATING_ENABLED, SCAN_PRINT_ONLY_MALICIOUS
)

if GATING_ENABLED:
    from models.routing_model import RoutingModel

def validate_path(path):
    if not path:
        return None

    normalized_path = os.path.normpath(path)

    if '\0' in normalized_path:
        return None

    abs_path = os.path.abspath(normalized_path)

    allowed_root = os.getenv(ENV_ALLOWED_SCAN_ROOT)
    if allowed_root:
        base = os.path.abspath(allowed_root)
        if not abs_path.startswith(base + os.sep) and abs_path != base:
            return None

    if not os.path.exists(abs_path):
        return None

    return abs_path

from features.statistics import extract_statistical_features

BASE_DIR = getattr(sys, '_MEIPASS', os.path.abspath('.'))




class MalwareScanner:
    def __init__(self, lightgbm_model_path, family_classifier_path,
                 max_file_size=DEFAULT_MAX_FILE_SIZE, cache_file=SCAN_CACHE_PATH, enable_cache=True,
                 print_only_malicious=None, print_malicious_paths=None):

        self.max_file_size = max_file_size
        self.cache_file = cache_file
        self.enable_cache = enable_cache
        self.binary_classifier = None
        self.routing_model = None
        self.print_only_malicious = SCAN_PRINT_ONLY_MALICIOUS if print_only_malicious is None else bool(print_only_malicious)
        self.print_malicious_paths = SCAN_PRINT_ONLY_MALICIOUS if print_malicious_paths is None else bool(print_malicious_paths)

        def _debug(msg):
            if not self.print_only_malicious:
                print(msg)
        self._debug = _debug

        if GATING_ENABLED:
            self._debug("[*] Gating System Enabled. Loading Routing Model...")
            try:
                self.routing_model = RoutingModel()
                self._debug("[+] Routing Model loaded")
            except Exception as e:
                self._debug(f"[!] Failed to load Routing Model: {e}. Falling back to single LightGBM model.")

        if not GATING_ENABLED or self.routing_model is None:
            self._debug("[*] Loading LightGBM binary classification model...")
            model_path = lightgbm_model_path
            if model_path.endswith('.gz'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
                    with gzip.open(model_path, 'rb') as gf:
                        shutil.copyfileobj(gf, tmp)
                    model_path = tmp.name
            self.binary_classifier = lgb.Booster(model_file=model_path)
            self._temp_model_path = model_path if model_path != lightgbm_model_path else None
            self._debug("[+] LightGBM binary classification model loaded")

        self._debug("[*] Loading family classifier...")
        self.family_classifier = FamilyClassifier()
        self.family_classifier.load(family_classifier_path)

        self.scan_cache = self._load_cache()
        if self.enable_cache:
            self._debug(f"[+] Scan cache loaded, total {len(self.scan_cache)} cached files")
        else:
            self._debug("[+] Scan cache disabled for this scanner instance")

        self._debug("[+] Malware scanner initialization completed")

    def _load_cache(self):

        if not self.enable_cache or not self.cache_file:
            return {}

        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self._debug(f"[!] Failed to load scan cache: {e}")
                return {}
        return {}

    def _save_cache(self):

        if not self.enable_cache or not self.cache_file:
            return

        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.scan_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._debug(f"[!] Failed to save scan cache: {e}")

    def _calculate_sha256(self, file_path):

        sha256_hash = hashlib.sha256()
        try:
            valid_path = validate_path(file_path)
            if not valid_path:
                return None

            with open(valid_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self._debug(f"[!] Failed to calculate SHA256 for file {file_path}: {e}")
            return None

    def _is_pe_file(self, file_path):

        try:
            valid_path = validate_path(file_path)
            if not valid_path:
                return False

            with open(valid_path, 'rb') as f:
                magic = f.read(2)
                if magic != b'MZ':
                    return False

                f.seek(0x3C)
                pe_offset_bytes = f.read(4)
                if len(pe_offset_bytes) < 4:
                    return False

                pe_offset = int.from_bytes(pe_offset_bytes, byteorder='little')

                f.seek(pe_offset)
                pe_signature = f.read(4)
                return pe_signature == b'PE\x00\x00'
        except Exception:
            return False

    def _preprocess_file(self, file_path):

        try:

            byte_sequence, pe_features, orig_length = extract_features_in_memory(file_path, self.max_file_size)
            if byte_sequence is None or pe_features is None:
                raise Exception("Failed to extract features in memory")
            from config.config import PE_FEATURE_VECTOR_DIM
            orig_pe_len = len(pe_features)
            status = 'ok'
            if orig_pe_len != PE_FEATURE_VECTOR_DIM:
                fixed_pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                copy_len = min(orig_pe_len, PE_FEATURE_VECTOR_DIM)
                fixed_pe_features[:copy_len] = pe_features[:copy_len]
                pe_features = fixed_pe_features
                status = 'padded' if orig_pe_len < PE_FEATURE_VECTOR_DIM else 'truncated'

            features = extract_statistical_features(byte_sequence, pe_features, orig_length)
            try:
                self._debug(f"[+] 完整特征维度={len(features)}，PE维度={len(pe_features)}，状态={status}")
            except Exception:
                pass
            return features
        except Exception as e:
            self._debug(f"[!] File preprocessing failed {file_path}: {e}")
            return None

    def _predict_malware_from_features(self, features):
        try:
            feature_vector = features.reshape(1, -1)
            
            if self.routing_model is not None:
                predictions, decisions = self.routing_model.predict(feature_vector)
                prediction_val = predictions[0]
            elif self.binary_classifier is not None:
                prediction_val = self.binary_classifier.predict(feature_vector)[0]
            else:
                raise Exception("No model loaded for prediction")

            is_malware = prediction_val > PREDICTION_THRESHOLD
            confidence = prediction_val if is_malware else (1 - prediction_val)

            return is_malware, confidence
        except Exception as e:
            self._debug(f"[!] Binary classification prediction failed: {e}")
            return False, 0.0

    def _predict_malware_batch(self, features_matrix):
        try:
            if self.routing_model is not None:
                predictions, decisions = self.routing_model.predict(features_matrix)
            elif self.binary_classifier is not None:
                predictions = self.binary_classifier.predict(features_matrix)
            else:
                raise Exception("No model loaded for prediction")
            
            is_malware_batch = predictions > PREDICTION_THRESHOLD
            confidence_batch = np.where(is_malware_batch, predictions, 1 - predictions)
            
            return is_malware_batch, confidence_batch
        except Exception as e:
            self._debug(f"[!] Batch prediction failed: {e}")
            count = len(features_matrix)
            return [False] * count, [0.0] * count

    def scan_batch(self, file_paths):
        final_results = [None] * len(file_paths)
        
        batch_indices = []
        batch_features = []
        batch_paths = []
        batch_hashes = []
        
        for i, file_path in enumerate(file_paths):
            valid_path = validate_path(file_path)
            if not valid_path:
                self._debug(f"[!] Invalid path: {file_path}")
                continue
            
            file_hash = self._calculate_sha256(valid_path)
            if not file_hash:
                continue
            
            if self.enable_cache and file_hash in self.scan_cache:
                cached_result = self.scan_cache[file_hash]
                self._debug(f"[*] Using cached result: {valid_path}")
                try:
                    if cached_result.get('is_malware') and self.print_malicious_paths:
                        print(valid_path)
                except Exception:
                    pass
                final_results[i] = cached_result
                continue
            
            if not self._is_pe_file(valid_path):
                self._debug(f"[-] Skipping non-PE file: {valid_path}")
                continue
                
            features = self._preprocess_file(valid_path)
            if features is None:
                continue
                
            batch_indices.append(i)
            batch_features.append(features)
            batch_paths.append(valid_path)
            batch_hashes.append(file_hash)
            
        if batch_features:
            matrix = np.array(batch_features)
            is_malware_list, confidence_list = self._predict_malware_batch(matrix)
            
            for idx, is_malware, confidence, path, fhash, feat in zip(batch_indices, is_malware_list, confidence_list, batch_paths, batch_hashes, batch_features):
                result = {
                    'file_path': path,
                    'file_name': os.path.basename(path),
                    'file_size': os.path.getsize(path),
                    'is_malware': bool(is_malware),
                    'confidence': float(confidence),
                }
                
                if is_malware:
                    cluster_id, family_name, is_new_family = self.predict_family(feat)
                    result.update({
                        'malware_family': {
                            'cluster_id': int(cluster_id) if cluster_id is not None else -1,
                            'family_name': family_name,
                            'is_new_family': bool(is_new_family)
                        }
                    })
                    try:
                        if self.print_malicious_paths:
                            print(path)
                    except Exception:
                        pass
                    if is_new_family:
                        self._debug(f"[+] New malware family discovered: {family_name}")
                    else:
                        self._debug(f"[+] Identified as known family: {family_name}")
                else:
                    self._debug(f"[+] Identified as benign software: {path}")
                    
                if self.enable_cache:
                    self.scan_cache[fhash] = result
                    
                final_results[idx] = result
                
        return [r for r in final_results if r is not None]

    def is_malware(self, file_path):
        features = self._preprocess_file(file_path)
        if features is None:
            return False, 0.0
        
        return self._predict_malware_from_features(features)

    def predict_family(self, features):
        return self.family_classifier.predict(features)

    def scan_file(self, file_path):

        valid_path = validate_path(file_path)
        if not valid_path:
            self._debug(f"[!] Invalid or non-existent file path: {file_path}")
            return None

        file_hash = self._calculate_sha256(valid_path)
        if file_hash is None:
            self._debug(f"[!] Unable to calculate file hash: {valid_path}")
            return None

        if self.enable_cache and file_hash in self.scan_cache:
            cached_result = self.scan_cache[file_hash]
            self._debug(f"[*] Using cached result: {valid_path}")
            try:
                if cached_result.get('is_malware') and self.print_malicious_paths:
                    print(valid_path)
            except Exception:
                pass
            return cached_result

        if not self._is_pe_file(valid_path):
            self._debug(f"[-] Skipping non-PE file: {valid_path}")
            return None

        self._debug(f"[*] Scanning file: {valid_path}")

        features = self._preprocess_file(valid_path)
        if features is None:
            return None

        is_malware, confidence = self._predict_malware_from_features(features)

        result = {
            'file_path': valid_path,
            'file_name': os.path.basename(valid_path),
            'file_size': os.path.getsize(valid_path),
            'is_malware': bool(is_malware),
            'confidence': float(confidence),
        }

        if is_malware:
            cluster_id, family_name, is_new_family = self.predict_family(features)

            result.update({
                'malware_family': {
                    'cluster_id': int(cluster_id) if cluster_id is not None else -1,
                    'family_name': family_name,
                    'is_new_family': bool(is_new_family)
                }
            })

            try:
                if self.print_malicious_paths:
                    print(valid_path)
            except Exception:
                pass
            if is_new_family:
                self._debug(f"[+] New malware family discovered: {family_name}")
            else:
                self._debug(f"[+] Identified as known family: {family_name}")
        else:
            self._debug(f"[+] Identified as benign software")

        if self.enable_cache:
            self.scan_cache[file_hash] = result

        return result

    def scan_directory(self, directory_path, recursive=False):

        results = []

        if recursive:
            files = Path(directory_path).rglob('*')
        else:
            files = Path(directory_path).glob('*')

        file_paths = [str(f) for f in files if f.is_file()]

        self._debug(f"[*] Scanning directory: {directory_path} ({'recursive' if recursive else 'non-recursive'})")
        self._debug(f"[*] Found {len(file_paths)} files")

        CHUNK_SIZE = 128
        for i in range(0, len(file_paths), CHUNK_SIZE):
            chunk = file_paths[i:i + CHUNK_SIZE]
            try:
                batch_results = self.scan_batch(chunk)
                results.extend(batch_results)
            except Exception as e:
                self._debug(f"[!] Batch processing failed for chunk {i}: {e}")

        return results

    def save_results(self, results, output_path):

        json_path = output_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self._debug(f"[+] Scan results saved to: {json_path}")

        csv_path = output_path + '.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = ['file_path', 'file_name', 'file_size', 'is_malware', 'confidence']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    flat_result = {
                        'file_path': result['file_path'],
                        'file_name': result['file_name'],
                        'file_size': result['file_size'],
                        'is_malware': result['is_malware'],
                        'confidence': result['confidence']
                    }
                    writer.writerow(flat_result)
        self._debug(f"[+] Scan results saved to: {csv_path}")

    def __del__(self):

        if getattr(self, 'enable_cache', False):
            self._save_cache()
        try:
            if hasattr(self, '_temp_model_path') and self._temp_model_path:
                os.unlink(self._temp_model_path)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Malware Scanner")

    parser.add_argument('--lightgbm-model-path', type=str, default=MODEL_PATH, help=HELP_LIGHTGBM_MODEL_PATH)
    parser.add_argument('--family-classifier-path', type=str, default=FAMILY_CLASSIFIER_PATH, help=HELP_FAMILY_CLASSIFIER_PATH)
    parser.add_argument('--cache-file', type=str, default=SCAN_CACHE_PATH, help=HELP_CACHE_FILE)

    parser.add_argument('--file-path', type=str, help=HELP_FILE_PATH)
    parser.add_argument('--dir-path', type=str, help=HELP_DIR_PATH)
    parser.add_argument('--recursive', '-r', action='store_true', help=HELP_RECURSIVE)
    parser.add_argument('--output-path', type=str, default=SCAN_OUTPUT_DIR, help=HELP_OUTPUT_PATH)

    parser.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)

    args = parser.parse_args()

    if not os.path.exists(args.lightgbm_model_path):
        print(f"[!] Error: LightGBM model file not found {args.lightgbm_model_path}")
        return

    if not os.path.exists(args.family_classifier_path):
        print(f"[!] Error: Family classifier file not found {args.family_classifier_path}")
        return

    scanner = MalwareScanner(
        lightgbm_model_path=args.lightgbm_model_path,
        family_classifier_path=args.family_classifier_path,
        max_file_size=args.max_file_size,
        cache_file=args.cache_file,
        enable_cache=True,
    )

    results = []
    if args.file_path:
        if not os.path.exists(args.file_path):
            print(f"[!] Error: File does not exist {args.file_path}")
            return
        result = scanner.scan_file(args.file_path)
        if result is not None:
            results.append(result)
    elif args.dir_path:
        if not os.path.exists(args.dir_path):
            print(f"[!] Error: Directory does not exist {args.dir_path}")
            return
        results = scanner.scan_directory(args.dir_path, args.recursive)
    else:
        print("[!] Error: Please specify a file or directory to scan")
        parser.print_help()
        return

    scanner.save_results(results, args.output_path)

    scanner._save_cache()

    malware_count = sum(1 for r in results if r['is_malware'])
    new_family_count = sum(1 for r in results if r['is_malware'] and
                          r.get('malware_family', {}).get('is_new_family', False))

    print(f"\n[*] Scan completion statistics:")
    print(f"    Total files: {len(results)}")
    print(f"    Malware: {malware_count}")
    print(f"    Newly discovered families: {new_family_count}")

if __name__ == '__main__':
    main()
''')

_register_embedded("scanner_service", r'''import os
import signal
import asyncio
import json
import struct
from contextlib import suppress
from threading import Lock
from typing import Optional, List, Any, Dict, Tuple

from scanner import MalwareScanner
from config.config import (
    MODEL_PATH,
    FAMILY_CLASSIFIER_PATH,
    SCAN_CACHE_PATH,
    DEFAULT_MAX_FILE_SIZE,
    ENV_LIGHTGBM_MODEL_PATH,
    ENV_FAMILY_CLASSIFIER_PATH,
    ENV_CACHE_PATH,
    ENV_MAX_FILE_SIZE,
    ENV_ALLOWED_SCAN_ROOT,
    SERVICE_CONCURRENCY_LIMIT,
    SERVICE_PRINT_MALICIOUS_PATHS,
    SERVICE_EXIT_COMMAND,
    SERVICE_ADMIN_TOKEN,
    SERVICE_CONTROL_LOCALHOSTS,
    ENV_SERVICE_ADMIN_TOKEN,
    ENV_SERVICE_EXIT_COMMAND,
    SERVICE_MAX_BATCH_SIZE,
    SERVICE_IPC_HOST,
    SERVICE_IPC_PORT,
    SERVICE_IPC_MAX_MESSAGE_BYTES,
    SERVICE_IPC_READ_TIMEOUT_SEC,
    SERVICE_IPC_WRITE_TIMEOUT_SEC,
    SERVICE_IPC_REQUEST_TIMEOUT_SEC,
    SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION,
    ENV_SERVICE_IPC_HOST,
    ENV_SERVICE_IPC_PORT,
    ENV_SERVICE_IPC_MAX_MESSAGE_BYTES,
    ENV_SERVICE_IPC_READ_TIMEOUT_SEC,
    ENV_SERVICE_IPC_WRITE_TIMEOUT_SEC,
    ENV_SERVICE_IPC_REQUEST_TIMEOUT_SEC,
    ENV_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION,
)
from utils.logging_utils import get_logger


ALLOWED_SCAN_ROOT = os.getenv(ENV_ALLOWED_SCAN_ROOT)
_logger = get_logger('scanner_service')
_IPC_PROTOCOL_VERSION = 1
_ipc_server: Optional[asyncio.AbstractServer] = None
_ipc_server_task: Optional[asyncio.Task] = None

def _validate_user_path(path: str) -> Optional[str]:
    if not path:
        return None
    normalized = os.path.normpath(path)
    if '\0' in normalized:
        return None
    abs_path = os.path.abspath(normalized)
    if ALLOWED_SCAN_ROOT:
        base = os.path.abspath(ALLOWED_SCAN_ROOT)
        if not abs_path.startswith(base + os.sep) and abs_path != base:
            return None
    if not os.path.exists(abs_path):
        return None
    return abs_path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_ipc_host() -> str:
    return os.getenv(ENV_SERVICE_IPC_HOST, SERVICE_IPC_HOST)


def _get_ipc_port() -> int:
    return _env_int(ENV_SERVICE_IPC_PORT, SERVICE_IPC_PORT)


def _get_ipc_max_message_bytes() -> int:
    return _env_int(ENV_SERVICE_IPC_MAX_MESSAGE_BYTES, SERVICE_IPC_MAX_MESSAGE_BYTES)


def _get_ipc_read_timeout_sec() -> float:
    return _env_float(ENV_SERVICE_IPC_READ_TIMEOUT_SEC, float(SERVICE_IPC_READ_TIMEOUT_SEC))


def _get_ipc_write_timeout_sec() -> float:
    return _env_float(ENV_SERVICE_IPC_WRITE_TIMEOUT_SEC, float(SERVICE_IPC_WRITE_TIMEOUT_SEC))


def _get_ipc_request_timeout_sec() -> float:
    return _env_float(ENV_SERVICE_IPC_REQUEST_TIMEOUT_SEC, float(SERVICE_IPC_REQUEST_TIMEOUT_SEC))


def _get_ipc_max_requests_per_connection() -> int:
    return _env_int(ENV_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION, SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION)

_scanner_lock = Lock()
_scanner_instance: Optional[MalwareScanner] = None
_scan_semaphore = asyncio.Semaphore(SERVICE_CONCURRENCY_LIMIT)


def _prefer_gz(path: str) -> str:
    gz = path + ('.gz' if not path.endswith('.gz') else '')
    return gz if os.path.exists(gz) and not os.path.exists(path) else path

def _build_scanner() -> MalwareScanner:
    lightgbm_model_path = _prefer_gz(os.getenv(ENV_LIGHTGBM_MODEL_PATH, MODEL_PATH))
    family_classifier_path = _prefer_gz(os.getenv(ENV_FAMILY_CLASSIFIER_PATH, FAMILY_CLASSIFIER_PATH))
    cache_file = os.getenv(ENV_CACHE_PATH, SCAN_CACHE_PATH)
    max_file_size = _env_int(ENV_MAX_FILE_SIZE, DEFAULT_MAX_FILE_SIZE)

    missing_paths: List[str] = [
        p for p in [lightgbm_model_path, family_classifier_path]
        if not os.path.exists(p)
    ]
    if missing_paths:
        raise RuntimeError(f'以下必需文件不存在: {missing_paths}')

    return MalwareScanner(
        lightgbm_model_path=lightgbm_model_path,
        family_classifier_path=family_classifier_path,
        max_file_size=max_file_size,
        cache_file=None,
        enable_cache=False,
        print_malicious_paths=SERVICE_PRINT_MALICIOUS_PATHS,
    )


def get_scanner() -> MalwareScanner:
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = _build_scanner()
    return _scanner_instance


def _get_exit_command() -> str:
    return os.getenv(ENV_SERVICE_EXIT_COMMAND, SERVICE_EXIT_COMMAND)


def _get_admin_token() -> str:
    return os.getenv(ENV_SERVICE_ADMIN_TOKEN, SERVICE_ADMIN_TOKEN)


def _cleanup_environment() -> None:
    global _scanner_instance

    with _scanner_lock:
        scanner = _scanner_instance
        _scanner_instance = None

    if scanner is None:
        return

    try:
        if getattr(scanner, 'enable_cache', False):
            scanner._save_cache()
    except Exception:
        pass

    try:
        if hasattr(scanner, 'scan_cache'):
            scanner.scan_cache.clear()
    except Exception:
        pass

    try:
        temp_model_path = getattr(scanner, '_temp_model_path', None)
        if temp_model_path:
            os.unlink(temp_model_path)
    except Exception:
        pass


def _trigger_process_exit() -> None:
    os.kill(os.getpid(), signal.SIGINT)


def _ipc_response_ok(request_id: Optional[str], payload: Any) -> Dict[str, Any]:
    return {
        'version': _IPC_PROTOCOL_VERSION,
        'id': request_id,
        'ok': True,
        'payload': payload,
    }


def _ipc_response_error(request_id: Optional[str], code: str, message: str, details: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {'code': code, 'message': message}
    if details is not None:
        err['details'] = details
    return {
        'version': _IPC_PROTOCOL_VERSION,
        'id': request_id,
        'ok': False,
        'error': err,
    }


async def _ipc_read_message(reader: asyncio.StreamReader, max_bytes: int, timeout_sec: float) -> Optional[Dict[str, Any]]:
    header = await asyncio.wait_for(reader.readexactly(4), timeout=timeout_sec)
    size = struct.unpack('>I', header)[0]
    if size <= 0:
        raise ValueError('message_size_invalid')
    if size > max_bytes:
        raise ValueError('message_too_large')
    body = await asyncio.wait_for(reader.readexactly(size), timeout=timeout_sec)
    try:
        decoded = body.decode('utf-8')
        obj = json.loads(decoded)
    except Exception as e:
        raise ValueError('message_decode_failed') from e
    if not isinstance(obj, dict):
        raise ValueError('message_not_object')
    return obj


async def _ipc_write_message(writer: asyncio.StreamWriter, message: Dict[str, Any], max_bytes: int, timeout_sec: float) -> None:
    encoded = json.dumps(message, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > max_bytes:
        encoded = json.dumps(
            _ipc_response_error(message.get('id'), 'response_too_large', '响应体超过大小限制'),
            ensure_ascii=False,
            separators=(',', ':'),
        ).encode('utf-8')
    frame = struct.pack('>I', len(encoded)) + encoded
    writer.write(frame)
    await asyncio.wait_for(writer.drain(), timeout=timeout_sec)


def _ipc_extract_timeout_sec(msg: Dict[str, Any], default_timeout_sec: float) -> float:
    value = msg.get('timeout_ms')
    if value is None:
        return default_timeout_sec
    try:
        ms = float(value)
    except Exception:
        return default_timeout_sec
    if ms <= 0:
        return default_timeout_sec
    return ms / 1000.0


async def _ipc_handle_message(msg: Dict[str, Any], client_host: Optional[str]) -> Dict[str, Any]:
    request_id = msg.get('id')
    msg_type = msg.get('type')
    payload = msg.get('payload') or {}
    if not isinstance(payload, dict):
        return _ipc_response_error(request_id, 'invalid_payload', 'payload必须是对象')

    if msg.get('version', _IPC_PROTOCOL_VERSION) != _IPC_PROTOCOL_VERSION:
        return _ipc_response_error(request_id, 'version_mismatch', '协议版本不匹配')

    if msg_type == 'health':
        return _ipc_response_ok(request_id, {'status': 'ok'})

    if msg_type == 'scan_file':
        file_path = payload.get('file_path')
        if not isinstance(file_path, str):
            return _ipc_response_error(request_id, 'invalid_argument', 'file_path必须是字符串')
        valid_path = _validate_user_path(file_path)
        if not valid_path:
            return _ipc_response_error(request_id, 'invalid_path', '路径不合法或不在允许的扫描目录内')
        scanner = get_scanner()
        async with _scan_semaphore:
            result = await asyncio.to_thread(scanner.scan_file, valid_path)
        if result is None:
            return _ipc_response_error(request_id, 'scan_failed', '文件不是有效的PE或扫描失败')
        result['virus_family'] = (result.get('malware_family') or {}).get('family_name')
        return _ipc_response_ok(request_id, result)

    if msg_type == 'scan_batch':
        file_paths = payload.get('file_paths')
        if not isinstance(file_paths, list):
            return _ipc_response_error(request_id, 'invalid_argument', 'file_paths必须是数组')
        if len(file_paths) > SERVICE_MAX_BATCH_SIZE:
            return _ipc_response_error(request_id, 'batch_too_large', f'批量扫描数量超过限制: {SERVICE_MAX_BATCH_SIZE}')
        valid_paths: List[str] = []
        for p in file_paths:
            if not isinstance(p, str):
                continue
            vp = _validate_user_path(p)
            if vp:
                valid_paths.append(vp)
        if not valid_paths:
            return _ipc_response_ok(request_id, [])
        scanner = get_scanner()
        async with _scan_semaphore:
            results = await asyncio.to_thread(scanner.scan_batch, valid_paths)
        for res in results:
            res['virus_family'] = (res.get('malware_family') or {}).get('family_name')
        return _ipc_response_ok(request_id, results)

    if msg_type == 'control':
        command = payload.get('command')
        token = payload.get('token')
        if command != _get_exit_command():
            return _ipc_response_error(request_id, 'unknown_command', '未知控制指令')
        expected = _get_admin_token()
        if expected:
            if token != expected:
                return _ipc_response_error(request_id, 'forbidden', '无权限执行控制指令')
        else:
            if client_host not in set(SERVICE_CONTROL_LOCALHOSTS):
                return _ipc_response_error(request_id, 'forbidden', '无权限执行控制指令')
        asyncio.create_task(asyncio.to_thread(_cleanup_environment))
        asyncio.create_task(asyncio.to_thread(_trigger_process_exit))
        return _ipc_response_ok(request_id, {'status': 'shutting_down'})

    return _ipc_response_error(request_id, 'unknown_type', '未知消息类型')


async def _ipc_handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info('peername')
    client_host = peer[0] if isinstance(peer, (tuple, list)) and peer else None
    max_bytes = _get_ipc_max_message_bytes()
    read_timeout_sec = _get_ipc_read_timeout_sec()
    write_timeout_sec = _get_ipc_write_timeout_sec()
    default_request_timeout_sec = _get_ipc_request_timeout_sec()
    max_requests = _get_ipc_max_requests_per_connection()

    _logger.info(f'IPC连接建立: {client_host}')
    handled = 0
    try:
        while handled < max_requests:
            try:
                msg = await _ipc_read_message(reader, max_bytes=max_bytes, timeout_sec=read_timeout_sec)
            except asyncio.IncompleteReadError:
                break
            except asyncio.TimeoutError:
                await _ipc_write_message(
                    writer,
                    _ipc_response_error(None, 'timeout', '读取请求超时'),
                    max_bytes=max_bytes,
                    timeout_sec=write_timeout_sec,
                )
                break
            except ValueError as e:
                code = str(e)
                await _ipc_write_message(
                    writer,
                    _ipc_response_error(None, code, '请求解析失败'),
                    max_bytes=max_bytes,
                    timeout_sec=write_timeout_sec,
                )
                break
            except Exception:
                await _ipc_write_message(
                    writer,
                    _ipc_response_error(None, 'internal_error', '读取请求失败'),
                    max_bytes=max_bytes,
                    timeout_sec=write_timeout_sec,
                )
                break

            handled += 1
            msg_type = msg.get('type')
            request_id = msg.get('id')
            request_timeout_sec = min(_ipc_extract_timeout_sec(msg, default_request_timeout_sec), default_request_timeout_sec)
            try:
                response = await asyncio.wait_for(_ipc_handle_message(msg, client_host), timeout=request_timeout_sec)
            except asyncio.TimeoutError:
                response = _ipc_response_error(request_id, 'timeout', '处理请求超时')
            except Exception as e:
                _logger.error(f'IPC处理异常 type={msg_type} id={request_id} err={e}')
                response = _ipc_response_error(request_id, 'internal_error', '处理请求失败')

            try:
                await _ipc_write_message(writer, response, max_bytes=max_bytes, timeout_sec=write_timeout_sec)
            except Exception:
                break
    finally:
        with suppress(Exception):
            writer.close()
        with suppress(Exception):
            await writer.wait_closed()
        _logger.info(f'IPC连接关闭: {client_host}')


async def start_ipc_server(host: Optional[str] = None, port: Optional[int] = None) -> Tuple[str, int]:
    global _ipc_server, _ipc_server_task
    if _ipc_server is not None:
        sock = _ipc_server.sockets[0] if _ipc_server.sockets else None
        if sock is None:
            return _get_ipc_host(), _get_ipc_port()
        addr = sock.getsockname()
        return addr[0], addr[1]

    bind_host = host if host is not None else _get_ipc_host()
    bind_port = port if port is not None else _get_ipc_port()
    server = await asyncio.start_server(_ipc_handle_client, host=bind_host, port=bind_port, start_serving=True)
    _ipc_server = server
    _ipc_server_task = asyncio.create_task(server.serve_forever())
    sock = server.sockets[0] if server.sockets else None
    addr = sock.getsockname() if sock else (bind_host, bind_port)
    _logger.info(f'IPC服务启动: {addr[0]}:{addr[1]}')
    return addr[0], addr[1]


async def stop_ipc_server() -> None:
    global _ipc_server, _ipc_server_task
    task = _ipc_server_task
    server = _ipc_server
    _ipc_server_task = None
    _ipc_server = None

    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    if server is not None:
        server.close()
        with suppress(Exception):
            await server.wait_closed()
        _logger.info('IPC服务已关闭')


async def run_ipc_forever(host: Optional[str] = None, port: Optional[int] = None) -> None:
    get_scanner()
    await start_ipc_server(host=host, port=port)
    try:
        await asyncio.Event().wait()
    finally:
        with suppress(Exception):
            await stop_ipc_server()
        _cleanup_environment()


if __name__ == '__main__':
    asyncio.run(run_ipc_forever())
''')

_register_embedded("collect_benign_pe", r'''import os
import shutil
import hashlib
import pefile
from typing import Optional

import config.config as cfg
from utils.path_utils import validate_path


def is_pe_file(file_path: str) -> bool:
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            return False
        with open(valid_path, 'rb') as f:
            sig = f.read(2)
            if sig != b'MZ':
                return False
        pe = pefile.PE(valid_path, fast_load=True)
        pe.close()
        return True
    except pefile.PEFormatError:
        return False
    except Exception:
        return False


def compute_sha256(file_path: str) -> Optional[str]:
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            return None
        h = hashlib.sha256()
        with open(valid_path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_pe_files(source_root: Optional[str] = None, dest_dir: Optional[str] = None) -> int:
    src_root = source_root or cfg.COLLECT_SOURCE_ROOT
    dst_dir = dest_dir or cfg.BENIGN_WHITELIST_PENDING_DIR
    os.makedirs(dst_dir, exist_ok=True)
    total_copied = 0
    for root, _, files in os.walk(src_root):
        for name in files:
            src_path = os.path.join(root, name)
            try:
                if cfg.COLLECT_MAX_FILE_SIZE and os.stat(src_path).st_size > cfg.COLLECT_MAX_FILE_SIZE:
                    continue
            except Exception:
                continue
            if not is_pe_file(src_path):
                continue
            digest = compute_sha256(src_path)
            if not digest:
                continue
            dst_path = os.path.join(dst_dir, digest)
            if os.path.exists(dst_path):
                continue
            try:
                shutil.copy2(src_path, dst_path)
                total_copied += 1
            except Exception:
                pass
    return total_copied


def main() -> None:
    copied = collect_pe_files()
    print(f"copied={copied}")


if __name__ == '__main__':
    main()
''')

_register_embedded("feature_extractor_enhanced", r'''import numpy as np
import pefile
import os
import sys
import json
import multiprocessing
import hashlib
import struct
from datetime import datetime
from config.config import DEFAULT_MAX_FILE_SIZE, PE_FEATURE_VECTOR_DIM, LIGHTWEIGHT_FEATURE_DIM, LIGHTWEIGHT_FEATURE_SCALE, SIZE_NORM_MAX, TIMESTAMP_MAX, TIMESTAMP_YEAR_BASE, TIMESTAMP_YEAR_MAX, SYSTEM_DLLS, COMMON_SECTIONS, ENTROPY_HIGH_THRESHOLD, ENTROPY_LOW_THRESHOLD, LARGE_TRAILING_DATA_SIZE

MAX_FILE_SIZE = DEFAULT_MAX_FILE_SIZE
PE_FEATURE_SIZE = 500

ERROR_COUNT = 0

def increment_error():
    global ERROR_COUNT
    ERROR_COUNT += 1

def validate_path(path):
    """
    Validates and normalizes a file path to prevent path traversal attacks.
    """
    if not path:
        return None
    
    normalized_path = os.path.normpath(path)
    
    if '\0' in normalized_path:
        return None
        
    if not os.path.exists(normalized_path):
        return None
        
    return normalized_path

def extract_byte_sequence(file_path):
    valid_path = validate_path(file_path)
    if not valid_path:
        return None

    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            raise ValueError("Invalid file path")

        with open(valid_path, 'rb') as f:
            f.seek(8)
            byte_sequence = np.fromfile(f, dtype=np.uint8, count=MAX_FILE_SIZE - 8)

        if len(byte_sequence) < MAX_FILE_SIZE - 8:
            padded_sequence = np.zeros(MAX_FILE_SIZE, dtype=np.uint8)
            padded_sequence[:len(byte_sequence)] = byte_sequence
            return padded_sequence

        full_sequence = np.zeros(MAX_FILE_SIZE, dtype=np.uint8)
        full_sequence[:len(byte_sequence)] = byte_sequence
        return full_sequence
    except Exception:
        increment_error()
        return None

def calculate_byte_entropy(byte_sequence, block_size=1024):
    if byte_sequence is None or len(byte_sequence) == 0:
        return 0, 0, 0, [], 0

    hist = np.bincount(byte_sequence, minlength=256)
    prob = hist / len(byte_sequence)
    prob = prob[prob > 0]
    overall_entropy = -np.sum(prob * np.log2(prob)) / 8

    block_entropies = []

    num_blocks = min(10, max(1, len(byte_sequence) // block_size))
    if num_blocks > 1:
        block_size = len(byte_sequence) // num_blocks
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size if i < num_blocks - 1 else len(byte_sequence)
            block = byte_sequence[start_idx:end_idx]
            if len(block) > 0:
                block_hist = np.bincount(block, minlength=256)
                block_prob = block_hist / len(block)
                block_prob = block_prob[block_prob > 0]
                if len(block_prob) > 0:
                    block_entropy = -np.sum(block_prob * np.log2(block_prob)) / 8
                    block_entropies.append(block_entropy)
    else:

        block = byte_sequence
        if len(block) > 0:
            block_hist = np.bincount(block, minlength=256)
            block_prob = block_hist / len(block)
            block_prob = block_prob[block_prob > 0]
            if len(block_prob) > 0:
                block_entropy = -np.sum(block_prob * np.log2(block_prob)) / 8
                block_entropies.append(block_entropy)

    if block_entropies:
        return overall_entropy, np.min(block_entropies), np.max(block_entropies), block_entropies, np.std(block_entropies)
    else:
        return overall_entropy, overall_entropy, overall_entropy, [], 0

def extract_file_attributes(file_path):
    features = {}
    missing_flags = {}

    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            raise ValueError("Invalid file path")

        stat = os.stat(valid_path)
        features['size'] = stat.st_size
        missing_flags['size_missing'] = 0

        features['log_size'] = np.log(stat.st_size + 1)
        missing_flags['log_size_missing'] = 0

        with open(valid_path, 'rb') as f:
            sample_data = np.fromfile(f, dtype=np.uint8, count=10240)

        avg_entropy, min_entropy, max_entropy, block_entropies, entropy_std = calculate_byte_entropy(sample_data)
        features['file_entropy_avg'] = avg_entropy
        features['file_entropy_min'] = min_entropy
        features['file_entropy_max'] = max_entropy
        features['file_entropy_range'] = max_entropy - min_entropy
        features['file_entropy_std'] = entropy_std
        missing_flags['file_entropy_missing'] = 0

        if block_entropies:
            features['file_entropy_q25'] = np.percentile(block_entropies, 25)
            features['file_entropy_q75'] = np.percentile(block_entropies, 75)
            features['file_entropy_median'] = np.median(block_entropies)
            missing_flags['file_entropy_percentiles_missing'] = 0

            high_entropy_count = sum(1 for e in block_entropies if e > ENTROPY_HIGH_THRESHOLD)
            features['high_entropy_ratio'] = high_entropy_count / len(block_entropies)
            low_entropy_count = sum(1 for e in block_entropies if e < ENTROPY_LOW_THRESHOLD)
            features['low_entropy_ratio'] = low_entropy_count / len(block_entropies)

            if len(block_entropies) > 1:
                entropy_changes = np.diff(block_entropies)
                features['entropy_change_rate'] = np.mean(np.abs(entropy_changes))
                features['entropy_change_std'] = np.std(entropy_changes)
            else:
                features['entropy_change_rate'] = 0
                features['entropy_change_std'] = 0
        else:
            features['file_entropy_q25'] = 0
            features['file_entropy_q75'] = 0
            features['file_entropy_median'] = 0
            features['high_entropy_ratio'] = 0
            features['low_entropy_ratio'] = 0
            features['entropy_change_rate'] = 0
            features['entropy_change_std'] = 0
            missing_flags['file_entropy_percentiles_missing'] = 1

        if len(sample_data) > 0:
            zero_ratio = np.sum(sample_data == 0) / len(sample_data)
            printable_ratio = np.sum((sample_data >= 32) & (sample_data <= 126)) / len(sample_data)
            features['zero_byte_ratio'] = zero_ratio
            features['printable_byte_ratio'] = printable_ratio
            missing_flags['byte_stats_missing'] = 0
        else:
            features['zero_byte_ratio'] = 0
            features['printable_byte_ratio'] = 0
            missing_flags['byte_stats_missing'] = 1

    except Exception:
        increment_error()

        feature_names = ['size', 'log_size', 'file_entropy_avg', 'file_entropy_min', 'file_entropy_max',
                        'file_entropy_range', 'file_entropy_std', 'file_entropy_q25', 'file_entropy_q75',
                        'file_entropy_median', 'high_entropy_ratio', 'low_entropy_ratio', 'entropy_change_rate',
                        'entropy_change_std', 'zero_byte_ratio', 'printable_byte_ratio']
        for name in feature_names:
            features[name] = 0
        missing_flags['size_missing'] = 1
        missing_flags['log_size_missing'] = 1
        missing_flags['file_entropy_missing'] = 1
        missing_flags['file_entropy_percentiles_missing'] = 1
        missing_flags['byte_stats_missing'] = 1

    features.update(missing_flags)
    return features

def extract_enhanced_pe_features(file_path):
    features = {}
    missing_flags = {}

    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            raise ValueError("Invalid file path")

        pe = pefile.PE(valid_path, fast_load=True)

        features['sections_count'] = len(pe.sections) if hasattr(pe, 'sections') else 0
        missing_flags['sections_count_missing'] = 0 if hasattr(pe, 'sections') else 1

        features['symbols_count'] = len(pe.SYMBOL_TABLE) if hasattr(pe, 'SYMBOL_TABLE') else 0
        missing_flags['symbols_count_missing'] = 0 if hasattr(pe, 'SYMBOL_TABLE') else 1

        features['imports_count'] = 0
        features['exports_count'] = 0

        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            imports = []
            api_names = []
            dll_names = []

            features['imports_count'] = len(pe.DIRECTORY_ENTRY_IMPORT)
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8').lower() if entry.dll else ''
                dll_names.append(dll_name)

                for imp in entry.imports:
                    if imp.name:
                        func_name = imp.name.decode('utf-8') if imp.name else ''
                        imports.append((dll_name, func_name))
                        api_names.append(func_name)

            features['unique_imports'] = len(set(imports))
            features['unique_dlls'] = len(set(dll_names))
            features['unique_apis'] = len(set(api_names))

            if dll_names:
                dll_name_lengths = [len(name) for name in dll_names if name]
                features['dll_name_avg_length'] = np.mean(dll_name_lengths)
                features['dll_name_max_length'] = np.max(dll_name_lengths)
                features['dll_name_min_length'] = np.min(dll_name_lengths)
                missing_flags['dll_stats_missing'] = 0
            else:
                missing_flags['dll_stats_missing'] = 1

            imported_system_dlls = set(dll.split('.')[0].lower() for dll in dll_names if dll) & set(SYSTEM_DLLS)
            features['imported_system_dlls_count'] = len(imported_system_dlls)
            missing_flags['imported_system_dlls_missing'] = 0
        else:
            features['unique_imports'] = 0
            features['unique_dlls'] = 0
            features['unique_apis'] = 0
            features['dll_name_avg_length'] = 0
            features['dll_name_max_length'] = 0
            features['dll_name_min_length'] = 0
            features['imported_system_dlls_count'] = 0
            missing_flags['dll_stats_missing'] = 1
            missing_flags['imported_system_dlls_missing'] = 1

        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            features['exports_count'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)

            export_names = []
            for symbol in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if symbol.name:
                    export_names.append(symbol.name.decode('utf-8'))

            if export_names:
                export_name_lengths = [len(name) for name in export_names]
                features['export_name_avg_length'] = np.mean(export_name_lengths)
                features['export_name_max_length'] = np.max(export_name_lengths)
                features['export_name_min_length'] = np.min(export_name_lengths)

                features['exports_density'] = len(export_names) / (features['size'] + 1)
                missing_flags['export_stats_missing'] = 0
            else:
                features['export_name_avg_length'] = 0
                features['export_name_max_length'] = 0
                features['export_name_min_length'] = 0
                features['exports_density'] = 0
                missing_flags['export_stats_missing'] = 1
        else:
            features['exports_count'] = 0
            features['export_name_avg_length'] = 0
            features['export_name_max_length'] = 0
            features['export_name_min_length'] = 0
            features['exports_density'] = 0
            missing_flags['export_stats_missing'] = 1

        if hasattr(pe, 'sections'):
            section_names = []
            section_sizes = []
            section_vsizes = []
            section_chars = []
            code_section_size = 0
            data_section_size = 0
            code_section_vsize = 0
            data_section_vsize = 0

            executable_sections_count = 0
            writable_sections_count = 0
            readable_sections_count = 0
            non_standard_executable_sections_count = 0
            rwx_sections_count = 0

            common_executable_section_names = {'.text', 'text', '.code'}

            for section in pe.sections:
                try:
                    name = section.Name.decode('utf-8').strip('\x00')
                    section_names.append(name)
                    section_sizes.append(section.SizeOfRawData)
                    section_vsizes.append(section.VirtualSize)
                    section_chars.append(section.Characteristics)

                    if section.Characteristics & 0x20000000:
                        executable_sections_count += 1
                        code_section_size += section.SizeOfRawData
                        code_section_vsize += section.VirtualSize

                        if name.lower() not in common_executable_section_names:
                            non_standard_executable_sections_count += 1
                    if section.Characteristics & 0x80000000:
                        writable_sections_count += 1
                    if section.Characteristics & 0x40000000:
                        readable_sections_count += 1
                        data_section_size += section.SizeOfRawData
                        data_section_vsize += section.VirtualSize

                    if (section.Characteristics & 0x20000000) and (section.Characteristics & 0x80000000):
                        features['executable_writable_sections'] = features.get('executable_writable_sections', 0) + 1
                        rwx_sections_count += 1
                except Exception:
                    increment_error()
                    pass

            features['section_names_count'] = len(section_names)
            features['section_total_size'] = sum(section_sizes)
            features['section_total_vsize'] = sum(section_vsizes)
            features['avg_section_size'] = np.mean(section_sizes) if section_sizes else 0
            features['avg_section_vsize'] = np.mean(section_vsizes) if section_vsizes else 0
            features['max_section_size'] = np.max(section_sizes) if section_sizes else 0
            features['min_section_size'] = np.min(section_sizes) if section_sizes else 0
            features['code_section_ratio'] = code_section_size / (features['section_total_size'] + 1)
            features['data_section_ratio'] = data_section_size / (features['section_total_size'] + 1)
            features['code_vsize_ratio'] = code_section_vsize / (features['section_total_vsize'] + 1)
            features['data_vsize_ratio'] = data_section_vsize / (features['section_total_vsize'] + 1)

            features['executable_sections_count'] = executable_sections_count
            features['writable_sections_count'] = writable_sections_count
            features['readable_sections_count'] = readable_sections_count
            features['executable_sections_ratio'] = executable_sections_count / (len(section_names) + 1)
            features['writable_sections_ratio'] = writable_sections_count / (len(section_names) + 1)
            features['readable_sections_ratio'] = readable_sections_count / (len(section_names) + 1)
            features['non_standard_executable_sections_count'] = non_standard_executable_sections_count
            features['rwx_sections_count'] = rwx_sections_count
            features['rwx_sections_ratio'] = rwx_sections_count / (len(section_names) + 1)

            if features['section_total_size'] > 0:
                features['executable_code_density'] = code_section_size / features['section_total_size']
            else:
                features['executable_code_density'] = 0

            if section_sizes:
                features['section_size_std'] = np.std(section_sizes)
                features['section_size_cv'] = np.std(section_sizes) / (np.mean(section_sizes) + 1e-8)
            else:
                features['section_size_std'] = 0
                features['section_size_cv'] = 0

            if section_names:
                section_name_lengths = [len(name) for name in section_names]
                features['section_name_avg_length'] = np.mean(section_name_lengths)
                features['section_name_max_length'] = np.max(section_name_lengths)
                features['section_name_min_length'] = np.min(section_name_lengths)
                missing_flags['section_name_stats_missing'] = 0
                lower_names = [n.lower() for n in section_names]
                features['has_upx_section'] = 1 if any('upx' in n for n in lower_names) else 0
                features['has_mpress_section'] = 1 if any('mpress' in n for n in lower_names) else 0
                features['has_aspack_section'] = 1 if any('aspack' in n for n in lower_names) else 0
                features['has_themida_section'] = 1 if any('themida' in n for n in lower_names) else 0
            else:
                features['section_name_avg_length'] = 0
                features['section_name_max_length'] = 0
                features['section_name_min_length'] = 0
                missing_flags['section_name_stats_missing'] = 1
                features['has_upx_section'] = 0
                features['has_mpress_section'] = 0
                features['has_aspack_section'] = 0
                features['has_themida_section'] = 0

            special_char_count = 0
            total_chars = 0
            for name in section_names:
                total_chars += len(name)
                for c in name:
                    if not (c.isalnum() or c in '_.'):
                        special_char_count += 1

            features['special_char_ratio'] = special_char_count / (total_chars + 1)

            long_sections = [name for name in section_names if len(name) > 6]
            short_sections = [name for name in section_names if len(name) < 3]
            features['long_sections_count'] = len(long_sections)
            features['short_sections_count'] = len(short_sections)
            features['long_sections_ratio'] = len(long_sections) / (len(section_names) + 1)
            features['short_sections_ratio'] = len(short_sections) / (len(section_names) + 1)
            missing_flags['sections_details_missing'] = 0
        else:
            features['section_name_avg_length'] = 0
            features['section_name_max_length'] = 0
            features['section_name_min_length'] = 0
            features['max_section_size'] = 0
            features['min_section_size'] = 0
            features['code_section_ratio'] = 0
            features['data_section_ratio'] = 0
            features['code_vsize_ratio'] = 0
            features['data_vsize_ratio'] = 0
            features['long_sections_count'] = 0
            features['short_sections_count'] = 0
            features['section_size_std'] = 0
            features['section_size_cv'] = 0
            features['executable_writable_sections'] = 0
            features['non_standard_executable_sections_count'] = 0
            features['rwx_sections_count'] = 0
            features['rwx_sections_ratio'] = 0.0
            missing_flags['section_name_stats_missing'] = 1
            missing_flags['sections_details_missing'] = 1

            for sec in COMMON_SECTIONS:
                features[f'has_{sec}_section'] = 0

        if hasattr(pe.OPTIONAL_HEADER, 'Subsystem'):
            features['subsystem'] = pe.OPTIONAL_HEADER.Subsystem
            missing_flags['subsystem_missing'] = 0
        else:
            features['subsystem'] = 0
            missing_flags['subsystem_missing'] = 1

        if hasattr(pe.OPTIONAL_HEADER, 'DllCharacteristics'):
            features['dll_characteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics

            features['has_nx_compat'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x100 else 0
            features['has_aslr'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x40 else 0
            features['has_seh'] = 1 if not (pe.OPTIONAL_HEADER.DllCharacteristics & 0x400) else 0
            features['has_guard_cf'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x4000 else 0
            missing_flags['dll_characteristics_missing'] = 0
        else:
            features['dll_characteristics'] = 0
            features['has_nx_compat'] = 0
            features['has_aslr'] = 0
            features['has_seh'] = 0
            features['has_guard_cf'] = 0
            missing_flags['dll_characteristics_missing'] = 1

        features['has_resources'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') else 0
        features['has_debug_info'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') else 0
        features['has_tls'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_TLS') else 0
        features['has_relocs'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') else 0
        features['has_exceptions'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_EXCEPTION') else 0
        try:
            sec_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_SECURITY']]
            sig_size = getattr(sec_dir, 'Size', 0)
            features['has_signature'] = 1 if sig_size and sig_size > 0 else 0
            features['signature_size'] = sig_size if sig_size else 0
            try:
                va = getattr(sec_dir, 'VirtualAddress', 0)
                sz = getattr(sec_dir, 'Size', 0)
                if va and sz:
                    with open(valid_path, 'rb') as f:
                        f.seek(va)
                        blob = f.read(sz)
                    has_st = (b'signingTime' in blob) or (b'1.2.840.113549.1.9.5' in blob)
                    features['signature_has_signing_time'] = 1 if has_st else 0
                else:
                    features['signature_has_signing_time'] = 0
            except Exception:
                features['signature_has_signing_time'] = 0
        except Exception:
            features['has_signature'] = 0
            features['signature_size'] = 0
            features['signature_has_signing_time'] = 0
        version_info_present = 0
        company_name_len = 0
        product_name_len = 0
        file_version_len = 0
        original_filename_len = 0
        try:
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']])
            if hasattr(pe, 'FileInfo'):
                for fi in pe.FileInfo:
                    if hasattr(fi, 'StringTable'):
                        for st in fi.StringTable:
                            if hasattr(st, 'entries'):
                                version_info_present = 1
                                for k, v in st.entries.items():
                                    key = k.strip().lower()
                                    val = v.strip() if isinstance(v, str) else ''
                                    if key == 'companyname':
                                        company_name_len = max(company_name_len, len(val))
                                    elif key == 'productname':
                                        product_name_len = max(product_name_len, len(val))
                                    elif key == 'fileversion':
                                        file_version_len = max(file_version_len, len(val))
                                    elif key == 'originalfilename':
                                        original_filename_len = max(original_filename_len, len(val))
        except Exception:
            pass
        features['version_info_present'] = version_info_present
        features['company_name_len'] = company_name_len
        features['product_name_len'] = product_name_len
        features['file_version_len'] = file_version_len
        features['original_filename_len'] = original_filename_len
        missing_flags['directory_entries_missing'] = 0 if (hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') or
                                                          hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') or
                                                          hasattr(pe, 'DIRECTORY_ENTRY_TLS') or
                                                          hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') or
                                                          hasattr(pe, 'DIRECTORY_ENTRY_EXCEPTION')) else 1

        try:
            with open(valid_path, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
                pe_end_offset = pe.sections[-1].PointerToRawData + pe.sections[-1].SizeOfRawData if hasattr(pe, 'sections') and pe.sections else file_size

                trailing_data_size = file_size - pe_end_offset
                features['trailing_data_size'] = trailing_data_size
                features['trailing_data_ratio'] = trailing_data_size / (file_size + 1)

                features['has_large_trailing_data'] = 1 if trailing_data_size > LARGE_TRAILING_DATA_SIZE else 0
                missing_flags['trailing_data_missing'] = 0
        except Exception:
            increment_error()
            features['trailing_data_size'] = 0
            features['trailing_data_ratio'] = 0
            features['has_large_trailing_data'] = 0
            missing_flags['trailing_data_missing'] = 1

        try:
            pe_header_size = pe.OPTIONAL_HEADER.SizeOfHeaders
            features['pe_header_size'] = pe_header_size
            features['header_size_ratio'] = pe_header_size / (features['size'] + 1)
            missing_flags['header_info_missing'] = 0
        except Exception:
            increment_error()
            features['pe_header_size'] = 0
            features['header_size_ratio'] = 0
            missing_flags['header_info_missing'] = 1

    except Exception as e:
        increment_error()
        default_keys = [
            'sections_count', 'symbols_count',
            'imports_count', 'exports_count', 'unique_imports', 'unique_dlls',
            'unique_apis', 'section_names_count', 'section_total_size',
            'section_total_vsize', 'avg_section_size', 'avg_section_vsize',
            'subsystem', 'dll_characteristics', 'code_section_ratio',
            'data_section_ratio', 'code_vsize_ratio', 'data_vsize_ratio',
            'has_nx_compat', 'has_aslr', 'has_seh', 'has_guard_cf', 'has_resources',
            'has_debug_info', 'has_tls', 'has_relocs', 'has_exceptions',
            'dll_name_avg_length', 'dll_name_max_length', 'dll_name_min_length',
            'section_name_avg_length', 'section_name_max_length', 'section_name_min_length',
            'export_name_avg_length', 'export_name_max_length', 'export_name_min_length',
            'max_section_size', 'min_section_size', 'entry_point_ratio',
            'long_sections_count', 'short_sections_count',
            'section_size_std', 'section_size_cv', 'executable_writable_sections',
            'file_entropy_avg', 'file_entropy_min', 'file_entropy_max', 'file_entropy_range',
            'zero_byte_ratio', 'printable_byte_ratio', 'trailing_data_size', 'trailing_data_ratio',
            'imported_system_dlls_count', 'exports_density',
            'has_large_trailing_data', 'pe_header_size', 'header_size_ratio',
            'file_entropy_std', 'file_entropy_q25', 'file_entropy_q75',
            'file_entropy_median', 'high_entropy_ratio', 'low_entropy_ratio',
            'entropy_change_rate', 'entropy_change_std',
            'executable_sections_count', 'writable_sections_count', 'readable_sections_count',
            'executable_sections_ratio', 'writable_sections_ratio', 'readable_sections_ratio',
            'executable_code_density',
            'non_standard_executable_sections_count', 'rwx_sections_count', 'rwx_sections_ratio',
            'special_char_ratio', 'long_sections_ratio', 'short_sections_ratio'
        ]

        for key in default_keys:
            features[key] = 0

        for sec in COMMON_SECTIONS:
            features[f'has_{sec}_section'] = 0

        missing_flag_names = ['sections_count_missing', 'symbols_count_missing', 'dll_stats_missing',
                             'imported_system_dlls_missing', 'export_stats_missing', 'section_name_stats_missing',
                             'sections_details_missing', 'subsystem_missing', 'dll_characteristics_missing',
                             'directory_entries_missing', 'trailing_data_missing', 'header_info_missing']
        for flag in missing_flag_names:
            missing_flags[flag] = 1

    features.update(missing_flags)
    return features

def extract_lightweight_pe_features(file_path):

    feature_vector = np.zeros(256, dtype=np.float32)
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            return feature_vector

        pe = pefile.PE(valid_path, fast_load=True)

        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                if entry.dll:
                    dll_name = entry.dll.decode('utf-8').lower()
                    dll_hash = int(hashlib.sha256(dll_name.encode('utf-8')).hexdigest(), 16)
                    feature_vector[dll_hash % 128] = 1

            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        api_name = imp.name.decode('utf-8')
                        api_hash = int(hashlib.sha256(api_name.encode('utf-8')).hexdigest(), 16)
                        feature_vector[128 + (api_hash % 128)] = 1

        if hasattr(pe, 'sections'):
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', 'ignore').strip('\x00')
                section_hash = int(hashlib.sha256(section_name.encode('utf-8')).hexdigest(), 16)
                feature_vector[section_hash % 32 + 224] = 1

        norm = np.linalg.norm(feature_vector)
        if norm > 0 and not np.isnan(norm):
            feature_vector /= norm

        return feature_vector

    except Exception:
        increment_error()
        return feature_vector

def extract_combined_pe_features(file_path):
    lightweight_features = extract_lightweight_pe_features(file_path)

    enhanced_features = extract_enhanced_pe_features(file_path)
    file_attrs = extract_file_attributes(file_path)

    all_features = {}
    all_features.update(enhanced_features)
    all_features.update(file_attrs)

    combined_vector = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)

    combined_vector[:LIGHTWEIGHT_FEATURE_DIM] = lightweight_features * LIGHTWEIGHT_FEATURE_SCALE

    max_file_size = SIZE_NORM_MAX
    max_timestamp = TIMESTAMP_MAX

    feature_order = [
        'size', 'log_size', 'sections_count', 'symbols_count', 'imports_count', 'exports_count',
        'unique_imports', 'unique_dlls', 'unique_apis', 'section_names_count', 'section_total_size',
        'section_total_vsize', 'avg_section_size', 'avg_section_vsize', 'subsystem', 'dll_characteristics',
        'code_section_ratio', 'data_section_ratio', 'code_vsize_ratio', 'data_vsize_ratio',
        'has_nx_compat', 'has_aslr', 'has_seh', 'has_guard_cf', 'has_resources', 'has_debug_info',
        'has_tls', 'has_relocs', 'has_exceptions', 'dll_name_avg_length', 'dll_name_max_length',
        'dll_name_min_length', 'section_name_avg_length', 'section_name_max_length', 'section_name_min_length',
        'export_name_avg_length', 'export_name_max_length', 'export_name_min_length', 'max_section_size',
        'min_section_size', 'long_sections_count', 'short_sections_count', 'section_size_std', 'section_size_cv',
        'executable_writable_sections', 'file_entropy_avg', 'file_entropy_min', 'file_entropy_max',
        'file_entropy_range', 'zero_byte_ratio', 'printable_byte_ratio', 'trailing_data_size',
        'trailing_data_ratio', 'imported_system_dlls_count', 'exports_density', 'has_large_trailing_data',
        'pe_header_size', 'header_size_ratio', 'file_entropy_std', 'file_entropy_q25', 'file_entropy_q75',
        'file_entropy_median', 'high_entropy_ratio', 'low_entropy_ratio', 'entropy_change_rate',
        'entropy_change_std', 'executable_sections_count', 'writable_sections_count', 'readable_sections_count',
        'executable_sections_ratio', 'writable_sections_ratio', 'readable_sections_ratio', 'executable_code_density',
        'non_standard_executable_sections_count', 'rwx_sections_count', 'rwx_sections_ratio',
        'special_char_ratio', 'long_sections_ratio', 'short_sections_ratio',

        'size_missing', 'log_size_missing', 'file_entropy_missing', 'file_entropy_percentiles_missing',
        'byte_stats_missing', 'sections_count_missing', 'symbols_count_missing', 'dll_stats_missing',
        'imported_system_dlls_missing', 'export_stats_missing', 'section_name_stats_missing',
        'sections_details_missing', 'subsystem_missing', 'dll_characteristics_missing',
        'directory_entries_missing', 'trailing_data_missing', 'header_info_missing'
    ]

    common_sections = ['.text', '.data', '.rdata', '.reloc', '.rsrc']
    for sec in common_sections:
        feature_order.append(f'has_{sec}_section')
    feature_order.extend([
        'has_signature','signature_size','signature_has_signing_time','version_info_present','company_name_len','product_name_len','file_version_len','original_filename_len',
        'has_upx_section','has_mpress_section','has_aspack_section','has_themida_section','timestamp','timestamp_year'
    ])

    for i, key in enumerate(feature_order):
        if 256 + i >= PE_FEATURE_VECTOR_DIM:
            break

        if key in all_features:
            val = all_features[key]
            if 'size' in key and isinstance(val, (int, float)):
                val = val / max_file_size
            elif key == 'timestamp' and isinstance(val, (int, float)):
                val = val / max_timestamp
            elif key == 'timestamp_year' and isinstance(val, (int, float)):
                val = (val - TIMESTAMP_YEAR_BASE) / (TIMESTAMP_YEAR_MAX - TIMESTAMP_YEAR_BASE)
            elif key.startswith('has_') and isinstance(val, (int, float)):
                val = float(val)
            elif key == 'log_size' and isinstance(val, (int, float)):
                val = val / np.log(max_file_size)

            combined_vector[256 + i] = val * 0.8 if isinstance(val, (int, float)) else 0

    norm = np.linalg.norm(combined_vector)
    if norm > 0 and not np.isnan(norm):
        combined_vector /= norm
    else:
        combined_vector = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)

    return combined_vector

def process_file_worker(args):
    file_path, label, output_dir = args
    before = ERROR_COUNT

    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            raise ValueError("Invalid file path")

        with open(valid_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        increment_error()
        return {'filename': 'unknown', 'status': 'failed', 'error': f'Could not read file: {file_path}', 'errors': ERROR_COUNT - before}

    filename = file_hash
    output_npz_path = os.path.join(output_dir, filename + '.npz')

    if os.path.exists(output_npz_path):
        return {'filename': filename, 'status': 'skipped', 'label': label, 'errors': ERROR_COUNT - before}

    byte_sequence = extract_byte_sequence(file_path)
    if byte_sequence is None:
        return {'filename': filename, 'status': 'failed', 'error': 'Could not read byte sequence.', 'errors': ERROR_COUNT - before}

    pe_features = extract_combined_pe_features(file_path)

    try:
        np.savez_compressed(
            output_npz_path,
            byte_sequence=byte_sequence,
            pe_features=pe_features
        )
        return {'filename': filename, 'status': 'success', 'label': label, 'errors': ERROR_COUNT - before}
    except Exception as e:
        increment_error()
        return {'filename': filename, 'status': 'failed', 'error': str(e), 'errors': ERROR_COUNT - before}

def extract_features_in_memory(input_file_path, max_file_size=DEFAULT_MAX_FILE_SIZE):

    global MAX_FILE_SIZE
    original_max_size = MAX_FILE_SIZE
    MAX_FILE_SIZE = max_file_size

    try:
        byte_sequence = extract_byte_sequence(input_file_path)
        if byte_sequence is None:
            raise Exception("Failed to extract byte sequence")

        pe_features = extract_combined_pe_features(input_file_path)

        return byte_sequence, pe_features
    except Exception as e:
        increment_error()
        print(f"[!] Failed to extract in-memory features for file {input_file_path}: {e}")
        return None, None
    finally:
        MAX_FILE_SIZE = original_max_size

def process_file_directory(input_file_path, output_file_path, max_file_size=DEFAULT_MAX_FILE_SIZE):

    byte_sequence, pe_features = extract_features_in_memory(input_file_path, max_file_size)
    if byte_sequence is None or pe_features is None:
        raise Exception(f"Failed to process file {input_file_path}")

    np.savez_compressed(
        output_file_path,
        byte_sequence=byte_sequence,
        pe_features=pe_features
    )

    #print(f"[+] Successfully processed file: {input_file_path} -> {output_file_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    from tqdm import tqdm

    BENIGN_DIR = os.path.join(base_dir, 'benign_samples')
    MALICIOUS_DIR = os.path.join(base_dir, 'malicious_samples')

    PROCESSED_DATA_DIR = os.path.join(base_dir, 'data', 'processed_lightgbm')
    METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'metadata.json')

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    def collect_tasks_recursive(base_directory, output_dir):
        tasks = []
        if not os.path.isdir(base_directory): return tasks
        label = 'benign' if 'benign_samples' in base_directory else 'malicious'
        for root, _, files in os.walk(base_directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                tasks.append((file_path, label, output_dir))
        return tasks

    benign_tasks = collect_tasks_recursive(BENIGN_DIR, PROCESSED_DATA_DIR)
    malicious_tasks = collect_tasks_recursive(MALICIOUS_DIR, PROCESSED_DATA_DIR)
    all_tasks = benign_tasks + malicious_tasks

    if not all_tasks:
        print("\n[!] No sample files found.")
        sys.exit()

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_file_worker, all_tasks), total=len(all_tasks)))

    file_to_label = {}

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            file_to_label = json.load(f)

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for r in results:
        status = r.get('status')
        if status == 'success':
            success_count += 1
            file_to_label[r['filename'] + '.npz'] = r['label']
        elif status == 'skipped':
            skipped_count += 1
            if r['filename'] + '.npz' not in file_to_label:
                 file_to_label[r['filename'] + '.npz'] = r['label']
        else:
            failed_count += 1

    with open(METADATA_FILE, 'w') as f:
        json.dump(file_to_label, f, indent=4)

    total_errors = sum(r.get('errors', 0) for r in results)
    print(f"\n[!] 本次运行共捕获到 {total_errors} 个错误")
    try:
        tds = getattr(pe.FILE_HEADER, 'TimeDateStamp', 0)
        features['timestamp'] = int(tds) if tds else 0
        from datetime import datetime
        features['timestamp_year'] = datetime.utcfromtimestamp(int(tds)).year if tds else 0
    except Exception:
        features['timestamp'] = 0
        features['timestamp_year'] = 0
''')

_register_embedded("cli", r'''import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from pipeline import main

if __name__ == "__main__":
    main()
''')

_install_embedded_importer()

from config.config import (
    MODEL_PATH, FAMILY_CLASSIFIER_PATH, FEATURES_PKL_PATH, PROCESSED_DATA_DIR, METADATA_FILE,
    BENIGN_SAMPLES_DIR, MALICIOUS_SAMPLES_DIR,
    DEFAULT_MAX_FILE_SIZE, DEFAULT_NUM_BOOST_ROUND, DEFAULT_INCREMENTAL_ROUNDS,
    DEFAULT_INCREMENTAL_EARLY_STOPPING, DEFAULT_MAX_FINETUNE_ITERATIONS,
    DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_MIN_SAMPLES, DEFAULT_MIN_FAMILY_SIZE,
    DEFAULT_TREAT_NOISE_AS_FAMILY,
    SCAN_CACHE_PATH, SCAN_OUTPUT_DIR, HDBSCAN_SAVE_DIR,
    HELP_MAX_FILE_SIZE, HELP_FAST_DEV_RUN, HELP_SAVE_FEATURES,
    HELP_FINETUNE_ON_FALSE_POSITIVES, HELP_INCREMENTAL_TRAINING,
    HELP_INCREMENTAL_DATA_DIR, HELP_INCREMENTAL_RAW_DATA_DIR, HELP_FILE_EXTENSIONS,
    HELP_LABEL_INFERENCE, HELP_NUM_BOOST_ROUND, HELP_INCREMENTAL_ROUNDS,
    HELP_INCREMENTAL_EARLY_STOPPING, HELP_MAX_FINETUNE_ITERATIONS,
    HELP_USE_EXISTING_FEATURES, HELP_DATA_DIR, HELP_FEATURES_PATH, HELP_SAVE_DIR,
    HELP_MIN_CLUSTER_SIZE, HELP_MIN_SAMPLES, HELP_MIN_FAMILY_SIZE, HELP_PLOT_PCA,
    HELP_EXPLAIN_DISCREPANCY, HELP_TREAT_NOISE_AS_FAMILY, HELP_LIGHTGBM_MODEL_PATH,
    HELP_FAMILY_CLASSIFIER_PATH, HELP_CACHE_FILE, HELP_FILE_PATH, HELP_DIR_PATH,
    HELP_RECURSIVE, HELP_OUTPUT_PATH,
    HELP_AUTOML_METHOD, HELP_AUTOML_TRIALS, HELP_AUTOML_CV, HELP_AUTOML_METRIC, HELP_AUTOML_FAST_DEV_RUN, HELP_SKIP_TUNING,
    AUTOML_METHOD_DEFAULT, AUTOML_TRIALS_DEFAULT, AUTOML_CV_FOLDS_DEFAULT, AUTOML_METRIC_DEFAULT,
    DETECTED_MALICIOUS_PATHS_REPORT_PATH
)
from utils.logging_utils import get_logger

def _serve_ipc_only() -> str:
    import scanner_service
    asyncio.run(scanner_service.run_ipc_forever())
    return 'ipc'

def main():
    logger = get_logger('kolo')
    parser = argparse.ArgumentParser(prog='KoloVirusDetector', description='KoloVirusDetector 项目入口')
    subs = parser.add_subparsers(dest='command', required=True)

    sp_pretrain = subs.add_parser('pretrain', help='预训练LightGBM模型')
    sp_pretrain.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)
    sp_pretrain.add_argument('--fast-dev-run', action='store_true', help=HELP_FAST_DEV_RUN)
    sp_pretrain.add_argument('--save-features', action='store_true', help=HELP_SAVE_FEATURES)
    sp_pretrain.add_argument('--finetune-on-false-positives', action='store_true', help=HELP_FINETUNE_ON_FALSE_POSITIVES)
    sp_pretrain.add_argument('--incremental-training', action='store_true', help=HELP_INCREMENTAL_TRAINING)
    sp_pretrain.add_argument('--incremental-data-dir', type=str, help=HELP_INCREMENTAL_DATA_DIR)
    sp_pretrain.add_argument('--incremental-raw-data-dir', type=str, help=HELP_INCREMENTAL_RAW_DATA_DIR)
    sp_pretrain.add_argument('--file-extensions', type=str, nargs='+', help=HELP_FILE_EXTENSIONS)
    sp_pretrain.add_argument('--label-inference', type=str, default='filename', choices=['filename', 'directory'], help=HELP_LABEL_INFERENCE)
    sp_pretrain.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND, help=HELP_NUM_BOOST_ROUND)
    sp_pretrain.add_argument('--incremental-rounds', type=int, default=DEFAULT_INCREMENTAL_ROUNDS, help=HELP_INCREMENTAL_ROUNDS)
    sp_pretrain.add_argument('--incremental-early-stopping', type=int, default=DEFAULT_INCREMENTAL_EARLY_STOPPING, help=HELP_INCREMENTAL_EARLY_STOPPING)
    sp_pretrain.add_argument('--max-finetune-iterations', type=int, default=DEFAULT_MAX_FINETUNE_ITERATIONS, help=HELP_MAX_FINETUNE_ITERATIONS)
    sp_pretrain.add_argument('--use-existing-features', action='store_true', help=HELP_USE_EXISTING_FEATURES)

    sp_finetune = subs.add_parser('finetune', help='HDBSCAN 家族发现与分类器训练')
    sp_finetune.add_argument('--data-dir', type=str, default=PROCESSED_DATA_DIR, help=HELP_DATA_DIR)
    sp_finetune.add_argument('--features-path', type=str, default=FEATURES_PKL_PATH, help=HELP_FEATURES_PATH)
    sp_finetune.add_argument('--save-dir', type=str, default=HDBSCAN_SAVE_DIR, help=HELP_SAVE_DIR)
    sp_finetune.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)
    sp_finetune.add_argument('--min-cluster-size', type=int, default=DEFAULT_MIN_CLUSTER_SIZE, help=HELP_MIN_CLUSTER_SIZE)
    sp_finetune.add_argument('--min-samples', type=int, default=DEFAULT_MIN_SAMPLES, help=HELP_MIN_SAMPLES)
    sp_finetune.add_argument('--min-family-size', type=int, default=DEFAULT_MIN_FAMILY_SIZE, help=HELP_MIN_FAMILY_SIZE)
    sp_finetune.add_argument('--plot-pca', action='store_true', help=HELP_PLOT_PCA)
    sp_finetune.add_argument('--explain-discrepancy', action='store_true', help=HELP_EXPLAIN_DISCREPANCY)
    sp_finetune.add_argument('--treat-noise-as-family', action='store_true', default=DEFAULT_TREAT_NOISE_AS_FAMILY, help=HELP_TREAT_NOISE_AS_FAMILY)

    sp_scan = subs.add_parser('scan', help='单次扫描或目录扫描')
    sp_scan.add_argument('--lightgbm-model-path', type=str, default=MODEL_PATH, help=HELP_LIGHTGBM_MODEL_PATH)
    sp_scan.add_argument('--family-classifier-path', type=str, default=FAMILY_CLASSIFIER_PATH, help=HELP_FAMILY_CLASSIFIER_PATH)
    sp_scan.add_argument('--cache-file', type=str, default=SCAN_CACHE_PATH, help=HELP_CACHE_FILE)
    sp_scan.add_argument('--file-path', type=str, help=HELP_FILE_PATH)
    sp_scan.add_argument('--dir-path', type=str, help=HELP_DIR_PATH)
    sp_scan.add_argument('--recursive', action='store_true', help=HELP_RECURSIVE)
    sp_scan.add_argument('--output-path', type=str, default=SCAN_OUTPUT_DIR, help=HELP_OUTPUT_PATH)
    sp_scan.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)

    sp_extract = subs.add_parser('extract', help='从默认样本目录提取并生成处理数据')
    sp_extract.add_argument('--output-dir', type=str, default=PROCESSED_DATA_DIR, help=HELP_SAVE_DIR)
    sp_extract.add_argument('--file-extensions', type=str, nargs='+', help=HELP_FILE_EXTENSIONS)
    sp_extract.add_argument('--label-inference', type=str, default='directory', choices=['filename', 'directory'], help=HELP_LABEL_INFERENCE)
    sp_extract.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)

    subs.add_parser('serve', help='启动IPC扫描服务')

    sp_train_routing = subs.add_parser('train-routing', help='训练路由门控与专家模型系统')
    sp_train_routing.add_argument('--use-existing-features', action='store_true', help=HELP_USE_EXISTING_FEATURES)
    sp_train_routing.add_argument('--save-features', action='store_true', help=HELP_SAVE_FEATURES)
    sp_train_routing.add_argument('--fast-dev-run', action='store_true', help=HELP_FAST_DEV_RUN)
    sp_train_routing.add_argument('--finetune-on-false-positives', action='store_true', help=HELP_FINETUNE_ON_FALSE_POSITIVES)
    sp_train_routing.add_argument('--incremental-training', action='store_true', help=HELP_INCREMENTAL_TRAINING)
    sp_train_routing.add_argument('--incremental-data-dir', type=str, help=HELP_INCREMENTAL_DATA_DIR)
    sp_train_routing.add_argument('--incremental-raw-data-dir', type=str, help=HELP_INCREMENTAL_RAW_DATA_DIR)
    sp_train_routing.add_argument('--file-extensions', type=str, nargs='+', help=HELP_FILE_EXTENSIONS)
    sp_train_routing.add_argument('--label-inference', type=str, default='filename', choices=['filename', 'directory'], help=HELP_LABEL_INFERENCE)
    sp_train_routing.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND, help=HELP_NUM_BOOST_ROUND)
    sp_train_routing.add_argument('--incremental-rounds', type=int, default=DEFAULT_INCREMENTAL_ROUNDS, help=HELP_INCREMENTAL_ROUNDS)
    sp_train_routing.add_argument('--incremental-early-stopping', type=int, default=DEFAULT_INCREMENTAL_EARLY_STOPPING, help=HELP_INCREMENTAL_EARLY_STOPPING)
    sp_train_routing.add_argument('--max-finetune-iterations', type=int, default=DEFAULT_MAX_FINETUNE_ITERATIONS, help=HELP_MAX_FINETUNE_ITERATIONS)
    sp_train_routing.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)
    
    sp_train_all = subs.add_parser('train-all', help='一键执行特征提取、模型训练、评估与聚类')
    sp_train_all.add_argument('--finetune-on-false-positives', action='store_true', help=HELP_FINETUNE_ON_FALSE_POSITIVES)
    sp_train_all.add_argument('--skip-tuning', action='store_true', help=HELP_SKIP_TUNING)
    
    sp_autotune = subs.add_parser('auto-tune', help='AutoML超参调优与交叉测试对比')
    sp_autotune.add_argument('--method', type=str, default=AUTOML_METHOD_DEFAULT, choices=['optuna', 'hyperopt'], help=HELP_AUTOML_METHOD)
    sp_autotune.add_argument('--trials', type=int, default=AUTOML_TRIALS_DEFAULT, help=HELP_AUTOML_TRIALS)
    sp_autotune.add_argument('--cv', type=int, default=AUTOML_CV_FOLDS_DEFAULT, help=HELP_AUTOML_CV)
    sp_autotune.add_argument('--metric', type=str, default=AUTOML_METRIC_DEFAULT, choices=['roc_auc', 'accuracy', 'f1', 'precision', 'recall'], help=HELP_AUTOML_METRIC)
    sp_autotune.add_argument('--use-existing-features', action='store_true', help=HELP_USE_EXISTING_FEATURES)
    sp_autotune.add_argument('--fast-dev-run', action='store_true', help=HELP_AUTOML_FAST_DEV_RUN)
    sp_autotune.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)

    args = parser.parse_args()

    if args.command == 'pretrain':
        import pretrain
        try:
            pretrain.main(args)
        except Exception as e:
            logger.error(f'预训练失败: {e}')
            raise
    elif args.command == 'finetune':
        import finetune
        try:
            finetune.main(args)
        except Exception as e:
            logger.error(f'微调失败: {e}')
            raise
    elif args.command == 'scan':
        import scanner
        try:
            scanner_instance = scanner.MalwareScanner(
                lightgbm_model_path=args.lightgbm_model_path,
                family_classifier_path=args.family_classifier_path,
                max_file_size=args.max_file_size,
                cache_file=args.cache_file,
                enable_cache=True,
            )
            results = []
            if args.file_path:
                result = scanner_instance.scan_file(args.file_path)
                if result is not None:
                    results.append(result)
            elif args.dir_path:
                results = scanner_instance.scan_directory(args.dir_path, args.recursive)
            else:
                logger.error('请指定 --file-path 或 --dir-path')
                return
            scanner_instance.save_results(results, args.output_path)
            malicious_paths = [r['file_path'] for r in results if r.get('is_malware')]
            os.makedirs(os.path.dirname(DETECTED_MALICIOUS_PATHS_REPORT_PATH), exist_ok=True)
            with open(DETECTED_MALICIOUS_PATHS_REPORT_PATH, 'w', encoding='utf-8') as f:
                for p in malicious_paths:
                    f.write(p + '\n')
            logger.info(f'恶意样本路径已保存: {DETECTED_MALICIOUS_PATHS_REPORT_PATH}，数量: {len(malicious_paths)}')
        except Exception as e:
            logger.error(f'扫描失败: {e}')
            raise
    elif args.command == 'extract':
        from training.data_loader import extract_features_from_raw_files
        try:
            sources = [BENIGN_SAMPLES_DIR, MALICIOUS_SAMPLES_DIR]
            all_files = []
            all_labels = []
            for src in sources:
                file_names, labels = extract_features_from_raw_files(
                    src, args.output_dir, args.max_file_size, args.file_extensions, args.label_inference
                )
                all_files.extend(file_names)
                all_labels.extend(labels)
            if all_files:
                import json
                os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
                mapping = {fn: ('benign' if lab == 0 else 'malicious') for fn, lab in zip(all_files, all_labels)}
                with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
                logger.info(f'已生成元数据: {METADATA_FILE}，样本数: {len(all_files)}')
        except Exception as e:
            logger.error(f'提取失败: {e}')
            raise
    elif args.command == 'serve':
        try:
            _serve_ipc_only()
        except Exception as e:
            logger.error(f'服务启动失败: {e}')
            raise
    elif args.command == 'train-routing':
        from training import train_routing
        try:
            train_routing.main(args)
        except Exception as e:
            logger.error(f'路由系统训练失败: {e}')
            raise
    elif args.command == 'train-all':
        import pretrain
        from training import train_routing
        import finetune
        try:
            if not os.path.exists(METADATA_FILE):
                logger.info(f"[*] 元数据文件不存在: {METADATA_FILE}，将自动开始特征提取...")
                from training.data_loader import extract_features_from_raw_files
                import json
                sources = [BENIGN_SAMPLES_DIR, MALICIOUS_SAMPLES_DIR]
                all_files = []
                all_labels = []
                for src in sources:
                    if os.path.exists(src):
                        file_names, labels = extract_features_from_raw_files(
                            src, PROCESSED_DATA_DIR, DEFAULT_MAX_FILE_SIZE, None, 'filename'
                        )
                        all_files.extend(file_names)
                        all_labels.extend(labels)
                if all_files:
                    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
                    mapping = {fn: ('benign' if lab == 0 else 'malicious') for fn, lab in zip(all_files, all_labels)}
                    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                        json.dump(mapping, f, ensure_ascii=False, indent=2)
                    logger.info(f'已自动生成元数据: {METADATA_FILE}，样本数: {len(all_files)}')
                else:
                    logger.warning("未找到原始样本，跳过自动提取。")

            pre_args = argparse.Namespace(
                max_file_size=DEFAULT_MAX_FILE_SIZE,
                fast_dev_run=False,
                save_features=True,
                finetune_on_false_positives=args.finetune_on_false_positives,
                incremental_training=False,
                incremental_data_dir=None,
                incremental_raw_data_dir=None,
                file_extensions=None,
                label_inference='filename',
                num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                incremental_rounds=DEFAULT_INCREMENTAL_ROUNDS,
                incremental_early_stopping=DEFAULT_INCREMENTAL_EARLY_STOPPING,
                max_finetune_iterations=DEFAULT_MAX_FINETUNE_ITERATIONS,
                use_existing_features=True
            )
            pretrain.main(pre_args)
            from training import automl
            override_params = None
            if not args.skip_tuning and os.path.exists(FEATURES_PKL_PATH):
                auto_args = argparse.Namespace(
                    method=AUTOML_METHOD_DEFAULT,
                    trials=AUTOML_TRIALS_DEFAULT,
                    cv=AUTOML_CV_FOLDS_DEFAULT,
                    metric=AUTOML_METRIC_DEFAULT,
                    use_existing_features=True,
                    fast_dev_run=True,
                    max_file_size=DEFAULT_MAX_FILE_SIZE
                )
                auto_result = automl.main(auto_args)
                override_params = auto_result.get('best_params', None)
                if override_params:
                    pre_args2 = argparse.Namespace(
                        max_file_size=DEFAULT_MAX_FILE_SIZE,
                        fast_dev_run=False,
                        save_features=True,
                        finetune_on_false_positives=args.finetune_on_false_positives,
                        incremental_training=False,
                        incremental_data_dir=None,
                        incremental_raw_data_dir=None,
                        file_extensions=None,
                        label_inference='filename',
                        num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                        incremental_rounds=DEFAULT_INCREMENTAL_ROUNDS,
                        incremental_early_stopping=DEFAULT_INCREMENTAL_EARLY_STOPPING,
                        max_finetune_iterations=DEFAULT_MAX_FINETUNE_ITERATIONS,
                        use_existing_features=True,
                        override_params=override_params
                    )
                    pretrain.main(pre_args2)
            routing_args = argparse.Namespace(
                use_existing_features=True,
                save_features=False,
                fast_dev_run=False,
                incremental_training=False,
                incremental_data_dir=None,
                incremental_raw_data_dir=None,
                file_extensions=None,
                label_inference='filename',
                num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                incremental_rounds=DEFAULT_INCREMENTAL_ROUNDS,
                incremental_early_stopping=DEFAULT_INCREMENTAL_EARLY_STOPPING,
                max_finetune_iterations=DEFAULT_MAX_FINETUNE_ITERATIONS,
                max_file_size=DEFAULT_MAX_FILE_SIZE
            )
            if override_params:
                routing_args.override_params = override_params
            train_routing.main(routing_args)
            fine_args = argparse.Namespace(
                data_dir=PROCESSED_DATA_DIR,
                features_path=FEATURES_PKL_PATH,
                save_dir=HDBSCAN_SAVE_DIR,
                max_file_size=DEFAULT_MAX_FILE_SIZE,
                min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
                min_samples=DEFAULT_MIN_SAMPLES,
                min_family_size=DEFAULT_MIN_FAMILY_SIZE,
                plot_pca=False,
                explain_discrepancy=False,
                treat_noise_as_family=DEFAULT_TREAT_NOISE_AS_FAMILY
            )
            finetune.main(fine_args)
            logger.info('训练与聚类流程已完成')
        except Exception as e:
            logger.error(f'一键训练失败: {e}')
            raise
    elif args.command == 'auto-tune':
        from training import automl
        try:
            result = automl.main(args)
            logger.info(f"AutoML完成: {result}")
        except Exception as e:
            logger.error(f'AutoML失败: {e}')
            raise

if __name__ == '__main__':
    main()
