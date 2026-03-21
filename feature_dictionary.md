# PE病毒检测特征字典

## 概述

本字典将特征名称映射到PE文件结构中的具体位置，便于理解每个特征的数据来源。

---

## 特征分类总览

| 类别 | 前缀/特征名 | 数量 | 重要性 |
|------|-------------|------|--------|
| 文件级统计 | byte_*, count_*, entropy相关 | ~15 | 中等 |
| PE结构特征 | header_*, pe_header_size | ~10 | 中等 |
| Section特征 | section_*, seg* | ~30 | 极高 |
| 导入表特征 | import_*, dll_*, unique_imports | ~20 | 高 |
| 导出表特征 | export_*, exports_* | ~8 | 中 |
| Section属性 | executable_*, writable_*, readable_* | ~15 | 高 |
| 安全标志 | has_* (ASLR, NX, SEH等) | ~15 | 高 |
| 尾部数据 | trailing_data_*, overlay_* | ~5 | 高 |
| 资源特征 | resource_* | ~3 | 中 |
| 字节序列 | chunk_* | ~25 | 低 |
| 轻量级哈希 | lw_* | 256 | 较低 |
| 交互特征 | *_ratio, *_score, *_risk | ~30 | 高 |

---

## 详细特征映射

### 1. 文件级统计特征 (Byte-level Statistics)

**数据来源**: 文件前10KB样本，按2KB分块计算

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| size | 25 | 74,607 | 文件整体 | 文件大小（字节） |
| log_size | 20 | 89,079 | 文件整体 | log(文件大小+1) |
| entropy | 28 | 67,549 | 前10KB样本 | 文件前10KB熵值 |
| seg0_entropy | 23 | 77,598 | 前2KB (偏移0-2047) | 第1个2KB块熵值 |
| seg1_entropy | 71 | 19,585 | 前2KB (偏移2048-4095) | 第2个2KB块熵值 |
| seg2_entropy | 19 | 101,263 | 前2KB (偏移4096-6143) | 第3个2KB块熵值 |
| seg0_mean | 61 | 26,982 | 前2KB | 第1块均值 |
| seg1_mean | 183 | 6,522 | 前2KB | 第2块均值 |
| seg2_mean | 113 | 11,026 | 前2KB | 第3块均值 |
| seg0_std | 76 | 17,272 | 前2KB | 第1块标准差 |
| seg1_std | 158 | 7,844 | 前2KB | 第2块标准差 |
| seg2_std | 111 | 11,173 | 前2KB | 第3块标准差 |
| file_entropy_avg | 63 | 24,368 | 全文件 | 平均熵值 |
| file_entropy_min | 41 | 40,889 | 全文件 | 最小熵值 |
| file_entropy_max | 47 | 35,449 | 全文件 | 最大熵值 |
| file_entropy_range | 104 | 12,030 | 全文件 | 熵值范围 (max-min) |
| file_entropy_std | 96 | 13,692 | 全文件 | 熵值标准差 |
| file_entropy_q25 | 69 | 20,000 | 全文件 | 熵值25%分位数 |
| file_entropy_q75 | 94 | 13,733 | 全文件 | 熵值75%分位数 |
| file_entropy_median | 82 | 16,426 | 全文件 | 熵值中位数 |
| high_entropy_ratio | 163 | 7,663 | 全文件 | 高熵(>0.8)块比例 |
| low_entropy_ratio | 309 | 996 | 全文件 | 低熵(<0.2)块比例 |
| entropy_change_rate | 90 | 14,682 | 块间差异 | 熵值变化率均值 |
| entropy_change_std | 92 | 14,572 | 块间差异 | 熵值变化率标准差 |
| zero_byte_ratio | 87 | 15,138 | 前10KB | 0x00字节比例 |
| printable_byte_ratio | 38 | 48,187 | 前10KB | 可打印ASCII字符比例 |
| count_0 | 98 | 13,481 | 前10KB | 0x00字节数量 |
| count_255 | 37 | 48,726 | 前10KB | 0xFF字节数量 |
| count_0x90 | 68 | 20,303 | 前10KB | 0x90(NOP)字节数量 |
| count_printable | 97 | 13,510 | 前10KB | 可打印字符数量 |
| byte_mean | 109 | 11,297 | 前10KB | 字节值均值 |
| byte_median | 114 | 11,014 | 前10KB | 字节值中位数 |
| byte_std | 95 | 13,694 | 前10KB | 字节值标准差 |
| byte_q25 | 80 | 16,621 | 前10KB | 字节值25%分位数 |
| byte_q75 | 74 | 17,826 | 前10KB | 字节值75%分位数 |
| byte_min | 387 | 0 | 前10KB | 字节值最小值 |
| byte_max | 388 | 0 | 前10KB | 字节值最大值 |
| special_char_ratio | 305 | 1,400 | 前10KB | 特殊字符比例 |

---

### 2. PE Header 特征

**数据来源**: DOS Header, PE Header, COFF Header, Optional Header

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| pe_header_size | 75 | 17,621 | PE Header区域 | PE Header总大小 |
| header_size_ratio | 53 | 31,654 | PE Header/文件大小 | Header占文件比例 |
| subsystem | 157 | 7,998 | Optional Header → Subsystem | 子系统类型 (GUI/CUI) |
| dll_characteristics | 17 | 115,695 | Optional Header → DllCharacteristics | DLL特性标志 |
| checksum | ~180 | ~7,000 | Optional Header → CheckSum | 校验和 |
| checksum_zero_flag | ~150 | ~8,000 | Optional Header → CheckSum | 校验和是否为0 |

---

### 3. 安全标志特征 (Security Flags)

**数据来源**: Optional Header → DllCharacteristics

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| has_aslr | 10 | 518,249 | DllCharacteristics[bit 6] | 是否启用ASLR地址空间布局随机化 |
| has_nx_compat | 31 | 59,503 | DllCharacteristics[bit 7] | 是否支持NX禁用兼容 |
| has_guard_cf | 11 | 509,293 | DllCharacteristics[bit 13] | 是否有Control Flow Guard |
| has_seh | 85 | 15,393 | COFF Header → Characteristics | 是否有结构化异常处理 |
| has_debug_info | 9 | 545,617 | DEBUG_DIRECTORY_ENTRY | 是否包含调试信息 |
| has_relocs | 24 | 76,042 | Section Table → .reloc | 是否有重定位信息 |
| has_tls | 83 | 16,353 | TLS Directory | 是否有线程局部存储 |
| has_exceptions | 398 | 0 | Section Table | 是否有异常处理(.pdata) |
| has_signature | ~160 | ~7,500 | Security Directory | 是否有数字签名 |

---

### 4. Section 特征

**数据来源**: Section Table (IMAGE_SECTION_HEADER)

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| section_entropy_max | **1** | **6,118,454** | 各Section Data熵值 | 🔥**最重要特征** |
| section_entropy_std | 26 | 74,452 | 各Section熵值标准差 | Section熵离散程度 |
| section_entropy_avg | 27 | 71,865 | 各Section平均熵值 | 平均Section熵 |
| section_entropy_min | 128 | 9,851 | 各Section最小熵值 | 最低Section熵 |
| packed_sections_ratio | **2** | **1,876,899** | 高熵Section/总数 | 高熵(>0.8)Section比例 |
| sections_count | 46 | 36,335 | NumberOfSections | Section数量 |
| section_total_size | 59 | 29,251 | 所有Section.SizeOfRawData之和 | Section原始数据总大小 |
| section_total_vsize | 21 | 83,223 | 所有Section.VirtualSize之和 | Section虚拟大小总和 |
| avg_section_size | 54 | 31,359 | 总Size/数量 | Section平均大小 |
| avg_section_vsize | 22 | 80,135 | 总VirtualSize/数量 | Section平均虚拟大小 |
| min_section_size | 73 | 18,129 | 最小的SizeOfRawData | 最小Section大小 |
| max_section_size | 81 | 16,454 | 最大的SizeOfRawData | 最大Section大小 |
| section_size_std | 170 | 7,168 | Section大小标准差 | Section大小离散度 |
| section_size_cv | 93 | 14,445 | Section大小变异系数 | Size变化幅度 |
| section_names_count | 122 | 10,667 | 有效Section名数量 | 有名称的Section数 |
| section_name_avg_length | 191 | 5,730 | Section名平均长度 | Section名平均字符数 |
| section_name_max_length | 288 | 2,320 | Section名最大长度 | 最长Section名 |
| section_name_min_length | 311 | 880 | Section名最小长度 | 最短Section名 |
| long_sections_count | 115 | 10,948 | 大于平均值2倍的Section数 | 大Section数量 |
| long_sections_ratio | 155 | 8,046 | 大Section/总数 | 大Section比例 |
| short_sections_count | 281 | 2,636 | 小于平均值1/2的Section数 | 小Section数量 |
| short_sections_ratio | 202 | 5,079 | 小Section/总数 | 小Section比例 |

---

### 5. Section 属性特征 (Section Flags)

**数据来源**: Section Header → Characteristics

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| executable_sections_ratio | 16 | 121,467 | IMAGE_SCN_CNT_CODE \| IMAGE_SCN_MEM_EXECUTE | 可执行Section比例 |
| writable_sections_ratio | 30 | 61,942 | IMAGE_SCN_MEM_WRITE | 可写Section比例 |
| readable_sections_ratio | 86 | 15,158 | IMAGE_SCN_MEM_READ | 可读Section比例 |
| rwx_sections_ratio | 135 | 9,251 | 同时EXECUTE+WRITE+READ | 完全权限Section比例 |
| rwx_sections_count | 130 | 9,636 | 同时EXECUTE+WRITE+READ | 完全权限Section数量 |
| executable_code_density | 29 | 63,430 | 可执行Section/总大小 | 可执行Section密度 |
| executable_writable_sections | 77 | 17,139 | EXECUTE+WRITE组合 | 危险:可执行可写Section |
| non_standard_executable_sections_count | 159 | 7,799 | 非标准Section可执行 | 非标准可执行Section数 |
| non_standard_executable_sections_ratio | ~165 | ~7,200 | 非标准可执行/总数 | 非标准可执行比例 |

---

### 6. 导入表特征 (Import Table)

**数据来源**: Import Directory (.idata Section)

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| imports_count | 84 | 15,477 | Import Directory → Import Lookup Table | 导入DLL数量 |
| unique_imports | 117 | 10,897 | 唯一导入函数名数量 | 不同API函数数量 |
| unique_dlls | 140 | 8,731 | 唯一DLL名数量 | 不同DLL数量 |
| import_ordinal_only_count | 62 | 24,599 | 仅用序号导入的函数数 | 无函数名只有序号 |
| import_ordinal_only_ratio | 148 | 8,464 | 序号导入/总导入 | 序号导入比例 |
| avg_imports_per_dll | ~130 | ~9,000 | imports_count/unique_dlls | 每DLL平均导入数 |
| imported_system_dlls_count | 34 | 52,036 | 系统DLL数量 | kernel32/user32等 |
| imported_system_dlls_ratio | 8 | 661,631 | 系统DLL/总DLL | 系统DLL占比 |
| dll_name_avg_length | 7 | 698,179 | DLL名平均长度 | 导入DLL名平均长度 |
| dll_name_max_length | 14 | 142,302 | DLL名最大长度 | 最长DLL名 |
| dll_name_min_length | 33 | 52,638 | DLL名最小长度 | 最短DLL名 |
| dll_imports_entropy | 133 | 9,570 | DLL名熵值 | DLL名的信息熵 |
| api_imports_entropy | 230 | 3,901 | API名熵值 | API名的信息熵 |
| imports_per_section | ~140 | ~8,500 | imports_count/sections_count | 每Section导入密度 |
| syscall_api_ratio | 270 | 2,737 | syscall相关API/总API | 系统调用API比例 |

---

### 7. 导出表特征 (Export Table)

**数据来源**: Export Directory (.edata Section)

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| exports_count | 15 | 130,756 | Export Directory → NumberOfFunctions | 导出函数数量 |
| exports_density | 18 | 108,460 | exports_count/size | 导出函数密度 |
| export_name_avg_length | 72 | 18,718 | 导出函数名平均长度 | 导出名平均长度 |
| export_name_max_length | 207 | 4,827 | 导出函数名最大长度 | 最长导出名 |
| export_name_min_length | 303 | 1,436 | 导出函数名最小长度 | 最短导出名 |
| exports_name_ratio | ~200 | ~4,500 | 有名的导出/总数 | 命名导出比例 |

---

### 8. 尾部数据特征 (Overlay/Trailing Data)

**数据来源**: PE文件末尾，Section之后的数据

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| trailing_data_size | 5 | 986,812 | 文件末尾 - Section结束位置 | 尾部数据大小 |
| trailing_data_ratio | 12 | 196,107 | trailing_data_size/文件大小 | 尾部数据比例 |
| has_large_trailing_data | 35 | 51,067 | trailing_data_size > 1MB | 是否有超大尾部数据 |
| overlay_entropy | ~120 | ~10,000 | 尾部数据熵值 | 尾部加密程度 |
| overlay_high_entropy_flag | ~125 | ~9,500 | overlay_entropy > 0.8 | 尾部是否高熵 |

---

### 9. 资源特征 (Resource Table)

**数据来源**: Resource Directory (.rsrc Section)

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| has_resources | 55 | 30,653 | Resource Directory存在 | 是否有资源 |
| resources_count | 13 | 159,849 | 资源类型数量 | 不同资源类型数 |
| resource_types_count | ~130 | ~9,200 | 资源条目总数 | 资源总数 |

---

### 10. 入口点特征 (Entry Point)

**数据来源**: Optional Header → AddressOfEntryPoint

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| entry_point_ratio | 4 | 1,078,173 | RVAOfEntryPoint / ImageBase | 入口点相对地址比例 |
| entry_in_nonstandard_section_flag | ~160 | ~7,500 | 入口点所在Section是否非标准 | 入口点是否在非标准Section |

---

### 11. 重定位特征 (Relocation)

**数据来源**: Relocation Directory (.reloc Section)

| 特征名 | 排名 | Gain值 | PE对应位置 | 说明 |
|--------|------|--------|------------|------|
| reloc_blocks_count | ~130 | ~9,000 | 重定位块数量 | 重定位表块数 |
| reloc_entries_count | ~145 | ~8,000 | 重定位条目数量 | 重定位表条目数 |

---

### 12. 字节序列特征 (Chunk Features)

**数据来源**: 将文件按字节分块后计算的统计特征

| 特征名 | 排名 | Gain值 | 说明 |
|--------|------|--------|------|
| chunk_mean_0 ~ chunk_mean_9 | 56, 45, 118, 124, 131, 119, 107, 125, 112, 137 | ~10,000-37,000 | 各块字节均值 |
| chunk_std_0 ~ chunk_std_9 | 67, 88, 99, 105, 149, 150, 146, 123, 126, 120 | ~8,000-22,000 | 各块字节标准差 |
| chunk_mean_diff_* | 121, 132, 147, 162, 168 | ~7,200-10,700 | 块间均值差异 |
| chunk_std_diff_* | 141, 145, 167, 185 | ~6,100-8,600 | 块间标准差差异 |

---

### 13. 轻量级哈希特征 (Lightweight Hash Features)

**数据来源**: DLL名称、API函数名、Section名称的SHA256最后一字节取模

| 范围 | 数量 | 内容 |
|------|------|------|
| lw_0 ~ lw_127 | 128 | DLL名称哈希 (sha256_last_byte % 128) |
| lw_128 ~ lw_255 | 128 | API函数名哈希 (sha256_last_byte % 128) |
| lw_224 ~ lw_255 | 32 | Section名称哈希 (sha256_last_byte % 32) |

**代表特征**:

| 特征名 | 排名 | Gain值 | 说明 |
|--------|------|--------|------|
| lw_232 | 3 | 1,110,997 | DLL名哈希 - 高排名 |
| lw_200 | 6 | 790,530 | API名哈希 - 高排名 |
| lw_123 | 32 | 55,658 | Section名哈希 |
| lw_97 | 42 | 40,004 | DLL名哈希 |
| lw_136 | 43 | 38,152 | DLL名哈希 |
| lw_235 | 44 | 37,193 | API名哈希 |
| lw_116 | 48 | 34,891 | API名哈希 |
| lw_34 | 49 | 34,145 | Section名哈希 |

**设计目的**: 将高维字符串压缩为固定256维二值向量，用于捕捉"类型"而非精确身份。

---

### 14. 复合交互特征 (Compound Features)

**数据来源**: 多个基础特征计算得出的交互特征

| 特征名 | 排名 | Gain值 | 计算公式 | 说明 |
|--------|------|--------|----------|------|
| entropy_packed_interaction | ~145 | ~8,500 | packed_ratio × high_entropy_ratio | 打包+高熵交互 |
| suspicious_section_score | ~155 | ~7,800 | rwx_ratio + nonstandard_exec_ratio + alignment_mismatch_ratio | 可疑Section评分 |
| api_behavior_mix_score | ~160 | ~7,500 | network_ratio + process_ratio + fs_ratio + reg_ratio | API行为混合评分 |
| overlay_entropy_weighted | ~150 | ~8,200 | overlay_entropy × trailing_data_ratio | 尾部熵加权 |
| security_mitigation_gap | ~165 | ~7,000 | (1-has_nx + 1-has_aslr + 1-has_seh + 1-has_guard_cf)/4 | 安全缓解差距 |
| suspicious_import_mix | ~170 | ~6,500 | syscall_ratio + ordinal_ratio - imported_system_ratio | 可疑导入混合 |
| header_consistency_risk | ~175 | ~6,000 | (checksum_zero + entry_nonstandard + alignment_mismatch)/3 | Header一致性风险 |
| section_balance_risk | ~180 | ~5,500 | |code_ratio - data_ratio| + rwx_ratio | Section平衡风险 |
| benign_metadata_score | ~185 | ~5,200 | 0.4×signature + 0.35×version_info + ... | 良性元数据评分 |
| packer_api_overlay_risk | ~190 | ~4,800 | 打包器风险组合 | 0.4×packed_ratio + 0.25×overlay_high + 0.35×api_behavior |
| import_execution_pressure | ~195 | ~4,500 | imports_per_section × (0.5 + executable_writable_ratio) | 导入执行压力 |
| entropy_structure_risk | ~200 | ~4,200 | 熵结构风险 | 0.45×packed_ratio + 0.25×max_entropy + 0.3×overlay_weighted |
| suspicious_vs_benign_margin | ~210 | ~3,800 | suspicious - 0.35×benign | 恶意vs良性边界 |
| timestamp_metadata_conflict | ~215 | ~3,500 | 时间戳元数据冲突 | 基于时间戳和签名状态 |
| alignment_exec_compound_risk | ~220 | ~3,200 | 对齐执行复合风险 | alignment_mismatch + nonstandard_exec |

---

### 15. API类别特征 (API Category Features)

**数据来源**: 导入函数名按类别统计

| 特征名 | 排名 | Gain值 | 类别关键词 | 说明 |
|--------|------|--------|-----------|------|
| api_network_ratio | ~175 | ~6,000 | Internet, Http, Socket, Connect等 | 网络相关API比例 |
| api_process_ratio | ~180 | ~5,500 | CreateProcess, OpenProcess等 | 进程相关API比例 |
| api_filesystem_ratio | ~185 | ~5,000 | CreateFile, ReadFile等 | 文件系统API比例 |
| api_registry_ratio | ~190 | ~4,500 | RegOpenKey, RegSetValue等 | 注册表API比例 |

---

## 特征优先级总结

### 🔥 极高重要性 (Top 20)

| 排名 | 特征名 | Gain值 | 类别 |
|------|--------|--------|------|
| 1 | section_entropy_max | 6,118,454 | Section熵 |
| 2 | packed_sections_ratio | 1,876,899 | Section属性 |
| 3 | lw_232 | 1,110,997 | 轻量级哈希 |
| 4 | entry_point_ratio | 1,078,173 | 入口点 |
| 5 | trailing_data_size | 986,812 | 尾部数据 |
| 6 | lw_200 | 790,530 | 轻量级哈希 |
| 7 | dll_name_avg_length | 698,179 | 导入表 |
| 8 | imported_system_dlls_ratio | 661,631 | 导入表 |
| 9 | has_debug_info | 545,617 | 安全标志 |
| 10 | has_aslr | 518,249 | 安全标志 |
| 11 | has_guard_cf | 509,293 | 安全标志 |
| 12 | trailing_data_ratio | 196,107 | 尾部数据 |
| 13 | resources_count | 159,849 | 资源 |
| 14 | dll_name_max_length | 142,302 | 导入表 |
| 15 | exports_count | 130,756 | 导出表 |
| 16 | executable_sections_ratio | 121,467 | Section属性 |
| 17 | dll_characteristics | 115,695 | PE Header |
| 18 | exports_density | 108,460 | 导出表 |
| 19 | seg2_entropy | 101,263 | 文件统计 |
| 20 | log_size | 89,079 | 文件统计 |

---

## PE结构与特征对照表

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PE FILE STRUCTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ DOS Header (MZ)              │  ← 相关特征: pe_header_size              │
│  │   e_magic = "MZ"             │                                         │
│  └───────────────────────────────┘                                         │
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ PE Header                    │  ← 相关特征: dll_characteristics,        │
│  │   Signature = "PE\0\0"       │           subsystem, checksum           │
│  │   COFF Header                │           has_debug_info                 │
│  │   Optional Header            │                                         │
│  └───────────────────────────────┘                                         │
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ Section Headers              │  ← 相关特征: sections_count,             │
│  │   .text (code)               │           section_*, seg*              │
│  │   .data (data)              │           executable_*, writable_*       │
│  │   .rdata (const)            │           has_relocs, has_tls            │
│  │   .rsrc (resources)         │           resources_count               │
│  │   .reloc (relocation)        │                                         │
│  └───────────────────────────────┘                                         │
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ Import Table (.idata)        │  ← 相关特征: imports_count,             │
│  │   DLL names                  │           dll_name_*, unique_imports    │
│  │   Function names/ordinals   │           imported_system_dlls_*        │
│  │                              │           import_ordinal_only_*         │
│  └───────────────────────────────┘                                         │
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ Export Table (.edata)        │  ← 相关特征: exports_count,             │
│  │   Function names            │           exports_density               │
│  │   Ordinals                   │           export_name_*                  │
│  └───────────────────────────────┘                                         │
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ TLS Directory                │  ← 相关特征: has_tls, tls_callbacks_*  │
│  └───────────────────────────────┘                                         │
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ Security Directory           │  ← 相关特征: has_signature              │
│  └───────────────────────────────┘                                         │
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ Debug Directory              │  ← 相关特征: has_debug_info             │
│  └───────────────────────────────┘                                         │
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ Overlay (Trailing Data)      │  ← 相关特征: trailing_data_size,       │
│  │   (appended after sections)  │           trailing_data_ratio,         │
│  │                              │           overlay_entropy               │
│  └───────────────────────────────┘                                         │
│                                                                             │
│  ┌───────────────────────────────┐                                         │
│  │ File-level Statistics        │  ← 相关特征: entropy, byte_*,          │
│  │   (first 10KB sampled)       │           count_*, file_entropy_*      │
│  └───────────────────────────────┘                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 特征名称命名规则

| 前缀/模式 | 含义 | 示例 |
|-----------|------|------|
| `has_*` | 二进制标志(0/1) | has_aslr, has_debug_info |
| `*_count` | 数量统计 | sections_count, exports_count |
| `*_ratio` | 比例(0-1) | writable_sections_ratio |
| `*_avg` | 平均值 | section_name_avg_length |
| `*_max` / `*_min` | 最大/最小值 | dll_name_max_length |
| `*_std` | 标准差 | section_entropy_std |
| `*_entropy` | 熵值相关 | file_entropy_avg |
| `lw_*` | 轻量级哈希 | lw_232, lw_200 |
| `seg*` | 文件分段特征 | seg0_entropy |
| `chunk_*` | 分块统计特征 | chunk_mean_0 |
| `*_density` | 密度相关 | executable_code_density |
| `*_risk` / `*_score` | 复合风险/评分 | security_mitigation_gap |

---

*文档生成时间: 2026-03-21*
*数据来源: features_pe.cpp, features_file_attributes.cpp*
