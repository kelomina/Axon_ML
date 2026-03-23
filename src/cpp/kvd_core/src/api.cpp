/*   Copyright 2025-2026 KoloStudio & Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
*/

#include "kvd/api.h"

#include "config.h"
#include "features.h"
#include "malware_scanner.h"
#include "onnx_infer.h"
#include "util_filesystem.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <windows.h>

#include <nlohmann/json.hpp>

#if defined(KVD_BUILD_SIGNATURE_ENGINE) && !defined(KVD_BUILD_AXON_ENGINE)
#define KVD_SIGNATURE_ONLY 1
#endif

#if defined(KVD_BUILD_AXON_ENGINE) && !defined(KVD_BUILD_SIGNATURE_ENGINE)
#define KVD_AXON_ONLY 1
#endif

struct kvd_handle {
  kvd::Config config;
  std::shared_ptr<int> token;
  std::shared_ptr<kvd::LightGbmModel> model;
  std::shared_ptr<kvd::LightGbmModel> model_normal;
  std::shared_ptr<kvd::LightGbmModel> model_packed;
  std::shared_ptr<kvd::FamilyClassifier> family_classifier;
  std::shared_ptr<kvd::OnnxModel> onnx_model;
  std::shared_ptr<kvd::OnnxModel> onnx_model_normal;
  std::shared_ptr<kvd::OnnxModel> onnx_model_packed;
};

static std::shared_ptr<kvd::LightGbmModel> get_shared_model(const std::string& path);
static std::shared_ptr<kvd::FamilyClassifier> get_shared_family_classifier(const std::string& path);
static constexpr std::size_t KVD_PE_FEATURE_VECTOR_DIM = 350;

static std::string kvd_temp_dir() {
  std::error_code ec;
  std::filesystem::path p = std::filesystem::temp_directory_path(ec);
  if (ec) {
    return std::string(".");
  }
  auto u8 = p.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

static std::string kvd_log_path() {
  std::filesystem::path base;
  auto temp = kvd_temp_dir();
  auto base_opt = kvd::to_filesystem_path(temp);
  if (base_opt) {
    base = *base_opt;
  } else {
    std::error_code ec;
    base = std::filesystem::temp_directory_path(ec);
    if (ec) base = std::filesystem::path(".");
  }
  std::filesystem::path p = base / "anxin_logs" / "crash";
  std::error_code ec;
  std::filesystem::create_directories(p, ec);
  auto u8 = (p / "kvd_native.log").u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

static void kvd_log_line(const std::string& line) {
  try {
    auto p = kvd::to_filesystem_path(kvd_log_path());
    if (!p) return;
    std::ofstream out(*p, std::ios::out | std::ios::app);
    out << line << "\n";
  } catch (...) {
  }
}

static void kvd_log_event(const std::string& tag, const std::string& detail) {
  std::string msg = "{\"tag\":\"" + tag + "\",\"detail\":\"" + detail + "\"}";
  kvd_log_line(msg);
}

static std::string kvd_basename(const char* p) {
  if (!p) return {};
  try {
    auto fs_path = kvd::to_filesystem_path(std::string(p));
    if (!fs_path) {
      return {};
    }
    auto name = fs_path->filename().u8string();
    return std::string(reinterpret_cast<const char*>(name.data()), name.size());
  } catch (...) {
    return {};
  }
}

static nlohmann::json to_json(const kvd::ScanResult& r) {
  nlohmann::json j;
  j["is_malware"] = r.is_malware;
  j["confidence"] = r.confidence;
  j["axon_malware"] = r.axon_malware;
  j["axon_score"] = r.axon_score;
  j["hardcase_triggered"] = r.hardcase_triggered;
  if (!r.hardcase_class.empty()) {
    j["hardcase_class"] = r.hardcase_class;
  }
  if (!r.hardcase_scores.empty()) {
    j["hardcase_scores"] = r.hardcase_scores;
  }
  j["signature_hit"] = r.signature_hit;
  j["signature_score"] = r.signature_score;
  if (!r.signature_reason.empty()) {
    j["signature_reason"] = r.signature_reason;
  }
  if (!r.error.empty()) {
    j["error"] = r.error;
  }
  if (r.family) {
    nlohmann::json f;
    f["family_name"] = r.family->family_name;
    f["cluster_id"] = r.family->cluster_id;
    f["is_new_family"] = r.family->is_new_family;
    j["malware_family"] = f;
  }
  return j;
}

kvd_handle* kvd_create(const kvd_config* config) {
  try {
    if (!config) {
      return nullptr;
    }
    kvd::Config cfg = kvd::config_from_api(
        config->model_path,
        config->model_normal_path,
        config->model_packed_path,
        config->family_classifier_json_path,
        config->allowed_scan_root,
        config->max_file_size,
        config->prediction_threshold,
        config->onnx_model_path,
        config->onnx_model_normal_path,
        config->onnx_model_packed_path);

#if defined(KVD_SIGNATURE_ONLY)
    kvd_handle* h = new (std::nothrow) kvd_handle{};
    if (!h) {
      return nullptr;
    }
    h->config = std::move(cfg);
    h->token = std::make_shared<int>(1);
    return h;
#else
    auto model = get_shared_model(cfg.model_path);
    if (!model) {
      return nullptr;
    }
    auto model_normal = get_shared_model(cfg.model_normal_path);
    auto model_packed = get_shared_model(cfg.model_packed_path);
    auto family_classifier = get_shared_family_classifier(cfg.family_classifier_json_path);
    kvd_handle* h = new (std::nothrow) kvd_handle{};
    if (!h) {
      return nullptr;
    }
    h->config = std::move(cfg);
    h->token = std::make_shared<int>(1);
    h->model = std::move(model);
    h->model_normal = std::move(model_normal);
    h->model_packed = std::move(model_packed);
    h->family_classifier = std::move(family_classifier);

    if (!cfg.onnx_model_path.empty()) {
      auto onnx_m = kvd::OnnxModel::load_from_file(cfg.onnx_model_path);
      if (onnx_m) {
        h->onnx_model = std::make_shared<kvd::OnnxModel>(std::move(*onnx_m));
      }
    }
    if (!cfg.onnx_model_normal_path.empty()) {
      auto onnx_m_normal = kvd::OnnxModel::load_from_file(cfg.onnx_model_normal_path);
      if (onnx_m_normal) {
        h->onnx_model_normal = std::make_shared<kvd::OnnxModel>(std::move(*onnx_m_normal));
      }
    }
    if (!cfg.onnx_model_packed_path.empty()) {
      auto onnx_m_packed = kvd::OnnxModel::load_from_file(cfg.onnx_model_packed_path);
      if (onnx_m_packed) {
        h->onnx_model_packed = std::make_shared<kvd::OnnxModel>(std::move(*onnx_m_packed));
      }
    }

    return h;
#endif
  } catch (const std::exception& e) {
    kvd_log_event("kvd_create_exception", e.what());
    return nullptr;
  } catch (...) {
    kvd_log_event("kvd_create_exception", "unknown");
    return nullptr;
  }
}

void kvd_destroy(kvd_handle* handle) {
  try {
    if (!handle) {
      return;
    }
    delete handle;
  } catch (const std::exception& e) {
    kvd_log_event("kvd_destroy_exception", e.what());
  } catch (...) {
    kvd_log_event("kvd_destroy_exception", "unknown");
  }
}

static int write_json_out(const nlohmann::json& j, char** out_json, size_t* out_len) {
  if (!out_json || !out_len) {
    return -1;
  }
  std::string s = j.dump();
  char* buf = static_cast<char*>(std::malloc(s.size() + 1));
  if (!buf) {
    return -2;
  }
  std::memcpy(buf, s.data(), s.size());
  buf[s.size()] = '\0';
  *out_json = buf;
  *out_len = s.size();
  return 0;
}

static int write_string_out(const std::string& s, char** out_json, size_t* out_len) {
  if (!out_json || !out_len) {
    return -1;
  }
  char* buf = static_cast<char*>(std::malloc(s.size() + 1));
  if (!buf) {
    return -2;
  }
  std::memcpy(buf, s.data(), s.size());
  buf[s.size()] = '\0';
  *out_json = buf;
  *out_len = s.size();
  return 0;
}

static int write_train_error(const std::string& code, char** out_json, size_t* out_len) {
  nlohmann::json j;
  j["ok"] = false;
  j["total"] = 0;
  j["trained"] = 0;
  j["failed"] = 0;
  j["error"] = code;
  return write_json_out(j, out_json, out_len);
}

static bool path_exists(const std::string& path) {
  std::error_code ec;
  auto p = kvd::to_filesystem_path(path);
  if (!p) {
    return false;
  }
  return std::filesystem::exists(*p, ec) && !ec;
}

static std::string to_utf8_string(const std::filesystem::path& p) {
  auto u8 = p.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

static bool is_onnx_path(const std::filesystem::path& p) {
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".onnx";
}

static std::string resolve_lightgbm_runtime_path(const std::string& path) {
  if (path.empty()) {
    return path;
  }
  auto fs_opt = kvd::to_filesystem_path(path);
  if (!fs_opt) {
    return path;
  }
  const auto& fs_path = *fs_opt;
  if (!is_onnx_path(fs_path)) {
    return path;
  }
  std::filesystem::path base_dir = fs_path.parent_path();
  std::string stem = fs_path.stem().string();
  std::filesystem::path train_txt_path = base_dir / (stem + ".train.txt");
  std::string train_txt_utf8 = to_utf8_string(train_txt_path);
  if (path_exists(train_txt_utf8)) {
    return train_txt_utf8;
  }
  std::filesystem::path legacy_txt_path = base_dir / (stem + ".txt");
  std::string legacy_txt_utf8 = to_utf8_string(legacy_txt_path);
  if (path_exists(legacy_txt_utf8)) {
    return legacy_txt_utf8;
  }
  return path;
}

static int validate_hardcase_manifest(const std::string& manifest_path, std::string& error_code) {
  if (manifest_path.empty()) {
    return KVD_MODEL_OK;
  }
  if (!path_exists(manifest_path)) {
    error_code = "hardcase_manifest_missing";
    return KVD_MODEL_ERR_HARDCASE_MANIFEST_MISSING;
  }
  auto manifest_fs = kvd::to_filesystem_path(manifest_path);
  if (!manifest_fs) {
    error_code = "hardcase_manifest_invalid";
    return KVD_MODEL_ERR_HARDCASE_MANIFEST_INVALID;
  }
  std::ifstream in(*manifest_fs);
  if (!in) {
    error_code = "hardcase_manifest_invalid";
    return KVD_MODEL_ERR_HARDCASE_MANIFEST_INVALID;
  }
  nlohmann::json j;
  try {
    in >> j;
  } catch (...) {
    error_code = "hardcase_manifest_invalid";
    return KVD_MODEL_ERR_HARDCASE_MANIFEST_INVALID;
  }
  if (!j.is_object()) {
    error_code = "hardcase_manifest_invalid";
    return KVD_MODEL_ERR_HARDCASE_MANIFEST_INVALID;
  }
  auto model_paths = j.value("class_model_paths", nlohmann::json::array());
  if (!model_paths.is_array() || model_paths.size() != 3) {
    error_code = "hardcase_manifest_invalid";
    return KVD_MODEL_ERR_HARDCASE_MANIFEST_INVALID;
  }
  std::filesystem::path base_dir = manifest_fs->parent_path();
  for (const auto& item : model_paths) {
    if (!item.is_string()) {
      error_code = "hardcase_manifest_invalid";
      return KVD_MODEL_ERR_HARDCASE_MANIFEST_INVALID;
    }
    std::filesystem::path model_path(item.get<std::string>());
    if (model_path.is_relative()) {
      model_path = base_dir / model_path;
    }
    auto u8 = model_path.u8string();
    std::string model_path_utf8(reinterpret_cast<const char*>(u8.data()), u8.size());
    std::string runtime_model_path = resolve_lightgbm_runtime_path(model_path_utf8);
    if (!path_exists(runtime_model_path)) {
      error_code = "hardcase_model_missing";
      return KVD_MODEL_ERR_HARDCASE_MODEL_MISSING;
    }
    if (!kvd::LightGbmModel::load_from_file(runtime_model_path)) {
      error_code = "hardcase_model_invalid";
      return KVD_MODEL_ERR_HARDCASE_MODEL_INVALID;
    }
  }
  return KVD_MODEL_OK;
}

static kvd::MalwareScanner* get_thread_scanner(kvd_handle* handle);

static void collect_train_targets(const std::string& path, std::vector<std::string>& out) {
  if (path.empty()) {
    return;
  }
  std::error_code ec;
  auto p_opt = kvd::to_filesystem_path(path);
  if (!p_opt) {
    return;
  }
  std::filesystem::path p = *p_opt;
  if (std::filesystem::is_regular_file(p, ec)) {
    auto u8 = p.u8string();
    out.push_back(std::string(reinterpret_cast<const char*>(u8.data()), u8.size()));
    return;
  }
  if (!std::filesystem::is_directory(p, ec)) {
    return;
  }
  std::filesystem::recursive_directory_iterator it(
      p, std::filesystem::directory_options::skip_permission_denied, ec);
  std::filesystem::recursive_directory_iterator end;
  for (; it != end; it.increment(ec)) {
    if (ec) {
      ec.clear();
      continue;
    }
    if (it->is_regular_file(ec)) {
      auto u8 = it->path().u8string();
      out.push_back(std::string(reinterpret_cast<const char*>(u8.data()), u8.size()));
    }
  }
}

static int kvd_train_internal(kvd_handle* handle, const std::vector<std::string>& paths, bool is_malware, char** out_json, size_t* out_len) {
  if (!handle) {
    return -1;
  }
  std::vector<std::string> files;
  for (const auto& p : paths) {
    if (p.empty()) continue;
    collect_train_targets(p, files);
  }
  std::size_t total = files.size();
  std::atomic<std::size_t> trained{0};
  if (total > 0) {
    std::size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    std::size_t threads = std::min<std::size_t>(hw, total);
    threads = std::min<std::size_t>(threads, 16);
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
      pool.emplace_back([&]() {
        auto scanner = get_thread_scanner(handle);
        if (!scanner) return;
        for (;;) {
          std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= files.size()) return;
          if (scanner->train_path(files[i], is_malware)) {
            trained.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }
    for (auto& th : pool) th.join();
  }
  std::size_t trained_n = trained.load(std::memory_order_relaxed);
  std::size_t failed = total >= trained_n ? (total - trained_n) : 0;
  if (trained_n > 0) {
    try {
      kvd::flush_signature_db();
    } catch (...) {
    }
  }
  nlohmann::json j;
  j["ok"] = trained_n > 0;
  j["total"] = total;
  j["trained"] = trained_n;
  j["failed"] = failed;
  return write_json_out(j, out_json, out_len);
}

static std::shared_ptr<kvd::LightGbmModel> get_shared_model(const std::string& path) {
  if (path.empty()) {
    return {};
  }
  std::string runtime_path = resolve_lightgbm_runtime_path(path);
  static std::mutex mu;
  static std::unordered_map<std::string, std::weak_ptr<kvd::LightGbmModel>> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(runtime_path);
  if (it != cache.end()) {
    auto sp = it->second.lock();
    if (sp) {
      return sp;
    }
  }
  auto model_opt = kvd::LightGbmModel::load_from_file(runtime_path);
  if (!model_opt) {
    return {};
  }
  auto sp = std::make_shared<kvd::LightGbmModel>(std::move(*model_opt));
  cache[runtime_path] = sp;
  return sp;
}

static std::shared_ptr<kvd::FamilyClassifier> get_shared_family_classifier(const std::string& path) {
  if (path.empty()) {
    return {};
  }
  static std::mutex mu;
  static std::unordered_map<std::string, std::weak_ptr<kvd::FamilyClassifier>> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(path);
  if (it != cache.end()) {
    auto sp = it->second.lock();
    if (sp) {
      return sp;
    }
  }
  auto fc_opt = kvd::FamilyClassifier::load_from_json(path);
  if (!fc_opt) {
    return {};
  }
  auto sp = std::make_shared<kvd::FamilyClassifier>(std::move(*fc_opt));
  cache[path] = sp;
  return sp;
}

struct TokenHash {
  std::size_t operator()(const std::shared_ptr<int>& p) const noexcept {
    return std::hash<int*>()(p.get());
  }
};

struct TokenEq {
  bool operator()(const std::shared_ptr<int>& a, const std::shared_ptr<int>& b) const noexcept {
    return a.get() == b.get();
  }
};

static kvd::MalwareScanner* get_thread_scanner(kvd_handle* handle) {
  if (!handle || !handle->token) {
    return nullptr;
  }
  thread_local std::unordered_map<std::shared_ptr<int>, std::unique_ptr<kvd::MalwareScanner>, TokenHash, TokenEq> cache;
  auto it = cache.find(handle->token);
  if (it != cache.end()) {
    return it->second.get();
  }
  auto scanner_opt = kvd::MalwareScanner::create_shared(
      handle->config,
      handle->model,
      handle->model_normal,
      handle->model_packed,
      handle->family_classifier,
      handle->onnx_model,
      handle->onnx_model_normal,
      handle->onnx_model_packed);
  if (!scanner_opt) {
    return nullptr;
  }
  auto ptr = std::make_unique<kvd::MalwareScanner>(std::move(*scanner_opt));
  auto raw = ptr.get();
  cache.emplace(handle->token, std::move(ptr));
  return raw;
}

int kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len) {
  try {
    if (!handle || !path) {
      return -1;
    }
    auto scanner = get_thread_scanner(handle);
    if (!scanner) {
      return -1;
    }
    kvd::ScanResult r = scanner->scan_path(path);
    return write_json_out(to_json(r), out_json, out_len);
  } catch (const std::exception& e) {
    kvd_log_event("kvd_scan_path_exception", kvd_basename(path) + ":" + e.what());
    return -1;
  } catch (...) {
    kvd_log_event("kvd_scan_path_exception", kvd_basename(path) + ":unknown");
    return -1;
  }
}

int kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len) {
  try {
    if (!handle || !bytes) {
      return -1;
    }
    std::vector<std::uint8_t> v(bytes, bytes + len);
    auto scanner = get_thread_scanner(handle);
    if (!scanner) {
      return -1;
    }
    kvd::ScanResult r = scanner->scan_bytes(v);
    return write_json_out(to_json(r), out_json, out_len);
  } catch (const std::exception& e) {
    kvd_log_event("kvd_scan_bytes_exception", e.what());
    return -1;
  } catch (...) {
    kvd_log_event("kvd_scan_bytes_exception", "unknown");
    return -1;
  }
}

int kvd_scan_paths(kvd_handle* handle, const char** paths, size_t count, char** out_json, size_t* out_len) {
  try {
    if (!handle) {
      return -1;
    }
    if (!paths && count > 0) {
      return -1;
    }
    std::vector<std::string> v;
    v.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (paths && paths[i]) {
        v.emplace_back(paths[i]);
      } else {
        v.emplace_back();
      }
    }
    auto scanner = get_thread_scanner(handle);
    if (!scanner) {
      return -1;
    }
    auto results = scanner->scan_paths(v);
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& r : results) {
      arr.push_back(to_json(r));
    }
    return write_json_out(arr, out_json, out_len);
  } catch (const std::exception& e) {
    kvd_log_event("kvd_scan_paths_exception", std::to_string(count) + ":" + e.what());
    return -1;
  } catch (...) {
    kvd_log_event("kvd_scan_paths_exception", std::to_string(count) + ":unknown");
    return -1;
  }
}

int kvd_train_path(kvd_handle* handle, const char* path, int label, char** out_json, size_t* out_len) {
  try {
    if (!handle || !path) {
      return write_train_error("invalid_argument", out_json, out_len);
    }
    std::vector<std::string> v;
    v.emplace_back(path);
    return kvd_train_internal(handle, v, label != 0, out_json, out_len);
  } catch (const std::exception& e) {
    kvd_log_event("kvd_train_path_exception", kvd_basename(path) + ":" + e.what());
    return write_train_error("exception", out_json, out_len);
  } catch (...) {
    kvd_log_event("kvd_train_path_exception", kvd_basename(path) + ":unknown");
    return write_train_error("exception", out_json, out_len);
  }
}

int kvd_train_paths(kvd_handle* handle, const char** paths, size_t count, int label, char** out_json, size_t* out_len) {
  try {
    if (!handle) {
      return write_train_error("invalid_argument", out_json, out_len);
    }
    if (!paths && count > 0) {
      return write_train_error("invalid_argument", out_json, out_len);
    }
    std::vector<std::string> v;
    v.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (paths && paths[i]) {
        v.emplace_back(paths[i]);
      } else {
        v.emplace_back();
      }
    }
    return kvd_train_internal(handle, v, label != 0, out_json, out_len);
  } catch (const std::exception& e) {
    kvd_log_event("kvd_train_paths_exception", std::to_string(count) + ":" + e.what());
    return write_train_error("exception", out_json, out_len);
  } catch (...) {
    kvd_log_event("kvd_train_paths_exception", std::to_string(count) + ":unknown");
    return write_train_error("exception", out_json, out_len);
  }
}

int kvd_train_from_path(kvd_handle* handle, const char* path, int label, char** out_json, size_t* out_len) {
  return kvd_train_path(handle, path, label, out_json, out_len);
}

void kvd_signature_flush(kvd_handle* handle) {
  try {
    if (!handle) return;
    kvd::flush_signature_db();
  } catch (...) {
  }
}

void kvd_free(char* p) {
  std::free(p);
}

int kvd_extract_pe_features(const char* path, float* out_features, size_t out_len) {
  try {
    if (!path || !out_features) {
      return -1;
    }
    if (out_len < KVD_PE_FEATURE_VECTOR_DIM) {
      return -1;
    }
    std::vector<float> features = kvd::extract_combined_pe_features_from_path(path, std::nullopt);
    if (features.size() != KVD_PE_FEATURE_VECTOR_DIM) {
      return -2;
    }
    std::copy(features.begin(), features.end(), out_features);
    return 0;
  } catch (const std::exception& e) {
    kvd_log_event("kvd_extract_pe_features_exception", kvd_basename(path) + ":" + e.what());
    return -2;
  } catch (...) {
    kvd_log_event("kvd_extract_pe_features_exception", kvd_basename(path) + ":unknown");
    return -2;
  }
}

int kvd_extract_pe_features_batch(
    const char** paths,
    size_t count,
    float* out_features,
    size_t feature_dim,
    int* out_status,
    unsigned int thread_count) {
  try {
    if (!paths || !out_features || !out_status) {
      return -1;
    }
    if (feature_dim < KVD_PE_FEATURE_VECTOR_DIM) {
      return -1;
    }
    if (count == 0) {
      return 0;
    }

    std::size_t hw = static_cast<std::size_t>(std::thread::hardware_concurrency());
    if (hw == 0) {
      hw = 8;
    }
    std::size_t workers = thread_count > 0 ? static_cast<std::size_t>(thread_count) : hw;
    if (workers == 0) {
      workers = 1;
    }
    std::size_t io_safe_cap = std::clamp<std::size_t>(hw / 2, 4, 16);
    workers = std::min<std::size_t>(workers, count);
    workers = std::min<std::size_t>(workers, io_safe_cap);

    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t t = 0; t < workers; ++t) {
      pool.emplace_back([&]() {
        for (;;) {
          std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= count) {
            return;
          }
          float* dst = out_features + i * feature_dim;
          std::fill_n(dst, feature_dim, 0.0f);
          const char* p = paths[i];
          if (!p) {
            out_status[i] = -1;
            continue;
          }
          try {
            std::vector<float> features = kvd::extract_combined_pe_features_from_path(p, std::nullopt);
            if (features.size() != KVD_PE_FEATURE_VECTOR_DIM) {
              out_status[i] = -2;
              continue;
            }
            std::copy(features.begin(), features.end(), dst);
            out_status[i] = 0;
          } catch (...) {
            out_status[i] = -2;
          }
        }
      });
    }
    for (auto& th : pool) {
      th.join();
    }
    return 0;
  } catch (...) {
    return -2;
  }
}

size_t kvd_get_pe_feature_dimension(void) {
  return KVD_PE_FEATURE_VECTOR_DIM;
}

int kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len) {
  try {
    if (!config) {
      return KVD_MODEL_ERR_INVALID_ARGUMENT;
    }
    if ((out_error && !out_len) || (!out_error && out_len)) {
      return KVD_MODEL_ERR_INVALID_ARGUMENT;
    }

    kvd::Config cfg = kvd::config_from_api(
        config->model_path,
        config->model_normal_path,
        config->model_packed_path,
        config->family_classifier_json_path,
        config->allowed_scan_root,
        config->max_file_size,
        config->prediction_threshold,
        config->onnx_model_path,
        config->onnx_model_normal_path,
        config->onnx_model_packed_path);

    auto write_error = [&](const std::string& code) -> int {
      if (!out_error && !out_len) {
        return 0;
      }
      int rc = write_string_out(code, out_error, out_len);
      return rc == 0 ? 0 : KVD_MODEL_ERR_OOM;
    };

#if defined(KVD_SIGNATURE_ONLY)
    int rc = write_error("ok");
    return rc == 0 ? KVD_MODEL_OK : rc;
#else

    if (cfg.model_path.empty()) {
      int rc = write_error("model_main_missing");
      return rc == 0 ? KVD_MODEL_ERR_MAIN_MISSING : rc;
    }
    std::string runtime_model_path = resolve_lightgbm_runtime_path(cfg.model_path);
    if (!path_exists(runtime_model_path)) {
      int rc = write_error("model_main_missing");
      return rc == 0 ? KVD_MODEL_ERR_MAIN_MISSING : rc;
    }
    if (!kvd::LightGbmModel::load_from_file(runtime_model_path)) {
      int rc = write_error("model_main_invalid");
      return rc == 0 ? KVD_MODEL_ERR_MAIN_INVALID : rc;
    }

    bool has_normal = !cfg.model_normal_path.empty();
    bool has_packed = !cfg.model_packed_path.empty();
    if (has_normal != has_packed) {
      int rc = write_error("model_route_incomplete");
      return rc == 0 ? KVD_MODEL_ERR_ROUTE_INCOMPLETE : rc;
    }

    if (has_normal) {
      std::string runtime_model_normal_path = resolve_lightgbm_runtime_path(cfg.model_normal_path);
      if (!path_exists(runtime_model_normal_path)) {
        int rc = write_error("model_normal_missing");
        return rc == 0 ? KVD_MODEL_ERR_NORMAL_MISSING : rc;
      }
      if (!kvd::LightGbmModel::load_from_file(runtime_model_normal_path)) {
        int rc = write_error("model_normal_invalid");
        return rc == 0 ? KVD_MODEL_ERR_NORMAL_INVALID : rc;
      }
    }

    if (has_packed) {
      std::string runtime_model_packed_path = resolve_lightgbm_runtime_path(cfg.model_packed_path);
      if (!path_exists(runtime_model_packed_path)) {
        int rc = write_error("model_packed_missing");
        return rc == 0 ? KVD_MODEL_ERR_PACKED_MISSING : rc;
      }
      if (!kvd::LightGbmModel::load_from_file(runtime_model_packed_path)) {
        int rc = write_error("model_packed_invalid");
        return rc == 0 ? KVD_MODEL_ERR_PACKED_INVALID : rc;
      }
    }

    if (!cfg.family_classifier_json_path.empty()) {
      if (!path_exists(cfg.family_classifier_json_path)) {
        int rc = write_error("family_classifier_missing");
        return rc == 0 ? KVD_MODEL_ERR_FAMILY_MISSING : rc;
      }
      if (!kvd::FamilyClassifier::load_from_json(cfg.family_classifier_json_path)) {
        int rc = write_error("family_classifier_invalid");
        return rc == 0 ? KVD_MODEL_ERR_FAMILY_INVALID : rc;
      }
    }
    if (!cfg.hardcase_manifest_path.empty()) {
      std::string hardcase_error;
      int hardcase_rc = validate_hardcase_manifest(cfg.hardcase_manifest_path, hardcase_error);
      if (hardcase_rc != KVD_MODEL_OK) {
        int rc = write_error(hardcase_error);
        return rc == 0 ? hardcase_rc : rc;
      }
    }

    if (!cfg.onnx_model_path.empty()) {
      if (!path_exists(cfg.onnx_model_path)) {
        int rc = write_error("onnx_model_main_missing");
        return rc == 0 ? KVD_MODEL_ERR_ONNX_MAIN_MISSING : rc;
      }
      if (!kvd::OnnxModel::load_from_file(cfg.onnx_model_path)) {
        int rc = write_error("onnx_model_main_invalid");
        return rc == 0 ? KVD_MODEL_ERR_ONNX_MAIN_INVALID : rc;
      }
    }
    if (!cfg.onnx_model_normal_path.empty()) {
      if (!path_exists(cfg.onnx_model_normal_path)) {
        int rc = write_error("onnx_model_normal_missing");
        return rc == 0 ? KVD_MODEL_ERR_ONNX_NORMAL_MISSING : rc;
      }
      if (!kvd::OnnxModel::load_from_file(cfg.onnx_model_normal_path)) {
        int rc = write_error("onnx_model_normal_invalid");
        return rc == 0 ? KVD_MODEL_ERR_ONNX_NORMAL_INVALID : rc;
      }
    }
    if (!cfg.onnx_model_packed_path.empty()) {
      if (!path_exists(cfg.onnx_model_packed_path)) {
        int rc = write_error("onnx_model_packed_missing");
        return rc == 0 ? KVD_MODEL_ERR_ONNX_PACKED_MISSING : rc;
      }
      if (!kvd::OnnxModel::load_from_file(cfg.onnx_model_packed_path)) {
        int rc = write_error("onnx_model_packed_invalid");
        return rc == 0 ? KVD_MODEL_ERR_ONNX_PACKED_INVALID : rc;
      }
    }

    int rc = write_error("ok");
    return rc == 0 ? KVD_MODEL_OK : rc;
#endif
  } catch (const std::exception& e) {
    kvd_log_event("kvd_validate_exception", e.what());
    return KVD_MODEL_ERR_INVALID_ARGUMENT;
  } catch (...) {
    kvd_log_event("kvd_validate_exception", "unknown");
    return KVD_MODEL_ERR_INVALID_ARGUMENT;
  }
}
