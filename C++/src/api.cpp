#include "kvd/api.h"

#include "config.h"
#include "malware_scanner.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

#include <nlohmann/json.hpp>

struct kvd_handle {
  kvd::Config config;
  std::shared_ptr<int> token;
  std::shared_ptr<kvd::LightGbmModel> model;
  std::shared_ptr<kvd::LightGbmModel> model_normal;
  std::shared_ptr<kvd::LightGbmModel> model_packed;
  std::shared_ptr<kvd::FamilyClassifier> family_classifier;
};

static nlohmann::json to_json(const kvd::ScanResult& r) {
  nlohmann::json j;
  j["is_malware"] = r.is_malware;
  j["confidence"] = r.confidence;
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
      config->prediction_threshold);

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
  return h;
}

void kvd_destroy(kvd_handle* handle) {
  delete handle;
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

static bool path_exists(const std::string& path) {
  std::error_code ec;
  return !path.empty() && std::filesystem::exists(std::filesystem::path(path), ec) && !ec;
}

static std::shared_ptr<kvd::LightGbmModel> get_shared_model(const std::string& path) {
  if (path.empty()) {
    return {};
  }
  static std::mutex mu;
  static std::unordered_map<std::string, std::weak_ptr<kvd::LightGbmModel>> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(path);
  if (it != cache.end()) {
    auto sp = it->second.lock();
    if (sp) {
      return sp;
    }
  }
  auto model_opt = kvd::LightGbmModel::load_from_file(path);
  if (!model_opt) {
    return {};
  }
  auto sp = std::make_shared<kvd::LightGbmModel>(std::move(*model_opt));
  cache[path] = sp;
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
      handle->family_classifier);
  if (!scanner_opt) {
    return nullptr;
  }
  auto ptr = std::make_unique<kvd::MalwareScanner>(std::move(*scanner_opt));
  auto raw = ptr.get();
  cache.emplace(handle->token, std::move(ptr));
  return raw;
}

int kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len) {
  if (!handle || !path) {
    return -1;
  }
  auto scanner = get_thread_scanner(handle);
  if (!scanner) {
    return -1;
  }
  kvd::ScanResult r = scanner->scan_path(path);
  return write_json_out(to_json(r), out_json, out_len);
}

int kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len) {
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
}

int kvd_scan_paths(kvd_handle* handle, const char** paths, size_t count, char** out_json, size_t* out_len) {
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
}

void kvd_free(char* p) {
  std::free(p);
}

int kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len) {
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
      config->prediction_threshold);

  auto write_error = [&](const std::string& code) -> int {
    if (!out_error && !out_len) {
      return 0;
    }
    int rc = write_string_out(code, out_error, out_len);
    return rc == 0 ? 0 : KVD_MODEL_ERR_OOM;
  };

  if (cfg.model_path.empty()) {
    int rc = write_error("model_main_missing");
    return rc == 0 ? KVD_MODEL_ERR_MAIN_MISSING : rc;
  }
  if (!path_exists(cfg.model_path)) {
    int rc = write_error("model_main_missing");
    return rc == 0 ? KVD_MODEL_ERR_MAIN_MISSING : rc;
  }
  if (!kvd::LightGbmModel::load_from_file(cfg.model_path)) {
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
    if (!path_exists(cfg.model_normal_path)) {
      int rc = write_error("model_normal_missing");
      return rc == 0 ? KVD_MODEL_ERR_NORMAL_MISSING : rc;
    }
    if (!kvd::LightGbmModel::load_from_file(cfg.model_normal_path)) {
      int rc = write_error("model_normal_invalid");
      return rc == 0 ? KVD_MODEL_ERR_NORMAL_INVALID : rc;
    }
  }

  if (has_packed) {
    if (!path_exists(cfg.model_packed_path)) {
      int rc = write_error("model_packed_missing");
      return rc == 0 ? KVD_MODEL_ERR_PACKED_MISSING : rc;
    }
    if (!kvd::LightGbmModel::load_from_file(cfg.model_packed_path)) {
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

  int rc = write_error("ok");
  return rc == 0 ? KVD_MODEL_OK : rc;
}
