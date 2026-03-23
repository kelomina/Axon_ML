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
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kvd {

struct Config {
  std::string model_path;
  std::string model_normal_path;
  std::string model_packed_path;
  std::string family_classifier_json_path;
  std::string hardcase_manifest_path;
  std::optional<std::string> allowed_scan_root;
  std::size_t max_file_size = 64 * 1024;
  float prediction_threshold = 0.98f;
  std::string onnx_model_path;
  std::string onnx_model_normal_path;
  std::string onnx_model_packed_path;
};

struct ScanResultFamily {
  std::string family_name;
  int cluster_id = -1;
  bool is_new_family = false;
};

struct ScanResult {
  bool is_malware = false;
  float confidence = 0.0f;
  bool axon_malware = false;
  float axon_score = 0.0f;
  bool hardcase_triggered = false;
  std::string hardcase_class;
  std::vector<float> hardcase_scores;
  std::optional<ScanResultFamily> family;
  bool signature_hit = false;
  float signature_score = 0.0f;
  std::string signature_reason;
  std::string error;
};

}
