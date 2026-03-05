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

#include "kvd_internal.h"

#include <optional>
#include <string>

namespace kvd {

Config config_from_api(
    const char* model_path,
    const char* model_normal_path,
    const char* model_packed_path,
    const char* family_classifier_json_path,
    const char* allowed_scan_root,
    unsigned int max_file_size,
    float prediction_threshold);

std::optional<std::string> getenv_string(const char* name);

}
