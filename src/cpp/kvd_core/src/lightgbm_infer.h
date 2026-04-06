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
#include <optional>
#include <string>
#include <vector>

namespace kvd {

class LightGbmModel {
public:
    LightGbmModel() = default;
    LightGbmModel(const LightGbmModel&) = delete;
    LightGbmModel& operator=(const LightGbmModel&) = delete;
    LightGbmModel(LightGbmModel&&) noexcept;
    LightGbmModel& operator=(LightGbmModel&&) noexcept;
    ~LightGbmModel();

    static std::optional<LightGbmModel> load_from_file(const std::string& path);

    std::optional<float> predict_one(const std::vector<float>& features) const;
    std::optional<std::vector<float>> predict_batch(const std::vector<float>& features, std::size_t row_count,
                                                    std::size_t num_features) const;

    bool ok() const;

private:
    void* handle_ = nullptr;
    int num_iterations_ = 0;
};

}  // namespace kvd
