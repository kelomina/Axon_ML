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
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kvd {

class OnnxModel {
public:
    OnnxModel();
    OnnxModel(const OnnxModel&) = delete;
    OnnxModel& operator=(const OnnxModel&) = delete;
    OnnxModel(OnnxModel&&) noexcept;
    OnnxModel& operator=(OnnxModel&&) noexcept;
    ~OnnxModel();

    static std::optional<OnnxModel> load_from_file(const std::string& path);

    std::optional<float> predict_one(const std::vector<float>& features) const;
    std::optional<std::vector<float>> predict_batch(const std::vector<float>& features, std::size_t row_count,
                                                    std::size_t num_features) const;

    bool ok() const;
    std::size_t get_input_size() const;
    std::size_t get_output_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::size_t input_size_ = 0;
    std::size_t output_size_ = 0;
};

}  // namespace kvd
