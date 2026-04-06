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

#include <optional>
#include <string>
#include <vector>

#include "kvd_internal.h"

namespace kvd {

class FamilyClassifier {
   public:
    static std::optional<FamilyClassifier> load_from_json(
        const std::string& path);

    std::optional<ScanResultFamily> predict(
        const std::vector<float>& features) const;

    bool ok() const;

   private:
    std::vector<int> cluster_ids_;
    std::vector<std::vector<float>> centroids_;
    std::vector<float> thresholds_;
    std::vector<std::string> family_names_;
    std::vector<float> scaler_mean_;
    std::vector<float> scaler_scale_;
};

}  // namespace kvd
