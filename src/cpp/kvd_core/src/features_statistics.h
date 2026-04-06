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
#include <vector>

namespace kvd {

struct ByteSequenceResult;

std::vector<float> extract_statistical_features(
    const std::vector<std::uint8_t>& padded_sequence, std::size_t orig_length);
std::vector<float> extract_statistical_features(const ByteSequenceResult& seq);

}  // namespace kvd
