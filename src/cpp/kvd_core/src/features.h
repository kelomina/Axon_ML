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

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "kvd_internal.h"

namespace kvd {

struct ByteSequenceStats {
    std::array<std::uint32_t, 256> hist{};
    double mean = 0.0;
    double m2 = 0.0;
    std::uint8_t min_val = 0;
    std::uint8_t max_val = 0;
    int count_0 = 0;
    int count_255 = 0;
    int count_90 = 0;
    int count_printable = 0;
    bool has_data = false;
};

struct ByteSequenceResult {
    std::vector<std::uint8_t> padded_sequence;
    std::size_t original_length = 0;
    ByteSequenceStats stats;
};

std::optional<ByteSequenceResult> extract_byte_sequence_from_path(
    const std::string& path, std::size_t max_file_size,
    const std::optional<std::string>& allowed_root);

std::optional<ByteSequenceResult> extract_byte_sequence_from_bytes(
    const std::uint8_t* bytes, std::size_t len, std::size_t max_file_size);

std::vector<float> extract_combined_pe_features_from_path(
    const std::string& path, const std::optional<std::string>& allowed_root);

std::optional<std::vector<std::string>> extract_import_sequence_from_path(
    const std::string& path, const std::optional<std::string>& allowed_root);

std::optional<std::size_t> pe_feature_index(const std::string& name);

}  // namespace kvd
