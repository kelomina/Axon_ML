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
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace kvd {

std::optional<std::string> validate_path(
    const std::string& path, const std::optional<std::string>& allowed_root);
std::optional<std::filesystem::path> to_filesystem_path(
    const std::string& path);
bool read_file_bytes(const std::string& path, std::vector<std::uint8_t>& out);
bool read_file_bytes_seek(const std::string& path, std::size_t offset,
                          std::size_t max_count,
                          std::vector<std::uint8_t>& out);

}  // namespace kvd
