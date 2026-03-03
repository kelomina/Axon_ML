#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace kvd {

std::optional<std::string> validate_path(const std::string& path, const std::optional<std::string>& allowed_root);
std::optional<std::filesystem::path> to_filesystem_path(const std::string& path);
bool read_file_bytes(const std::string& path, std::vector<std::uint8_t>& out);
bool read_file_bytes_seek(const std::string& path, std::size_t offset, std::size_t max_count, std::vector<std::uint8_t>& out);

}
