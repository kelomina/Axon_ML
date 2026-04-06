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
#include "util_filesystem.h"

#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cwctype>
#include <filesystem>
#include <fstream>

namespace kvd {

static bool contains_nul(const std::string& s) {
    return std::find(s.begin(), s.end(), '\0') != s.end();
}

static std::optional<std::wstring> multi_byte_to_wide(const std::string& s, unsigned int code_page, DWORD flags) {
    if (s.empty()) { return std::wstring(); }
    int len = MultiByteToWideChar(code_page, flags, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (len <= 0) { return std::nullopt; }
    std::wstring w;
    w.resize(static_cast<std::size_t>(len));
    int ok = MultiByteToWideChar(code_page, flags, s.data(), static_cast<int>(s.size()), w.data(), len);
    if (ok != len) { return std::nullopt; }
    return w;
}

static std::optional<std::wstring> utf8_or_ansi_to_wide(const std::string& s) {
    auto utf8 = multi_byte_to_wide(s, CP_UTF8, MB_ERR_INVALID_CHARS);
    if (utf8) { return utf8; }
    return multi_byte_to_wide(s, CP_ACP, 0);
}

static std::wstring to_lower(std::wstring s) {
    std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
    return s;
}

std::optional<std::filesystem::path> to_filesystem_path(const std::string& path) {
    if (path.empty() || contains_nul(path)) { return std::nullopt; }
    auto w = utf8_or_ansi_to_wide(path);
    if (!w) { return std::nullopt; }
    return std::filesystem::path(*w);
}

std::optional<std::string> validate_path(const std::string& path, const std::optional<std::string>& allowed_root) {
    auto p_opt = to_filesystem_path(path);
    if (!p_opt) { return std::nullopt; }
    std::error_code ec;
    std::filesystem::path p = std::filesystem::absolute(*p_opt, ec);
    if (ec) { return std::nullopt; }
    p = std::filesystem::weakly_canonical(p, ec);
    if (ec) { return std::nullopt; }
    if (!std::filesystem::exists(p, ec) || ec) { return std::nullopt; }
    if (allowed_root) {
        auto root_opt = to_filesystem_path(*allowed_root);
        if (!root_opt) { return std::nullopt; }
        std::filesystem::path root = std::filesystem::absolute(*root_opt, ec);
        if (ec) { return std::nullopt; }
        root = std::filesystem::weakly_canonical(root, ec);
        if (ec) { return std::nullopt; }

        std::wstring abs_p = p.wstring();
        std::wstring base = root.wstring();
        abs_p = to_lower(std::move(abs_p));
        base = to_lower(std::move(base));

        if (abs_p == base) {
        } else {
            if (!base.empty() && base.back() != L'\\' && base.back() != L'/') { base.push_back(L'\\'); }
            if (abs_p.size() < base.size()) { return std::nullopt; }
            if (abs_p.compare(0, base.size(), base) != 0) { return std::nullopt; }
        }
    }
    auto u8 = p.u8string();
    return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

bool read_file_bytes(const std::string& path, std::vector<std::uint8_t>& out) {
    out.clear();
    auto p = to_filesystem_path(path);
    if (!p) { return false; }
    std::ifstream f(*p, std::ios::binary);
    if (!f) { return false; }
    f.seekg(0, std::ios::end);
    std::streamoff size = f.tellg();
    if (size < 0) { return false; }
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(size));
    if (!out.empty()) {
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
        if (!f) {
            out.clear();
            return false;
        }
    }
    return true;
}

bool read_file_bytes_seek(const std::string& path, std::size_t offset, std::size_t max_count,
                          std::vector<std::uint8_t>& out) {
    out.clear();
    auto p = to_filesystem_path(path);
    if (!p) { return false; }
    std::ifstream f(*p, std::ios::binary);
    if (!f) { return false; }
    f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!f) { return false; }
    out.resize(max_count);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(max_count));
    std::streamsize read_count = f.gcount();
    if (read_count < 0) {
        out.clear();
        return false;
    }
    out.resize(static_cast<std::size_t>(read_count));
    return true;
}

}  // namespace kvd
