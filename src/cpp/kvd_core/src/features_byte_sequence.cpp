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
#include <algorithm>
#include <fstream>

#include "features.h"
#include "util_filesystem.h"

namespace kvd {

static ByteSequenceStats compute_stats(const std::vector<std::uint8_t>& data,
                                       std::size_t n) {
    ByteSequenceStats s;
    s.hist.fill(0);
    if (n == 0) {
        return s;
    }
    s.has_data = true;
    s.min_val = data[0];
    s.max_val = data[0];
    double mean = 0.0;
    double m2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint8_t b = data[i];
        double x = static_cast<double>(b);
        double delta = x - mean;
        mean += delta / static_cast<double>(i + 1);
        double delta2 = x - mean;
        m2 += delta * delta2;
        if (b < s.min_val) s.min_val = b;
        if (b > s.max_val) s.max_val = b;
        if (b == 0) s.count_0++;
        if (b == 255) s.count_255++;
        if (b == 0x90) s.count_90++;
        if (b >= 32 && b <= 126) s.count_printable++;
        s.hist[b]++;
    }
    s.mean = mean;
    s.m2 = m2;
    return s;
}

std::optional<ByteSequenceResult> extract_byte_sequence_from_path(
    const std::string& path, std::size_t max_file_size,
    const std::optional<std::string>& allowed_root) {
    auto valid = validate_path(path, allowed_root);
    if (!valid) {
        return std::nullopt;
    }
    std::size_t offset = 8;
    if (max_file_size <= offset) {
        return std::nullopt;
    }
    ByteSequenceResult r;
    r.padded_sequence.assign(max_file_size, 0);
    {
        auto fs_path = to_filesystem_path(*valid);
        if (!fs_path) {
            return std::nullopt;
        }
        std::ifstream f(*fs_path, std::ios::binary);
        if (!f) {
            return std::nullopt;
        }
        f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!f) {
            return std::nullopt;
        }
        std::size_t want = max_file_size - offset;
        f.read(reinterpret_cast<char*>(r.padded_sequence.data()),
               static_cast<std::streamsize>(want));
        std::streamsize read_n = f.gcount();
        if (read_n < 0) {
            return std::nullopt;
        }
        r.original_length = static_cast<std::size_t>(read_n);
    }
    r.stats = compute_stats(r.padded_sequence, r.original_length);
    return r;
}

std::optional<ByteSequenceResult> extract_byte_sequence_from_bytes(
    const std::uint8_t* bytes, std::size_t len, std::size_t max_file_size) {
    if (!bytes) {
        return std::nullopt;
    }
    std::size_t offset = 8;
    if (max_file_size <= offset) {
        return std::nullopt;
    }
    if (len <= offset) {
        ByteSequenceResult r;
        r.original_length = 0;
        r.padded_sequence.assign(max_file_size, 0);
        r.stats = compute_stats(r.padded_sequence, r.original_length);
        return r;
    }
    std::size_t avail = len - offset;
    std::size_t want = max_file_size - offset;
    std::size_t take = avail < want ? avail : want;

    ByteSequenceResult r;
    r.original_length = take;
    r.padded_sequence.assign(max_file_size, 0);
    if (take > 0) {
        std::copy(bytes + offset, bytes + offset + take,
                  r.padded_sequence.begin());
    }
    r.stats = compute_stats(r.padded_sequence, r.original_length);
    return r;
}

}  // namespace kvd
