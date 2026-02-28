#pragma once

#include "kvd_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
    const std::string& path,
    std::size_t max_file_size,
    const std::optional<std::string>& allowed_root);

std::optional<ByteSequenceResult> extract_byte_sequence_from_bytes(
    const std::uint8_t* bytes,
    std::size_t len,
    std::size_t max_file_size);

std::vector<float> extract_combined_pe_features_from_path(
    const std::string& path,
    const std::optional<std::string>& allowed_root);

std::optional<std::vector<std::string>> extract_import_sequence_from_path(
    const std::string& path,
    const std::optional<std::string>& allowed_root);

std::optional<std::size_t> pe_feature_index(const std::string& name);

}
