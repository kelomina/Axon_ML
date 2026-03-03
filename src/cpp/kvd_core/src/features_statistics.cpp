#include "features_statistics.h"

#include "features.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace kvd {

static double entropy_from_hist(const std::array<std::uint32_t, 256>& counts, std::size_t n) {
  if (n == 0) return 0.0;
  double inv = 1.0 / static_cast<double>(n);
  double s = 0.0;
  for (std::size_t i = 0; i < 256; ++i) {
    if (counts[i] == 0) continue;
    double p = static_cast<double>(counts[i]) * inv;
    s += p * std::log2(p);
  }
  return (-s) / 8.0;
}

static float percentile_from_hist(const std::array<std::uint32_t, 256>& counts, std::size_t n, double q) {
  if (n == 0) return 0.0f;
  double pos = (static_cast<double>(n) - 1.0) * q;
  std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  double w = pos - static_cast<double>(lo);
  std::size_t acc = 0;
  int a = 0;
  for (int i = 0; i < 256; ++i) {
    acc += counts[i];
    if (acc > lo) {
      a = i;
      break;
    }
  }
  acc = 0;
  int b = a;
  for (int i = 0; i < 256; ++i) {
    acc += counts[i];
    if (acc > hi) {
      b = i;
      break;
    }
  }
  double av = static_cast<double>(a);
  double bv = static_cast<double>(b);
  return static_cast<float>(av + (bv - av) * w);
}

static float std_f32(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  double m = 0.0;
  for (float x : v) m += static_cast<double>(x);
  m /= static_cast<double>(v.size());
  double acc = 0.0;
  for (float x : v) {
    double d = static_cast<double>(x) - m;
    acc += d * d;
  }
  return static_cast<float>(std::sqrt(acc / static_cast<double>(v.size())));
}

static ByteSequenceStats compute_stats_from_sequence(const std::vector<std::uint8_t>& padded_sequence, std::size_t n) {
  ByteSequenceStats s;
  s.hist.fill(0);
  if (n == 0) {
    return s;
  }
  s.has_data = true;
  s.min_val = padded_sequence[0];
  s.max_val = padded_sequence[0];
  double mean = 0.0;
  double m2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    std::uint8_t b = padded_sequence[i];
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

std::vector<float> extract_statistical_features(const ByteSequenceResult& seq) {
  std::size_t n = seq.original_length;
  if (n > seq.padded_sequence.size()) n = seq.padded_sequence.size();

  std::vector<float> features;
  features.reserve(49);

  ByteSequenceStats stats = seq.stats;
  if (!stats.has_data && n > 0) {
    stats = compute_stats_from_sequence(seq.padded_sequence, n);
  }

  double mean_val = stats.mean;
  double m2 = stats.m2;
  std::uint8_t min_val = stats.min_val;
  std::uint8_t max_val = stats.max_val;
  int count_0 = stats.count_0;
  int count_255 = stats.count_255;
  int count_90 = stats.count_90;
  int count_printable = stats.count_printable;
  const std::array<std::uint32_t, 256>& hist = stats.hist;
  double std_val = n > 0 ? std::sqrt(m2 / static_cast<double>(n)) : 0.0;

  float median_val = 0.0f;
  float q25 = 0.0f;
  float q75 = 0.0f;
  if (n > 0) {
    median_val = percentile_from_hist(hist, n, 0.5);
    q25 = percentile_from_hist(hist, n, 0.25);
    q75 = percentile_from_hist(hist, n, 0.75);
  }

  features.push_back(static_cast<float>(mean_val));
  features.push_back(static_cast<float>(std_val));
  features.push_back(static_cast<float>(min_val));
  features.push_back(static_cast<float>(max_val));
  features.push_back(median_val);
  features.push_back(q25);
  features.push_back(q75);

  features.push_back(static_cast<float>(count_0));
  features.push_back(static_cast<float>(count_255));
  features.push_back(static_cast<float>(count_90));
  features.push_back(static_cast<float>(count_printable));

  features.push_back(static_cast<float>(entropy_from_hist(hist, n)));

  std::size_t one_third = 0;
  if (n >= 3) one_third = n / 3;

  for (int seg_i = 0; seg_i < 3; ++seg_i) {
    std::size_t start = 0;
    std::size_t end = n;
    if (n >= 3) {
      if (seg_i == 0) {
        start = 0;
        end = one_third;
      } else if (seg_i == 1) {
        start = one_third;
        end = 2 * one_third;
      } else {
        start = 2 * one_third;
        end = n;
      }
    }

    std::size_t seg_len = end > start ? (end - start) : 0;
    if (seg_len == 0) {
      features.push_back(0.0f);
      features.push_back(0.0f);
      features.push_back(0.0f);
    } else {
      double m = 0.0;
      double sd = 0.0;
      std::array<std::uint32_t, 256> h{};
      h.fill(0);
      for (std::size_t i = start; i < end; ++i) {
        double x = static_cast<double>(seq.padded_sequence[i]);
        m += x;
        sd += x * x;
        h[seq.padded_sequence[i]]++;
      }
      m /= static_cast<double>(seg_len);
      sd = std::sqrt(std::max(0.0, sd / static_cast<double>(seg_len) - m * m));
      features.push_back(static_cast<float>(m));
      features.push_back(static_cast<float>(sd));
      features.push_back(static_cast<float>(entropy_from_hist(h, seg_len)));
    }
  }

  static constexpr std::size_t STAT_CHUNK_COUNT = 10;
  std::size_t chunk_size = n / STAT_CHUNK_COUNT;
  if (chunk_size < 1) chunk_size = 1;

  std::vector<float> chunk_means;
  std::vector<float> chunk_stds;
  chunk_means.reserve(STAT_CHUNK_COUNT);
  chunk_stds.reserve(STAT_CHUNK_COUNT);

  std::vector<double> prefix_sum(n + 1, 0.0);
  std::vector<double> prefix_sq(n + 1, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double x = static_cast<double>(seq.padded_sequence[i]);
    prefix_sum[i + 1] = prefix_sum[i] + x;
    prefix_sq[i + 1] = prefix_sq[i] + x * x;
  }

  for (std::size_t i = 0; i < STAT_CHUNK_COUNT; ++i) {
    std::size_t start = i * chunk_size;
    std::size_t end = (i < STAT_CHUNK_COUNT - 1) ? (start + chunk_size) : n;
    if (start > n) start = n;
    if (end > n) end = n;
    std::size_t len = end > start ? (end - start) : 0;
    if (len == 0) {
      chunk_means.push_back(0.0f);
      chunk_stds.push_back(0.0f);
    } else {
      double sum = prefix_sum[end] - prefix_sum[start];
      double sq = prefix_sq[end] - prefix_sq[start];
      double m = sum / static_cast<double>(len);
      double var = std::max(0.0, sq / static_cast<double>(len) - m * m);
      double sd = std::sqrt(var);
      chunk_means.push_back(static_cast<float>(m));
      chunk_stds.push_back(static_cast<float>(sd));
    }
  }

  for (float x : chunk_means) features.push_back(x);
  for (float x : chunk_stds) features.push_back(x);

  if (chunk_means.size() > 1) {
    std::vector<float> mean_diffs;
    std::vector<float> std_diffs;
    mean_diffs.reserve(chunk_means.size() - 1);
    std_diffs.reserve(chunk_stds.size() - 1);
    for (std::size_t i = 1; i < chunk_means.size(); ++i) {
      mean_diffs.push_back(chunk_means[i] - chunk_means[i - 1]);
      std_diffs.push_back(chunk_stds[i] - chunk_stds[i - 1]);
    }

    double mean_abs = 0.0;
    for (float x : mean_diffs) mean_abs += std::abs(static_cast<double>(x));
    mean_abs /= static_cast<double>(mean_diffs.size());

    double std_abs = 0.0;
    for (float x : std_diffs) std_abs += std::abs(static_cast<double>(x));
    std_abs /= static_cast<double>(std_diffs.size());

    auto mm1 = std::minmax_element(mean_diffs.begin(), mean_diffs.end());
    auto mm2 = std::minmax_element(std_diffs.begin(), std_diffs.end());

    features.push_back(static_cast<float>(mean_abs));
    features.push_back(std_f32(mean_diffs));
    features.push_back(*mm1.second);
    features.push_back(*mm1.first);

    features.push_back(static_cast<float>(std_abs));
    features.push_back(std_f32(std_diffs));
    features.push_back(*mm2.second);
    features.push_back(*mm2.first);
  } else {
    for (int i = 0; i < 8; ++i) features.push_back(0.0f);
  }

  return features;
}

std::vector<float> extract_statistical_features(const std::vector<std::uint8_t>& padded_sequence, std::size_t orig_length) {
  std::size_t n = orig_length;
  if (n > padded_sequence.size()) n = padded_sequence.size();

  ByteSequenceResult seq;
  seq.padded_sequence = padded_sequence;
  seq.original_length = orig_length;
  seq.stats = compute_stats_from_sequence(padded_sequence, n);
  return extract_statistical_features(seq);
}

}
