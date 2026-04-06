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
#include "onnx_infer.h"

#include <onnxruntime_cxx_api.h>

#include <memory>

namespace kvd {

struct OnnxModel::Impl {
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::SessionOptions> options;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator;
};

OnnxModel::OnnxModel() = default;

OnnxModel::OnnxModel(OnnxModel&& other) noexcept
    : impl_(std::move(other.impl_)), input_size_(other.input_size_), output_size_(other.output_size_) {
    other.input_size_ = 0;
    other.output_size_ = 0;
}

OnnxModel& OnnxModel::operator=(OnnxModel&& other) noexcept {
    if (this == &other) return *this;
    impl_ = std::move(other.impl_);
    input_size_ = other.input_size_;
    output_size_ = other.output_size_;
    other.input_size_ = 0;
    other.output_size_ = 0;
    return *this;
}

OnnxModel::~OnnxModel() = default;

std::optional<OnnxModel> OnnxModel::load_from_file(const std::string& path) {
    try {
        auto env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "KoloVirusDetector");
        auto options = std::make_unique<Ort::SessionOptions>();
        options->SetIntraOpNumThreads(1);
        options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        std::wstring wpath = std::wstring(path.begin(), path.end());
        auto session = std::make_unique<Ort::Session>(*env, wpath.c_str(), *options);
        Ort::AllocatorWithDefaultOptions allocator;

        std::size_t input_count = session->GetInputCount();
        std::size_t output_count = session->GetOutputCount();

        if (input_count == 0 || output_count == 0) { return std::nullopt; }

        auto input_name_ptr = session->GetInputNameAllocated(0, allocator);
        auto output_name_ptr = session->GetOutputNameAllocated(0, allocator);

        auto input_type_info = session->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_shape = input_tensor_info.GetShape();
        auto output_type_info = session->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_shape = output_tensor_info.GetShape();

        std::size_t input_size = 1;
        for (auto dim : input_shape) {
            if (dim > 0) { input_size *= static_cast<std::size_t>(dim); }
        }

        std::size_t output_size = 1;
        for (auto dim : output_shape) {
            if (dim > 0) { output_size *= static_cast<std::size_t>(dim); }
        }

        OnnxModel m;
        m.impl_ = std::make_unique<Impl>();
        m.impl_->env = std::move(env);
        m.impl_->options = std::move(options);
        m.impl_->session = std::move(session);
        m.impl_->allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        m.input_size_ = input_size;
        m.output_size_ = output_size;
        return m;
    } catch (...) { return std::nullopt; }
}

std::optional<float> OnnxModel::predict_one(const std::vector<float>& features) const {
    if (!impl_ || !impl_->session) { return std::nullopt; }
    if (features.size() != input_size_) { return std::nullopt; }
    try {
        Ort::Session* session = impl_->session.get();
        Ort::AllocatorWithDefaultOptions* allocator = impl_->allocator.get();

        auto input_name_ptr = session->GetInputNameAllocated(0, *allocator);
        auto output_name_ptr = session->GetOutputNameAllocated(0, *allocator);

        std::vector<float> output(1);
        std::vector<const char*> input_names = {input_name_ptr.get()};
        std::vector<const char*> output_names = {output_name_ptr.get()};

        std::array<std::int64_t, 2> input_shape = {1, static_cast<std::int64_t>(input_size_)};
        std::array<std::int64_t, 2> output_shape = {1, 1};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(*allocator, input_shape.data(), input_shape.size());
        float* input_data = input_tensor.GetTensorMutableData<float>();
        std::copy(features.begin(), features.end(), input_data);

        Ort::Value output_tensor =
            Ort::Value::CreateTensor<float>(*allocator, output_shape.data(), output_shape.size());
        float* output_data = output_tensor.GetTensorMutableData<float>();
        output_data[0] = 0.0f;

        session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(),
                     &output_tensor, 1);

        return output_data[0];
    } catch (...) { return std::nullopt; }
}

std::optional<std::vector<float>> OnnxModel::predict_batch(const std::vector<float>& features, std::size_t row_count,
                                                           std::size_t num_features) const {
    if (!impl_ || !impl_->session) { return std::nullopt; }
    if (row_count == 0 || num_features != input_size_) { return std::nullopt; }
    if (features.size() != row_count * num_features) { return std::nullopt; }
    try {
        Ort::Session* session = impl_->session.get();
        Ort::AllocatorWithDefaultOptions* allocator = impl_->allocator.get();

        auto input_name_ptr = session->GetInputNameAllocated(0, *allocator);
        auto output_name_ptr = session->GetOutputNameAllocated(0, *allocator);

        std::vector<float> output(row_count);
        std::vector<const char*> input_names = {input_name_ptr.get()};
        std::vector<const char*> output_names = {output_name_ptr.get()};

        std::array<std::int64_t, 2> input_shape = {static_cast<std::int64_t>(row_count),
                                                   static_cast<std::int64_t>(num_features)};
        std::array<std::int64_t, 2> output_shape = {static_cast<std::int64_t>(row_count),
                                                    static_cast<std::int64_t>(output_size_)};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(*allocator, input_shape.data(), input_shape.size());
        float* input_data = input_tensor.GetTensorMutableData<float>();
        std::copy(features.begin(), features.end(), input_data);

        std::vector<Ort::Value> output_tensors;
        output_tensors.emplace_back(
            Ort::Value::CreateTensor<float>(*allocator, output_shape.data(), output_shape.size()));
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::fill(output_data, output_data + row_count * output_size_, 0.0f);

        session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(),
                     output_tensors.data(), 1);

        for (std::size_t i = 0; i < row_count; ++i) { output[i] = output_data[i]; }
        return output;
    } catch (...) { return std::nullopt; }
}

bool OnnxModel::ok() const {
    return impl_ && impl_->session != nullptr;
}

std::size_t OnnxModel::get_input_size() const {
    return input_size_;
}

std::size_t OnnxModel::get_output_size() const {
    return output_size_;
}

}  // namespace kvd
