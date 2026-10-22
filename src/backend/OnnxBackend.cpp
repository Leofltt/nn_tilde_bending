#include "OnnxBackend.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

// Include official ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

struct OnnxBackend::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "NetBender_OnnxBackend"};
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Dynamic graph metadata
    std::vector<std::string> inputNames;
    std::vector<const char*> inputNamePtrs;
    std::vector<std::string> outputNames;
    std::vector<const char*> outputNamePtrs;

    // Preallocated tensor shape storage to avoid allocations on audio thread
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;
};

OnnxBackend::OnnxBackend()
    : m_impl(std::make_unique<Impl>())
{
    // Optimize session for low latency audio threads
    m_impl->sessionOptions.SetIntraOpNumThreads(1);
    m_impl->sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

OnnxBackend::~OnnxBackend() = default;

int OnnxBackend::load(std::string path, double sampleRate, const std::string* target_method) {
    std::lock_guard<std::mutex> lock(m_sessionMutex);
    m_loaded = false;
    m_path = path;
    m_sampleRate = sampleRate > 0.0 ? sampleRate : 44100.0;

    try {
#if defined(_WIN32)
        std::wstring wpath(path.begin(), path.end());
        m_impl->session = std::make_unique<Ort::Session>(m_impl->env, wpath.c_str(), m_impl->sessionOptions);
#else
        m_impl->session = std::make_unique<Ort::Session>(m_impl->env, path.c_str(), m_impl->sessionOptions);
#endif
        introspectGraph();
        m_loaded = true;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[OnnxBackend] Error loading model " << path << ": " << e.what() << std::endl;
        m_loaded = false;
        return -1;
    }
}

int OnnxBackend::reload() {
    if (m_path.empty()) return -1;
    return load(m_path, m_sampleRate);
}

bool OnnxBackend::is_loaded() {
    return m_loaded;
}

void OnnxBackend::set_sample_rate(double sampleRate) {
    if (sampleRate > 0.0) m_sampleRate = sampleRate;
}

double OnnxBackend::get_sample_rate() {
    return m_sampleRate;
}

void OnnxBackend::use_gpu(bool value) {
    m_useGpu = value;
}

void OnnxBackend::introspectGraph() {
    if (!m_impl->session) return;

    m_availableLayers.clear();
    m_parameters.clear();
    m_audioInputs.clear();
    m_audioOutputs.clear();
    m_impl->inputNames.clear();
    m_impl->inputNamePtrs.clear();
    m_impl->outputNames.clear();
    m_impl->outputNamePtrs.clear();

    size_t numInputs = m_impl->session->GetInputCount();
    size_t numOutputs = m_impl->session->GetOutputCount();

    for (size_t i = 0; i < numInputs; ++i) {
        auto inputName = m_impl->session->GetInputNameAllocated(i, m_impl->allocator);
        std::string name = inputName.get();
        m_impl->inputNames.push_back(name);

        auto typeInfo = m_impl->session->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensorInfo.GetShape();

        // Categorize input:
        // Audio stream inputs typically have negative/symbolic dynamic sample length, or batch dim
        // Parameter inputs have fixed positive weights dimensions (e.g. conv kernels, bias vectors)
        bool isAudioStream = (i == 0) || (shape.size() >= 3 && (shape.back() <= 0 || shape.back() > 2048));

        if (isAudioStream && m_audioInputs.empty()) {
            StreamInfo stream;
            stream.name = name;
            stream.inputIndex = i;
            stream.shape = shape;
            m_audioInputs.push_back(stream);
            m_inChannels = (shape.size() >= 2 && shape[1] > 0) ? (int)shape[1] : 1;
        } else {
            // Parameter input (Exposed Port)
            ParameterInfo param;
            param.name = name;
            param.shape = shape;
            param.inputIndex = i;

            size_t total = 1;
            for (auto d : shape) {
                total *= (d > 0 ? d : 1);
            }
            param.totalElements = total;
            param.originalWeights.assign(total, 0.0f);
            param.currentWeights.assign(total, 0.0f);

            m_parameters[name] = param;
            m_availableLayers.push_back(name);
        }
    }

    // Refresh pointers after vector is fully populated
    for (const auto& n : m_impl->inputNames) {
        m_impl->inputNamePtrs.push_back(n.c_str());
    }

    // Outputs
    for (size_t i = 0; i < numOutputs; ++i) {
        auto outputName = m_impl->session->GetOutputNameAllocated(i, m_impl->allocator);
        std::string name = outputName.get();
        m_impl->outputNames.push_back(name);

        auto typeInfo = m_impl->session->GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensorInfo.GetShape();

        StreamInfo stream;
        stream.name = name;
        stream.inputIndex = i;
        stream.shape = shape;
        m_audioOutputs.push_back(stream);

        if (i == 0) {
            m_outChannels = (shape.size() >= 2 && shape[1] > 0) ? (int)shape[1] : 1;
        }
    }

    for (const auto& n : m_impl->outputNames) {
        m_impl->outputNamePtrs.push_back(n.c_str());
    }

    std::cout << "[OnnxBackend] Loaded " << m_path << " | Audio In: " << m_inChannels 
              << " | Audio Out: " << m_outChannels << " | Bendable Layers: " 
              << m_availableLayers.size() << std::endl;
}

void OnnxBackend::perform(std::vector<float*>& in_buffer,
                          std::vector<float*>& out_buffer,
                          std::string method,
                          int n_batches, int n_out_channels, int n_vec) {
    if (!m_loaded || !m_impl->session) return;
    std::unique_lock<std::mutex> lock(m_sessionMutex, std::try_to_lock);
    if (!lock.owns_lock()) return;

    try {
        std::vector<Ort::Value> inputTensors;
        inputTensors.reserve(m_impl->inputNames.size());

        // 1. Bind audio input tensor
        int inCh = std::min((int)in_buffer.size(), m_inChannels);
        std::vector<int64_t> inShape = { 1, inCh, n_vec };

        // For planar audio in_buffer [ch][samples], wrap the first channel pointer
        // (or contiguous interleaved block depending on model layout)
        float* audioInputPtr = in_buffer.empty() ? nullptr : in_buffer[0];
        if (audioInputPtr) {
            inputTensors.push_back(Ort::Value::CreateTensor<float>(
                m_impl->memoryInfo,
                audioInputPtr,
                inCh * n_vec,
                inShape.data(),
                inShape.size()
            ));
        }

        // 2. Bind parameter tensors (Input-Promoted zero-copy weights)
        for (size_t i = 1; i < m_impl->inputNames.size(); ++i) {
            const std::string& paramName = m_impl->inputNames[i];
            auto it = m_parameters.find(paramName);
            if (it != m_parameters.end()) {
                ParameterInfo& p = it->second;
                inputTensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memoryInfo,
                    p.currentWeights.data(),
                    p.totalElements,
                    p.shape.data(),
                    p.shape.size()
                ));
            }
        }

        // 3. Run session
        auto outputTensors = m_impl->session->Run(
            Ort::RunOptions{nullptr},
            m_impl->inputNamePtrs.data(),
            inputTensors.data(),
            inputTensors.size(),
            m_impl->outputNamePtrs.data(),
            m_impl->outputNamePtrs.size()
        );

        // 4. Copy computed output back to out_buffer
        if (!outputTensors.empty()) {
            float* outData = outputTensors[0].GetTensorMutableData<float>();
            int outCh = std::min((int)out_buffer.size(), m_outChannels);
            for (int c = 0; c < outCh; ++c) {
                if (out_buffer[c]) {
                    std::memcpy(out_buffer[c], outData + (c * n_vec), n_vec * sizeof(float));
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[OnnxBackend::perform] Execution error: " << e.what() << std::endl;
    }
}

void OnnxBackend::perform_forward(std::vector<float*>& in_buffer,
                                  std::vector<float*>& out_buffer,
                                  int n_batches, int n_out_channels, int n_vec,
                                  LatentHook latent_hook) {
    perform(in_buffer, out_buffer, "forward", n_batches, n_out_channels, n_vec);
}

void OnnxBackend::perform_prior_decode(std::vector<float*>& out_buffer,
                                       int n_batches, int n_out_channels, int n_vec) {
    std::vector<float*> empty_in;
    perform(empty_in, out_buffer, "prior", n_batches, n_out_channels, n_vec);
}

void OnnxBackend::perform_decode_gen(std::vector<float*>& out_buffer,
                                     int n_batches, int n_out_channels, int n_vec,
                                     LatentHook latent_hook) {
    std::vector<float*> empty_in;
    perform(empty_in, out_buffer, "decode gen", n_batches, n_out_channels, n_vec);
}

bool OnnxBackend::has_method(std::string method_name) {
    return method_name == "forward";
}

bool OnnxBackend::has_prior_decode() {
    return false;
}

bool OnnxBackend::has_autoencode() {
    return true;
}

std::vector<std::string> OnnxBackend::get_plugin_modes() {
    return { "forward" };
}

std::vector<int> OnnxBackend::get_mode_params(std::string mode) {
    // [in_dim, in_ratio, out_dim, out_ratio]
    return { m_inChannels, 1, m_outChannels, 1 };
}

std::vector<std::string> OnnxBackend::get_available_methods(LockedModel* model) {
    return { "forward" };
}

std::vector<int> OnnxBackend::get_method_params(std::string method) {
    return { m_inChannels, 1, m_outChannels, 1 };
}

int OnnxBackend::get_higher_ratio() {
    return m_higherRatio;
}

// Parameter Introspection & Bending
std::vector<std::string> OnnxBackend::get_available_layers() {
    return m_availableLayers;
}

std::vector<float> OnnxBackend::get_layer_weights(std::string layer_name) {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        return it->second.currentWeights;
    }
    return {};
}

std::vector<float> OnnxBackend::get_original_layer_weights(std::string layer_name) {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        return it->second.originalWeights;
    }
    return {};
}

std::vector<int64_t> OnnxBackend::get_layer_shape(std::string layer_name) {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        return it->second.shape;
    }
    return {};
}

void OnnxBackend::set_layer_weights(std::string layer_name, const std::vector<float>& weights) {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        size_t count = std::min(it->second.currentWeights.size(), weights.size());
        std::memcpy(it->second.currentWeights.data(), weights.data(), count * sizeof(float));
    }
}

void OnnxBackend::reset_layer_weights(std::string layer_name) {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        it->second.currentWeights = it->second.originalWeights;
    }
}

void OnnxBackend::reset_all_layer_weights() {
    for (auto& pair : m_parameters) {
        pair.second.currentWeights = pair.second.originalWeights;
    }
}

int OnnxBackend::save_model(std::string path) {
    return 0; // Exporting mutated ONNX will be handled in Phase 5
}

// Stubs for Max/MSP & PureData Attributes
std::vector<std::string> OnnxBackend::get_available_attributes() { return {}; }
std::vector<std::string> OnnxBackend::get_settable_attributes() { return {}; }
bool OnnxBackend::has_settable_attribute(std::string attribute) { return false; }
std::string OnnxBackend::get_attribute_as_string(std::string attribute_name) { return ""; }
void OnnxBackend::set_attribute(std::string, std::vector<std::string>, const BufferMap&) {}
std::vector<std::string> OnnxBackend::get_buffer_attributes() { return {}; }
bool OnnxBackend::is_buffer_element_of_attribute(std::string, int) { return false; }
bool OnnxBackend::is_tensor_element_of_attribute(std::string, int) { return false; }
std::string OnnxBackend::get_buffer_name(std::string, int) { return ""; }
int OnnxBackend::update_buffer(std::string, StaticBuffer<float>&) { return 0; }
int OnnxBackend::reset_buffer(std::string) { return 0; }
ModelInfo OnnxBackend::get_model_info() { return {}; }
