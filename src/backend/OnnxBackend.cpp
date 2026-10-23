#include "OnnxBackend.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstring>

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
    size_t numOverridables = m_impl->session->GetOverridableInitializerCount();

    // 1. Process primary required graph inputs (audio stream)
    for (size_t i = 0; i < numInputs; ++i) {
        auto inputName = m_impl->session->GetInputNameAllocated(i, m_impl->allocator);
        std::string name = inputName.get();
        m_impl->inputNames.push_back(name);

        auto typeInfo = m_impl->session->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensorInfo.GetShape();

        bool isAudioStream = (i == 0) || (shape.size() >= 3 && (shape.back() <= 0 || shape.back() > 2048));

        if (isAudioStream && m_audioInputs.empty()) {
            StreamInfo stream;
            stream.name = name;
            stream.inputIndex = i;
            stream.shape = shape;
            m_audioInputs.push_back(stream);
            m_inChannels = (shape.size() >= 2 && shape[1] > 0) ? (int)shape[1] : 1;
        } else {
            // Free parameter input
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

    // 2. Process Overridable Initializers (Promoted layers with model baseline defaults)
    for (size_t i = 0; i < numOverridables; ++i) {
        auto initName = m_impl->session->GetOverridableInitializerNameAllocated(i, m_impl->allocator);
        std::string name = initName.get();

        if (m_parameters.find(name) == m_parameters.end()) {
            auto typeInfo = m_impl->session->GetOverridableInitializerTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = tensorInfo.GetShape();

            ParameterInfo param;
            param.name = name;
            param.shape = shape;
            param.inputIndex = m_impl->inputNames.size();

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

    // 3. Load initial baseline weights from high-speed binary sidecar (.bin) if available
    std::string sidecarPath = m_path;
    size_t extPos = sidecarPath.find_last_of('.');
    if (extPos != std::string::npos) {
        sidecarPath = sidecarPath.substr(0, extPos) + ".bin";
    } else {
        sidecarPath += ".bin";
    }

    std::ifstream binFile(sidecarPath, std::ios::binary);
    if (binFile.is_open()) {
        char magic[4];
        binFile.read(magic, 4);
        if (std::memcmp(magic, "NBND", 4) == 0) {
            uint32_t version = 0;
            binFile.read(reinterpret_cast<char*>(&version), sizeof(version));
            uint32_t numTensors = 0;
            binFile.read(reinterpret_cast<char*>(&numTensors), sizeof(numTensors));

            for (uint32_t t = 0; t < numTensors; ++t) {
                uint32_t nameLen = 0;
                binFile.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
                std::string tName(nameLen, '\0');
                binFile.read(&tName[0], nameLen);

                uint32_t numDims = 0;
                binFile.read(reinterpret_cast<char*>(&numDims), sizeof(numDims));
                std::vector<int64_t> tShape(numDims);
                for (uint32_t d = 0; d < numDims; ++d) {
                    binFile.read(reinterpret_cast<char*>(&tShape[d]), sizeof(int64_t));
                }

                uint64_t numElems = 0;
                binFile.read(reinterpret_cast<char*>(&numElems), sizeof(numElems));

                std::vector<float> data(numElems);
                binFile.read(reinterpret_cast<char*>(data.data()), numElems * sizeof(float));

                auto it = m_parameters.find(tName);
                if (it != m_parameters.end()) {
                    it->second.originalWeights = data;
                    it->second.currentWeights = data;
                    it->second.totalElements = numElems;
                    it->second.shape = tShape;
                }
            }
            std::cout << "[OnnxBackend] Successfully loaded initial baseline weights from " << sidecarPath << std::endl;
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
        std::vector<const char*> runInputNames;
        std::vector<Ort::Value> runInputTensors;

        // 1. Bind audio input tensor
        int inCh = std::min((int)in_buffer.size(), m_inChannels);
        std::vector<int64_t> inShape = { 1, inCh, n_vec };

        float* audioInputPtr = in_buffer.empty() ? nullptr : in_buffer[0];
        if (audioInputPtr && !m_audioInputs.empty()) {
            runInputNames.push_back(m_audioInputs[0].name.c_str());
            runInputTensors.push_back(Ort::Value::CreateTensor<float>(
                m_impl->memoryInfo,
                audioInputPtr,
                inCh * n_vec,
                inShape.data(),
                inShape.size()
            ));
        }

        // 2. Bind parameter tensors (Only override initializers that have actually been modified/bent)
        for (auto& pair : m_parameters) {
            ParameterInfo& p = pair.second;
            // Check if parameter has been bent/modified away from original weights
            bool isModified = false;
            if (p.currentWeights.size() == p.originalWeights.size() && !p.currentWeights.empty()) {
                if (std::memcmp(p.currentWeights.data(), p.originalWeights.data(), p.currentWeights.size() * sizeof(float)) != 0) {
                    isModified = true;
                }
            }

            if (isModified) {
                runInputNames.push_back(pair.first.c_str());
                runInputTensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memoryInfo,
                    p.currentWeights.data(),
                    p.totalElements,
                    p.shape.data(),
                    p.shape.size()
                ));
            }
        }

        // 3. Run session with dynamic feed list
        auto outputTensors = m_impl->session->Run(
            Ort::RunOptions{nullptr},
            runInputNames.data(),
            runInputTensors.data(),
            runInputTensors.size(),
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
