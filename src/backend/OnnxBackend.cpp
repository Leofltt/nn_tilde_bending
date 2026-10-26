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

    // Companion method sessions (e.g. encode, decode, prior)
    struct CompanionSession {
        std::unique_ptr<Ort::Session> session;
        std::vector<std::string> inputNames;
        std::vector<const char*> inputNamePtrs;
        std::vector<std::string> outputNames;
        std::vector<const char*> outputNamePtrs;
        std::vector<StreamInfo> inputs;
        std::vector<StreamInfo> outputs;
    };
    std::unordered_map<std::string, CompanionSession> companions;

    // Dynamic graph metadata for primary session (forward)
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
    m_impl->companions.clear();

    try {
#if defined(_WIN32)
        std::wstring wpath(path.begin(), path.end());
        m_impl->session = std::make_unique<Ort::Session>(m_impl->env, wpath.c_str(), m_impl->sessionOptions);
#else
        m_impl->session = std::make_unique<Ort::Session>(m_impl->env, path.c_str(), m_impl->sessionOptions);
#endif
        introspectGraph();

        // Check for companion sessions in the same directory: _encode.onnx, _decode.onnx, _prior.onnx
        std::string base = path;
        size_t dotPos = base.find_last_of('.');
        if (dotPos != std::string::npos) base = base.substr(0, dotPos);

        const std::vector<std::string> potentialMethods = { "encode", "decode", "prior" };
        for (const auto& meth : potentialMethods) {
            std::string companionPath = base + "_" + meth + ".onnx";
            std::ifstream testFile(companionPath);
            if (testFile.good()) {
                testFile.close();
                try {
                    Impl::CompanionSession cs;
#if defined(_WIN32)
                    std::wstring wcomp(companionPath.begin(), companionPath.end());
                    cs.session = std::make_unique<Ort::Session>(m_impl->env, wcomp.c_str(), m_impl->sessionOptions);
#else
                    cs.session = std::make_unique<Ort::Session>(m_impl->env, companionPath.c_str(), m_impl->sessionOptions);
#endif
                    size_t inCount = cs.session->GetInputCount();
                    for (size_t i = 0; i < inCount; ++i) {
                        auto inName = cs.session->GetInputNameAllocated(i, m_impl->allocator);
                        cs.inputNames.push_back(inName.get());
                    }

                    size_t outCount = cs.session->GetOutputCount();
                    for (size_t i = 0; i < outCount; ++i) {
                        auto outName = cs.session->GetOutputNameAllocated(i, m_impl->allocator);
                        cs.outputNames.push_back(outName.get());
                    }
                    m_impl->companions[meth] = std::move(cs);
                    std::cout << "[OnnxBackend] Loaded companion method '" << meth << "' from " << companionPath << std::endl;
                } catch (const std::exception& ce) {
                    std::cerr << "[OnnxBackend] Note: Could not load companion " << companionPath << ": " << ce.what() << std::endl;
                }
            }
        }

        // Re-anchor companion pointers across all loaded companions once map insertion is fully finished
        for (auto& pair : m_impl->companions) {
            auto& comp = pair.second;
            comp.inputNamePtrs.clear();
            for (const auto& n : comp.inputNames) {
                comp.inputNamePtrs.push_back(n.c_str());
            }
            comp.outputNamePtrs.clear();
            for (const auto& n : comp.outputNames) {
                comp.outputNamePtrs.push_back(n.c_str());
            }
        }
        std::cout << "[OnnxBackend] Finished companion loading." << std::endl;

        // 1. Check for embedded model metadata properties (netbender.ratio, netbender.sample_rate, etc.)
        if (m_impl->session) {
            try {
                auto modelMetadata = m_impl->session->GetModelMetadata();
                auto lookupMeta = [&](const char* key) -> std::string {
                    try {
                        auto val = modelMetadata.LookupCustomMetadataMapAllocated(key, m_impl->allocator);
                        return val ? std::string(val.get()) : std::string();
                    } catch (...) {
                        return std::string();
                    }
                };

                try {
                    std::string metaRatio = lookupMeta("netbender.ratio");
                    if (!metaRatio.empty()) {
                        m_higherRatio = std::max(1, std::stoi(metaRatio));
                    }
                } catch (...) {}

                try {
                    std::string metaSr = lookupMeta("netbender.sample_rate");
                    if (!metaSr.empty()) {
                        double sr = std::stod(metaSr);
                        if (sr > 0.0) m_sampleRate = sr;
                    }
                } catch (...) {}

                try {
                    std::string metaLatent = lookupMeta("netbender.latent_dim");
                    if (!metaLatent.empty()) {
                        m_latentDim = std::max(1, std::stoi(metaLatent));
                    }
                } catch (...) {}
            } catch (...) {
                // Model metadata not present or unsupported
            }
        }

        // 2. Introspect latent dimensions from companion models if not already established
        try {
            if (m_impl->companions.find("decode") != m_impl->companions.end()) {
                auto& dec = m_impl->companions["decode"];
                if (dec.session && dec.session->GetInputCount() > 0) {
                    try {
                        auto typeInfo = dec.session->GetInputTypeInfo(0);
                        auto tInfo = typeInfo.GetTensorTypeAndShapeInfo();
                        if (tInfo) {
                            auto shp = tInfo.GetShape();
                            if (shp.size() >= 2 && shp[1] > 0) {
                                m_latentDim = (int)shp[1];
                            }
                        }
                    } catch (...) {}
                }
            } else if (m_impl->companions.find("encode") != m_impl->companions.end()) {
                auto& enc = m_impl->companions["encode"];
                if (enc.session && enc.session->GetOutputCount() > 0) {
                    try {
                        auto typeInfo = enc.session->GetOutputTypeInfo(0);
                        auto tInfo = typeInfo.GetTensorTypeAndShapeInfo();
                        if (tInfo) {
                            auto shp = tInfo.GetShape();
                            if (shp.size() >= 2 && shp[1] > 0) {
                                m_latentDim = (int)shp[1];
                            }
                        }
                    } catch (...) {}
                }
            }
        } catch (...) {}

        std::cout << "[OnnxBackend] Latent Dimension: " << m_latentDim
                  << " | Stride / Higher Ratio: " << m_higherRatio 
                  << " | Native Sample Rate: " << m_sampleRate << std::endl;

        m_availableMethods.clear();
        m_availableMethods.push_back("forward");
        for (const auto& pair : m_impl->companions) {
            m_availableMethods.push_back(pair.first);
        }

        // Initialize history buffer for causal receptive field models (e.g. RAVE with ratio 2048)
        m_audioHistory.clear();
        m_historySamples = (m_higherRatio >= 1024) ? 4096 : 0;
        if (m_historySamples > 0) {
            m_audioHistory.resize(std::max(1, m_inChannels), std::vector<float>(m_historySamples, 0.0f));
            std::cout << "[OnnxBackend] Enabled " << m_historySamples << "-sample streaming history buffer for causal receptive field" << std::endl;
        }

        m_loaded = true;
        m_lastError.clear();
        std::cout << "[OnnxBackend] Finished load() successfully for " << path << std::endl;
        std::cout << std::flush;
        return 0;
    } catch (const std::exception& e) {
        m_lastError = e.what();
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

        std::vector<int64_t> shape;
        try {
            auto typeInfo = m_impl->session->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            if (tensorInfo) {
                shape = tensorInfo.GetShape();
            }
        } catch (...) {}

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
            try {
                auto typeInfo = m_impl->session->GetOverridableInitializerTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                std::vector<int64_t> shape;
                if (tensorInfo) {
                    shape = tensorInfo.GetShape();
                }

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
            } catch (...) {}
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

        std::vector<int64_t> shape;
        try {
            auto typeInfo = m_impl->session->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            if (tensorInfo) {
                shape = tensorInfo.GetShape();
            }
        } catch (...) {}

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
    if (!m_loaded) return;
    std::unique_lock<std::mutex> lock(m_sessionMutex, std::try_to_lock);
    if (!lock.owns_lock()) return;

    // Check if target method belongs to a companion session (e.g. encode, decode, prior)
    auto compIt = m_impl->companions.find(method);
    if (compIt != m_impl->companions.end()) {
        auto& comp = compIt->second;
        if (!comp.session) return;
        try {
            std::vector<const char*> compInNames = comp.inputNamePtrs;
            std::vector<Ort::Value> compInTensors;

            // Bind input stream
            int inCh = std::min((int)in_buffer.size(), m_inChannels);
            if (method == "decode") inCh = m_latentDim;
            std::vector<int64_t> inShape = { 1, inCh, n_vec };

            float* inPtr = in_buffer.empty() ? nullptr : in_buffer[0];
            if (inPtr && !compInNames.empty()) {
                compInTensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memoryInfo,
                    inPtr,
                    inCh * n_vec,
                    inShape.data(),
                    inShape.size()
                ));
            }

            auto compOutputs = comp.session->Run(
                Ort::RunOptions{nullptr},
                compInNames.data(),
                compInTensors.data(),
                compInTensors.size(),
                comp.outputNamePtrs.data(),
                comp.outputNamePtrs.size()
            );

            if (!compOutputs.empty()) {
                float* outData = compOutputs[0].GetTensorMutableData<float>();
                auto outInfo = compOutputs[0].GetTensorTypeAndShapeInfo();
                auto outShape = outInfo.GetShape();
                int outSamples = (outShape.size() >= 3 && outShape[2] > 0) ? (int)outShape[2] : n_vec;
                int copy_samples = std::min(n_vec, outSamples);
                int outCh = std::min((int)out_buffer.size(), (int)n_out_channels);

                for (int c = 0; c < outCh; ++c) {
                    if (out_buffer[c]) {
                        std::memcpy(out_buffer[c], outData + (c * outSamples), copy_samples * sizeof(float));
                        if (copy_samples < n_vec) {
                            std::memset(out_buffer[c] + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
                        }
                    }
                }
            }
        } catch (const std::exception& ce) {
            std::cerr << "[OnnxBackend::perform:" << method << "] Error: " << ce.what() << std::endl;
        }
        return;
    }

    // If primary session has no audio inputs but companion encode/decode exist, route through autoencode
    if (m_audioInputs.empty()) {
        if (has_autoencode()) {
            auto& enc = m_impl->companions["encode"];
            auto& dec = m_impl->companions["decode"];
            if (!enc.session || !dec.session) return;

            try {
                int inCh = std::min((int)in_buffer.size(), m_inChannels);
                std::vector<int64_t> inShape = { 1, inCh, n_vec };
                float* inPtr = in_buffer.empty() ? nullptr : in_buffer[0];
                if (!inPtr || enc.inputNamePtrs.empty()) return;

                std::vector<Ort::Value> encInTensors;
                encInTensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memoryInfo,
                    inPtr,
                    inCh * n_vec,
                    inShape.data(),
                    inShape.size()
                ));

                auto encOutputs = enc.session->Run(
                    Ort::RunOptions{nullptr},
                    enc.inputNamePtrs.data(),
                    encInTensors.data(),
                    encInTensors.size(),
                    enc.outputNamePtrs.data(),
                    enc.outputNamePtrs.size()
                );

                if (!encOutputs.empty() && !dec.inputNamePtrs.empty()) {
                    auto decOutputs = dec.session->Run(
                        Ort::RunOptions{nullptr},
                        dec.inputNamePtrs.data(),
                        encOutputs.data(),
                        1,
                        dec.outputNamePtrs.data(),
                        dec.outputNamePtrs.size()
                    );

                    if (!decOutputs.empty()) {
                        float* outData = decOutputs[0].GetTensorMutableData<float>();
                        auto outInfo = decOutputs[0].GetTensorTypeAndShapeInfo();
                        auto outShape = outInfo.GetShape();
                        int outSamples = (outShape.size() >= 3 && outShape[2] > 0) ? (int)outShape[2] : n_vec;
                        int copy_samples = std::min(n_vec, outSamples);
                        int outCh = std::min((int)out_buffer.size(), (int)n_out_channels);

                        for (int c = 0; c < outCh; ++c) {
                            if (out_buffer[c]) {
                                std::memcpy(out_buffer[c], outData + (c * outSamples), copy_samples * sizeof(float));
                                if (copy_samples < n_vec) {
                                    std::memset(out_buffer[c] + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
                                }
                            }
                        }
                    }
                }
            } catch (const std::exception& ae) {
                std::cerr << "[OnnxBackend::perform:autoencode] Error: " << ae.what() << std::endl;
            }
            return;
        }
        return;
    }

    if (!m_impl->session) return;

    try {
        std::vector<const char*> runInputNames;
        std::vector<Ort::Value> runInputTensors;

        // 1. Bind audio input tensor (with causal receptive field history if required)
        int inCh = std::min((int)in_buffer.size(), m_inChannels);
        float* audioInputPtr = in_buffer.empty() ? nullptr : in_buffer[0];

        std::vector<float> compositeAudio;
        int streamLen = n_vec;
        if (audioInputPtr && !m_audioInputs.empty()) {
            runInputNames.push_back(m_audioInputs[0].name.c_str());

            if (m_historySamples > 0) {
                streamLen = (int)m_historySamples + n_vec;
                compositeAudio.resize(inCh * streamLen);
                for (int c = 0; c < inCh; ++c) {
                    float* histData = (c < (int)m_audioHistory.size()) ? m_audioHistory[c].data() : nullptr;
                    float* inData = in_buffer[c];
                    float* dest = compositeAudio.data() + (c * streamLen);
                    // Copy historical context
                    if (histData) {
                        std::memcpy(dest, histData, m_historySamples * sizeof(float));
                    } else {
                        std::memset(dest, 0, m_historySamples * sizeof(float));
                    }
                    // Copy current block
                    if (inData) {
                        std::memcpy(dest + m_historySamples, inData, n_vec * sizeof(float));
                    } else {
                        std::memset(dest + m_historySamples, 0, n_vec * sizeof(float));
                    }
                }
                std::vector<int64_t> inShape = { 1, inCh, (int64_t)streamLen };
                runInputTensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memoryInfo,
                    compositeAudio.data(),
                    compositeAudio.size(),
                    inShape.data(),
                    inShape.size()
                ));
            } else {
                std::vector<int64_t> inShape = { 1, inCh, n_vec };
                runInputTensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memoryInfo,
                    audioInputPtr,
                    inCh * n_vec,
                    inShape.data(),
                    inShape.size()
                ));
            }
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

        if (runInputNames.empty() || m_impl->outputNamePtrs.empty()) return;

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
            auto outInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
            auto outShape = outInfo.GetShape();
            int outSamples = (outShape.size() >= 3 && outShape[2] > 0) ? (int)outShape[2] : streamLen;
            int copy_samples = std::min(n_vec, outSamples);
            int outCh = std::min((int)out_buffer.size(), m_outChannels);

            // Receptive field compensation: extract the latest n_vec samples from the synthesized stream
            int offset = (outSamples > copy_samples) ? (outSamples - copy_samples) : 0;

            for (int c = 0; c < outCh; ++c) {
                if (out_buffer[c]) {
                    std::memcpy(out_buffer[c], outData + (c * outSamples) + offset, copy_samples * sizeof(float));
                    if (copy_samples < n_vec) {
                        std::memset(out_buffer[c] + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
                    }
                }
            }

            // Update history buffer with current input audio for next block
            if (m_historySamples > 0) {
                for (int c = 0; c < inCh && c < (int)m_audioHistory.size(); ++c) {
                    if (in_buffer[c]) {
                        if (n_vec >= (int)m_historySamples) {
                            std::memcpy(m_audioHistory[c].data(), in_buffer[c] + (n_vec - m_historySamples), m_historySamples * sizeof(float));
                        } else {
                            size_t shift = m_historySamples - n_vec;
                            std::memmove(m_audioHistory[c].data(), m_audioHistory[c].data() + n_vec, shift * sizeof(float));
                            std::memcpy(m_audioHistory[c].data() + shift, in_buffer[c], n_vec * sizeof(float));
                        }
                    }
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
    if (!m_loaded) return;
    std::unique_lock<std::mutex> lock(m_sessionMutex, std::try_to_lock);
    if (!lock.owns_lock()) return;

    auto priorIt = m_impl->companions.find("prior");
    auto decIt = m_impl->companions.find("decode");

    if (priorIt != m_impl->companions.end() && decIt != m_impl->companions.end() &&
        priorIt->second.session && decIt->second.session) {
        auto& priorComp = priorIt->second;
        auto& decComp = decIt->second;

        try {
            // 1. Run Prior companion session to generate latents z
            std::vector<Ort::Value> priorInTensors;
            std::vector<float> dummyInput;
            std::vector<int64_t> inShape = { 1, 1, (int64_t)n_vec };
            if (!priorComp.inputNamePtrs.empty()) {
                dummyInput.assign(n_vec, 0.0f);
                priorInTensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memoryInfo,
                    dummyInput.data(),
                    dummyInput.size(),
                    inShape.data(),
                    inShape.size()
                ));
            }

            auto priorOutputs = priorComp.session->Run(
                Ort::RunOptions{nullptr},
                priorComp.inputNamePtrs.data(),
                priorInTensors.data(),
                priorInTensors.size(),
                priorComp.outputNamePtrs.data(),
                priorComp.outputNamePtrs.size()
            );

            // 2. Feed generated latents z into Decode companion session
            if (!priorOutputs.empty() && !decComp.inputNamePtrs.empty()) {
                auto decOutputs = decComp.session->Run(
                    Ort::RunOptions{nullptr},
                    decComp.inputNamePtrs.data(),
                    priorOutputs.data(),
                    1,
                    decComp.outputNamePtrs.data(),
                    decComp.outputNamePtrs.size()
                );

                if (!decOutputs.empty()) {
                    float* outData = decOutputs[0].GetTensorMutableData<float>();
                    auto outInfo = decOutputs[0].GetTensorTypeAndShapeInfo();
                    auto outShape = outInfo.GetShape();
                    int outSamples = (outShape.size() >= 3 && outShape[2] > 0) ? (int)outShape[2] : n_vec;
                    int copy_samples = std::min(n_vec, outSamples);
                    int outCh = std::min((int)out_buffer.size(), (int)n_out_channels);

                    for (int c = 0; c < outCh; ++c) {
                        if (out_buffer[c]) {
                            std::memcpy(out_buffer[c], outData + (c * outSamples), copy_samples * sizeof(float));
                            if (copy_samples < n_vec) {
                                std::memset(out_buffer[c] + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
                            }
                        }
                    }
                    return;
                }
            }
        } catch (const std::exception& pe) {
            std::cerr << "[OnnxBackend::perform_prior_decode] Prior error: " << pe.what() << ", falling back to autonomous decode gen" << std::endl;
            if (lock.owns_lock()) lock.unlock();
            perform_decode_gen(out_buffer, n_batches, n_out_channels, n_vec);
            return;
        }
    }

    // Fallback: autonomous decode gen
    if (lock.owns_lock()) lock.unlock();
    perform_decode_gen(out_buffer, n_batches, n_out_channels, n_vec);
}

void OnnxBackend::perform_decode_gen(std::vector<float*>& out_buffer,
                                     int n_batches, int n_out_channels, int n_vec,
                                     LatentHook latent_hook) {
    if (!m_loaded) return;
    std::unique_lock<std::mutex> lock(m_sessionMutex, std::try_to_lock);
    if (!lock.owns_lock()) return;

    auto compIt = m_impl->companions.find("decode");
    if (compIt == m_impl->companions.end() || !compIt->second.session) return;
    auto& comp = compIt->second;

    int ratio = std::max(1, m_higherRatio);
    // Enforce minimum frames for codecs with wide receptive field kernels (e.g. EnCodec reflect pad 6, RAVE causal delay)
    int min_frames = (m_higherRatio <= 512 || m_latentDim >= 64) ? 8 : 4;
    int latent_steps = std::max(min_frames, n_vec / ratio);
    int latent_dim = std::max(1, m_latentDim);

    // Determine nominal latent standard deviation based on codec architecture
    // EnCodec SEANet operates at sigma ~ 8.5 to 10.0; RAVE operates at sigma ~ 1.0
    float latent_scale = (m_latentDim >= 64 || m_higherRatio <= 512) ? 8.5f : 1.0f;

    // Synthesize autonomous Gaussian latent excitation
    static uint32_t rng_state = 1337;
    std::vector<float> latentData(latent_dim * latent_steps);
    for (size_t i = 0; i < latentData.size(); ++i) {
        // Uniform to Gaussian via Box-Muller approximation
        rng_state = rng_state * 1664525u + 1013904223u;
        float u1 = std::max(1e-7f, (float)(rng_state & 0x00FFFFFF) / 16777216.0f);
        rng_state = rng_state * 1664525u + 1013904223u;
        float u2 = (float)(rng_state & 0x00FFFFFF) / 16777216.0f;
        float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
        latentData[i] = z * latent_scale;
    }

    if (latent_hook) {
        IBackend::LatentBuffer buf{ latentData.data(), (int64_t)latent_dim, (int64_t)latent_steps };
        latent_hook(buf);
    }

    try {
        std::vector<int64_t> inShape = { 1, (int64_t)latent_dim, (int64_t)latent_steps };
        std::vector<Ort::Value> compInTensors;
        compInTensors.push_back(Ort::Value::CreateTensor<float>(
            m_impl->memoryInfo,
            latentData.data(),
            latentData.size(),
            inShape.data(),
            inShape.size()
        ));

        auto compOutputs = comp.session->Run(
            Ort::RunOptions{nullptr},
            comp.inputNamePtrs.data(),
            compInTensors.data(),
            compInTensors.size(),
            comp.outputNamePtrs.data(),
            comp.outputNamePtrs.size()
        );

        if (!compOutputs.empty()) {
            float* outData = compOutputs[0].GetTensorMutableData<float>();
            auto outInfo = compOutputs[0].GetTensorTypeAndShapeInfo();
            auto outShape = outInfo.GetShape();
            int outSamples = (outShape.size() >= 3 && outShape[2] > 0) ? (int)outShape[2] : n_vec;
            int copy_samples = std::min(n_vec, outSamples);
            int outCh = std::min((int)out_buffer.size(), (int)n_out_channels);

            // Extract from the warm tail of the synthesized stream past the initial receptive field latency
            int offset = (outSamples > copy_samples) ? (outSamples - copy_samples) : 0;

            for (int c = 0; c < outCh; ++c) {
                if (out_buffer[c]) {
                    std::memcpy(out_buffer[c], outData + (c * outSamples) + offset, copy_samples * sizeof(float));
                    if (copy_samples < n_vec) {
                        std::memset(out_buffer[c] + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
                    }
                }
            }
        }
    } catch (const std::exception& ce) {
        std::cerr << "[OnnxBackend::perform_decode_gen] Error: " << ce.what() << std::endl;
    }
}

bool OnnxBackend::has_method(std::string method_name) {
    if (method_name == "forward") return (m_impl->session != nullptr && !m_audioInputs.empty());
    return m_impl->companions.find(method_name) != m_impl->companions.end();
}

bool OnnxBackend::has_prior_decode() {
    return has_method("prior");
}

bool OnnxBackend::has_autoencode() {
    return has_method("encode") && has_method("decode");
}

std::vector<std::string> OnnxBackend::get_plugin_modes() {
    std::vector<std::string> modes;
    if (!is_loaded()) return modes;

    if (has_method("forward") || has_autoencode()) {
        modes.push_back("forward");
    }
    if (has_method("decode")) {
        modes.push_back("decode gen");
    }
    if (has_method("prior")) {
        modes.push_back("prior");
    }
    return modes;
}

std::vector<int> OnnxBackend::get_mode_params(std::string mode) {
    // [in_dim, in_ratio, out_dim, out_ratio]
    if (mode == "forward") {
        return { m_inChannels, 1, m_outChannels, 1 };
    } else if (mode == "decode gen" || mode == "prior") {
        return { 0, 1, m_outChannels, 1 };
    } else if (mode == "encode") {
        return { m_inChannels, 1, m_latentDim, m_higherRatio };
    } else if (mode == "decode") {
        return { m_latentDim, m_higherRatio, m_outChannels, 1 };
    }
    return { m_inChannels, 1, m_outChannels, 1 };
}

std::vector<std::string> OnnxBackend::get_available_methods(LockedModel* model) {
    return m_availableMethods;
}

std::vector<int> OnnxBackend::get_method_params(std::string method) {
    return get_mode_params(method);
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
