#include "OnnxBackend.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>

// Include official ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif

struct OnnxBackend::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "NetBender_OnnxBackend"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Companion method sessions (e.g. encode, decode)
    struct CompanionSession {
        std::unique_ptr<Ort::Session> session;
        std::vector<std::string> input_names;
        std::vector<const char*> input_name_ptrs;
        std::vector<std::string> output_names;
        std::vector<const char*> output_name_ptrs;
        std::vector<StreamInfo> inputs;
        std::vector<StreamInfo> outputs;
    };
    std::unordered_map<std::string, CompanionSession> companions;

    // Dynamic graph metadata for primary session (forward)
    std::vector<std::string> input_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<std::string> output_names;
    std::vector<const char*> output_name_ptrs;

    // Preallocated tensor shape storage to avoid allocations on audio thread
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
};

OnnxBackend::OnnxBackend()
    : m_impl(std::make_unique<Impl>())
{
    // Optimize session for multi-core audio thread execution
    unsigned int hw_threads = std::thread::hardware_concurrency();
    int intra_threads = std::max(1, (int)std::min(hw_threads, 4u));
    m_impl->session_options.SetIntraOpNumThreads(intra_threads);
    m_impl->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

OnnxBackend::~OnnxBackend() = default;

auto OnnxBackend::load(std::string path, double sample_rate, const std::string* target_method) -> int {
    std::lock_guard<std::mutex> lock(m_session_mutex);
    m_loaded = false;
    m_path = path;
    m_sample_rate = sample_rate > 0.0 ? sample_rate : 44100.0;
    m_model_sample_rate = m_sample_rate;
    m_impl->companions.clear();

    try {
#if defined(_WIN32)
        std::wstring wpath(path.begin(), path.end());
        m_impl->session = std::make_unique<Ort::Session>(m_impl->env, wpath.c_str(), m_impl->session_options);
#else
        m_impl->session = std::make_unique<Ort::Session>(m_impl->env, path.c_str(), m_impl->session_options);
#endif
        introspect_graph();

        // Check for companion sessions in the same directory: _encode.onnx, _decode.onnx, _prior.onnx
        std::string base = path;
        size_t dot_pos = base.find_last_of('.');
        if (dot_pos != std::string::npos) base = base.substr(0, dot_pos);

        const std::vector<std::string> potential_methods = { "encode", "decode", "prior" };
        for (const auto& meth : potential_methods) {
            std::string companion_path = base + "_" + meth + ".onnx";
            std::ifstream test_file(companion_path);
            if (test_file.good()) {
                test_file.close();
                try {
                    Impl::CompanionSession cs;
#if defined(_WIN32)
                    std::wstring wcomp(companion_path.begin(), companion_path.end());
                    cs.session = std::make_unique<Ort::Session>(m_impl->env, wcomp.c_str(), m_impl->session_options);
#else
                    cs.session = std::make_unique<Ort::Session>(m_impl->env, companion_path.c_str(), m_impl->session_options);
#endif
                    size_t in_count = cs.session->GetInputCount();
                    for (size_t i = 0; i < in_count; ++i) {
                        auto in_name = cs.session->GetInputNameAllocated(i, m_impl->allocator);
                        cs.input_names.push_back(in_name.get());
                    }

                    size_t out_count = cs.session->GetOutputCount();
                    for (size_t i = 0; i < out_count; ++i) {
                        auto out_name = cs.session->GetOutputNameAllocated(i, m_impl->allocator);
                        cs.output_names.push_back(out_name.get());
                    }
                    m_impl->companions[meth] = std::move(cs);
                    std::cout << "[OnnxBackend] Loaded companion method '" << meth << "' from " << companion_path << std::endl;
                } catch (const std::exception& ce) {
                    std::cerr << "[OnnxBackend] Note: Could not load companion " << companion_path << ": " << ce.what() << std::endl;
                }
            }
        }

        // Re-anchor companion pointers across all loaded companions once map insertion is fully finished
        for (auto& pair : m_impl->companions) {
            auto& comp = pair.second;
            comp.input_name_ptrs.clear();
            for (const auto& n : comp.input_names) {
                comp.input_name_ptrs.push_back(n.c_str());
            }
            comp.output_name_ptrs.clear();
            for (const auto& n : comp.output_names) {
                comp.output_name_ptrs.push_back(n.c_str());
            }
        }
        std::cout << "[OnnxBackend] Finished companion loading." << std::endl;

        // Check for embedded model metadata properties (netbender.ratio, netbender.sample_rate, etc.)
        if (m_impl->session) {
            try {
                auto model_metadata = m_impl->session->GetModelMetadata();
                auto lookup_meta = [&](const char* key) -> std::string {
                    try {
                        auto val = model_metadata.LookupCustomMetadataMapAllocated(key, m_impl->allocator);
                        return val ? std::string(val.get()) : std::string();
                    } catch (...) {
                        return std::string();
                    }
                };

                try {
                    std::string meta_ratio = lookup_meta("netbender.ratio");
                    if (!meta_ratio.empty()) {
                        m_higher_ratio = std::max(1, std::stoi(meta_ratio));
                    }
                } catch (...) {}

                try {
                    std::string meta_sr = lookup_meta("netbender.sample_rate");
                    if (!meta_sr.empty()) {
                        double sr = std::stod(meta_sr);
                        if (sr > 0.0) {
                            m_model_sample_rate = sr;
                            m_sample_rate = sr;
                        }
                    }
                } catch (...) {}

                try {
                    std::string meta_latent = lookup_meta("netbender.latent_dim");
                    if (!meta_latent.empty()) {
                        m_latent_dim = std::max(1, std::stoi(meta_latent));
                    }
                } catch (...) {}

                try {
                    std::string meta_hist = lookup_meta("netbender.history_samples");
                    if (!meta_hist.empty()) {
                        m_history_samples = (size_t)std::max(0, std::stoi(meta_hist));
                    }
                } catch (...) {}

                try {
                    std::string meta_offset = lookup_meta("netbender.output_offset");
                    if (!meta_offset.empty()) {
                        m_output_offset = (size_t)std::max(0, std::stoi(meta_offset));
                    }
                } catch (...) {}
            } catch (...) {
                // Model metadata not present or unsupported
            }
        }

        // Introspect latent dimensions from companion models if not already established
        try {
            if (m_impl->companions.find("decode") != m_impl->companions.end()) {
                auto& dec = m_impl->companions["decode"];
                if (dec.session && dec.session->GetInputCount() > 0) {
                    try {
                        auto type_info = dec.session->GetInputTypeInfo(0);
                        auto t_info = type_info.GetTensorTypeAndShapeInfo();
                        if (t_info) {
                            auto shp = t_info.GetShape();
                            if (shp.size() >= 2 && shp[1] > 0) {
                                m_latent_dim = (int)shp[1];
                            }
                        }
                    } catch (...) {}
                }
            } else if (m_impl->companions.find("encode") != m_impl->companions.end()) {
                auto& enc = m_impl->companions["encode"];
                if (enc.session && enc.session->GetOutputCount() > 0) {
                    try {
                        auto type_info = enc.session->GetOutputTypeInfo(0);
                        auto t_info = type_info.GetTensorTypeAndShapeInfo();
                        if (t_info) {
                            auto shp = t_info.GetShape();
                            if (shp.size() >= 2 && shp[1] > 0) {
                                m_latent_dim = (int)shp[1];
                            }
                        }
                    } catch (...) {}
                }
            }
        } catch (...) {}

        std::cout << "[OnnxBackend] Latent Dimension: " << m_latent_dim
                  << " | Stride / Higher Ratio: " << m_higher_ratio 
                  << " | Native Sample Rate: " << m_sample_rate << std::endl;

        m_available_methods.clear();
        m_available_methods.push_back("forward");
        for (const auto& pair : m_impl->companions) {
            m_available_methods.push_back(pair.first);
        }

        m_audio_history.clear();
        
        // Backwards compatibility fallback if model does not contain explicit netbender.history_samples metadata
        if (m_history_samples == 0 && m_output_offset == 0) {
            bool is_rave = false;
            for (const auto& pair : m_parameters) {
                if (pair.first.find("rave") != std::string::npos || pair.first.find("_rave") != std::string::npos) {
                    is_rave = true;
                    break;
                }
            }
            if (is_rave && m_higher_ratio >= 1024) {
                m_history_samples = (m_latent_dim >= 8) ? 28672 : 4096;
                m_output_offset = m_history_samples;
            }
        } else if (m_output_offset == 0 && m_history_samples > 0) {
            m_output_offset = m_history_samples;
        }

        if (m_history_samples > 0) {
            m_audio_history.resize(std::max(1, m_in_channels), std::vector<float>(m_history_samples, 0.0f));
            std::cout << "[OnnxBackend] Enabled " << m_history_samples << "-sample streaming history buffer (output offset: " << m_output_offset << ")" << std::endl;
        } else {
            std::cout << "[OnnxBackend] Zero-delay direct streaming enabled (output offset: " << m_output_offset << ")" << std::endl;
        }

        m_loaded = true;
        clear_decode_gen_fifo();
        m_last_error.clear();
        std::cout << "[OnnxBackend] Finished load() successfully for " << path << std::endl;
        std::cout << std::flush;
        return 0;
    } catch (const std::exception& e) {
        m_last_error = e.what();
        std::cerr << "[OnnxBackend] Error loading model " << path << ": " << e.what() << std::endl;
        m_loaded = false;
        return -1;
    }
}

auto OnnxBackend::reload() -> int {
    if (m_path.empty()) return -1;
    return load(m_path, m_sample_rate);
}

auto OnnxBackend::is_loaded() -> bool {
    return m_loaded;
}

auto OnnxBackend::set_sample_rate(double sample_rate) -> void {
    if (sample_rate > 0.0) m_sample_rate = sample_rate;
}

auto OnnxBackend::get_sample_rate() -> double {
    return (m_model_sample_rate > 0.0) ? m_model_sample_rate : m_sample_rate;
}

auto OnnxBackend::use_gpu(bool value) -> void {
    if (m_use_gpu == value) return;
    m_use_gpu = value;

    m_impl->session_options = Ort::SessionOptions();
    unsigned int hw_threads = std::thread::hardware_concurrency();
    int intra_threads = std::max(1, (int)std::min(hw_threads, 4u));
    m_impl->session_options.SetIntraOpNumThreads(intra_threads);
    m_impl->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#if defined(__APPLE__)
    if (m_use_gpu) {
        uint32_t coreml_flags = COREML_FLAG_ENABLE_ON_SUBGRAPH;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(m_impl->session_options, coreml_flags);
        if (status != nullptr) {
            std::cerr << "[OnnxBackend] Warning: CoreML EP could not be enabled, falling back to CPU." << std::endl;
            Ort::GetApi().ReleaseStatus(status);
        } else {
            std::cout << "[OnnxBackend] CoreML Execution Provider enabled." << std::endl;
        }
    }
#endif

    if (m_loaded && !m_path.empty()) {
        reload();
    }
}

auto OnnxBackend::introspect_graph() -> void {
    if (!m_impl->session) return;

    m_available_layers.clear();
    m_parameters.clear();
    m_audio_inputs.clear();
    m_audio_outputs.clear();
    m_impl->input_names.clear();
    m_impl->input_name_ptrs.clear();
    m_impl->output_names.clear();
    m_impl->output_name_ptrs.clear();

    size_t num_inputs = m_impl->session->GetInputCount();
    size_t num_outputs = m_impl->session->GetOutputCount();
    size_t num_overridables = m_impl->session->GetOverridableInitializerCount();

    // Process primary required graph inputs (audio stream)
    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = m_impl->session->GetInputNameAllocated(i, m_impl->allocator);
        std::string name = input_name.get();
        m_impl->input_names.push_back(name);

        std::vector<int64_t> shape;
        try {
            auto type_info = m_impl->session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            if (tensor_info) {
                shape = tensor_info.GetShape();
            }
        } catch (...) {}

        bool is_audio_stream = (i == 0) || (shape.size() >= 3 && (shape.back() <= 0 || shape.back() > 2048));

        if (is_audio_stream && m_audio_inputs.empty()) {
            StreamInfo stream;
            stream.name = name;
            stream.input_index = i;
            stream.shape = shape;
            m_audio_inputs.push_back(stream);
            m_in_channels = (shape.size() >= 2 && shape[1] > 0) ? (int)shape[1] : 1;
        } else {
            // Free parameter input
            ParameterInfo param;
            param.name = name;
            param.shape = shape;
            param.input_index = i;

            size_t total = 1;
            for (auto d : shape) {
                total *= (d > 0 ? d : 1);
            }
            param.total_elements = total;
            param.original_weights.assign(total, 0.0f);
            param.current_weights.assign(total, 0.0f);

            m_parameters[name] = param;
            m_available_layers.push_back(name);
        }
    }

    // Process Overridable Initializers (Promoted layers with model baseline defaults)
    for (size_t i = 0; i < num_overridables; ++i) {
        auto init_name = m_impl->session->GetOverridableInitializerNameAllocated(i, m_impl->allocator);
        std::string name = init_name.get();

        if (m_parameters.find(name) == m_parameters.end()) {
            try {
                auto type_info = m_impl->session->GetOverridableInitializerTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                std::vector<int64_t> shape;
                if (tensor_info) {
                    shape = tensor_info.GetShape();
                }

                ParameterInfo param;
                param.name = name;
                param.shape = shape;
                param.input_index = m_impl->input_names.size();

                size_t total = 1;
                for (auto d : shape) {
                    total *= (d > 0 ? d : 1);
                }
                param.total_elements = total;
                param.original_weights.assign(total, 0.0f);
                param.current_weights.assign(total, 0.0f);

                m_parameters[name] = param;
                m_available_layers.push_back(name);
            } catch (...) {}
        }
    }

    // Load initial baseline weights from high-speed binary sidecar (.bin) if available
    std::string sidecar_path = m_path;
    size_t ext_pos = sidecar_path.find_last_of('.');
    if (ext_pos != std::string::npos) {
        sidecar_path = sidecar_path.substr(0, ext_pos) + ".bin";
    } else {
        sidecar_path += ".bin";
    }

    std::ifstream bin_file(sidecar_path, std::ios::binary);
    if (bin_file.is_open()) {
        char magic[4];
        bin_file.read(magic, 4);
        if (std::memcmp(magic, "NBND", 4) == 0) {
            uint32_t version = 0;
            bin_file.read(reinterpret_cast<char*>(&version), sizeof(version));
            uint32_t num_tensors = 0;
            bin_file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

            for (uint32_t t = 0; t < num_tensors; ++t) {
                uint32_t name_len = 0;
                bin_file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
                std::string t_name(name_len, '\0');
                bin_file.read(&t_name[0], name_len);

                uint32_t num_dims = 0;
                bin_file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
                std::vector<int64_t> t_shape(num_dims);
                for (uint32_t d = 0; d < num_dims; ++d) {
                    bin_file.read(reinterpret_cast<char*>(&t_shape[d]), sizeof(int64_t));
                }

                uint64_t num_elems = 0;
                bin_file.read(reinterpret_cast<char*>(&num_elems), sizeof(num_elems));

                std::vector<float> data(num_elems);
                bin_file.read(reinterpret_cast<char*>(data.data()), num_elems * sizeof(float));

                auto it = m_parameters.find(t_name);
                if (it != m_parameters.end()) {
                    it->second.original_weights = data;
                    it->second.current_weights = data;
                    it->second.total_elements = num_elems;
                    it->second.shape = t_shape;
                }
            }
            std::cout << "[OnnxBackend] Successfully loaded initial baseline weights from " << sidecar_path << std::endl;
        }
    }

    // Refresh pointers after vector is fully populated
    for (const auto& n : m_impl->input_names) {
        m_impl->input_name_ptrs.push_back(n.c_str());
    }

    // Outputs
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = m_impl->session->GetOutputNameAllocated(i, m_impl->allocator);
        std::string name = output_name.get();
        m_impl->output_names.push_back(name);

        std::vector<int64_t> shape;
        try {
            auto type_info = m_impl->session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            if (tensor_info) {
                shape = tensor_info.GetShape();
            }
        } catch (...) {}

        StreamInfo stream;
        stream.name = name;
        stream.input_index = i;
        stream.shape = shape;
        m_audio_outputs.push_back(stream);

        if (i == 0) {
            m_out_channels = (shape.size() >= 2 && shape[1] > 0) ? (int)shape[1] : 1;
        }
    }

    for (const auto& n : m_impl->output_names) {
        m_impl->output_name_ptrs.push_back(n.c_str());
    }

    std::cout << "[OnnxBackend] Loaded " << m_path << " | Audio In: " << m_in_channels 
              << " | Audio Out: " << m_out_channels << " | Bendable Layers: " 
              << m_available_layers.size() << std::endl;
}

auto OnnxBackend::perform(std::vector<float*>& in_buffer,
                          std::vector<float*>& out_buffer,
                          std::string method,
                          int n_batches, int n_out_channels, int n_vec) -> void {
    if (!m_loaded) return;
    std::unique_lock<std::mutex> lock(m_session_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;

    // Check if target method belongs to a companion session (e.g. encode, decode, prior)
    auto comp_it = m_impl->companions.find(method);
    if (comp_it != m_impl->companions.end()) {
        auto& comp = comp_it->second;
        if (!comp.session) return;
        try {
            std::vector<const char*> comp_in_names = comp.input_name_ptrs;
            std::vector<Ort::Value> comp_in_tensors;

            // Bind input stream
            int in_ch = std::min((int)in_buffer.size(), m_in_channels);
            if (method == "decode") in_ch = m_latent_dim;
            std::vector<int64_t> in_shape = { 1, in_ch, n_vec };

            float* in_ptr = in_buffer.empty() ? nullptr : in_buffer[0];
            if (in_ptr && !comp_in_names.empty()) {
                comp_in_tensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memory_info,
                    in_ptr,
                    in_ch * n_vec,
                    in_shape.data(),
                    in_shape.size()
                ));
            }

            auto comp_outputs = comp.session->Run(
                Ort::RunOptions{nullptr},
                comp_in_names.data(),
                comp_in_tensors.data(),
                comp_in_tensors.size(),
                comp.output_name_ptrs.data(),
                comp.output_name_ptrs.size()
            );

            if (!comp_outputs.empty()) {
                float* out_data = comp_outputs[0].GetTensorMutableData<float>();
                auto out_info = comp_outputs[0].GetTensorTypeAndShapeInfo();
                auto out_shape = out_info.GetShape();
                int out_samples = (out_shape.size() >= 3 && out_shape[2] > 0) ? (int)out_shape[2] : n_vec;
                int copy_samples = std::min(n_vec, out_samples);
                int out_ch = std::min((int)out_buffer.size(), (int)n_out_channels);

                for (int c = 0; c < out_ch; ++c) {
                    if (out_buffer[c]) {
                        std::memcpy(out_buffer[c], out_data + (c * out_samples), copy_samples * sizeof(float));
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
    if (m_audio_inputs.empty()) {
        if (has_autoencode()) {
            auto& enc = m_impl->companions["encode"];
            auto& dec = m_impl->companions["decode"];
            if (!enc.session || !dec.session) return;

            try {
                int in_ch = std::min((int)in_buffer.size(), m_in_channels);
                std::vector<int64_t> in_shape = { 1, in_ch, n_vec };
                float* in_ptr = in_buffer.empty() ? nullptr : in_buffer[0];
                if (!in_ptr || enc.input_name_ptrs.empty()) return;

                std::vector<Ort::Value> enc_in_tensors;
                enc_in_tensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memory_info,
                    in_ptr,
                    in_ch * n_vec,
                    in_shape.data(),
                    in_shape.size()
                ));

                auto enc_outputs = enc.session->Run(
                    Ort::RunOptions{nullptr},
                    enc.input_name_ptrs.data(),
                    enc_in_tensors.data(),
                    enc_in_tensors.size(),
                    enc.output_name_ptrs.data(),
                    enc.output_name_ptrs.size()
                );

                if (!enc_outputs.empty() && !dec.input_name_ptrs.empty()) {
                    auto dec_outputs = dec.session->Run(
                        Ort::RunOptions{nullptr},
                        dec.input_name_ptrs.data(),
                        enc_outputs.data(),
                        1,
                        dec.output_name_ptrs.data(),
                        dec.output_name_ptrs.size()
                    );

                    if (!dec_outputs.empty()) {
                        float* out_data = dec_outputs[0].GetTensorMutableData<float>();
                        auto out_info = dec_outputs[0].GetTensorTypeAndShapeInfo();
                        auto out_shape = out_info.GetShape();
                        int out_samples = (out_shape.size() >= 3 && out_shape[2] > 0) ? (int)out_shape[2] : n_vec;
                        int copy_samples = std::min(n_vec, out_samples);
                        int out_ch = std::min((int)out_buffer.size(), (int)n_out_channels);

                        for (int c = 0; c < out_ch; ++c) {
                            if (out_buffer[c]) {
                                std::memcpy(out_buffer[c], out_data + (c * out_samples), copy_samples * sizeof(float));
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
        std::vector<const char*> run_input_names;
        std::vector<Ort::Value> run_input_tensors;

        // 1. Bind audio input tensor (with causal receptive field history if required)
        int in_ch = std::min((int)in_buffer.size(), m_in_channels);
        float* audio_input_ptr = in_buffer.empty() ? nullptr : in_buffer[0];

        std::vector<float> composite_audio;
        int stream_len = n_vec;
        if (audio_input_ptr && !m_audio_inputs.empty()) {
            run_input_names.push_back(m_audio_inputs[0].name.c_str());

            if (m_history_samples > 0 || in_ch > 1) {
                stream_len = (int)m_history_samples + n_vec;
                composite_audio.resize(in_ch * stream_len);
                for (int c = 0; c < in_ch; ++c) {
                    float* hist_data = (m_history_samples > 0 && c < (int)m_audio_history.size()) ? m_audio_history[c].data() : nullptr;
                    float* in_data = (c < (int)in_buffer.size()) ? in_buffer[c] : nullptr;
                    float* dest = composite_audio.data() + (c * stream_len);
                    // Copy historical context
                    if (m_history_samples > 0) {
                        if (hist_data) {
                            std::memcpy(dest, hist_data, m_history_samples * sizeof(float));
                        } else {
                            std::memset(dest, 0, m_history_samples * sizeof(float));
                        }
                    }
                    // Copy current block
                    if (in_data) {
                        std::memcpy(dest + m_history_samples, in_data, n_vec * sizeof(float));
                    } else {
                        std::memset(dest + m_history_samples, 0, n_vec * sizeof(float));
                    }
                }
                std::vector<int64_t> in_shape = { 1, in_ch, (int64_t)stream_len };
                run_input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memory_info,
                    composite_audio.data(),
                    composite_audio.size(),
                    in_shape.data(),
                    in_shape.size()
                ));
            } else {
                std::vector<int64_t> in_shape = { 1, in_ch, n_vec };
                run_input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memory_info,
                    audio_input_ptr,
                    in_ch * n_vec,
                    in_shape.data(),
                    in_shape.size()
                ));
            }
        }

        // 2. Bind parameter tensors (Only override initializers that have actually been modified/bent)
        for (auto& pair : m_parameters) {
            ParameterInfo& p = pair.second;
            if (p.is_modified) {
                run_input_names.push_back(pair.first.c_str());
                run_input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memory_info,
                    p.current_weights.data(),
                    p.total_elements,
                    p.shape.data(),
                    p.shape.size()
                ));
            }
        }

        if (run_input_names.empty() || m_impl->output_name_ptrs.empty()) return;

        // 3. Run session with dynamic feed list
        auto output_tensors = m_impl->session->Run(
            Ort::RunOptions{nullptr},
            run_input_names.data(),
            run_input_tensors.data(),
            run_input_tensors.size(),
            m_impl->output_name_ptrs.data(),
            m_impl->output_name_ptrs.size()
        );

        // 4. Copy computed output back to out_buffer
        if (!output_tensors.empty()) {
            float* out_data = output_tensors[0].GetTensorMutableData<float>();
            auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto out_shape = out_info.GetShape();
            int out_samples = (out_shape.size() >= 3 && out_shape[2] > 0) ? (int)out_shape[2] : stream_len;
            int offset = (int)m_output_offset;
            if (offset >= out_samples) offset = 0;
            int copy_samples = std::min(n_vec, std::max(0, out_samples - offset));
            int out_ch = std::min((int)out_buffer.size(), m_out_channels);

            for (int c = 0; c < out_ch; ++c) {
                if (out_buffer[c]) {
                    if (copy_samples > 0) {
                        std::memcpy(out_buffer[c], out_data + (c * out_samples) + offset, copy_samples * sizeof(float));
                    }
                    if (copy_samples < n_vec) {
                        std::memset(out_buffer[c] + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
                    }
                }
            }

            // Update history buffer with current input audio for next block
            if (m_history_samples > 0) {
                for (int c = 0; c < in_ch && c < (int)m_audio_history.size(); ++c) {
                    if (in_buffer[c]) {
                        if (n_vec >= (int)m_history_samples) {
                            std::memcpy(m_audio_history[c].data(), in_buffer[c] + (n_vec - m_history_samples), m_history_samples * sizeof(float));
                        } else {
                            size_t shift = m_history_samples - n_vec;
                            std::memmove(m_audio_history[c].data(), m_audio_history[c].data() + n_vec, shift * sizeof(float));
                            std::memcpy(m_audio_history[c].data() + shift, in_buffer[c], n_vec * sizeof(float));
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[OnnxBackend::perform] Execution error: " << e.what() << std::endl;
    }
}

auto OnnxBackend::perform_forward(std::vector<float*>& in_buffer,
                                  std::vector<float*>& out_buffer,
                                  int n_batches, int n_out_channels, int n_vec,
                                  LatentHook latent_hook) -> void {
    if (latent_hook) {
        auto enc_it = m_impl->companions.find("encode");
        auto dec_it = m_impl->companions.find("decode");
        if (enc_it != m_impl->companions.end() && dec_it != m_impl->companions.end() &&
            enc_it->second.session && dec_it->second.session) {
            std::unique_lock<std::mutex> lock(m_session_mutex);
            auto& enc_comp = enc_it->second;
            auto& dec_comp = dec_it->second;

            try {
                int in_ch = std::min((int)in_buffer.size(), m_in_channels);
                if (in_ch <= 0 || n_vec <= 0) return;

                std::vector<float> audio_in(in_ch * n_vec);
                for (int c = 0; c < in_ch; ++c) {
                    if (in_buffer[c]) {
                        std::memcpy(audio_in.data() + (c * n_vec), in_buffer[c], n_vec * sizeof(float));
                    } else {
                        std::memset(audio_in.data() + (c * n_vec), 0, n_vec * sizeof(float));
                    }
                }
                std::vector<int64_t> in_shape = { 1, in_ch, (int64_t)n_vec };
                std::vector<Ort::Value> enc_in_tensors;
                enc_in_tensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memory_info,
                    audio_in.data(),
                    audio_in.size(),
                    in_shape.data(),
                    in_shape.size()
                ));

                auto enc_outputs = enc_comp.session->Run(
                    Ort::RunOptions{nullptr},
                    enc_comp.input_name_ptrs.data(),
                    enc_in_tensors.data(),
                    enc_in_tensors.size(),
                    enc_comp.output_name_ptrs.data(),
                    enc_comp.output_name_ptrs.size()
                );

                if (!enc_outputs.empty()) {
                    float* latent_ptr = enc_outputs[0].GetTensorMutableData<float>();
                    auto latent_info = enc_outputs[0].GetTensorTypeAndShapeInfo();
                    auto latent_shape = latent_info.GetShape();
                    int64_t lat_dim = (latent_shape.size() >= 2) ? latent_shape[1] : 1;
                    int64_t lat_steps = (latent_shape.size() >= 3) ? latent_shape[2] : 1;

                    LatentBuffer buf{ latent_ptr, lat_dim, lat_steps };
                    latent_hook(buf);

                    auto dec_outputs = dec_comp.session->Run(
                        Ort::RunOptions{nullptr},
                        dec_comp.input_name_ptrs.data(),
                        enc_outputs.data(),
                        1,
                        dec_comp.output_name_ptrs.data(),
                        dec_comp.output_name_ptrs.size()
                    );

                    if (!dec_outputs.empty()) {
                        float* out_data = dec_outputs[0].GetTensorMutableData<float>();
                        auto out_info = dec_outputs[0].GetTensorTypeAndShapeInfo();
                        auto out_shape = out_info.GetShape();
                        int num_channels = (out_shape.size() >= 2 && out_shape[1] > 0) ? (int)out_shape[1] : 1;
                        int out_samples = (out_shape.size() >= 3 && out_shape[2] > 0) ? (int)out_shape[2] : n_vec;
                        int copy_samples = std::min(n_vec, out_samples);
                        int out_ch = std::min((int)out_buffer.size(), std::min(num_channels, (int)n_out_channels));

                        for (int c = 0; c < out_ch; ++c) {
                            if (out_buffer[c]) {
                                std::memcpy(out_buffer[c], out_data + (c * out_samples), copy_samples * sizeof(float));
                                if (copy_samples < n_vec) {
                                    std::memset(out_buffer[c] + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
                                }
                            }
                        }
                        return;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[OnnxBackend::perform_forward] Latent hook encode/decode error: " << e.what() << std::endl;
            }
        }
    }

    perform(in_buffer, out_buffer, "forward", n_batches, n_out_channels, n_vec);
}

auto OnnxBackend::perform_prior_decode(std::vector<float*>& out_buffer,
                                       int n_batches, int n_out_channels, int n_vec) -> void {
    if (!m_loaded) return;
    std::unique_lock<std::mutex> lock(m_session_mutex);

    auto prior_it = m_impl->companions.find("prior");
    auto dec_it = m_impl->companions.find("decode");

    if (prior_it != m_impl->companions.end() && dec_it != m_impl->companions.end() &&
        prior_it->second.session && dec_it->second.session) {
        auto& prior_comp = prior_it->second;
        auto& dec_comp = dec_it->second;

        try {
            // 1. Run Prior companion session to generate latents z
            std::vector<Ort::Value> prior_in_tensors;
            std::vector<float> dummy_input;
            std::vector<int64_t> in_shape = { 1, 1, (int64_t)n_vec };
            if (!prior_comp.input_name_ptrs.empty()) {
                dummy_input.assign(n_vec, 0.0f);
                prior_in_tensors.push_back(Ort::Value::CreateTensor<float>(
                    m_impl->memory_info,
                    dummy_input.data(),
                    dummy_input.size(),
                    in_shape.data(),
                    in_shape.size()
                ));
            }

            auto prior_outputs = prior_comp.session->Run(
                Ort::RunOptions{nullptr},
                prior_comp.input_name_ptrs.data(),
                prior_in_tensors.data(),
                prior_in_tensors.size(),
                prior_comp.output_name_ptrs.data(),
                prior_comp.output_name_ptrs.size()
            );

            // 2. Feed generated latents z into Decode companion session
            if (!prior_outputs.empty() && !dec_comp.input_name_ptrs.empty()) {
                auto dec_outputs = dec_comp.session->Run(
                    Ort::RunOptions{nullptr},
                    dec_comp.input_name_ptrs.data(),
                    prior_outputs.data(),
                    1,
                    dec_comp.output_name_ptrs.data(),
                    dec_comp.output_name_ptrs.size()
                );

                if (!dec_outputs.empty()) {
                    float* out_data = dec_outputs[0].GetTensorMutableData<float>();
                    auto out_info = dec_outputs[0].GetTensorTypeAndShapeInfo();
                    auto out_shape = out_info.GetShape();
                    int out_samples = (out_shape.size() >= 3 && out_shape[2] > 0) ? (int)out_shape[2] : n_vec;
                    int copy_samples = std::min(n_vec, out_samples);
                    int out_ch = std::min((int)out_buffer.size(), (int)n_out_channels);

                    for (int c = 0; c < out_ch; ++c) {
                        if (out_buffer[c]) {
                            std::memcpy(out_buffer[c], out_data + (c * out_samples), copy_samples * sizeof(float));
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

auto OnnxBackend::perform_decode_gen(std::vector<float*>& out_buffer,
                                     int n_batches, int n_out_channels, int n_vec,
                                     LatentHook latent_hook) -> void {
    if (!m_loaded) return;

    auto comp_it = m_impl->companions.find("decode");
    if (comp_it == m_impl->companions.end() || !comp_it->second.session) return;
    auto& comp = comp_it->second;

    int ratio = std::max(1, m_higher_ratio);
    int out_ch = std::min((int)out_buffer.size(), (int)n_out_channels);

    // Check available samples in FIFO buffer
    size_t fifo_samples = (m_decode_gen_fifo.empty() || m_decode_gen_fifo[0].empty()) 
        ? 0 : (m_decode_gen_fifo[0].size() - m_decode_gen_fifo_read_idx);

    // If FIFO does not have enough samples to fulfill n_vec + crossfade cushion, synthesize another block
    const size_t kCrossfadeSamples = 256;
    if (fifo_samples < (size_t)(n_vec + kCrossfadeSamples)) {
        std::unique_lock<std::mutex> lock(m_session_mutex);

        // Compact FIFO by erasing already consumed samples
        if (m_decode_gen_fifo_read_idx > 0) {
            for (auto& ch : m_decode_gen_fifo) {
                if (m_decode_gen_fifo_read_idx < ch.size()) {
                    ch.erase(ch.begin(), ch.begin() + m_decode_gen_fifo_read_idx);
                } else {
                    ch.clear();
                }
            }
            m_decode_gen_fifo_read_idx = 0;
        }

        int min_frames = 8;
        if (m_higher_ratio >= 1024 && m_latent_dim >= 8 && m_history_samples > 4096) {
            min_frames = 16;
        }
        int latent_steps = std::max(min_frames, (n_vec + (int)kCrossfadeSamples) / ratio);
        if (m_use_gpu && ratio > 0) {
            // CoreML Execution Provider has a 16,384-sample dynamic sequence limit on 1D ConvTranspose layers
            int max_coreml_frames = std::max(1, 16384 / ratio);
            latent_steps = std::min(latent_steps, max_coreml_frames);
        }
        int latent_dim = std::max(1, m_latent_dim);

        float latent_scale = 1.0f;
        if (m_latent_dim >= 128 && m_higher_ratio <= 640) {
            latent_scale = 5.0f; // EnCodec SEANet calibrated (prevents tripping 2.0 fuse)
        } else if (m_higher_ratio >= 1024) {
            latent_scale = 1.0f; // VAEs (Stable Audio 64-dim, Speech RAVE 8/16-dim) standard unit normal latents
        }

        // Continuous Ornstein-Uhlenbeck smooth latent random walk
        const float alpha = (ratio <= 640) ? 0.94f : 0.86f;
        const float beta = std::sqrt(1.0f - alpha * alpha);

        static std::mt19937 rng_gen(1337);
        std::normal_distribution<float> norm_dist(0.0f, latent_scale);

        if (m_decode_gen_latent_state.size() != (size_t)latent_dim) {
            m_decode_gen_latent_state.resize(latent_dim);
            for (int c = 0; c < latent_dim; ++c) {
                m_decode_gen_latent_state[c] = norm_dist(rng_gen);
            }
        }

        std::vector<float> latent_data(latent_dim * latent_steps);
        for (int c = 0; c < latent_dim; ++c) {
            float prev = m_decode_gen_latent_state[c];
            float* ch_ptr = latent_data.data() + (c * latent_steps);
            for (int t = 0; t < latent_steps; ++t) {
                float eps = norm_dist(rng_gen);
                prev = alpha * prev + beta * eps;
                ch_ptr[t] = prev;
            }
            m_decode_gen_latent_state[c] = prev;
        }

        // Apply latent hook directly on contiguous [latent_dim, latent_steps] buffer
        if (latent_hook) {
            IBackend::LatentBuffer buf{ latent_data.data(), (int64_t)latent_dim, (int64_t)latent_steps };
            latent_hook(buf);
        }

        try {
            std::vector<int64_t> in_shape = { 1, (int64_t)latent_dim, (int64_t)latent_steps };
            std::vector<Ort::Value> comp_in_tensors;
            comp_in_tensors.push_back(Ort::Value::CreateTensor<float>(
                m_impl->memory_info,
                latent_data.data(),
                latent_data.size(),
                in_shape.data(),
                in_shape.size()
            ));

            auto comp_outputs = comp.session->Run(
                Ort::RunOptions{nullptr},
                comp.input_name_ptrs.data(),
                comp_in_tensors.data(),
                comp_in_tensors.size(),
                comp.output_name_ptrs.data(),
                comp.output_name_ptrs.size()
            );

            if (!comp_outputs.empty()) {
                float* out_data = comp_outputs[0].GetTensorMutableData<float>();
                auto out_info = comp_outputs[0].GetTensorTypeAndShapeInfo();
                auto out_shape = out_info.GetShape();
                int num_channels = (out_shape.size() >= 2) ? (int)out_shape[1] : 1;
                int out_samples = (out_shape.size() >= 3 && out_shape[2] > 0) ? (int)out_shape[2] : (latent_steps * ratio);

                if (m_decode_gen_fifo.size() < (size_t)num_channels) {
                    m_decode_gen_fifo.resize(num_channels);
                }

                // Smoothly crossfade seam with previous unconsumed block in FIFO
                size_t prev_available = m_decode_gen_fifo[0].size();
                size_t overlap = std::min(kCrossfadeSamples, std::min(prev_available, (size_t)out_samples / 2));

                for (int c = 0; c < num_channels; ++c) {
                    if (overlap > 0) {
                        size_t tail_idx = prev_available - overlap;
                        for (size_t i = 0; i < overlap; ++i) {
                            float w = 0.5f - 0.5f * std::cos(3.14159265f * (float)(i + 0.5f) / (float)overlap);
                            float old_s = m_decode_gen_fifo[c][tail_idx + i];
                            float new_s = out_data[c * out_samples + i];
                            m_decode_gen_fifo[c][tail_idx + i] = old_s * (1.0f - w) + new_s * w;
                        }
                        m_decode_gen_fifo[c].insert(m_decode_gen_fifo[c].end(),
                                                  out_data + (c * out_samples) + overlap,
                                                  out_data + (c * out_samples) + out_samples);
                    } else {
                        m_decode_gen_fifo[c].insert(m_decode_gen_fifo[c].end(),
                                                  out_data + (c * out_samples),
                                                  out_data + (c * out_samples) + out_samples);
                    }
                }
            }
        } catch (const std::exception& ce) {
            std::cerr << "[OnnxBackend::perform_decode_gen] Error: " << ce.what() << std::endl;
        }
    }

    // Transfer n_vec samples from FIFO to destination out_buffer
    int fifo_channels = (int)m_decode_gen_fifo.size();
    size_t available = (fifo_channels > 0) ? (m_decode_gen_fifo[0].size() - m_decode_gen_fifo_read_idx) : 0;
    int copy_samples = std::min((size_t)n_vec, available);

    for (int c = 0; c < out_ch; ++c) {
        if (out_buffer[c]) {
            if (c < fifo_channels && copy_samples > 0) {
                const float* src = m_decode_gen_fifo[c].data() + m_decode_gen_fifo_read_idx;
                std::memcpy(out_buffer[c], src, copy_samples * sizeof(float));
                if (copy_samples < n_vec) {
                    std::memset(out_buffer[c] + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
                }
            } else {
                std::memset(out_buffer[c], 0, n_vec * sizeof(float));
            }
        }
    }

    m_decode_gen_fifo_read_idx += copy_samples;
    if (fifo_channels > 0 && m_decode_gen_fifo_read_idx >= m_decode_gen_fifo[0].size()) {
        for (auto& ch : m_decode_gen_fifo) ch.clear();
        m_decode_gen_fifo_read_idx = 0;
    }
}

auto OnnxBackend::has_method(std::string method_name) -> bool {
    if (method_name == "forward") return (m_impl->session != nullptr && !m_audio_inputs.empty());
    return m_impl->companions.find(method_name) != m_impl->companions.end();
}

auto OnnxBackend::has_prior_decode() -> bool {
    return has_method("prior");
}

auto OnnxBackend::has_autoencode() -> bool {
    return has_method("encode") && has_method("decode");
}

auto OnnxBackend::get_plugin_modes() -> std::vector<std::string> {
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

auto OnnxBackend::get_mode_params(std::string mode) -> std::vector<int> {
    // [in_dim, in_ratio, out_dim, out_ratio]
    if (mode == "forward") {
        return { m_in_channels, 1, m_out_channels, 1 };
    } else if (mode == "decode gen" || mode == "prior") {
        return { 0, 1, m_out_channels, 1 };
    } else if (mode == "encode") {
        return { m_in_channels, 1, m_latent_dim, m_higher_ratio };
    } else if (mode == "decode") {
        return { m_latent_dim, m_higher_ratio, m_out_channels, 1 };
    }
    return { m_in_channels, 1, m_out_channels, 1 };
}

auto OnnxBackend::get_available_methods(LockedModel* model) -> std::vector<std::string> {
    return m_available_methods;
}

auto OnnxBackend::get_method_params(std::string method) -> std::vector<int> {
    return get_mode_params(method);
}

auto OnnxBackend::get_higher_ratio() -> int {
    return m_higher_ratio;
}

// Parameter Introspection & Bending
auto OnnxBackend::get_available_layers() -> std::vector<std::string> {
    return m_available_layers;
}

auto OnnxBackend::get_layer_weights(std::string layer_name) -> std::vector<float> {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        return it->second.current_weights;
    }
    return {};
}

auto OnnxBackend::get_original_layer_weights(std::string layer_name) -> std::vector<float> {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        return it->second.original_weights;
    }
    return {};
}

auto OnnxBackend::get_layer_shape(std::string layer_name) -> std::vector<int64_t> {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        return it->second.shape;
    }
    return {};
}

auto OnnxBackend::set_layer_weights(std::string layer_name, const std::vector<float>& weights) -> void {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        size_t count = std::min(it->second.current_weights.size(), weights.size());
        std::memcpy(it->second.current_weights.data(), weights.data(), count * sizeof(float));
        it->second.is_modified = true;
    }
}

auto OnnxBackend::reset_layer_weights(std::string layer_name) -> void {
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        it->second.current_weights = it->second.original_weights;
        it->second.is_modified = false;
    }
}

auto OnnxBackend::reset_all_layer_weights() -> void {
    for (auto& pair : m_parameters) {
        pair.second.current_weights = pair.second.original_weights;
        pair.second.is_modified = false;
    }
}

auto OnnxBackend::save_model(std::string path) -> int {
    return 0; // Exporting mutated ONNX will be handled in Phase 5
}

// Stubs for Max/MSP & PureData Attributes
auto OnnxBackend::get_available_attributes() -> std::vector<std::string> { return {}; }
auto OnnxBackend::get_settable_attributes() -> std::vector<std::string> { return {}; }
auto OnnxBackend::has_settable_attribute(std::string attribute) -> bool { return false; }
auto OnnxBackend::get_attribute_as_string(std::string attribute_name) -> std::string { return ""; }
auto OnnxBackend::set_attribute(std::string, std::vector<std::string>, const BufferMap&) -> void {}
auto OnnxBackend::get_buffer_attributes() -> std::vector<std::string> { return {}; }
auto OnnxBackend::is_buffer_element_of_attribute(std::string, int) -> bool { return false; }
auto OnnxBackend::is_tensor_element_of_attribute(std::string, int) -> bool { return false; }
auto OnnxBackend::get_buffer_name(std::string, int) -> std::string { return ""; }
auto OnnxBackend::update_buffer(std::string, StaticBuffer<float>&) -> int { return 0; }
auto OnnxBackend::reset_buffer(std::string) -> int { return 0; }
auto OnnxBackend::get_model_info() -> ModelInfo { return {}; }
