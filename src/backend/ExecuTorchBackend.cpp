#include "ExecuTorchBackend.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <random>

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#endif

struct ExecuTorchBackend::Impl {
#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    std::unique_ptr<executorch::extension::Module> module;
    std::unordered_map<std::string, std::unique_ptr<executorch::extension::Module>> companions;
#endif
};

ExecuTorchBackend::ExecuTorchBackend()
    : m_impl(std::make_unique<Impl>())
{
}

ExecuTorchBackend::~ExecuTorchBackend() = default;

auto ExecuTorchBackend::get_backend_name() const -> std::string {
    std::string del = m_delegate;
    std::transform(del.begin(), del.end(), del.begin(), ::toupper);
    if (del.empty() || del == "PORTABLE") {
        return "ExecuTorch";
    }
    return "ExecuTorch (" + del + ")";
}

auto ExecuTorchBackend::load_metadata_and_sidecar(const std::string& base_path) -> void {
    m_available_layers.clear();
    m_parameters.clear();
    m_method_param_inputs.clear();
    m_available_methods.clear();

    // 1. Read optional JSON metadata sidecar (<base>.json) exported by pte_bend_prep / migrate_ts_to_pte
    std::string json_path = base_path + ".json";
    std::ifstream json_file(json_path);
    if (json_file.is_open()) {
        try {
            nlohmann::json meta;
            json_file >> meta;

            if (meta.contains("sample_rate") && meta["sample_rate"].is_number()) {
                double sr = meta["sample_rate"].get<double>();
                if (sr > 0.0) {
                    m_model_sample_rate = sr;
                    m_sample_rate = sr;
                }
            } else if (meta.contains("sr") && meta["sr"].is_number()) {
                double sr = meta["sr"].get<double>();
                if (sr > 0.0) {
                    m_model_sample_rate = sr;
                    m_sample_rate = sr;
                }
            }
            if (meta.contains("ratio") && meta["ratio"].is_number_integer()) {
                m_higher_ratio = std::max(1, meta["ratio"].get<int>());
            } else if (meta.contains("higher_ratio") && meta["higher_ratio"].is_number_integer()) {
                m_higher_ratio = std::max(1, meta["higher_ratio"].get<int>());
            }
            if (meta.contains("latent_dim") && meta["latent_dim"].is_number_integer()) {
                m_latent_dim = std::max(1, meta["latent_dim"].get<int>());
            } else if (meta.contains("encode_out_dim") && meta["encode_out_dim"].is_number_integer()) {
                m_latent_dim = std::max(1, meta["encode_out_dim"].get<int>());
            } else if (meta.contains("decode_in_dim") && meta["decode_in_dim"].is_number_integer()) {
                m_latent_dim = std::max(1, meta["decode_in_dim"].get<int>());
            }
            if (meta.contains("in_channels") && meta["in_channels"].is_number_integer()) {
                m_in_channels = std::max(1, meta["in_channels"].get<int>());
            } else if (meta.contains("in_dim") && meta["in_dim"].is_number_integer()) {
                m_in_channels = std::max(1, meta["in_dim"].get<int>());
            }
            if (meta.contains("out_channels") && meta["out_channels"].is_number_integer()) {
                m_out_channels = std::max(1, meta["out_channels"].get<int>());
            } else if (meta.contains("out_dim") && meta["out_dim"].is_number_integer()) {
                m_out_channels = std::max(1, meta["out_dim"].get<int>());
            }
            if (meta.contains("history_samples") && meta["history_samples"].is_number_integer()) {
                m_history_samples = (size_t)std::max(0, meta["history_samples"].get<int>());
            }
            if (meta.contains("output_offset") && meta["output_offset"].is_number_integer()) {
                m_output_offset = (size_t)std::max(0, meta["output_offset"].get<int>());
            }
            if (meta.contains("buffer_size") && meta["buffer_size"].is_number_integer()) {
                m_buffer_size = std::max(1, meta["buffer_size"].get<int>());
            }
            if (meta.contains("decode_latent_steps") && meta["decode_latent_steps"].is_number_integer()) {
                m_decode_latent_steps = std::max(1, meta["decode_latent_steps"].get<int>());
            } else {
                int ratio = std::max(1, m_higher_ratio);
                m_decode_latent_steps = std::max(1, ((int)m_history_samples + m_buffer_size + ratio - 1) / ratio);
            }
            if (meta.contains("delegate") && meta["delegate"].is_string()) {
                m_delegate = meta["delegate"].get<std::string>();
            } else if (meta.contains("netbender.delegate") && meta["netbender.delegate"].is_string()) {
                m_delegate = meta["netbender.delegate"].get<std::string>();
            }
            if (meta.contains("methods") && meta["methods"].is_array()) {
                for (const auto& m : meta["methods"]) {
                    if (m.is_string()) {
                        m_available_methods.push_back(m.get<std::string>());
                    }
                }
            }
            if (meta.contains("method_param_inputs") && meta["method_param_inputs"].is_object()) {
                for (auto it = meta["method_param_inputs"].begin(); it != meta["method_param_inputs"].end(); ++it) {
                    if (it.value().is_array()) {
                        std::vector<std::string> names;
                        for (const auto& item : it.value()) {
                            if (item.is_string()) names.push_back(item.get<std::string>());
                        }
                        m_method_param_inputs[it.key()] = std::move(names);
                    }
                }
            } else if (meta.contains("promoted_inputs") && meta["promoted_inputs"].is_array()) {
                std::vector<std::string> names;
                for (const auto& item : meta["promoted_inputs"]) {
                    if (item.is_string()) names.push_back(item.get<std::string>());
                }
                m_method_param_inputs["forward"] = std::move(names);
            }
        } catch (const std::exception& e) {
            std::cerr << "[ExecuTorchBackend] Warning parsing metadata " << json_path << ": " << e.what() << std::endl;
        }
    }

    // 2. Read initial baseline weights from high-speed binary sidecar (<base>.bin)
    std::string sidecar_path = base_path + ".bin";
    std::ifstream bin_file(sidecar_path, std::ios::binary);
    if (bin_file.is_open()) {
        char magic[4];
        bin_file.read(magic, 4);
        if (bin_file.good() && std::memcmp(magic, "NBND", 4) == 0) {
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

                ParameterInfo param;
                param.name = t_name;
                param.shape = t_shape;
                param.total_elements = (size_t)num_elems;
                param.input_order = t;
                param.original_weights = data;
                param.current_weights = std::move(data);
                param.is_modified = false;

                m_parameters[t_name] = std::move(param);
                m_available_layers.push_back(t_name);
            }
            std::cout << "[ExecuTorchBackend] Loaded " << m_available_layers.size()
                      << " bendable layers from " << sidecar_path << std::endl;
        }
    }

    // Default forward parameter input order if not explicitly provided in JSON
    if (m_method_param_inputs.find("forward") == m_method_param_inputs.end() && !m_available_layers.empty()) {
        m_method_param_inputs["forward"] = m_available_layers;
    }
}

auto ExecuTorchBackend::load(std::string path, double sample_rate, const std::string* target_method) -> int {
    std::lock_guard<std::mutex> lock(m_session_mutex);
    m_loaded = false;
    m_path = path;
    m_sample_rate = sample_rate > 0.0 ? sample_rate : 44100.0;
    m_model_sample_rate = m_sample_rate;

    std::ifstream check_file(path, std::ios::binary);
    if (!check_file.good()) {
        m_last_error = "Model file does not exist: " + path;
        std::cerr << "[ExecuTorchBackend] " << m_last_error << std::endl;
        return -1;
    }
    check_file.close();

    std::string base = path;
    size_t dot_pos = base.find_last_of('.');
    if (dot_pos != std::string::npos) {
        base = base.substr(0, dot_pos);
    }

    load_metadata_and_sidecar(base);

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    try {
        m_impl->companions.clear();
        m_impl->module = std::make_unique<executorch::extension::Module>(
            path, executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors
        );
        auto err = m_impl->module->load();
        if (err != executorch::runtime::Error::Ok) {
            m_last_error = "ExecuTorch failed to load .pte program (error " + std::to_string((int)err) + ")";
            std::cerr << "[ExecuTorchBackend] " << m_last_error << std::endl;
            return -1;
        }

        // Introspect methods directly from .pte program if not in JSON
        auto method_names_res = m_impl->module->method_names();
        if (method_names_res.ok()) {
            m_available_methods.clear();
            for (const auto& name : method_names_res.get()) {
                // Skip constant attribute getter methods when populating audio methods
                if (name == "sr" || name == "get_sample_rate" || name == "encode_params" || name == "decode_params" || name == "forward_params") {
                    continue;
                }
                m_available_methods.push_back(name);
            }
        }

        // Also check for companion _encode.pte, _decode.pte, _prior.pte files
        const std::vector<std::string> potential_companions = { "encode", "decode", "prior" };
        for (const auto& meth : potential_companions) {
            if ( std::find(m_available_methods.begin(), m_available_methods.end(), meth) != m_available_methods.end() ) {
                continue;
            }
            std::string comp_path = base + "_" + meth + ".pte";
            std::ifstream f(comp_path, std::ios::binary);
            if (f.good()) {
                f.close();
                auto comp_mod = std::make_unique<executorch::extension::Module>(
                    comp_path, executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors
                );
                if (comp_mod->load() == executorch::runtime::Error::Ok) {
                    m_impl->companions[meth] = std::move(comp_mod);
                    m_available_methods.push_back(meth);
                }
            }
        }
    } catch (const std::exception& e) {
        m_last_error = e.what();
        std::cerr << "[ExecuTorchBackend] Exception loading " << path << ": " << e.what() << std::endl;
        return -1;
    }
#else
    // Check companion .pte files on disk so method introspection works even without C++ runtime linked
    if (m_available_methods.empty()) {
        m_available_methods.push_back("forward");
        const std::vector<std::string> potential_companions = { "encode", "decode", "prior" };
        for (const auto& meth : potential_companions) {
            std::string comp_path = base + "_" + meth + ".pte";
            std::ifstream f(comp_path, std::ios::binary);
            if (f.good()) {
                m_available_methods.push_back(meth);
            }
        }
    }
#endif

    if (m_available_methods.empty()) {
        m_available_methods.push_back("forward");
    }

    m_audio_history.clear();
    if (m_history_samples > 0) {
        if (m_output_offset == 0) m_output_offset = m_history_samples;
        m_audio_history.resize(std::max(1, m_in_channels), std::vector<float>(m_history_samples, 0.0f));
    }

    m_loaded = true;
    clear_decode_gen_fifo();
    m_last_error.clear();

    std::cout << "[ExecuTorchBackend] Loaded " << path
              << " | Delegate: " << m_delegate
              << " | SR: " << m_sample_rate
              << " | Ratio: " << m_higher_ratio
              << " | Latent Dim: " << m_latent_dim
              << " | Bendable Layers: " << m_available_layers.size() << std::endl;
    return 0;
}

auto ExecuTorchBackend::reload() -> int {
    if (m_path.empty()) return -1;
    return load(m_path, m_sample_rate);
}

auto ExecuTorchBackend::is_loaded() -> bool {
    return m_loaded;
}

auto ExecuTorchBackend::set_sample_rate(double sample_rate) -> void {
    if (sample_rate > 0.0) m_sample_rate = sample_rate;
}

auto ExecuTorchBackend::get_sample_rate() -> double {
    return (m_model_sample_rate > 0.0) ? m_model_sample_rate : m_sample_rate;
}

auto ExecuTorchBackend::use_gpu(bool value) -> void {
    m_use_gpu = value;
}

auto ExecuTorchBackend::perform(std::vector<float*>& in_buffer,
                                std::vector<float*>& out_buffer,
                                std::string method,
                                int n_batches, int n_out_channels, int n_vec) -> void {
    if (!m_loaded) return;
    std::unique_lock<std::mutex> lock(m_session_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    executorch::extension::Module* target_mod = m_impl->module.get();
    std::string exec_method = method;

    auto comp_it = m_impl->companions.find(method);
    if (comp_it != m_impl->companions.end() && comp_it->second) {
        target_mod = comp_it->second.get();
        exec_method = "forward";
    }
    if (!target_mod) return;

    try {
        int in_ch = (method == "decode") ? m_latent_dim : std::min((int)in_buffer.size(), m_in_channels);
        if (in_ch <= 0 || n_vec <= 0) return;

        int stream_len = n_vec;
        if (method == "forward" || method == "encode") {
            stream_len = (int)m_history_samples + n_vec;
        } else if (method == "decode") {
            stream_len = std::max(1, m_decode_latent_steps);
        }

        std::vector<float> audio_in((size_t)(in_ch * stream_len), 0.0f);

        if (method == "decode") {
            if (in_buffer.size() == 1 && in_buffer[0]) {
                std::memcpy(audio_in.data(), in_buffer[0], (size_t)(in_ch * stream_len) * sizeof(float));
            } else {
                for (int c = 0; c < in_ch && c < (int)in_buffer.size(); ++c) {
                    if (in_buffer[c]) {
                        std::memcpy(audio_in.data() + (c * stream_len), in_buffer[c], stream_len * sizeof(float));
                    }
                }
            }
        } else {
            for (int c = 0; c < in_ch; ++c) {
                float* dest = audio_in.data() + (c * stream_len);
                if ((method == "forward" || method == "encode") && m_history_samples > 0 && c < (int)m_audio_history.size()) {
                    std::memcpy(dest, m_audio_history[c].data(), m_history_samples * sizeof(float));
                    if (c < (int)in_buffer.size() && in_buffer[c]) {
                        std::memcpy(dest + m_history_samples, in_buffer[c], n_vec * sizeof(float));
                    }
                } else if (c < (int)in_buffer.size() && in_buffer[c]) {
                    std::memcpy(dest, in_buffer[c], n_vec * sizeof(float));
                }
            }
        }

        std::vector<executorch::aten::SizesType> in_sizes = { 1, (executorch::aten::SizesType)in_ch, (executorch::aten::SizesType)stream_len };
        auto audio_tensor = executorch::extension::from_blob(audio_in.data(), in_sizes);

        std::vector<executorch::runtime::EValue> inputs;
        inputs.push_back(audio_tensor);

        // Bind any promoted bendable parameter tensors for this method
        std::vector<executorch::extension::TensorPtr> param_tensors;
        auto pm_it = m_method_param_inputs.find(method);
        if (pm_it != m_method_param_inputs.end()) {
            param_tensors.reserve(pm_it->second.size());
            for (const auto& pname : pm_it->second) {
                auto pit = m_parameters.find(pname);
                if (pit != m_parameters.end()) {
                    std::vector<executorch::aten::SizesType> psizes;
                    for (auto d : pit->second.shape) psizes.push_back((executorch::aten::SizesType)d);
                    auto pt = executorch::extension::from_blob(pit->second.current_weights.data(), psizes);
                    param_tensors.push_back(pt);
                    inputs.push_back(pt);
                }
            }
        }

        auto result = target_mod->execute(exec_method, inputs);
        if (result.ok() && !result.get().empty() && result.get()[0].isTensor()) {
            auto out_tensor = result.get()[0].toTensor();
            const float* out_data = out_tensor.const_data_ptr<float>();
            int out_samples = (out_tensor.dim() >= 3) ? (int)out_tensor.size(2) : n_vec;
            int out_ch = std::min((int)out_buffer.size(), n_out_channels);

            if (method == "encode") {
                int target_latent_steps = std::max(1, m_decode_latent_steps);
                int copy_samples = std::min(target_latent_steps, out_samples);
                for (int c = 0; c < out_ch; ++c) {
                    if (out_buffer[c]) {
                        if (copy_samples > 0) {
                            std::memcpy(out_buffer[c], out_data + (c * out_samples), copy_samples * sizeof(float));
                        }
                        if (copy_samples < target_latent_steps) {
                            std::memset(out_buffer[c] + copy_samples, 0, (target_latent_steps - copy_samples) * sizeof(float));
                        }
                    }
                }
            } else {
                int offset = (out_samples > n_vec && (int)m_output_offset < out_samples) ? (int)m_output_offset : 0;
                int copy_samples = std::min(n_vec, std::max(0, out_samples - offset));
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
            }

            if ((method == "forward" || method == "encode") && m_history_samples > 0) {
                for (int c = 0; c < in_ch && c < (int)m_audio_history.size(); ++c) {
                    if (c < (int)in_buffer.size() && in_buffer[c]) {
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
        std::cerr << "[ExecuTorchBackend::perform:" << method << "] Error: " << e.what() << std::endl;
    }
#else
    for (int c = 0; c < (int)out_buffer.size() && c < n_out_channels; ++c) {
        if (out_buffer[c]) std::memset(out_buffer[c], 0, n_vec * sizeof(float));
    }
#endif
}

auto ExecuTorchBackend::perform_forward(std::vector<float*>& in_buffer,
                                        std::vector<float*>& out_buffer,
                                        int n_batches, int n_out_channels, int n_vec,
                                        LatentHook latent_hook) -> void {
    if (latent_hook && has_autoencode()) {
#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
        int latent_steps = std::max(1, m_decode_latent_steps);
        std::vector<std::vector<float>> latent_storage(m_latent_dim, std::vector<float>(latent_steps, 0.0f));
        std::vector<float*> latent_ptrs(m_latent_dim);
        for (int i = 0; i < m_latent_dim; ++i) latent_ptrs[i] = latent_storage[i].data();

        perform(in_buffer, latent_ptrs, "encode", n_batches, m_latent_dim, n_vec);

        std::vector<float> flat_latents((size_t)(m_latent_dim * latent_steps), 0.0f);
        for (int i = 0; i < m_latent_dim; ++i) {
            std::memcpy(flat_latents.data() + (i * latent_steps), latent_storage[i].data(), latent_steps * sizeof(float));
        }

        LatentBuffer buf{ flat_latents.data(), m_latent_dim, latent_steps };
        latent_hook(buf);

        std::vector<float*> decode_in_ptrs = { flat_latents.data() };
        perform(decode_in_ptrs, out_buffer, "decode", n_batches, n_out_channels, n_vec);
        return;
#endif
    }
    perform(in_buffer, out_buffer, "forward", n_batches, n_out_channels, n_vec);
}

auto ExecuTorchBackend::perform_prior_decode(std::vector<float*>& out_buffer,
                                             int n_batches, int n_out_channels, int n_vec) -> void {
    perform_decode_gen(out_buffer, n_batches, n_out_channels, n_vec, nullptr);
}

auto ExecuTorchBackend::perform_decode_gen(std::vector<float*>& out_buffer,
                                           int n_batches, int n_out_channels, int n_vec,
                                           LatentHook latent_hook) -> void {
    if (!m_loaded || !has_method("decode")) return;

    int ratio = std::max(1, m_higher_ratio);
    int out_ch = std::min((int)out_buffer.size(), n_out_channels);

    size_t fifo_samples = (m_decode_gen_fifo.empty() || m_decode_gen_fifo[0].empty())
        ? 0 : (m_decode_gen_fifo[0].size() - m_decode_gen_fifo_read_idx);

    const size_t kCrossfadeSamples = 256;
    if (m_decode_gen_fifo_read_idx > 0 && fifo_samples < (size_t)(n_vec + kCrossfadeSamples)) {
        for (auto& ch : m_decode_gen_fifo) {
            if (m_decode_gen_fifo_read_idx < ch.size()) {
                ch.erase(ch.begin(), ch.begin() + m_decode_gen_fifo_read_idx);
            } else {
                ch.clear();
            }
        }
        m_decode_gen_fifo_read_idx = 0;
    }

    int latent_steps = std::max(1, m_decode_latent_steps);
    int latent_dim = std::max(1, m_latent_dim);
    int gen_samples = std::max(1, m_buffer_size);

    const float alpha = (ratio <= 640) ? 0.94f : 0.86f;
    const float beta = std::sqrt(1.0f - alpha * alpha);
    const float latent_scale = (latent_dim >= 128) ? 5.0f : 1.0f;
    if ((int)m_decode_gen_latent_state.size() != latent_dim) {
        m_decode_gen_latent_state.assign(latent_dim, 0.0f);
    }

    static thread_local std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<float> dist(0.0f, 1.0f);

    while (fifo_samples < (size_t)(n_vec + kCrossfadeSamples)) {
        std::vector<float> z_data((size_t)(latent_dim * latent_steps), 0.0f);
        for (int t = 0; t < latent_steps; ++t) {
            for (int d = 0; d < latent_dim; ++d) {
                m_decode_gen_latent_state[d] = alpha * m_decode_gen_latent_state[d] + beta * dist(gen);
                z_data[d * latent_steps + t] = m_decode_gen_latent_state[d] * latent_scale;
            }
        }

        if (latent_hook) {
            LatentBuffer buf{ z_data.data(), latent_dim, latent_steps };
            latent_hook(buf);
        }

        std::vector<std::vector<float>> gen_out(out_ch, std::vector<float>(gen_samples, 0.0f));
        std::vector<float*> gen_out_ptrs(out_ch);
        for (int c = 0; c < out_ch; ++c) gen_out_ptrs[c] = gen_out[c].data();
        std::vector<float*> z_in_ptrs = { z_data.data() };

        perform(z_in_ptrs, gen_out_ptrs, "decode", 1, out_ch, gen_samples);

        if ((int)m_decode_gen_fifo.size() < out_ch) {
            m_decode_gen_fifo.resize(out_ch);
        }
        for (int c = 0; c < out_ch; ++c) {
            auto& fifo = m_decode_gen_fifo[c];
            size_t xfade = std::min({kCrossfadeSamples, fifo.size(), (size_t)gen_samples});
            if (xfade > 0) {
                size_t tail_start = fifo.size() - xfade;
                for (size_t i = 0; i < xfade; ++i) {
                    float t = (float)(i + 1) / (float)(xfade + 1);
                    float w_in = 0.5f * (1.0f - std::cos(3.14159265358979323846f * t));
                    float w_out = 1.0f - w_in;
                    fifo[tail_start + i] = fifo[tail_start + i] * w_out + gen_out[c][i] * w_in;
                }
                fifo.insert(fifo.end(), gen_out[c].begin() + xfade, gen_out[c].end());
            } else {
                fifo.insert(fifo.end(), gen_out[c].begin(), gen_out[c].end());
            }
        }
        fifo_samples = (m_decode_gen_fifo.empty() || m_decode_gen_fifo[0].empty())
            ? 0 : (m_decode_gen_fifo[0].size() - m_decode_gen_fifo_read_idx);
    }

    size_t avail = (m_decode_gen_fifo.empty() || m_decode_gen_fifo[0].empty())
        ? 0 : (m_decode_gen_fifo[0].size() - m_decode_gen_fifo_read_idx);
    int to_copy = std::min(n_vec, (int)avail);

    for (int c = 0; c < out_ch; ++c) {
        if (out_buffer[c]) {
            if (to_copy > 0) {
                std::memcpy(out_buffer[c], m_decode_gen_fifo[c].data() + m_decode_gen_fifo_read_idx, to_copy * sizeof(float));
            }
            if (to_copy < n_vec) {
                std::memset(out_buffer[c] + to_copy, 0, (n_vec - to_copy) * sizeof(float));
            }
        }
    }
    m_decode_gen_fifo_read_idx += to_copy;
}

auto ExecuTorchBackend::has_method(std::string method_name) -> bool {
    return std::find(m_available_methods.begin(), m_available_methods.end(), method_name) != m_available_methods.end();
}

auto ExecuTorchBackend::has_prior_decode() -> bool {
    return has_method("prior") && has_method("decode");
}

auto ExecuTorchBackend::has_autoencode() -> bool {
    return has_method("encode") && has_method("decode");
}

auto ExecuTorchBackend::get_plugin_modes() -> std::vector<std::string> {
    std::vector<std::string> modes;
    if (has_method("forward") || has_autoencode()) {
        modes.push_back("forward");
    }
    if (has_method("decode")) {
        modes.push_back("decode gen");
    }
    if (has_prior_decode()) {
        modes.push_back("prior");
    }
    if (modes.empty()) {
        modes.push_back("forward");
    }
    return modes;
}

auto ExecuTorchBackend::get_mode_params(std::string mode) -> std::vector<int> {
    if (mode == "decode gen" || mode == "prior" || mode == "generate") {
        return { 0, 1, m_out_channels, 1 };
    }
    return get_method_params(mode);
}

auto ExecuTorchBackend::get_available_methods(LockedModel* model) -> std::vector<std::string> {
    return m_available_methods;
}

auto ExecuTorchBackend::get_method_params(std::string method) -> std::vector<int> {
    if (method == "encode") {
        return { m_in_channels, 1, m_latent_dim, m_higher_ratio };
    }
    if (method == "decode") {
        return { m_latent_dim, m_higher_ratio, m_out_channels, 1 };
    }
    if (method == "prior") {
        return { 0, 1, m_latent_dim, m_higher_ratio };
    }
    return { m_in_channels, 1, m_out_channels, 1 };
}

auto ExecuTorchBackend::get_higher_ratio() -> int {
    return std::max(1, m_higher_ratio);
}

auto ExecuTorchBackend::get_available_layers() -> std::vector<std::string> {
    return m_available_layers;
}

auto ExecuTorchBackend::get_layer_weights(std::string layer_name) -> std::vector<float> {
    std::lock_guard<std::mutex> lock(m_session_mutex);
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        return it->second.current_weights;
    }
    return {};
}

auto ExecuTorchBackend::get_original_layer_weights(std::string layer_name) -> std::vector<float> {
    std::lock_guard<std::mutex> lock(m_session_mutex);
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        return it->second.original_weights;
    }
    return {};
}

auto ExecuTorchBackend::get_layer_shape(std::string layer_name) -> std::vector<int64_t> {
    std::lock_guard<std::mutex> lock(m_session_mutex);
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        return it->second.shape;
    }
    return {};
}

auto ExecuTorchBackend::set_layer_weights(std::string layer_name, const std::vector<float>& weights) -> void {
    std::lock_guard<std::mutex> lock(m_session_mutex);
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        size_t copy_len = std::min(it->second.current_weights.size(), weights.size());
        if (copy_len > 0) {
            std::memcpy(it->second.current_weights.data(), weights.data(), copy_len * sizeof(float));
            it->second.is_modified = true;
        }
    }
}

auto ExecuTorchBackend::reset_layer_weights(std::string layer_name) -> void {
    std::lock_guard<std::mutex> lock(m_session_mutex);
    auto it = m_parameters.find(layer_name);
    if (it != m_parameters.end()) {
        it->second.current_weights = it->second.original_weights;
        it->second.is_modified = false;
    }
}

auto ExecuTorchBackend::reset_all_layer_weights() -> void {
    std::lock_guard<std::mutex> lock(m_session_mutex);
    for (auto& pair : m_parameters) {
        pair.second.current_weights = pair.second.original_weights;
        pair.second.is_modified = false;
    }
}

auto ExecuTorchBackend::save_model(std::string path) -> int {
    std::lock_guard<std::mutex> lock(m_session_mutex);
    if (!m_loaded || m_path.empty()) return -1;

    try {
        if (path != m_path) {
            std::ifstream src(m_path, std::ios::binary);
            std::ofstream dst(path, std::ios::binary);
            if (!src.good() || !dst.good()) return -1;
            dst << src.rdbuf();
        }

        std::string base = path;
        size_t dot_pos = base.find_last_of('.');
        if (dot_pos != std::string::npos) base = base.substr(0, dot_pos);
        std::string sidecar_path = base + ".bin";

        std::ofstream bin_out(sidecar_path, std::ios::binary);
        if (!bin_out.good()) return -1;

        bin_out.write("NBND", 4);
        uint32_t version = 1;
        bin_out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        uint32_t num_tensors = (uint32_t)m_available_layers.size();
        bin_out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

        for (const auto& name : m_available_layers) {
            const auto& param = m_parameters[name];
            uint32_t name_len = (uint32_t)name.size();
            bin_out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            bin_out.write(name.data(), name_len);

            uint32_t num_dims = (uint32_t)param.shape.size();
            bin_out.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));
            for (int64_t d : param.shape) {
                bin_out.write(reinterpret_cast<const char*>(&d), sizeof(int64_t));
            }

            uint64_t num_elems = (uint64_t)param.current_weights.size();
            bin_out.write(reinterpret_cast<const char*>(&num_elems), sizeof(num_elems));
            bin_out.write(reinterpret_cast<const char*>(param.current_weights.data()), num_elems * sizeof(float));
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

auto ExecuTorchBackend::get_available_attributes() -> std::vector<std::string> { return {}; }
auto ExecuTorchBackend::get_settable_attributes() -> std::vector<std::string> { return {}; }
auto ExecuTorchBackend::has_settable_attribute(std::string attribute) -> bool { return false; }
auto ExecuTorchBackend::get_attribute_as_string(std::string attribute_name) -> std::string { return ""; }
auto ExecuTorchBackend::set_attribute(std::string attribute_name,
                                      std::vector<std::string> attribute_args,
                                      const BufferMap& buffer_array) -> void {}
auto ExecuTorchBackend::get_buffer_attributes() -> std::vector<std::string> { return {}; }
auto ExecuTorchBackend::is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx) -> bool { return false; }
auto ExecuTorchBackend::is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx) -> bool { return false; }
auto ExecuTorchBackend::get_buffer_name(std::string attribute_name, int attribute_elt_idx) -> std::string { return ""; }
auto ExecuTorchBackend::update_buffer(std::string buffer_id, StaticBuffer<float>& buffer) -> int { return 0; }
auto ExecuTorchBackend::reset_buffer(std::string buffer_id) -> int { return 0; }

auto ExecuTorchBackend::get_model_info() -> ModelInfo {
    ModelInfo info;
    for (const auto& m : m_available_methods) {
        auto p = get_method_params(m);
        if (p.size() >= 4) {
            MethodProperties props;
            props.name = m;
            props.channels_in = p[0];
            props.ratio_in = p[1];
            props.channels_out = p[2];
            props.ratio_out = p[3];
            info.method_properties[m] = props;
        }
    }
    return info;
}
