#pragma once
#include "IBackend.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

/**
 * ExecuTorchBackend
 *
 * Hardware-accelerated neural audio inference engine powered by Meta's ExecuTorch (.pte),
 * inspired by jasper-zheng/neural_tilde and extended with NetBender's real-time
 * Input-Promoted Parameter bending and Latent Terrain hooks.
 *
 * Supports hardware delegates:
 *   - CoreML (Apple Neural Engine / GPU / CPU)
 *   - MPS (Apple Metal Performance Shaders)
 *   - XNNPACK (SIMD-accelerated CPU)
 *   - Portable (Reference fallback)
 */
class ExecuTorchBackend : public IBackend {
public:
    ExecuTorchBackend();
    ~ExecuTorchBackend() override;

    // Lifecycle
    auto load(std::string path, double sample_rate, const std::string* target_method = nullptr) -> int override;
    auto reload() -> int override;
    auto is_loaded() -> bool override;
    auto set_sample_rate(double sample_rate) -> void override;
    auto get_sample_rate() -> double override;
    auto use_gpu(bool value) -> void override;
    auto get_last_error() const -> std::string override { return m_last_error; }
    auto get_backend_name() const -> std::string override;

    // Inference Execution
    auto perform(std::vector<float*>& in_buffer,
                 std::vector<float*>& out_buffer,
                 std::string method,
                 int n_batches, int n_out_channels, int n_vec) -> void override;

    auto perform_forward(std::vector<float*>& in_buffer,
                         std::vector<float*>& out_buffer,
                         int n_batches, int n_out_channels, int n_vec,
                         LatentHook latent_hook = nullptr) -> void override;

    auto perform_prior_decode(std::vector<float*>& out_buffer,
                              int n_batches, int n_out_channels, int n_vec) -> void override;

    auto perform_decode_gen(std::vector<float*>& out_buffer,
                            int n_batches, int n_out_channels, int n_vec,
                            LatentHook latent_hook = nullptr) -> void override;

    // Mode & Method Introspection
    auto has_method(std::string method_name) -> bool override;
    auto has_prior_decode() -> bool override;
    auto has_autoencode() -> bool override;
    auto get_plugin_modes() -> std::vector<std::string> override;
    auto get_mode_params(std::string mode) -> std::vector<int> override;
    auto get_available_methods(LockedModel* model = nullptr) -> std::vector<std::string> override;
    auto get_method_params(std::string method) -> std::vector<int> override;
    auto get_higher_ratio() -> int override;

    // Parameter Introspection & Bending
    auto get_available_layers() -> std::vector<std::string> override;
    auto get_layer_weights(std::string layer_name) -> std::vector<float> override;
    auto get_original_layer_weights(std::string layer_name) -> std::vector<float> override;
    auto get_layer_shape(std::string layer_name) -> std::vector<int64_t> override;
    auto set_layer_weights(std::string layer_name, const std::vector<float>& weights) -> void override;
    auto reset_layer_weights(std::string layer_name) -> void override;
    auto reset_all_layer_weights() -> void override;
    auto save_model(std::string path) -> int override;

    // Max/MSP & PureData Attributes
    auto get_available_attributes() -> std::vector<std::string> override;
    auto get_settable_attributes() -> std::vector<std::string> override;
    auto has_settable_attribute(std::string attribute) -> bool override;
    auto get_attribute_as_string(std::string attribute_name) -> std::string override;
    auto set_attribute(std::string attribute_name,
                       std::vector<std::string> attribute_args,
                       const BufferMap& buffer_array) -> void override;
    auto get_buffer_attributes() -> std::vector<std::string> override;
    auto is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx) -> bool override;
    auto is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx) -> bool override;
    auto get_buffer_name(std::string attribute_name, int attribute_elt_idx) -> std::string override;
    auto update_buffer(std::string buffer_id, StaticBuffer<float>& buffer) -> int override;
    auto reset_buffer(std::string buffer_id) -> int override;
    auto get_model_info() -> ModelInfo override;

private:
    struct ParameterInfo {
        std::string name;
        std::vector<int64_t> shape;
        size_t total_elements { 0 };
        size_t input_order { 0 };
        std::vector<float> original_weights;
        std::vector<float> current_weights;
        bool is_modified { false };
    };

    std::string m_path;
    std::string m_delegate { "portable" };
    double m_sample_rate { 44100.0 };
    double m_model_sample_rate { 44100.0 };
    bool m_loaded { false };
    bool m_use_gpu { false };
    std::string m_last_error;
    std::mutex m_session_mutex;

    struct Impl;
    std::unique_ptr<Impl> m_impl;

    std::vector<std::string> m_available_layers;
    std::unordered_map<std::string, ParameterInfo> m_parameters;
    std::unordered_map<std::string, std::vector<std::string>> m_method_param_inputs;

    int m_in_channels { 1 };
    int m_out_channels { 1 };
    int m_higher_ratio { 2048 };
    int m_latent_dim { 64 };
    int m_buffer_size { 2048 };
    int m_decode_latent_steps { 1 };

    std::vector<std::vector<float>> m_audio_history;
    size_t m_history_samples { 0 };
    size_t m_output_offset { 0 };

    std::vector<std::string> m_available_methods;

    auto load_metadata_and_sidecar(const std::string& base_path) -> void;
};

