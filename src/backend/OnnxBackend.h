#pragma once
#include "IBackend.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

// Forward declaration of ONNX Runtime types to avoid heavy include in header
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
    struct MemoryInfo;
    struct Value;
    struct AllocatorWithDefaultOptions;
}

/**
 * OnnxBackend
 * 
 * High-performance, zero-copy ONNX inference engine supporting real-time parameter bending.
 *
 * Implements the "Input-Promoted Parameter pattern":
 * Weights, biases, and normalization parameters are promoted to graph inputs.
 * At inference time, mutated parameters from layerState.scratchFinal are bound as non-owning
 * Ort::Value tensors, allowing audio-rate modulation with zero heap allocation and zero session reloads.
 */
class OnnxBackend : public IBackend {
public:
    OnnxBackend();
    ~OnnxBackend() override;

    // Lifecycle
    int load(std::string path, double sampleRate, const std::string* target_method = nullptr) override;
    int reload() override;
    bool is_loaded() override;
    void set_sample_rate(double sampleRate) override;
    double get_sample_rate() override;
    void use_gpu(bool value) override;

    // Inference Execution
    void perform(std::vector<float*>& in_buffer,
                 std::vector<float*>& out_buffer,
                 std::string method,
                 int n_batches, int n_out_channels, int n_vec) override;

    void perform_forward(std::vector<float*>& in_buffer,
                         std::vector<float*>& out_buffer,
                         int n_batches, int n_out_channels, int n_vec,
                         LatentHook latent_hook = nullptr) override;

    void perform_prior_decode(std::vector<float*>& out_buffer,
                              int n_batches, int n_out_channels, int n_vec) override;

    void perform_decode_gen(std::vector<float*>& out_buffer,
                            int n_batches, int n_out_channels, int n_vec,
                            LatentHook latent_hook = nullptr) override;

    // Mode & Method Introspection
    bool has_method(std::string method_name) override;
    bool has_prior_decode() override;
    bool has_autoencode() override;
    std::vector<std::string> get_plugin_modes() override;
    std::vector<int> get_mode_params(std::string mode) override;
    std::vector<std::string> get_available_methods(LockedModel* model = nullptr) override;
    std::vector<int> get_method_params(std::string method) override;
    int get_higher_ratio() override;

    // Parameter Introspection & Bending
    std::vector<std::string> get_available_layers() override;
    std::vector<float> get_layer_weights(std::string layer_name) override;
    std::vector<float> get_original_layer_weights(std::string layer_name) override;
    std::vector<int64_t> get_layer_shape(std::string layer_name) override;
    void set_layer_weights(std::string layer_name, const std::vector<float>& weights) override;
    void reset_layer_weights(std::string layer_name) override;
    void reset_all_layer_weights() override;
    int save_model(std::string path) override;

    // Max/MSP & PureData Attributes
    std::vector<std::string> get_available_attributes() override;
    std::vector<std::string> get_settable_attributes() override;
    bool has_settable_attribute(std::string attribute) override;
    std::string get_attribute_as_string(std::string attribute_name) override;
    void set_attribute(std::string attribute_name,
                       std::vector<std::string> attribute_args,
                       const BufferMap& buffer_array) override;
    std::vector<std::string> get_buffer_attributes() override;
    bool is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx) override;
    bool is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx) override;
    std::string get_buffer_name(std::string attribute_name, int attribute_elt_idx) override;
    int update_buffer(std::string buffer_id, StaticBuffer<float>& buffer) override;
    int reset_buffer(std::string buffer_id) override;
    ModelInfo get_model_info() override;

private:
    struct ParameterInfo {
        std::string name;
        std::vector<int64_t> shape;
        size_t totalElements { 0 };
        size_t inputIndex { 0 };
        std::vector<float> originalWeights;
        std::vector<float> currentWeights;
    };

    struct StreamInfo {
        std::string name;
        size_t inputIndex { 0 };
        std::vector<int64_t> shape;
    };

    std::string m_path;
    double m_sampleRate { 44100.0 };
    bool m_loaded { false };
    bool m_useGpu { false };
    std::mutex m_sessionMutex;

    // Internal implementation pointers (PIMPL to keep ORT types out of public API)
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    std::vector<std::string> m_availableLayers;
    std::unordered_map<std::string, ParameterInfo> m_parameters;
    std::vector<StreamInfo> m_audioInputs;
    std::vector<StreamInfo> m_audioOutputs;

    int m_inChannels { 1 };
    int m_outChannels { 1 };
    int m_higherRatio { 1 };

    void introspectGraph();
};
