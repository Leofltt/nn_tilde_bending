#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <functional>
#include <cstdint>
#include "../shared/static_buffer.h"

struct MethodProperties {
  std::string name = ""; 
  int channels_in = -1;
  int channels_out = -1;
  int ratio_in = -1;
  int ratio_out = -1;
};

struct AttributeProperties {
  std::string name = ""; 
  std::vector<std::string> attribute_types = {};
};

struct ModelInfo {
  using MethodDict = std::unordered_map<std::string, MethodProperties>;
  using AttributeDict = std::unordered_map<std::string, AttributeProperties>;
  MethodDict method_properties = {};
  AttributeDict attribute_properties = {}; 
};

struct LockedModel;

/**
 * Abstract interface representing a Neural Network Audio Backend.
 * Allows decoupling high-level audio DSP and bending tools from specific inference runtimes
 * (such as LibTorch TorchScript, ONNX Runtime, or RT-Neural).
 */
class IBackend {
public:
  using DataType = float;
  using BufferMap = std::map<std::string, StaticBuffer<float>>;

  virtual ~IBackend() = default;

  // Lifecycle & Model Loading
  virtual auto load(std::string path, double sample_rate, const std::string* target_method = nullptr) -> int = 0;
  virtual auto reload() -> int = 0;
  virtual auto is_loaded() -> bool = 0;
  virtual auto set_sample_rate(double sample_rate) -> void = 0;
  virtual auto get_sample_rate() -> double = 0;
  virtual auto use_gpu(bool value) -> void = 0;
  virtual auto get_last_error() const -> std::string { return ""; }
  virtual auto get_backend_name() const -> std::string { return "Generic"; }
  virtual auto get_input_label(const std::string& method, int index) -> std::string {
    return "(signal) model input " + std::to_string(index);
  }
  virtual auto get_output_label(const std::string& method, int index) -> std::string {
    return "(signal) model output " + std::to_string(index);
  }

  // Latent Hook Abstraction (backend agnostic: operates on raw float pointer, channels, length)
  struct LatentBuffer {
    float* data;
    int64_t channels;
    int64_t length;
  };
  using LatentHook = std::function<void(LatentBuffer&)>;

  // Inference Execution
  virtual auto perform(std::vector<float*>& in_buffer,
                       std::vector<float*>& out_buffer,
                       std::string method,
                       int n_batches, int n_out_channels, int n_vec) -> void = 0;

  virtual auto perform_forward(std::vector<float*>& in_buffer,
                               std::vector<float*>& out_buffer,
                               int n_batches, int n_out_channels, int n_vec,
                               LatentHook latent_hook = nullptr) -> void {}

  virtual auto perform_prior_decode(std::vector<float*>& out_buffer,
                                    int n_batches, int n_out_channels, int n_vec) -> void = 0;

  virtual auto perform_decode_gen(std::vector<float*>& out_buffer,
                                  int n_batches, int n_out_channels, int n_vec,
                                  LatentHook latent_hook = nullptr) -> void {}

  virtual auto clear_decode_gen_fifo() -> void {
    for (auto& ch : m_decode_gen_fifo) ch.clear();
    m_decode_gen_fifo.clear();
    m_decode_gen_fifo_read_idx = 0;
    m_decode_gen_latent_state.clear();
  }

  // Method & Mode Introspection
  virtual auto has_method(std::string method_name) -> bool = 0;
  virtual auto has_prior_decode() -> bool = 0;
  virtual auto has_autoencode() -> bool = 0;
  virtual auto get_plugin_modes() -> std::vector<std::string> = 0;
  virtual auto get_mode_params(std::string mode) -> std::vector<int> = 0;
  virtual auto get_available_methods(LockedModel* model = nullptr) -> std::vector<std::string> = 0;
  virtual auto get_method_params(std::string method) -> std::vector<int> = 0;
  virtual auto get_higher_ratio() -> int = 0;

  // Parameter Introspection & Bending
  virtual auto get_available_layers() -> std::vector<std::string> = 0;
  virtual auto get_layer_weights(std::string layer_name) -> std::vector<float> = 0;
  virtual auto get_original_layer_weights(std::string layer_name) -> std::vector<float> = 0;
  virtual auto get_layer_shape(std::string layer_name) -> std::vector<int64_t> = 0;
  virtual auto set_layer_weights(std::string layer_name, const std::vector<float>& weights) -> void = 0;
  virtual auto reset_layer_weights(std::string layer_name) -> void = 0;
  virtual auto reset_all_layer_weights() -> void = 0;
  virtual auto reset_all_weights() -> void { reset_all_layer_weights(); }
  virtual auto save_model(std::string path) -> int = 0;

  // Attributes & Buffer Management (Max/MSP & PureData compatibility)
  virtual auto get_available_attributes() -> std::vector<std::string> = 0;
  virtual auto get_settable_attributes() -> std::vector<std::string> = 0;
  virtual auto has_settable_attribute(std::string attribute) -> bool = 0;
  virtual auto get_attribute_as_string(std::string attribute_name) -> std::string = 0;
  virtual auto set_attribute(std::string attribute_name,
                             std::vector<std::string> attribute_args,
                             const BufferMap& buffer_array) -> void = 0;
  virtual auto get_buffer_attributes() -> std::vector<std::string> = 0;
  virtual auto is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx) -> bool = 0;
  virtual auto is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx) -> bool = 0;
  virtual auto get_buffer_name(std::string attribute_name, int attribute_elt_idx) -> std::string = 0;
  virtual auto update_buffer(std::string buffer_id, StaticBuffer<float>& buffer) -> int = 0;
  virtual auto reset_buffer(std::string buffer_id) -> int = 0;
  virtual auto get_model_info() -> ModelInfo = 0;

protected:
  // Decode Gen output FIFO buffer per channel for gapless streaming & latent random walk state
  std::vector<std::vector<float>> m_decode_gen_fifo;
  size_t                          m_decode_gen_fifo_read_idx { 0 };
  std::vector<float>              m_decode_gen_latent_state;
};
