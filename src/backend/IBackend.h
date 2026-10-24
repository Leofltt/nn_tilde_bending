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
  virtual int load(std::string path, double sampleRate, const std::string* target_method = nullptr) = 0;
  virtual int reload() = 0;
  virtual bool is_loaded() = 0;
  virtual void set_sample_rate(double sampleRate) = 0;
  virtual double get_sample_rate() = 0;
  virtual void use_gpu(bool value) = 0;
  virtual std::string get_last_error() const { return ""; }

  // Latent Hook Abstraction (backend agnostic: operates on raw float pointer, channels, length)
  struct LatentBuffer {
    float* data;
    int64_t channels;
    int64_t length;
  };
  using LatentHook = std::function<void(LatentBuffer&)>;

  // Inference Execution
  virtual void perform(std::vector<float *> &in_buffer,
                       std::vector<float *> &out_buffer, 
                       std::string method, 
                       int n_batches, int n_out_channels, int n_vec) = 0;

  virtual void perform_forward(std::vector<float *> &in_buffer,
                               std::vector<float *> &out_buffer,
                               int n_batches, int n_out_channels, int n_vec,
                               LatentHook latent_hook = nullptr) {}

  virtual void perform_prior_decode(std::vector<float *> &out_buffer,
                                    int n_batches, int n_out_channels, int n_vec) = 0;

  virtual void perform_decode_gen(std::vector<float *> &out_buffer,
                                  int n_batches, int n_out_channels, int n_vec,
                                  LatentHook latent_hook = nullptr) {}

  // Method & Mode Introspection
  virtual bool has_method(std::string method_name) = 0;
  virtual bool has_prior_decode() = 0;
  virtual bool has_autoencode() = 0;
  virtual std::vector<std::string> get_plugin_modes() = 0;
  virtual std::vector<int> get_mode_params(std::string mode) = 0;
  virtual std::vector<std::string> get_available_methods(LockedModel *model = nullptr) = 0;
  virtual std::vector<int> get_method_params(std::string method) = 0;
  virtual int get_higher_ratio() = 0;

  // Parameter Introspection & Bending
  virtual std::vector<std::string> get_available_layers() = 0;
  virtual std::vector<float> get_layer_weights(std::string layer_name) = 0;
  virtual std::vector<float> get_original_layer_weights(std::string layer_name) = 0;
  virtual std::vector<int64_t> get_layer_shape(std::string layer_name) = 0;
  virtual void set_layer_weights(std::string layer_name, const std::vector<float>& weights) = 0;
  virtual void reset_layer_weights(std::string layer_name) = 0;
  virtual void reset_all_layer_weights() = 0;
  virtual int save_model(std::string path) = 0;

  // Attributes & Buffer Management (Max/MSP & PureData compatibility)
  virtual std::vector<std::string> get_available_attributes() = 0;
  virtual std::vector<std::string> get_settable_attributes() = 0;
  virtual bool has_settable_attribute(std::string attribute) = 0;
  virtual std::string get_attribute_as_string(std::string attribute_name) = 0;
  virtual void set_attribute(std::string attribute_name,
                             std::vector<std::string> attribute_args, 
                             const BufferMap &buffer_array) = 0;
  virtual std::vector<std::string> get_buffer_attributes() = 0;
  virtual bool is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx) = 0;
  virtual bool is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx) = 0;
  virtual std::string get_buffer_name(std::string attribute_name, int attribute_elt_idx) = 0;
  virtual int update_buffer(std::string buffer_id, StaticBuffer<float> &buffer) = 0;
  virtual int reset_buffer(std::string buffer_id) = 0;
  virtual ModelInfo get_model_info() = 0;
};
