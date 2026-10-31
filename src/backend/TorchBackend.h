#pragma once
#include "IBackend.h"
#include <mutex>
#include <string>
#include <torch/script.h>
#include "../shared/static_buffer.h"
#include <torch/torch.h>
#include <vector>
#include <functional>

struct LockedModel {
  torch::jit::script::Module* model; 
  std::mutex mutex;
};

/**
 * TorchBackend (TorchScript execution engine)
 * Implements IBackend for LibTorch / TorchScript models (.ts)
 */
class TorchBackend : public IBackend {

protected:
  torch::jit::script::Module m_model;
  int m_loaded, m_in_dim, m_in_ratio, m_out_dim, m_out_ratio;
  std::string m_path;
  std::mutex m_model_mutex;
  std::vector<std::string> m_available_methods;
  std::vector<std::string> m_buffer_attributes;
  c10::DeviceType m_device;
  bool m_use_gpu;
  auto retrieve_buffer_attributes() -> std::vector<std::string>;
  std::unique_ptr<std::thread> set_attribute_thread; 
  double m_sr; 
  std::unordered_map<std::string, std::vector<float>> m_original_weights;
  std::unordered_map<std::string, torch::Tensor> m_parameter_cache;
  auto prepare_io_tensor(std::vector<float*>& in_buffer, int in_dim, int n_batches, int in_ratio, int n_vec) -> torch::Tensor;
  std::string m_last_error;

public:
  auto get_last_error() const -> std::string override { return m_last_error; }
  auto get_backend_name() const -> std::string override {
    return m_use_gpu ? "TorchScript (GPU)" : "TorchScript";
  }
  auto get_input_label(const std::string& method, int index) -> std::string override {
    try {
      return m_model.attr(method + "_input_labels").toList().get(index).toStringRef();
    } catch (...) {
      return "(signal) model input " + std::to_string(index);
    }
  }
  auto get_output_label(const std::string& method, int index) -> std::string override {
    try {
      return m_model.attr(method + "_output_labels").toList().get(index).toStringRef();
    } catch (...) {
      return "(signal) model output " + std::to_string(index);
    }
  }
  using DataType = float; 
  using ArgsType = std::vector<c10::IValue>;
  using KwargsType = std::unordered_map<std::string, c10::IValue>;
  using BufferMap = std::map<std::string, StaticBuffer<float>>;
  using LatentHook = IBackend::LatentHook;

  TorchBackend();
  virtual ~TorchBackend() override = default;

  auto perform(std::vector<float*>& in_buffer,
               std::vector<float*>& out_buffer,
               std::string method,
               int n_batches, int n_out_channels, int n_vec) -> void override;

  auto perform_forward(std::vector<float*>& in_buffer,
                       std::vector<float*>& out_buffer,
                       int n_batches, int n_out_channels, int n_vec,
                       IBackend::LatentHook latent_hook = nullptr) -> void override;

  auto perform_prior_decode(std::vector<float*>& out_buffer,
                            int n_batches, int n_out_channels, int n_vec) -> void override;

  auto perform_decode_gen(std::vector<float*>& out_buffer,
                          int n_batches, int n_out_channels, int n_vec,
                          IBackend::LatentHook latent_hook = nullptr) -> void override;

  auto has_method(std::string method_name) -> bool override;
  auto has_prior_decode() -> bool override;
  auto has_autoencode() -> bool override;
  auto get_plugin_modes() -> std::vector<std::string> override;
  auto get_mode_params(std::string mode) -> std::vector<int> override;
  auto has_settable_attribute(std::string attribute) -> bool override;
  auto get_available_methods(LockedModel* model = nullptr) -> std::vector<std::string> override;
  auto get_available_attributes() -> std::vector<std::string> override;
  auto get_settable_attributes() -> std::vector<std::string> override;
  auto get_attribute(std::string attribute_name) -> std::vector<c10::IValue>;
  auto get_attribute_as_string(std::string attribute_name) -> std::string override;
  auto set_attribute(std::string attribute_name,
                     std::vector<std::string> attribute_args,
                     const BufferMap& buffer_array) -> void override;

  // buffer attributes
  auto is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx) -> bool override;
  auto is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx) -> bool override;
  auto get_buffer_name(std::string attribute_name, int attribute_elt_idx) -> std::string override;
  auto update_buffer(std::string buffer_id, StaticBuffer<float>& buffer) -> int override;
  auto reset_buffer(std::string buffer_id) -> int override;

  auto get_method_params(std::string method) -> std::vector<int> override;
  auto get_higher_ratio() -> int override;
  auto load(std::string path, double sample_rate, const std::string* target_method = nullptr) -> int override;
  auto reload() -> int override;
  auto set_sample_rate(double sample_rate) -> void override;
  auto get_sample_rate() -> double override;
  auto is_loaded() -> bool override;
  auto get_model() -> torch::jit::script::Module { return m_model; }
  auto use_gpu(bool value) -> void override;
  auto get_buffer_attributes() -> std::vector<std::string> override;
  
  auto empty_args() -> ArgsType { return ArgsType(); }
  auto empty_kwargs() -> KwargsType { return KwargsType(); }
  auto empty_inputs() -> std::pair<ArgsType, KwargsType> {
    return std::make_pair(empty_args(), empty_kwargs());
  } 

  auto get_available_layers() -> std::vector<std::string> override;
  auto get_layer_weights(std::string layer_name) -> std::vector<float> override;
  auto get_original_layer_weights(std::string layer_name) -> std::vector<float> override;
  auto get_layer_shape(std::string layer_name) -> std::vector<int64_t> override;
  auto set_layer_weights(std::string layer_name, const std::vector<float>& weights) -> void override;
  auto reset_layer_weights(std::string layer_name) -> void override;
  auto reset_all_layer_weights() -> void override;
  auto save_model(std::string path) -> int override;

  auto get_model_info() -> ModelInfo override;
  const std::unordered_map<int, std::string> id_to_string_hash = {
    {0, "bool"}, 
    {1, "int"}, 
    {2, "float"},
    {3, "string"}, 
    {4, "tensor"}, 
    {5, "buffer"} 
  };
};

// Backward-compatible alias for existing frontend code
using Backend = TorchBackend;
