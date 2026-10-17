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
  std::vector<std::string> retrieve_buffer_attributes();
  std::unique_ptr<std::thread> set_attribute_thread; 
  double m_sr; 
  std::unordered_map<std::string, std::vector<float>> m_original_weights;
  std::unordered_map<std::string, torch::Tensor> m_parameter_cache;
  torch::Tensor prepare_io_tensor(std::vector<float *> &in_buffer, int in_dim, int n_batches, int in_ratio, int n_vec);

public:
  using DataType = float; 
  using ArgsType = std::vector<c10::IValue>;
  using KwargsType = std::unordered_map<std::string, c10::IValue>;
  using BufferMap = std::map<std::string, StaticBuffer<float>>;
  using TorchLatentHook = std::function<at::Tensor(at::Tensor)>;
  // For backward compatibility with existing code calling Backend::LatentHook:
  using LatentHook = TorchLatentHook;

  TorchBackend();
  virtual ~TorchBackend() override = default;

  void perform(std::vector<float *> &in_buffer,
               std::vector<float *> &out_buffer, 
               std::string method, 
               int n_batches, int n_out_channels, int n_vec) override;

  // Generic IBackend forward hook (converts LatentBuffer to/from tensor under the hood)
  void perform_forward(std::vector<float *> &in_buffer,
                       std::vector<float *> &out_buffer,
                       int n_batches, int n_out_channels, int n_vec,
                       IBackend::LatentHook latent_hook = nullptr) override;

  // TorchLatentHook overload for perform_forward
  void perform_forward(std::vector<float *> &in_buffer,
                       std::vector<float *> &out_buffer,
                       int n_batches, int n_out_channels, int n_vec,
                       TorchLatentHook latent_hook);

  void perform_forward_torch(std::vector<float *> &in_buffer,
                             std::vector<float *> &out_buffer,
                             int n_batches, int n_out_channels, int n_vec,
                             TorchLatentHook latent_hook = nullptr);

  void perform_prior_decode(std::vector<float *> &out_buffer,
                            int n_batches, int n_out_channels, int n_vec) override;

  // Generic IBackend decode gen hook
  void perform_decode_gen(std::vector<float *> &out_buffer,
                          int n_batches, int n_out_channels, int n_vec,
                          IBackend::LatentHook latent_hook = nullptr) override;

  // TorchLatentHook overload for perform_decode_gen
  void perform_decode_gen(std::vector<float *> &out_buffer,
                          int n_batches, int n_out_channels, int n_vec,
                          TorchLatentHook latent_hook);

  void perform_decode_gen_torch(std::vector<float *> &out_buffer,
                                int n_batches, int n_out_channels, int n_vec,
                                TorchLatentHook latent_hook = nullptr);

  bool has_method(std::string method_name) override;
  bool has_prior_decode() override;
  bool has_autoencode() override;
  std::vector<std::string> get_plugin_modes() override;
  std::vector<int> get_mode_params(std::string mode) override;
  bool has_settable_attribute(std::string attribute) override;
  std::vector<std::string> get_available_methods(LockedModel *model = nullptr) override;
  std::vector<std::string> get_available_attributes() override;
  std::vector<std::string> get_settable_attributes() override;
  std::vector<c10::IValue> get_attribute(std::string attribute_name);
  std::string get_attribute_as_string(std::string attribute_name) override;
  void set_attribute(std::string attribute_name,
                     std::vector<std::string> attribute_args, 
                     const BufferMap &buffer_array) override;

  // buffer attributes
  bool is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx) override;
  bool is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx) override;
  std::string get_buffer_name(std::string attribute_name, int attribute_elt_idx) override;
  int update_buffer(std::string buffer_id, StaticBuffer<float> &buffer) override;
  int reset_buffer(std::string) override;

  std::vector<int> get_method_params(std::string method) override;
  int get_higher_ratio() override;
  int load(std::string path, double sampleRate, const std::string* target_method = nullptr) override;
  int reload() override;
  void set_sample_rate(double sampleRate) override;
  double get_sample_rate() override;
  bool is_loaded() override;
  torch::jit::script::Module get_model() { return m_model; }
  void use_gpu(bool value) override;
  std::vector<std::string> get_buffer_attributes() override;
  
  ArgsType empty_args() { return ArgsType(); }
  KwargsType empty_kwargs() { return KwargsType(); }
  std::pair<ArgsType, KwargsType> empty_inputs() {
    return std::make_pair(empty_args(), empty_kwargs());
  } 

  std::vector<std::string> get_available_layers() override;
  std::vector<float> get_layer_weights(std::string layer_name) override;
  std::vector<float> get_original_layer_weights(std::string layer_name) override;
  std::vector<int64_t> get_layer_shape(std::string layer_name) override;
  void set_layer_weights(std::string layer_name, const std::vector<float>& weights) override;
  void reset_layer_weights(std::string layer_name) override;
  void reset_all_layer_weights() override;
  int save_model(std::string path) override;

  ModelInfo get_model_info() override;
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
