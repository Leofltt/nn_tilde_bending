#include "backend.h"
#include <exception>
#include "parsing_utils.h"
#include <algorithm>
#include <iostream>
#include <exception>
#include <stdio.h>
#include <stdlib.h>

#define CPU torch::kCPU
#define CUDA torch::kCUDA
#define MPS torch::kMPS

#define REFRESH_THREAD_INTERVAL 50


std::string tensor_to_str(torch::Tensor &tsr) {
  std::stringstream ss; 
  ss << "Tensor(dim: ";
  for (int i = 0; i < tsr.dim(); i++) {
    ss << tsr.size(i); 
    if (i != tsr.dim() - 1)
      ss << ", ";
  }
  ss << ")"; 
  return ss.str(); 
}


TorchBackend::TorchBackend() : m_loaded(0), m_device(CPU), m_use_gpu(false) {
  at::init_num_threads();
}

torch::Tensor TorchBackend::prepare_io_tensor(std::vector<float *> &in_buffer, int in_dim, int n_batches, int in_ratio, int n_vec) {
  // Directly allocate one contiguous tensor of shape [in_dim * n_batches, n_vec]
  // This avoids per-channel .clone() heap thrashing and subsequent torch::cat() allocations!
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto cat_tensor_in = torch::empty({in_dim * n_batches, n_vec}, options);
  float* tensor_data = cat_tensor_in.data_ptr<float>();

  int total_channels = in_dim * n_batches;
  for (int i = 0; i < total_channels; ++i) {
    float* dest = tensor_data + i * n_vec;
    if (i < (int)in_buffer.size() && in_buffer[i] != nullptr) {
      std::memcpy(dest, in_buffer[i], n_vec * sizeof(float));
    } else {
      std::memset(dest, 0, n_vec * sizeof(float));
    }
  }

  // Reshape and permute to desired [n_batches, in_dim, samples]
  cat_tensor_in = cat_tensor_in.reshape({in_dim, n_batches, -1, in_ratio});
  cat_tensor_in = cat_tensor_in.select(-1, -1);
  cat_tensor_in = cat_tensor_in.permute({1, 0, 2});
  return cat_tensor_in;
}

void TorchBackend::perform(std::vector<float *> &in_buffer,
                      std::vector<float *> &out_buffer, 
                      std::string method, 
                      int n_batches, int n_out_channels, int n_vec) {
  c10::InferenceMode guard;

  auto params = get_method_params(method);
  if (!params.size())
    return;

  auto in_dim = params[0];
  auto in_ratio = params[1];
  auto out_dim = params[2];
  auto out_ratio = params[3];

  if (!m_loaded)
    return;

  // COPY BUFFER INTO A SINGLE PRE-ALLOCATED CONTIGUOUS TENSOR (Zero clone / Zero cat)
  auto cat_tensor_in = prepare_io_tensor(in_buffer, in_dim, n_batches, in_ratio, n_vec);

  // SEND TENSOR TO DEVICE
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  cat_tensor_in = cat_tensor_in.to(m_device);
  std::vector<torch::jit::IValue> inputs = {cat_tensor_in};
  auto kwargs = empty_kwargs();

  // PROCESS TENSOR
  at::Tensor tensor_out;
  try {
    tensor_out = m_model.get_method(method)(inputs, kwargs).toTensor();
    tensor_out = tensor_out.repeat_interleave(out_ratio).reshape(
        {n_batches, out_dim, -1});
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return;
  }
  model_lock.unlock();

  int out_batches(tensor_out.size(0)), out_channels(tensor_out.size(1)),
      out_n_vec(tensor_out.size(2));

  tensor_out = tensor_out.contiguous().to(CPU);
  const float* out_data = tensor_out.data_ptr<float>();

  int copy_samples = std::min(n_vec, out_n_vec);
  for (int j = 0; j < n_batches; j++) {
    for (int i = 0; i < n_out_channels; i++) {
      float* dest = out_buffer[j * n_out_channels + i];
      if (dest == nullptr) continue;

      if (i < out_channels) {
        const float* src = out_data + (j * out_channels + i) * out_n_vec;
        std::memcpy(dest, src, copy_samples * sizeof(float));
        if (copy_samples < n_vec) {
          std::memset(dest + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
        }
      } else {
        std::memset(dest, 0, n_vec * sizeof(float));
      }
    }
  }
}

void TorchBackend::perform_forward(std::vector<float *> &in_buffer,
                              std::vector<float *> &out_buffer,
                              int n_batches, int n_out_channels, int n_vec,
                              IBackend::LatentHook latent_hook) {
  if (latent_hook) {
    TorchLatentHook torch_hook = [&latent_hook](at::Tensor z) -> at::Tensor {
      if (!z.defined() || z.numel() == 0) return z;
      auto z_contig = z.contiguous();
      float* ptr = z_contig.data_ptr<float>();
      int64_t channels = z.dim() > 1 ? z.size(1) : 1;
      int64_t length = z.dim() > 2 ? z.size(2) : z.numel();
      IBackend::LatentBuffer buf{ ptr, channels, length };
      latent_hook(buf);
      return z_contig;
    };
    perform_forward_torch(in_buffer, out_buffer, n_batches, n_out_channels, n_vec, torch_hook);
  } else {
    perform_forward_torch(in_buffer, out_buffer, n_batches, n_out_channels, n_vec, nullptr);
  }
}

void TorchBackend::perform_forward(std::vector<float *> &in_buffer,
                                   std::vector<float *> &out_buffer,
                                   int n_batches, int n_out_channels, int n_vec,
                                   TorchLatentHook latent_hook) {
  perform_forward_torch(in_buffer, out_buffer, n_batches, n_out_channels, n_vec, latent_hook);
}

void TorchBackend::perform_forward_torch(std::vector<float *> &in_buffer,
                                   std::vector<float *> &out_buffer,
                                   int n_batches, int n_out_channels, int n_vec,
                                   TorchLatentHook latent_hook) {
  // If no latent hook is active and the model has a native forward method, invoke it directly
  if (!latent_hook && has_method("forward")) {
    perform(in_buffer, out_buffer, "forward", n_batches, n_out_channels, n_vec);
    return;
  }

  // If model has encode and decode, execute encode -> (latent_hook) -> decode
  // so latent bending and autoencoding are fully supported.
  bool has_enc_dec = has_method("encode") && has_method("decode");
  if (has_enc_dec) {
    c10::InferenceMode guard;
    if (!m_loaded) return;

    auto encode_params = get_method_params("encode");
    auto decode_params = get_method_params("decode");
    if (encode_params.empty() || decode_params.empty()) return;

    auto in_dim = encode_params[0];
    auto out_dim = decode_params[2];
    auto out_ratio = decode_params[3];

    // Audio input to encoder must never be temporally decimated (in_ratio = 1)
    auto cat_tensor_in = prepare_io_tensor(in_buffer, in_dim, n_batches, 1, n_vec);

    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    cat_tensor_in = cat_tensor_in.to(m_device);
    std::vector<torch::jit::IValue> encode_inputs = {cat_tensor_in};
    auto kwargs = empty_kwargs();

    at::Tensor latent_tensor;
    try {
      latent_tensor = m_model.get_method("encode")(encode_inputs, kwargs).toTensor();
    } catch (const std::exception &e) {
      std::cerr << "Forward encode error: " << e.what() << '\n';
      return;
    }

    if (latent_hook) {
      try {
        latent_tensor = latent_hook(latent_tensor);
      } catch (const std::exception &e) {
        std::cerr << "Forward latent hook error: " << e.what() << '\n';
      }
    }

    std::vector<torch::jit::IValue> decode_inputs = {latent_tensor};
    at::Tensor tensor_out;
    try {
      tensor_out = m_model.get_method("decode")(decode_inputs, kwargs).toTensor();
      int effective_out_ratio = (out_ratio > 1 && tensor_out.size(-1) >= n_vec) ? 1 : out_ratio;
      tensor_out = tensor_out.repeat_interleave(effective_out_ratio).reshape({n_batches, out_dim, -1});
    } catch (const std::exception &e) {
      std::cerr << "Forward decode error: " << e.what() << '\n';
      return;
    }
    model_lock.unlock();

    int out_batches(tensor_out.size(0)), out_channels(tensor_out.size(1)),
        out_n_vec(tensor_out.size(2));
    tensor_out = tensor_out.contiguous().to(CPU);
    const float* out_data = tensor_out.data_ptr<float>();

    int copy_samples = std::min(n_vec, out_n_vec);
    for (int j = 0; j < n_batches; j++) {
      for (int i = 0; i < n_out_channels; i++) {
        float* dest = out_buffer[j * n_out_channels + i];
        if (dest == nullptr) continue;

        if (i < out_channels) {
          const float* src = out_data + (j * out_channels + i) * out_n_vec;
          std::memcpy(dest, src, copy_samples * sizeof(float));
          if (copy_samples < n_vec) {
            std::memset(dest + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
          }
        } else {
          std::memset(dest, 0, n_vec * sizeof(float));
        }
      }
    }
  } else {
    // Native forward method
    perform(in_buffer, out_buffer, "forward", n_batches, n_out_channels, n_vec);
  }
}

void TorchBackend::perform_prior_decode(std::vector<float *> &out_buffer,
                                  int n_batches, int n_out_channels, int n_vec) {
  c10::InferenceMode guard;
  if (!m_loaded) return;

  std::string prior_method = has_method("prior") ? "prior" : (has_method("generate") ? "generate" : "");
  if (prior_method.empty()) return;

  // If decode exists and prior generates latents, chain prior -> decode
  if (has_method("decode")) {
    auto prior_params = get_method_params(prior_method);
    auto decode_params = get_method_params("decode");
    if (decode_params.empty()) return;

    auto out_dim = decode_params[2];
    auto out_ratio = decode_params[3];

    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto kwargs = empty_kwargs();
    std::vector<torch::jit::IValue> prior_inputs;

    if (!prior_params.empty() && prior_params[0] > 0) {
      int p_in_dim = prior_params[0];
      int p_in_ratio = prior_params[1];
      std::vector<float*> empty_in;
      auto cat_tensor_in = prepare_io_tensor(empty_in, p_in_dim, n_batches, p_in_ratio, n_vec).to(m_device);
      prior_inputs.push_back(cat_tensor_in);
    }

    at::Tensor latent_tensor;
    try {
      latent_tensor = m_model.get_method(prior_method)(prior_inputs, kwargs).toTensor();
    } catch (const std::exception &e) {
      std::cerr << "Prior error: " << e.what() << '\n';
      return;
    }

    std::vector<torch::jit::IValue> decode_inputs = {latent_tensor};
    at::Tensor tensor_out;
    try {
      tensor_out = m_model.get_method("decode")(decode_inputs, kwargs).toTensor();
      tensor_out = tensor_out.repeat_interleave(out_ratio).reshape({n_batches, out_dim, -1});
    } catch (const std::exception &e) {
      std::cerr << "Prior decode error: " << e.what() << '\n';
      return;
    }
    model_lock.unlock();

    int out_batches(tensor_out.size(0)), out_channels(tensor_out.size(1)),
        out_n_vec(tensor_out.size(2));
    tensor_out = tensor_out.contiguous().to(CPU);
    const float* out_data = tensor_out.data_ptr<float>();

    int copy_samples = std::min(n_vec, out_n_vec);
    for (int j = 0; j < n_batches; j++) {
      for (int i = 0; i < n_out_channels; i++) {
        float* dest = out_buffer[j * n_out_channels + i];
        if (dest == nullptr) continue;

        if (i < out_channels) {
          const float* src = out_data + (j * out_channels + i) * out_n_vec;
          std::memcpy(dest, src, copy_samples * sizeof(float));
          if (copy_samples < n_vec) {
            std::memset(dest + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
          }
        } else {
          std::memset(dest, 0, n_vec * sizeof(float));
        }
      }
    }
  } else {
    // If no decode method, call prior directly with perform
    std::vector<float*> empty_in_buf;
    perform(empty_in_buf, out_buffer, prior_method, n_batches, n_out_channels, n_vec);
  }
}

void TorchBackend::perform_decode_gen(std::vector<float *> &out_buffer,
                                int n_batches, int n_out_channels, int n_vec,
                                IBackend::LatentHook latent_hook) {
  if (latent_hook) {
    TorchLatentHook torch_hook = [&latent_hook](at::Tensor z) -> at::Tensor {
      if (!z.defined() || z.numel() == 0) return z;
      auto z_contig = z.contiguous();
      float* ptr = z_contig.data_ptr<float>();
      int64_t channels = z.dim() > 1 ? z.size(1) : 1;
      int64_t length = z.dim() > 2 ? z.size(2) : z.numel();
      IBackend::LatentBuffer buf{ ptr, channels, length };
      latent_hook(buf);
      return z_contig;
    };
    perform_decode_gen_torch(out_buffer, n_batches, n_out_channels, n_vec, torch_hook);
  } else {
    perform_decode_gen_torch(out_buffer, n_batches, n_out_channels, n_vec, nullptr);
  }
}

void TorchBackend::perform_decode_gen(std::vector<float *> &out_buffer,
                                      int n_batches, int n_out_channels, int n_vec,
                                      TorchLatentHook latent_hook) {
  perform_decode_gen_torch(out_buffer, n_batches, n_out_channels, n_vec, latent_hook);
}

void TorchBackend::perform_decode_gen_torch(std::vector<float *> &out_buffer,
                                      int n_batches, int n_out_channels, int n_vec,
                                      TorchLatentHook latent_hook) {
  c10::InferenceMode guard;
  if (!m_loaded || !has_method("decode")) return;

  auto decode_params = get_method_params("decode");
  if (decode_params.size() < 4) return;

  int latent_dim = decode_params[0];
  int latent_ratio = decode_params[1];
  int out_dim = decode_params[2];
  int out_ratio = decode_params[3];

  int min_steps = (latent_ratio <= 640 && latent_dim >= 128) ? 8 : (latent_ratio >= 1024 && latent_dim >= 8 ? 16 : 4);
  int latent_steps = std::max(min_steps, n_vec / std::max(1, latent_ratio));

  at::Tensor latent_tensor;
  std::string prior_method = has_method("prior") ? "prior" : (has_method("generate") ? "generate" : "");

  if (!prior_method.empty()) {
    // If model has a learned prior (RAVE / Darbouka), seed base latents using prior!
    auto prior_params = get_method_params(prior_method);
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto kwargs = empty_kwargs();
    std::vector<torch::jit::IValue> prior_inputs;

    if (!prior_params.empty() && prior_params[0] > 0) {
      int p_in_dim = prior_params[0];
      int p_in_ratio = prior_params[1];
      std::vector<float*> empty_in;
      auto cat_tensor_in = prepare_io_tensor(empty_in, p_in_dim, n_batches, p_in_ratio, n_vec).to(m_device);
      prior_inputs.push_back(cat_tensor_in);
    }

    try {
      latent_tensor = m_model.get_method(prior_method)(prior_inputs, kwargs).toTensor().to(m_device);
    } catch (...) {
      latent_tensor = at::Tensor();
    }
    model_lock.unlock();
  }

  // Fallback: If no prior or prior failed, synthesize base latent buffer with calibrated Gaussian excitation
  // EnCodec SEANet operates at sigma ~ 8.5 to 10.0; VAEs (Stable Audio, RAVE) operate at unit normal sigma ~ 1.0
  if (!latent_tensor.defined() || latent_tensor.numel() == 0) {
    float latent_scale = (latent_dim >= 128 && latent_ratio <= 640) ? 8.5f : 1.0f;
    latent_tensor = torch::randn({n_batches, latent_dim, latent_steps},
                                 torch::TensorOptions().dtype(torch::kFloat32).device(m_device)) * latent_scale;
  }

  // Apply latent hook (2D Latent Vector Pad, Drift, Harmonics, RFF Orbit)
  if (latent_hook) {
    try {
      latent_tensor = latent_hook(latent_tensor);
    } catch (const std::exception &e) {
      std::cerr << "Decode Gen latent hook error: " << e.what() << '\n';
    }
  }

  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  auto kwargs = empty_kwargs();
  std::vector<torch::jit::IValue> decode_inputs = {latent_tensor.to(m_device)};
  at::Tensor tensor_out;
  try {
    tensor_out = m_model.get_method("decode")(decode_inputs, kwargs).toTensor();
    int effective_out_ratio = (out_ratio > 1 && tensor_out.size(-1) >= n_vec) ? 1 : out_ratio;
    tensor_out = tensor_out.repeat_interleave(effective_out_ratio).reshape({n_batches, out_dim, -1});
  } catch (const std::exception &e) {
    std::cerr << "Decode Gen decode error: " << e.what() << '\n';
    return;
  }
  model_lock.unlock();

  int out_batches(tensor_out.size(0)), out_channels(tensor_out.size(1)),
      out_n_vec(tensor_out.size(2));
  tensor_out = tensor_out.contiguous().to(CPU);
  const float* out_data = tensor_out.data_ptr<float>();

  int copy_samples = std::min(n_vec, out_n_vec);
  for (int j = 0; j < n_batches; j++) {
    for (int i = 0; i < n_out_channels; i++) {
      float* dest = out_buffer[j * n_out_channels + i];
      if (dest == nullptr) continue;

      if (i < out_channels) {
        const float* src = out_data + (j * out_channels + i) * out_n_vec;
        std::memcpy(dest, src, copy_samples * sizeof(float));
        if (copy_samples < n_vec) {
          std::memset(dest + copy_samples, 0, (n_vec - copy_samples) * sizeof(float));
        }
      } else {
        std::memset(dest, 0, n_vec * sizeof(float));
      }
    }
  }
}

bool TorchBackend::has_prior_decode() {
  return (has_method("prior") || has_method("generate")) && has_method("decode");
}

bool TorchBackend::has_autoencode() {
  return has_method("encode") && has_method("decode");
}

std::vector<std::string> TorchBackend::get_plugin_modes() {
  std::vector<std::string> modes;
  if (!is_loaded())
    return modes;

  if (has_method("forward") || (has_method("encode") && has_method("decode"))) {
    modes.push_back("forward");
  }
  if (has_method("decode")) {
    modes.push_back("decode gen");
  }
  if (has_method("prior")) {
    modes.push_back("prior");
  } else if (has_method("generate")) {
    modes.push_back("generate");
  }

  // If none of the known modes match, fallback to available methods
  if (modes.empty()) {
    for (const auto &m : m_available_methods) {
      modes.push_back(m);
    }
  }

  return modes;
}

std::vector<int> TorchBackend::get_mode_params(std::string mode) {
  if (mode == "forward") {
    if (has_method("forward")) {
      return get_method_params("forward");
    } else if (has_method("encode") && has_method("decode")) {
      auto ep = get_method_params("encode");
      auto dp = get_method_params("decode");
      if (ep.size() >= 4 && dp.size() >= 4) {
        return { ep[0], ep[1], dp[2], dp[3] };
      }
    }
  }
  if ((mode == "decode gen" || mode == "prior" || mode == "generate") && has_method("decode")) {
    auto dp = get_method_params("decode");
    if (dp.size() >= 4) {
      // 0 inputs, audio out channels from decode
      return { 0, 1, dp[2], dp[3] };
    }
  }
  return get_method_params(mode);
}

int TorchBackend::load(std::string path, double sampleRate, const std::string* target_method) {
  try {
    auto model = torch::jit::load(path);
    if (target_method != nullptr) {
      if (!(*target_method).empty()) {
        // if target_method is not null, check if loaded model has it
        auto locked_model = LockedModel(); 
        locked_model.model = &model;  
        auto methods = get_available_methods(&locked_model);
        if (std::find(methods.begin(), methods.end(), *target_method) == methods.end()) {
          throw "path " + path + " does not contain target method " + (*target_method); 
        }
      }
    }
    model.eval();
    model.to(m_device);

    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    m_model = model;
    model_lock.unlock();

    m_available_methods = get_available_methods();
    m_buffer_attributes = retrieve_buffer_attributes();
    m_path = path;
    set_sample_rate(sampleRate);
    m_loaded = 1;

    // Cache original weights and parameter tensor handles for bending / reset
    m_original_weights.clear();
    m_parameter_cache.clear();
    for (const auto &layer : m_model.named_parameters()) {
      m_parameter_cache[layer.name] = layer.value;
      auto t = layer.value.contiguous().to(CPU);
      m_original_weights[layer.name] = std::vector<float>(
          t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
    }

    m_last_error.clear();
    return 0;
  } catch (const std::exception &e) {
    m_last_error = e.what();
    throw "problem loading model " + path + ". Exception : " + e.what();
  } catch (const std::string &s) {
    m_last_error = s;
    throw;
  } catch (...) {
    m_last_error = "Unknown error loading TorchScript model.";
    throw;
  }
}

void TorchBackend::set_sample_rate(double sampleRate) {
  m_sr = sampleRate;
  if (has_method("set_sample_rate")) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto args = empty_args(); args.push_back((int)sampleRate);
    auto kwargs = empty_kwargs();
    std::string method_name = "set_sample_rate";
    try {
      m_model.get_method(method_name)(args, kwargs);
    } catch (...) {}
    model_lock.unlock();
  } 
}

double TorchBackend::get_sample_rate() {
  if (has_method("get_sample_rate")) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto args = empty_args();
    auto kwargs = empty_kwargs();
    try {
      auto res = m_model.get_method("get_sample_rate")(args, kwargs);
      model_lock.unlock();
      if (res.isInt()) {
        int sr = res.toInt();
        if (sr > 0) return (double)sr;
      } else if (res.isDouble()) {
        double sr = res.toDouble();
        if (sr > 0) return sr;
      }
    } catch (...) {
      model_lock.unlock();
    }
  }
  try {
    for (const auto& buf : m_model.named_buffers()) {
      if (buf.name.find("sampling_rate") != std::string::npos || buf.name.find("sample_rate") != std::string::npos) {
        if (buf.value.numel() == 1) {
          int64_t sr = buf.value.item<int64_t>();
          if (sr > 0) return (double)sr;
        }
      }
    }
  } catch (...) {}
  return m_sr > 0 ? m_sr : 44100.0;
}

int TorchBackend::reload() {
  auto return_code = load(m_path, m_sr);
  return return_code;
}

bool TorchBackend::has_method(std::string method_name) {
  if (!is_loaded()){
    return false; 
  }
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  for (const auto &m : m_model.get_methods()) {
    if (m.name() == method_name)
      return true;
  }
  return false;
}

bool TorchBackend::has_settable_attribute(std::string attribute) {
  for (const auto &a : get_settable_attributes()) {
    if (a == attribute)
      return true;
  }
  return false;
}

std::vector<std::string> TorchBackend::get_available_methods(LockedModel *target_model) {
  std::vector<std::string> methods;
  torch::jit::script::Module *model;
  std::mutex *mutex; 
  if (target_model == nullptr) {
    model = &m_model;
    mutex = &m_model_mutex;    
  } else {
    model = target_model->model;
    mutex = &target_model->mutex;
  }
  try {
    std::vector<c10::IValue> dumb_input = {};
    std::unique_lock<std::mutex> model_lock(*mutex);
    auto methods_from_model =
        model->get_method("get_methods")(dumb_input).toList();
    model_lock.unlock();

    for (int i = 0; i < methods_from_model.size(); i++) {
      methods.push_back(methods_from_model.get(i).toStringRef());
    }
  } catch (...) {
    std::unique_lock<std::mutex> model_lock(*mutex);
    for (const auto &m : model->get_methods()) {
      try {
        auto method_params = model->attr(m.name() + "_params");
        methods.push_back(m.name());
      } catch (...) {
      }
    }
    model_lock.unlock();
  }
  return methods;
}

std::vector<std::string> TorchBackend::get_available_attributes() {
  std::vector<std::string> attributes;
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  for (const auto &attribute : m_model.named_attributes())
    attributes.push_back(attribute.name);
  return attributes;
}

std::vector<std::string> TorchBackend::get_settable_attributes() {
  std::vector<std::string> attributes;
  try {
    std::vector<c10::IValue> dumb_input = {};
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto methods_from_model =
        m_model.get_method("get_attributes")(dumb_input).toList();
    model_lock.unlock();
    for (int i = 0; i < methods_from_model.size(); i++) {
      auto attr_name = methods_from_model.get(i).toStringRef();
      if (attr_name != "none") {
        attributes.push_back(attr_name);
      }
    }
  } catch (...) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    for (const auto &a : m_model.named_attributes()) {
      try {
        auto method_params = m_model.attr(a.name + "_params");
        if (a.name != "none") {
          attributes.push_back(a.name);
        }
      } catch (...) {
      }
    }
    model_lock.unlock();
  }
  return attributes;
}

std::vector<c10::IValue> TorchBackend::get_attribute(std::string attribute_name) {
  std::string attribute_getter_name = "get_" + attribute_name;
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  try {
    auto attribute_getter = m_model.get_method(attribute_getter_name);
  } catch (...) {
    model_lock.unlock();
    throw "getter for attribute " + attribute_name + " not found in model";
  }
  std::vector<c10::IValue> getter_inputs = {}, attributes;
  auto kwargs = empty_kwargs();
  try {
    try {
      auto output_list = m_model.get_method(attribute_getter_name)(getter_inputs, kwargs).toList();
      attributes = output_list.vec();
    } catch (...) {
      auto output_tuple_ref =
          m_model.get_method(attribute_getter_name)(getter_inputs, kwargs).toTuple();
      attributes = (*output_tuple_ref.get()).elements();
    }
  } catch (...) {
    model_lock.unlock();
    throw ("could not access method " + attribute_getter_name + " from original script.");
  }
  model_lock.unlock();
  return attributes;
}

std::string TorchBackend::get_attribute_as_string(std::string attribute_name) {
  std::vector<c10::IValue> getter_outputs = get_attribute(attribute_name);
  // finstringd arguments
  torch::Tensor setter_params;
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  try {
    setter_params = m_model.attr(attribute_name + "_params").toTensor();
    model_lock.unlock();
  } catch (...) {
    model_lock.unlock();
    throw "parameters to get attribute " + attribute_name +
        " not found in model";
  }
  std::string current_attr = "";
  for (int i = 0; i < setter_params.size(0); i++) {
    int current_id = setter_params[i].item().toInt();
    switch (current_id) {
    // bool case
    case 0: {
      current_attr += (getter_outputs[i].toBool()) ? "true" : "false";
      break;
    }
    // int case
    case 1: {
      current_attr += std::to_string(getter_outputs[i].toInt());
      break;
    }
    // float case
    case 2: {
      float result = getter_outputs[i].to<float>();
      current_attr += std::to_string(result);
      break;
    }
    // str case
    case 3: {
      current_attr += getter_outputs[i].toStringRef();
      break;
    }
    // tensor case
    case 4: {
      auto tensor = getter_outputs[i].toTensor();
      current_attr += tensor_to_str(tensor);
      break; 
    }
    case 5: {
      current_attr += getter_outputs[i].toStringRef();
      break;
    }
    default: {
      throw "bad type id : " + std::to_string(current_id) + " at index " +
          std::to_string(i);
      break;
    }
    }
    if (i < setter_params.size(0) - 1)
      current_attr += " ";
  }
  return current_attr;
}


void TorchBackend::set_attribute(std::string attribute_name,
                            std::vector<std::string> attribute_args, 
                            const TorchBackend::BufferMap &buffer_array) {
  // find setter
  std::string attribute_setter_name = "set_" + attribute_name;
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  try {
    auto attribute_setter = m_model.get_method(attribute_setter_name);
  } catch (...) {
    model_lock.unlock();
    throw "setter for attribute " + attribute_name + " not found in model";
  }
  // find arguments
  torch::Tensor setter_params;
  try {
    setter_params = m_model.attr(attribute_name + "_params").toTensor();
  } catch (...) {
    model_lock.unlock();
    throw "parameters to set attribute " + attribute_name +
        " not found in model";
  }
  model_lock.unlock();

  // check attribute params
  int setter_elements = setter_params.numel();
  if (attribute_args.size() != setter_elements) {
    std::stringstream exception;
    exception << "wrong number of elements : setter seems to have " << std::to_string(setter_params.numel());
    exception << ", but " << attribute_args.size() << " was given";
    throw exception.str();
  }

  // process inputs
  std::vector<c10::IValue> setter_inputs = {};
  auto n_args = setter_params.size(0);
  for (int i = 0; i < n_args; i++) {
    int current_id = setter_params[i].item().toInt();
    switch (current_id) {
      // bool case
      case 0:
        setter_inputs.push_back(c10::IValue(to_bool(attribute_args[i])));
        break;
      // int case
      case 1:
        setter_inputs.push_back(c10::IValue(to_int(attribute_args[i])));
        break;
      // float case
      case 2:
        setter_inputs.push_back(c10::IValue(to_float(attribute_args[i])));
        break;
      // str case
      case 3:
        setter_inputs.push_back(c10::IValue(attribute_args[i]));
        break;
      // tensor case
      case 4: {
        auto buffer_name = get_buffer_name(attribute_name, i);
        if (!buffer_array.contains(buffer_name)) {
          throw std::string("buffer for argument ") + buffer_name + std::string(" not found. Did you initialise it?");
        } else {
          auto buffer = buffer_array.at(buffer_name);
          auto buf_tensor = buffer.to_tensor().index({0});
          setter_inputs.push_back(buf_tensor.clone()); 
        }
        break;
      }
      case 5: {
        auto buffer_name = get_buffer_name(attribute_name, i);
        if (!buffer_array.contains(buffer_name)) {
          throw std::string("buffer for argument ") + buffer_name + std::string(" not found. Did you initialise it?");
        } else {
          auto buffer = buffer_array.at(buffer_name);
          update_buffer(buffer_name, buffer); 
          setter_inputs.push_back(c10::IValue(buffer_name));
        }
        break;
      }
      default:
        throw "bad type id : " + std::to_string(current_id) + "at index " +
            std::to_string(i);
        break;
    }
  }
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto kwargs = empty_kwargs();
    auto setter_out = m_model.get_method(attribute_setter_name)(setter_inputs, kwargs);
    model_lock.unlock();
    int setter_result = setter_out.toInt();
    if (setter_result != 0) {
      throw "setter returned -1";
    }
  } catch (std::string &e) {
    throw "setter for " + attribute_name + " failed : " + e;
  } catch (std::exception &e) {
    throw "setter for " + attribute_name + " failed : " + e.what();
  } catch (...) {
    throw "setter for " + attribute_name + " failed";
  }
}

std::vector<int> TorchBackend::get_method_params(std::string method) {
  std::vector<int> params;

  if (std::find(m_available_methods.begin(), m_available_methods.end(),
                method) != m_available_methods.end()) {
    try {
      std::unique_lock<std::mutex> model_lock(m_model_mutex);
      auto p = m_model.attr(method + "_params").toTensor();
      model_lock.unlock();
      for (int i(0); i < 4; i++)
        params.push_back(p[i].item().to<int>());
    } catch (...) {
    }
  }
  return params;
}

int TorchBackend::get_higher_ratio() {
  int higher_ratio = 1;
  for (const auto &method : m_available_methods) {
    auto params = get_method_params(method);
    if (!params.size())
      continue; // METHOD NOT USABLE, SKIPPING
    int max_ratio = std::max(params[1], params[3]);
    higher_ratio = std::max(higher_ratio, max_ratio);
  }
  return higher_ratio;
}

bool TorchBackend::is_loaded() { return m_loaded; }

void TorchBackend::use_gpu(bool value) {
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  if (value) {
    if (torch::hasCUDA()) {
      std::cout << "sending model to cuda" << std::endl;
      m_device = CUDA;
    } else if (torch::hasMPS()) {
      std::cout << "sending model to mps" << std::endl;
      m_device = MPS;
    } else {
      std::cout << "sending model to cpu" << std::endl;
      m_device = CPU;
    }
  } else {
    m_device = CPU;
  }
  m_model.to(m_device);
}

std::string TorchBackend::get_buffer_name(std::string attribute_name, int attribute_elt_idx) {
  return attribute_name + "#" + std::to_string(attribute_elt_idx);
}

bool TorchBackend::is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx) {
  std::string target_buffer_name = get_buffer_name(attribute_name, attribute_elt_idx);
  // std::cout << "current registered buffers : " << m_buffer_attributes.size() << std::endl; 
  for (auto buffer_name: m_buffer_attributes) {
    // std::cout << "target buffer name : " << target_buffer_name << "; compared buffer name : " << buffer_name << std::endl;
    if (buffer_name == target_buffer_name) {
      return true;
    }
  }
  return false;
}

bool TorchBackend::is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx) {
  auto attribute_params = m_model.attr(attribute_name + "_params").toTensor(); 
  if (id_to_string_hash.at(attribute_params.item().toInt()) == "tensor") {
    return true;
  } else {
    return false;
  }
}

std::vector<std::string> TorchBackend::retrieve_buffer_attributes() {
  std::vector<std::string> bufferNames = {};
  std::string getter_name = "get_buffer_attributes";
  if (has_method(getter_name)) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    try {
      auto model_method = m_model.get_method(getter_name);
      auto model_out = model_method(empty_args(), empty_kwargs()).toList().vec();
      for (int i = 0; i < model_out.size(); i++) {
        // std::cout << "adding " << model_out[i].toStringRef() << " to buffer list" << std::endl; 
        bufferNames.push_back(model_out[i].toStringRef());
      }
      model_lock.unlock();
    } catch (std::exception& e) {
      model_lock.unlock();
      throw e;
    }
  } 
  return bufferNames;
}

std::vector<std::string> TorchBackend::get_buffer_attributes() {
  return m_buffer_attributes;
}

int TorchBackend::reset_buffer(std::string buffer_name) {
  std::string clear_method = "clear_buffer_attribute";
  auto args = empty_args(); auto kwargs = empty_kwargs();
  args.push_back(buffer_name);
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  auto model_out = m_model.get_method(clear_method)(args, kwargs);
  int result = model_out.toInt();
  return result;  
}

int TorchBackend::update_buffer(std::string buffer_name, StaticBuffer<float> &buffer) {
  std::string setter_method = "set_buffer_attribute";
  auto args = empty_args(); auto kwargs = empty_kwargs();
  auto tensor = buffer.to_tensor().to(m_device); auto sr = static_cast<int>(buffer.sr());
  // auto tensor_min = tensor.min(); auto tensor_max = tensor.max(); 
  auto tensor_size = {tensor.size(0), tensor.size(1)};
  args.push_back(buffer_name); args.push_back(tensor); args.push_back(sr);
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  auto model_out = m_model.get_method(setter_method)(args, kwargs);
  model_lock.unlock();
  int result = model_out.toInt();
  return result; 
}

ModelInfo TorchBackend::get_model_info() {
  if (!m_loaded) {
    throw "no model loaded; cannot retrieve model information";
  }

  auto methods = get_available_methods();
  auto attributes = get_settable_attributes();
  auto method_dict = ModelInfo::MethodDict();
  auto attribute_dict = ModelInfo::AttributeDict();
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  model_lock.unlock();
  try {
    for (auto method : methods) {
      auto method_properties = MethodProperties();
      model_lock.lock();
      auto setter_params = m_model.attr(method + "_params").toTensor();
      model_lock.unlock();
      method_properties.name = method; 
      method_properties.channels_in = setter_params[0].item().toInt(); 
      method_properties.ratio_in = setter_params[1].item().toInt(); 
      method_properties.channels_out = setter_params[2].item().toInt(); 
      method_properties.ratio_out = setter_params[3].item().toInt(); 
      method_dict.insert({method, method_properties});
    }
    for (auto attribute: attributes) {
      model_lock.lock();
      auto attribute_params = m_model.attr(attribute + "_params").toTensor();
      model_lock.unlock();
      auto attribute_properties = AttributeProperties();
      attribute_properties.name = attribute;
      auto attribute_types = std::vector<std::string>();
      for (int i = 0; i < attribute_params.size(0); i++) {
        auto param_idx = attribute_params[i].item().toInt();
        auto param_str = this->id_to_string_hash.at(param_idx);
        attribute_types.push_back(param_str);
      }
      attribute_properties.attribute_types = attribute_types;
      attribute_dict.insert({attribute, attribute_properties});
    }
    auto model_info = ModelInfo();
    model_info.method_properties = method_dict;
    model_info.attribute_properties = attribute_dict;
    return model_info;
  } catch (std::exception& e) {
    model_lock.unlock();
    throw "error fetching model information for model " + m_path + ". Caught error : " + e.what();
  }
}

std::vector<std::string> TorchBackend::get_available_layers() {
  std::vector<std::string> layers;
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  for (const auto &layer : m_model.named_parameters())
    layers.push_back(layer.name);
  model_lock.unlock();
  return layers;
}


std::vector<float> TorchBackend::get_layer_weights(std::string layer_name) {
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  torch::Tensor m_tensor;
  for (const auto &layer : m_model.named_parameters())
    if (layer.name == layer_name)
      m_tensor = layer.value.contiguous().to(CPU);
  model_lock.unlock();

  auto m_weights = std::vector(m_tensor.data_ptr<float>(), m_tensor.data_ptr<float>()+m_tensor.numel());
  return m_weights;
}

std::vector<int64_t> TorchBackend::get_layer_shape(std::string layer_name) {
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  std::vector<int64_t> shape;
  for (const auto &layer : m_model.named_parameters()) {
    if (layer.name == layer_name) {
      auto sizes = layer.value.sizes();
      shape.assign(sizes.begin(), sizes.end());
      break;
    }
  }
  return shape;
}

void TorchBackend::set_layer_weights(std::string layer_name,
                                const std::vector<float>& weights) {
  std::unique_lock<std::mutex> model_lock(m_model_mutex);

  // O(1) parameter lookup using cached tensor references
  auto it = m_parameter_cache.find(layer_name);
  if (it != m_parameter_cache.end()) {
    torch::NoGradGuard no_grad;
    it->second.copy_(torch::from_blob(const_cast<float*>(weights.data()), it->second.sizes()));
    return;
  }

  // Fallback linear search in case cache was bypassed
  for (const auto &layer : m_model.named_parameters()) {
    if (layer.name == layer_name) {
      torch::NoGradGuard no_grad;
      layer.value.copy_(torch::from_blob(const_cast<float*>(weights.data()), layer.value.sizes()));
      m_parameter_cache[layer_name] = layer.value;
      break;
    }
  }
}

std::vector<float> TorchBackend::get_original_layer_weights(std::string layer_name) {
  auto it = m_original_weights.find(layer_name);
  if (it != m_original_weights.end()) {
    return it->second;
  }
  return get_layer_weights(layer_name);
}

void TorchBackend::reset_layer_weights(std::string layer_name) {
  auto it = m_original_weights.find(layer_name);
  if (it != m_original_weights.end()) {
    set_layer_weights(layer_name, it->second);
  }
}

void TorchBackend::reset_all_layer_weights() {
  for (const auto &pair : m_original_weights) {
    set_layer_weights(pair.first, pair.second);
  }
}

int TorchBackend::save_model(std::string path) {
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  try {
    m_model.save(path);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Failed to save model: " << e.what() << '\n';
    return -1;
  }
}