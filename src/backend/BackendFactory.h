#pragma once
#include "IBackend.h"
#include "backend.h"
#include "OnnxBackend.h"
#include "ExecuTorchBackend.h"
#include <memory>
#include <string>
#include <algorithm>

/**
 * BackendFactory
 * Responsible for instantiating the appropriate neural inference backend
 * based on model file extension (.pte -> ExecuTorchBackend, .onnx -> OnnxBackend, .ts -> TorchBackend).
 */
class BackendFactory {
public:
  static auto create_for_file(const std::string& path) -> std::unique_ptr<IBackend> {
    std::string ext = "";
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos != std::string::npos) {
      ext = path.substr(dot_pos);
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    if (ext == ".pte") {
      return std::make_unique<ExecuTorchBackend>();
    }
    if (ext == ".onnx") {
      return std::make_unique<OnnxBackend>();
    }

    // Default to TorchScript backend (.ts or fallback)
    return std::make_unique<TorchBackend>();
  }

  static auto create_default() -> std::unique_ptr<IBackend> {
    return std::make_unique<TorchBackend>();
  }

  // Backward compatibility aliases
  static auto createForFile(const std::string& path) -> std::unique_ptr<IBackend> {
    return create_for_file(path);
  }

  static auto createDefault() -> std::unique_ptr<IBackend> {
    return create_default();
  }
};
