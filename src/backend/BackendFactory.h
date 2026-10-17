#pragma once
#include "IBackend.h"
#include "backend.h"
#include <memory>
#include <string>
#include <algorithm>

/**
 * BackendFactory
 * Responsible for instantiating the appropriate neural inference backend
 * based on model file extension (.ts -> TorchBackend, .onnx -> OnnxBackend).
 */
class BackendFactory {
public:
  static std::unique_ptr<IBackend> createForFile(const std::string& path) {
    std::string ext = "";
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos != std::string::npos) {
      ext = path.substr(dot_pos);
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    if (ext == ".onnx") {
      // Phase 3 will return std::make_unique<OnnxBackend>()
      // Fallback or placeholder until OnnxBackend is compiled:
      return nullptr;
    }

    // Default to TorchScript backend (.ts or fallback)
    return std::make_unique<TorchBackend>();
  }

  static std::unique_ptr<IBackend> createDefault() {
    return std::make_unique<TorchBackend>();
  }
};
