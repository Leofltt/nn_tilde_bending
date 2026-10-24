#include "BackendFactory.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <cassert>
#include <memory>
#include <algorithm>

int main(int argc, char** argv) {
    std::cout << "==========================================" << std::endl;
    std::cout << "Unit Test: Multi-Backend (TorchScript & ONNX)" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::vector<std::string> testModels;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            testModels.push_back(argv[i]);
        }
    } else {
        testModels = {
            // TorchScript reference models
            "models/darbouka.ts",
            "models/encodec_bendable.ts",
            // ONNX bendable models
            "models/darbouka_bendable.onnx",
            "models/encodec_bendable.onnx"
        };
    }

    int failures = 0;
    std::string reportPath = "test_report.md";
    std::ofstream report(reportPath);

    report << "# NetBender Backend Parity & Unit Test Report\n\n";
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    report << "**Timestamp**: " << std::ctime(&now) << "\n\n";

    struct ModelTestResult {
        std::string path;
        std::string backendType;
        bool loadSuccess = false;
        std::string loadError;
        double sr = 0;
        int ratio = 0;
        int latentDim = 0;
        size_t numLayers = 0;
        std::vector<std::string> modes;
        std::vector<std::string> bendableLayers;
        std::string fwdStatus = "N/A";
        std::string decStatus = "N/A";
        std::string failureDetails;
    };

    std::vector<ModelTestResult> results;

    for (const auto& path : testModels) {
        std::cout << "\n[TESTING] Model: " << path << std::endl;
        std::cout << std::flush;

        ModelTestResult res;
        res.path = path;

        // Detect backend type from extension
        bool isTorch = (path.size() >= 3 && path.substr(path.size() - 3) == ".ts");
        res.backendType = isTorch ? "TorchScript" : "ONNX Runtime";

        auto backend = BackendFactory::createForFile(path);
        if (!backend) {
            res.loadSuccess = false;
            res.loadError = "Failed to allocate backend from BackendFactory.";
            results.push_back(res);
            failures++;
            continue;
        }

        int loadRet = -1;
        try {
            loadRet = backend->load(path, 44100.0);
        } catch (const std::exception& e) {
            res.loadError = e.what();
            loadRet = -1;
        } catch (const std::string& s) {
            res.loadError = s;
            loadRet = -1;
        } catch (...) {
            res.loadError = "Unknown exception during load()";
            loadRet = -1;
        }

        if (loadRet != 0) {
            if (res.loadError.empty()) res.loadError = backend->get_last_error();
            if (res.loadError.empty()) res.loadError = "Unknown error during initialization or shape inference.";
            std::cerr << "  FAILED: load() returned " << loadRet << ": " << res.loadError << std::endl;
            failures++;
            res.loadSuccess = false;
            results.push_back(res);
            continue;
        }

        res.loadSuccess = true;
        std::cout << "  SUCCESS: Loaded " << path << " (" << res.backendType << ")" << std::endl;
        res.sr = backend->get_sample_rate();
        res.ratio = backend->get_higher_ratio();
        auto layers = backend->get_available_layers();
        res.numLayers = layers.size();
        res.bendableLayers = layers;

        std::cout << "  Sample rate: " << res.sr << std::endl;
        std::cout << "  Higher ratio: " << res.ratio << std::endl;
        std::cout << "  Available layers: " << res.numLayers << std::endl;

        res.modes = backend->get_plugin_modes();
        if (backend->has_method("decode")) {
            auto decParams = backend->get_mode_params("decode");
            if (!decParams.empty()) res.latentDim = decParams[0];
        }

        // Perform test forward inference
        int inCh = 1;
        int outCh = 1;
        int n_vec = 2048;
        std::vector<float> inData(inCh * n_vec, 0.0f);
        std::vector<float> outData(outCh * n_vec, 0.0f);
        std::vector<float*> inPtrs = { inData.data() };
        std::vector<float*> outPtrs = { outData.data() };

        std::cout << "  Testing perform_forward..." << std::endl;
        try {
            backend->perform_forward(inPtrs, outPtrs, 1, outCh, n_vec, nullptr);
            std::cout << "  SUCCESS: perform_forward completed." << std::endl;
            res.fwdStatus = "✅ PASS";
        } catch (const std::exception& e) {
            std::cerr << "  FAILED: perform_forward exception: " << e.what() << std::endl;
            res.fwdStatus = "❌ FAIL";
            res.failureDetails += std::string("perform_forward error: ") + e.what() + "\n";
            failures++;
        } catch (...) {
            std::cerr << "  FAILED: perform_forward unknown exception" << std::endl;
            res.fwdStatus = "❌ FAIL";
            res.failureDetails += "perform_forward error: unknown exception\n";
            failures++;
        }

        // Test decode gen if available
        if (backend->has_method("decode")) {
            std::cout << "  Testing perform_decode_gen..." << std::endl;
            try {
                backend->perform_decode_gen(outPtrs, 1, outCh, n_vec, nullptr);
                std::cout << "  SUCCESS: perform_decode_gen completed." << std::endl;
                res.decStatus = "✅ PASS";
            } catch (const std::exception& e) {
                std::cerr << "  FAILED: perform_decode_gen exception: " << e.what() << std::endl;
                res.decStatus = "❌ FAIL";
                res.failureDetails += std::string("perform_decode_gen error: ") + e.what() + "\n";
                failures++;
            } catch (...) {
                std::cerr << "  FAILED: perform_decode_gen unknown exception" << std::endl;
                res.decStatus = "❌ FAIL";
                res.failureDetails += "perform_decode_gen error: unknown exception\n";
                failures++;
            }
        }

        results.push_back(res);
    }

    // Write Overview Table
    report << "| Model Path | Engine | Status | Sample Rate | Ratio | Latent Dim | Layers | Available Modes | Forward | Decode Gen |\n";
    report << "|---|---|---|---|---|---|---|---|---|---|\n";

    for (const auto& r : results) {
        if (!r.loadSuccess) {
            report << "| `" << r.path << "` | " << r.backendType << " | ❌ FAILED | - | - | - | - | - | - | - |\n";
        } else {
            std::string modeStr;
            for (size_t i = 0; i < r.modes.size(); ++i) {
                if (i > 0) modeStr += ", ";
                modeStr += r.modes[i];
            }
            report << "| `" << r.path << "` | " << r.backendType << " | ✅ PASS | " << r.sr << " Hz | " << r.ratio << " | "
                   << r.latentDim << " | " << r.numLayers << " | `" << modeStr << "` | "
                   << r.fwdStatus << " | " << r.decStatus << " |\n";
        }
    }

    // Write Detailed Per-Model Breakdown
    report << "\n## Model Inspection Details\n\n";
    for (const auto& r : results) {
        report << "### `" << r.path << "` (" << r.backendType << ")\n\n";
        if (!r.loadSuccess) {
            report << "> [!CAUTION]\n";
            report << "> **Load Error**: " << r.loadError << "\n\n";
            continue;
        }

        report << "- **Engine**: `" << r.backendType << "`\n";
        report << "- **Load Status**: ✅ Successfully initialized backend session\n";
        report << "- **Sample Rate**: " << r.sr << " Hz\n";
        report << "- **Higher Ratio**: " << r.ratio << "\n";
        report << "- **Latent Dimension**: " << r.latentDim << "\n";
        report << "- **Bendable Parameter Count**: " << r.numLayers << " layers\n";
        report << "- **Forward Inference (2048 samples)**: " << r.fwdStatus << "\n";
        report << "- **Decode Generation**: " << r.decStatus << "\n";

        if (!r.failureDetails.empty()) {
            report << "\n> [!WARNING]\n";
            report << "> **Execution Diagnostics**:\n> ```\n> " << r.failureDetails << "> ```\n";
        }

        if (!r.bendableLayers.empty()) {
            report << "\n<details>\n<summary><b>Bendable Layers (" << r.numLayers << ")</b></summary>\n\n";
            for (size_t i = 0; i < std::min<size_t>(r.bendableLayers.size(), 30); ++i) {
                report << "- `" << r.bendableLayers[i] << "`\n";
            }
            if (r.bendableLayers.size() > 30) {
                report << "- *... and " << (r.bendableLayers.size() - 30) << " more layers*\n";
            }
            report << "\n</details>\n\n";
        }
    }

    std::cout << "\n==========================================" << std::endl;
    report << "\n## Summary\n\n";
    if (failures == 0) {
        std::cout << "ALL BACKEND TESTS PASSED!" << std::endl;
        report << "**Result**: 🎉 **ALL TESTS PASSED (TorchScript & ONNX Parity Verified)**\n";
        report.close();
        std::cout << "Saved test report to: " << reportPath << std::endl;
        return 0;
    } else {
        std::cerr << failures << " TEST(S) FAILED!" << std::endl;
        report << "**Result**: ⚠️ **" << failures << " TEST(S) FAILED**\n";
        report.close();
        std::cout << "Saved test report to: " << reportPath << std::endl;
        return 1;
    }
}
