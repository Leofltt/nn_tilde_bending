#include "BackendFactory.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <cassert>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iomanip>

int main(int argc, char** argv) {
    std::cout << "==========================================" << std::endl;
    std::cout << "Unit Test: Multi-Backend Audio & Sentry Parity" << std::endl;
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
            "models/speech.ts",
            "models/stable_audio_1_vae.ts",
            // ONNX bendable models
            "models/darbouka_bendable.onnx",
            "models/encodec_bendable.onnx",
            "models/speech_bendable.onnx",
            "models/stable_audio_1_vae_bendable.onnx"
        };
    }

    int failures = 0;
    std::string reportPath = "test_report.md";
    std::ofstream report(reportPath);

    report << "# NetBender Audio Parity & Sentry Verification Report\n\n";
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
        
        // Audio Quality & Sentry Status
        std::string fwdStatus = "N/A";
        float fwdPeak = 0.0f;
        float fwdRms = 0.0f;
        bool fwdFuseTripped = false;
        bool fwdHasNan = false;

        std::string decStatus = "N/A";
        float decPeak = 0.0f;
        float decRms = 0.0f;
        bool decFuseTripped = false;
        bool decHasNan = false;

        std::string priorStatus = "N/A";
        float priorPeak = 0.0f;
        float priorRms = 0.0f;
        bool priorFuseTripped = false;
        bool priorHasNan = false;

        std::string failureDetails;
    };

    std::vector<ModelTestResult> results;

    for (const auto& path : testModels) {
        std::cout << "\n[TESTING] Model: " << path << std::endl;
        std::cout << std::flush;

        ModelTestResult res;
        res.path = path;

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

        res.modes = backend->get_plugin_modes();
        if (backend->has_method("decode")) {
            auto decParams = backend->get_mode_params("decode");
            if (!decParams.empty()) res.latentDim = decParams[0];
        }

        int inCh = 1;
        int outCh = 1;
        if (backend->has_method("forward")) {
            auto fwdParams = backend->get_mode_params("forward");
            if (fwdParams.size() >= 3) {
                inCh = std::max(1, fwdParams[0]);
                outCh = std::max(1, fwdParams[2]);
            }
        } else if (backend->has_method("decode")) {
            auto decParams = backend->get_mode_params("decode");
            if (decParams.size() >= 3) {
                outCh = std::max(1, decParams[2]);
            }
        }
        int n_vec = 2048;

        // -------------------------------------------------------------
        // 1. STREAMING FORWARD TEST (8 consecutive blocks of test audio)
        // -------------------------------------------------------------
        int numFwdBlocks = (res.latentDim >= 8 && res.ratio >= 1024) ? 16 : 8;
        std::cout << "  Testing perform_forward (" << numFwdBlocks << "-block audio stream, " << inCh << "->" << outCh << " ch)..." << std::endl;
        try {
            float maxPeak = 0.0f;
            double sumSq = 0.0;
            size_t totalSamples = 0;
            bool hadNan = false;
            bool fuseTripped = false;

            for (int b = 0; b < numFwdBlocks; ++b) {
                std::vector<std::vector<float>> inChannels(inCh, std::vector<float>(n_vec));
                std::vector<std::vector<float>> outChannels(outCh, std::vector<float>(n_vec, 0.0f));
                for (int c = 0; c < inCh; ++c) {
                    for (int i = 0; i < n_vec; ++i) {
                        float t = (float)(b * n_vec + i) / 44100.0f;
                        inChannels[c][i] = std::sin(6.2831853f * (100.0f + c * 50.0f) * t) * 0.7f;
                        if (i % 512 == 0) inChannels[c][i] += 0.2f;
                    }
                }

                std::vector<float*> inPtrs(inCh);
                std::vector<float*> outPtrs(outCh);
                for (int c = 0; c < inCh; ++c) inPtrs[c] = inChannels[c].data();
                for (int c = 0; c < outCh; ++c) outPtrs[c] = outChannels[c].data();

                backend->perform_forward(inPtrs, outPtrs, 1, outCh, n_vec, nullptr);

                for (int c = 0; c < outCh; ++c) {
                    for (int i = 0; i < n_vec; ++i) {
                        float s = outChannels[c][i];
                        if (std::isnan(s) || std::isinf(s)) {
                            hadNan = true;
                        } else {
                            float abs_s = std::abs(s);
                            if (abs_s > maxPeak) maxPeak = abs_s;
                            if (abs_s > 2.0f) fuseTripped = true;
                            sumSq += (double)s * (double)s;
                            totalSamples++;
                        }
                    }
                }
            }

            res.fwdPeak = maxPeak;
            res.fwdRms = totalSamples > 0 ? (float)std::sqrt(sumSq / totalSamples) : 0.0f;
            res.fwdHasNan = hadNan;
            res.fwdFuseTripped = fuseTripped;

            std::cout << "    Forward: Peak=" << std::fixed << std::setprecision(4) << maxPeak
                      << " | RMS=" << res.fwdRms << " | NaN=" << (hadNan ? "YES" : "NO")
                      << " | FuseCutoff(>2.0)=" << (fuseTripped ? "💥 BLOWN" : "OK") << std::endl;

            if (hadNan) {
                res.fwdStatus = "❌ FAIL (NaN)";
                res.failureDetails += "Forward produced NaN/Inf\n";
                failures++;
            } else if (fuseTripped) {
                res.fwdStatus = "💥 BLOWN FUSE";
                res.failureDetails += "Forward peak (" + std::to_string(maxPeak) + ") exceeded 2.0 safety fuse\n";
                failures++;
            } else if (res.fwdRms < 1e-5f) {
                res.fwdStatus = "⚠️ SILENT";
                res.failureDetails += "Forward produced near-zero audio (RMS < 1e-5)\n";
                failures++;
            } else {
                res.fwdStatus = "✅ PASS";
            }
        } catch (const std::exception& e) {
            std::cerr << "  FAILED: perform_forward exception: " << e.what() << std::endl;
            res.fwdStatus = "❌ FAIL";
            res.failureDetails += std::string("perform_forward error: ") + e.what() + "\n";
            failures++;
        }

        // -------------------------------------------------------------
        // 2. DECODE GEN TEST (4 blocks)
        // -------------------------------------------------------------
        if (backend->has_method("decode")) {
            std::cout << "  Testing perform_decode_gen (4 blocks)..." << std::endl;
            try {
                float maxPeak = 0.0f;
                double sumSq = 0.0;
                size_t totalSamples = 0;
                bool hadNan = false;
                bool fuseTripped = false;

                for (int b = 0; b < 4; ++b) {
                    std::vector<std::vector<float>> outChannels(outCh, std::vector<float>(n_vec, 0.0f));
                    std::vector<float*> outPtrs(outCh);
                    for (int c = 0; c < outCh; ++c) outPtrs[c] = outChannels[c].data();

                    backend->perform_decode_gen(outPtrs, 1, outCh, n_vec, nullptr);

                    for (int c = 0; c < outCh; ++c) {
                        for (int i = 0; i < n_vec; ++i) {
                            float s = outChannels[c][i];
                            if (std::isnan(s) || std::isinf(s)) {
                                hadNan = true;
                            } else {
                                float abs_s = std::abs(s);
                                if (abs_s > maxPeak) maxPeak = abs_s;
                                if (abs_s > 2.0f) fuseTripped = true;
                                sumSq += (double)s * (double)s;
                                totalSamples++;
                            }
                        }
                    }
                }

                res.decPeak = maxPeak;
                res.decRms = totalSamples > 0 ? (float)std::sqrt(sumSq / totalSamples) : 0.0f;
                res.decHasNan = hadNan;
                res.decFuseTripped = fuseTripped;

                std::cout << "    Decode Gen: Peak=" << std::fixed << std::setprecision(4) << maxPeak
                          << " | RMS=" << res.decRms << " | NaN=" << (hadNan ? "YES" : "NO")
                          << " | FuseCutoff(>2.0)=" << (fuseTripped ? "💥 BLOWN" : "OK") << std::endl;

                if (hadNan) {
                    res.decStatus = "❌ FAIL (NaN)";
                    res.failureDetails += "Decode Gen produced NaN/Inf\n";
                    failures++;
                } else if (fuseTripped) {
                    res.decStatus = "💥 BLOWN FUSE";
                    res.failureDetails += "Decode Gen peak (" + std::to_string(maxPeak) + ") exceeded 2.0 safety fuse\n";
                    failures++;
                } else if (res.decRms < 1e-5f) {
                    res.decStatus = "⚠️ SILENT";
                    res.failureDetails += "Decode Gen produced near-zero audio (RMS < 1e-5)\n";
                    failures++;
                } else {
                    res.decStatus = "✅ PASS";
                }
            } catch (const std::exception& e) {
                std::cerr << "  FAILED: perform_decode_gen exception: " << e.what() << std::endl;
                res.decStatus = "❌ FAIL";
                res.failureDetails += std::string("perform_decode_gen error: ") + e.what() + "\n";
                failures++;
            }
        }

        // -------------------------------------------------------------
        // 3. PRIOR DECODE TEST (4 blocks)
        // -------------------------------------------------------------
        if (backend->has_method("prior")) {
            std::cout << "  Testing perform_prior_decode (4 blocks)..." << std::endl;
            try {
                float maxPeak = 0.0f;
                double sumSq = 0.0;
                size_t totalSamples = 0;
                bool hadNan = false;
                bool fuseTripped = false;

                for (int b = 0; b < 4; ++b) {
                    std::vector<std::vector<float>> outChannels(outCh, std::vector<float>(n_vec, 0.0f));
                    std::vector<float*> outPtrs(outCh);
                    for (int c = 0; c < outCh; ++c) outPtrs[c] = outChannels[c].data();

                    backend->perform_prior_decode(outPtrs, 1, outCh, n_vec);

                    for (int c = 0; c < outCh; ++c) {
                        for (int i = 0; i < n_vec; ++i) {
                            float s = outChannels[c][i];
                            if (std::isnan(s) || std::isinf(s)) {
                                hadNan = true;
                            } else {
                                float abs_s = std::abs(s);
                                if (abs_s > maxPeak) maxPeak = abs_s;
                                if (abs_s > 2.0f) fuseTripped = true;
                                sumSq += (double)s * (double)s;
                                totalSamples++;
                            }
                        }
                    }
                }

                res.priorPeak = maxPeak;
                res.priorRms = totalSamples > 0 ? (float)std::sqrt(sumSq / totalSamples) : 0.0f;
                res.priorHasNan = hadNan;
                res.priorFuseTripped = fuseTripped;

                std::cout << "    Prior: Peak=" << std::fixed << std::setprecision(4) << maxPeak
                          << " | RMS=" << res.priorRms << " | NaN=" << (hadNan ? "YES" : "NO")
                          << " | FuseCutoff(>2.0)=" << (fuseTripped ? "💥 BLOWN" : "OK") << std::endl;

                if (hadNan) {
                    res.priorStatus = "❌ FAIL (NaN)";
                    res.failureDetails += "Prior produced NaN/Inf\n";
                    failures++;
                } else if (fuseTripped) {
                    res.priorStatus = "💥 BLOWN FUSE";
                    res.failureDetails += "Prior peak (" + std::to_string(maxPeak) + ") exceeded 2.0 safety fuse\n";
                    failures++;
                } else if (res.priorRms < 1e-5f) {
                    res.priorStatus = "⚠️ SILENT";
                    res.failureDetails += "Prior produced near-zero audio (RMS < 1e-5)\n";
                    failures++;
                } else {
                    res.priorStatus = "✅ PASS";
                }
            } catch (const std::exception& e) {
                std::cerr << "  FAILED: perform_prior_decode exception: " << e.what() << std::endl;
                res.priorStatus = "❌ FAIL";
                res.failureDetails += std::string("perform_prior_decode error: ") + e.what() + "\n";
                failures++;
            }
        }

        results.push_back(res);
    }

    // Write Overview Table
    report << "| Model Path | Engine | Status | Forward Peak (RMS) | Decode Gen (RMS) | Prior (RMS) |\n";
    report << "|---|---|---|---|---|---|\n";

    for (const auto& r : results) {
        if (!r.loadSuccess) {
            report << "| `" << r.path << "` | " << r.backendType << " | ❌ LOAD FAIL | - | - | - |\n";
        } else {
            auto fmtStat = [](const std::string& st, float peak, float rms) {
                if (st == "N/A") return std::string("N/A");
                return st + " (Pk:" + std::to_string(peak).substr(0, 5) + ", RMS:" + std::to_string(rms).substr(0, 6) + ")";
            };
            report << "| `" << r.path << "` | " << r.backendType << " | "
                   << (r.failureDetails.empty() ? "✅ PASS" : "⚠️ ISSUES") << " | "
                   << fmtStat(r.fwdStatus, r.fwdPeak, r.fwdRms) << " | "
                   << fmtStat(r.decStatus, r.decPeak, r.decRms) << " | "
                   << fmtStat(r.priorStatus, r.priorPeak, r.priorRms) << " |\n";
        }
    }

    // Write Detailed Per-Model Breakdown
    report << "\n## Detailed Model Diagnostics\n\n";
    for (const auto& r : results) {
        report << "### `" << r.path << "` (" << r.backendType << ")\n\n";
        if (!r.loadSuccess) {
            report << "> [!CAUTION]\n";
            report << "> **Load Error**: " << r.loadError << "\n\n";
            continue;
        }

        report << "- **Engine**: `" << r.backendType << "`\n";
        report << "- **Sample Rate**: " << r.sr << " Hz | **Ratio**: " << r.ratio << " | **Latent Dim**: " << r.latentDim << "\n";
        report << "- **Forward Stream (8 blk)**: " << r.fwdStatus
               << " (Max Peak: " << r.fwdPeak << ", RMS: " << r.fwdRms << ")\n";
        report << "- **Decode Gen (4 blk)**: " << r.decStatus
               << " (Max Peak: " << r.decPeak << ", RMS: " << r.decRms << ")\n";
        if (r.priorStatus != "N/A") {
            report << "- **Prior Mode (4 blk)**: " << r.priorStatus
                   << " (Max Peak: " << r.priorPeak << ", RMS: " << r.priorRms << ")\n";
        }

        if (!r.failureDetails.empty()) {
            report << "\n> [!WARNING]\n";
            report << "> **Detected Audio / Sentry Issues**:\n> ```\n> " << r.failureDetails << "> ```\n";
        }
    }

    std::cout << "\n==========================================" << std::endl;
    report << "\n## Summary\n\n";
    if (failures == 0) {
        std::cout << "ALL BACKEND TESTS PASSED!" << std::endl;
        report << "**Result**: 🎉 **ALL TESTS PASSED WITH REAL AUDIO VALIDATION**\n";
        report.close();
        std::cout << "Saved test report to: " << reportPath << std::endl;
        return 0;
    } else {
        std::cerr << failures << " TEST(S) FAILED OR TRIPPED SENTRY!" << std::endl;
        report << "**Result**: ⚠️ **" << failures << " TEST(S) FAILED OR TRIPPED SENTRY**\n";
        report.close();
        std::cout << "Saved test report to: " << reportPath << std::endl;
        return 1;
    }
}
