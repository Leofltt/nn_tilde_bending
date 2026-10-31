# NetBender — Multi-Backend Neural Audio Circuit-Bending Engine

> [!TIP]
> **Pre-built Installers & Nightly Binaries**:
> Looking to quickly install the VST3, AU, or Standalone plugins without building from source? Download pre-built installers and packages from the latest completed run of the [Release Plugins GitHub Action Workflow](https://github.com/Leofltt/nn_tilde_bending/actions/workflows/release_plugins.yaml) (under the **Artifacts** section).

**NetBender** (`nn_tilde_bending`) is a performative neural audio synthesis and live circuit-bending engine evolved from [IRCAM's `nn~`](https://github.com/acids-ircam/nn_tilde) and inspired by modern edge runtimes ([`neural_tilde`](https://github.com/jasper-zheng/neural_tilde)). It transforms deep learning audio models—such as [RAVE](https://github.com/acids-ircam/RAVE), [Stable Audio Open VAE](https://huggingface.co/stabilityai/stable-audio-open-1.0), and [Meta EnCodec](https://github.com/facebookresearch/encodec)—into expressive, tactile instruments via real-time weight mutation, layer-selective trace isolation, momentary short-circuiting, thermal drift, resonant cross-talk, latent space terrain navigation, and safety circuit breakers.

---

## Multi-Backend Runtime Architecture

Because TorchScript (`torch.jit`) is deprecated in PyTorch 2.x+, NetBender uses a unified, runtime-agnostic `IBackend` architecture (`BackendFactory`) across all frontends (JUCE VST3/AU/Standalone, Max/MSP, and PureData). Simply load any `.pte`, `.onnx`, or `.ts` model file and NetBender automatically dispatches to the appropriate engine:

| Extension | Backend Class | Hardware Acceleration / Delegates | Weight-Bending Mechanism |
| --- | --- | --- | --- |
| **`.pte`** | `ExecuTorchBackend` | **CoreML** (Apple Neural Engine / GPU), **MPS** (Metal), **XNNPACK** (ARM NEON / x86 AVX), **Portable** | Input-Promoted Parameters (`torch.func.functional_call`) + `NBND` `.bin` sidecar & `.json` metadata |
| **`.onnx`** | `OnnxBackend` | **CoreML Execution Provider** (macOS `@gpu 1` / GPU toggle), **CPU** multi-threaded | Promoted Graph Initializers (`onnx_bend_prep.py`) + zero-copy `Ort::Value` bindings & `NBND` `.bin` sidecar |
| **`.ts`** | `TorchBackend` | **MPS** (macOS), **CUDA** (Linux/Windows), **CPU** (LibTorch fallback) | Direct in-place `at::Tensor` parameter mutation + shadow baseline cloning |

### The `NBND` Sidecar Format (`.bin` + `.json`)

Compiled edge runtimes (ExecuTorch `.pte` and ONNX Runtime `.onnx`) achieve their speedups by freezing static weights inside hardware-compiled subgraphs. To preserve **zero-allocation, sample-accurate weight bending** without re-compiling sessions on the audio thread:

1. Selected bendable parameters are promoted to explicit graph inputs during export (`pte_bend_prep.py` / `onnx_bend_prep.py`).
2. Their baseline float32 weights and tensor shapes are written to a companion `<model>.bin` file (`NBND` binary header).
3. At runtime, NetBender memory-maps the `NBND` sidecar into contiguous `float` buffers and binds them directly to the ExecuTorch or ONNX Runtime execution inputs with zero copies.

---

## Frontends & Build Targets

NetBender provides native frontends for DAWs, standalone desktop performance, and modular patching environments:

| Target | Description | Output / Installation |
| --- | --- | --- |
| `plugin` | **JUCE VST3 & AudioUnit (AU)** (`NetBender`) | Bundled dylibs, signed with `SIGN_ID`, installed to `~/Library/Audio/Plug-Ins/VST3` and `~/Library/Audio/Plug-Ins/Components` (macOS). |
| `standalone` | **JUCE Standalone Application** (`NetBender`) | Desktop app with direct audio/MIDI I/O and full interactive circuit-bending UI. |
| `maxmsp` | **Max/MSP Externals** (`netBender~`, `mc.netBender~`, `mcs.netBender~`, `netBender.info`) | Built as `.mxo` / `.mxe64` bundles for Max 8+. |
| `puredata` | **PureData External** (`netBender~`, with legacy `nn~` alias) | Built as `netBender~.pd_darwin` / `netBender~.pd_linux` / `netBender~.dll`. |
| `all` (default) | Builds all enabled frontend targets | Builds all configured targets. |

---

## Exporting & Migrating Models (`python_tools/`)

NetBender includes a unified Python CLI (`python_tools/export_cli.py`) and standalone utilities (`python_tools/migrate_ts_to_pte.py`, `python_tools/rave_ts_to_onnx.py`, `python_tools/stable_audio_ts_to_onnx.py`) to export models from checkpoints or migrate existing `.ts` models to `.pte` (ExecuTorch) and `.onnx` (ONNX Runtime).

### Python Virtual Environment Setup (`.venv`)

Always install Python conversion dependencies inside `.venv` (CMake also auto-discovers ExecuTorch's C++ headers and delegate libraries directly from `.venv`):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 1. Exporting to ExecuTorch (`.pte`) & Migrating `.ts` Models

ExecuTorch exports support two bend promotion modes (`--bend-mode`) and four hardware delegates (`--delegate {coreml,mps,xnnpack,portable}`):

- **`--bend-mode delegate` (Default, Recommended for CoreML / XNNPACK)**:
  Promotes 1D/channelwise normalization (`norm`, `weight_g`, $\gamma$, $\beta$), biases, and activation parameters (`snake` $\alpha/\beta$, channel scales) to dynamic inputs while keeping heavy convolution kernels static inside the delegated CoreML/XNNPACK subgraph. This gives you full Apple Neural Engine / GPU acceleration **and** expressive, stable real-time circuit bending.
- **`--bend-mode full` (Recommended for `xnnpack` / `portable` when bending every kernel)**:
  Promotes **all** floating-point weights (including 1D/2D convolution matrices) to dynamic graph inputs.
- **`--bend-mode none`**:
  Exports a pure static `.pte` graph with no promoted weight inputs (maximum raw inference speed).

```bash
# Convert all 4 reference models to CoreML-accelerated .pte files (GPU / Apple Neural Engine):
.venv/bin/python -m python_tools.migrate_ts_to_pte models/darbouka.ts \
    -o models/darbouka_coreml.pte --delegate coreml --bend-mode delegate
.venv/bin/python -m python_tools.migrate_ts_to_pte models/speech.ts \
    -o models/speech_coreml.pte --delegate coreml --bend-mode delegate
.venv/bin/python -m python_tools.migrate_ts_to_pte models/encodec_bendable.ts \
    -o models/encodec_coreml.pte --delegate coreml --bend-mode delegate
.venv/bin/python -m python_tools.migrate_ts_to_pte models/stable_audio_1_vae.ts \
    -o models/stable_audio_1_vae_coreml.pte --delegate coreml --bend-mode delegate

# Convert all 4 reference models to XNNPACK-accelerated .pte files (CPU SIMD):
.venv/bin/python -m python_tools.migrate_ts_to_pte models/darbouka.ts \
    -o models/darbouka_xnnpack.pte --delegate xnnpack --bend-mode delegate
.venv/bin/python -m python_tools.migrate_ts_to_pte models/speech.ts \
    -o models/speech_xnnpack.pte --delegate xnnpack --bend-mode delegate
.venv/bin/python -m python_tools.migrate_ts_to_pte models/encodec_bendable.ts \
    -o models/encodec_xnnpack.pte --delegate xnnpack --bend-mode delegate
.venv/bin/python -m python_tools.migrate_ts_to_pte models/stable_audio_1_vae.ts \
    -o models/stable_audio_1_vae_xnnpack.pte --delegate xnnpack --bend-mode delegate

# Or use the unified export_cli wrapper / custom layer filters:
.venv/bin/python -m python_tools.export_cli migrate_pte \
    -i models/stable_audio_1_vae.ts \
    -o models/stable_audio_1_vae_coreml.pte \
    --delegate coreml \
    --bend-mode delegate \
    --filter snake norm bias
```

> [!NOTE]
> **How `.ts` $\rightarrow$ `.pte` Lowering Works**:
> `migrate_ts_to_pte.py` first translates the TorchScript `inlined_graph` IR directly into an eager PyTorch `nn.Module` (`TorchScriptGraphEagerModule`) and exports it via `torch.export.export()` without touching ONNX (used by `encodec_bendable.ts` and `stable_audio_1_vae.ts`). If a model contains non-exportable complex/FFT ops (such as RAVE's filtered-noise synthesizer in `darbouka.ts` and `speech.ts`), it automatically falls back to NetBender's RAVE graph sanitizer to prune the non-exportable branch while preserving the tonal waveform and stateful streaming history window (`history_samples`).

### 2. Exporting & Converting Models to Bendable ONNX (`.onnx`)

```bash
# Convert RAVE TorchScript (.ts) models to bendable ONNX (exports forward, encode, decode + .bin sidecar):
.venv/bin/python -m python_tools.export_cli rave_to_onnx \
    -i models/darbouka.ts \
    -o models/darbouka_bendable.onnx \
    --buffer-size 2048

.venv/bin/python -m python_tools.export_cli rave_to_onnx \
    -i models/speech.ts \
    -o models/speech_bendable.onnx \
    --buffer-size 2048

# Convert Stable Audio Open / 1.0 VAE (.ts) or EnCodec (.ts) to bendable ONNX:
.venv/bin/python -m python_tools.export_cli stable_audio_to_onnx \
    -i models/stable_audio_1_vae.ts \
    -o models/stable_audio_1_vae_bendable.onnx \
    --buffer-size 8192

.venv/bin/python -m python_tools.export_cli stable_audio_to_onnx \
    -i models/encodec_bendable.ts \
    -o models/encodec_bendable.onnx \
    --buffer-size 2048 --channels 1

# Export directly from Hugging Face or local checkpoints (Stable Audio, EnCodec, RAVE):
.venv/bin/python -m python_tools.export_cli stable_audio --onnx -o models/sa_vae.onnx
.venv/bin/python -m python_tools.export_cli encodec --onnx --output models/encodec_24k.onnx
.venv/bin/python -m python_tools.export_cli rave -i path/to/rave.ckpt -o models/my_rave.onnx --onnx
```

### 3. Promoting Arbitrary `.onnx` or PyTorch Models (`onnx_bend_prep.py` & `pte_bend_prep.py`)

If you already have a standard `.onnx` model, you can promote its weights and generate its `NBND` `.bin` sidecar with `onnx_bend_prep.py`:

```bash
# Inspect an ONNX model's inputs, outputs, and initializers:
python python_tools/onnx_bend_prep.py -i model.onnx --inspect

# Promote all float initializers (or filter specific layers) and generate <model>_bendable.onnx + .bin:
python python_tools/onnx_bend_prep.py -i model.onnx -o model_bendable.onnx --filter norm bias snake
```

In Python scripts, you can export any custom `torch.nn.Module` to a bendable `.pte` bundle using `export_bendable_pte` from `python_tools/pte_bend_prep.py`:

```python
import torch
from python_tools.pte_bend_prep import export_bendable_pte

sample_audio = torch.zeros(1, 1, 2048)
export_bendable_pte(
    module=my_audio_model,
    sample_audio_input=sample_audio,
    output_pte_path="models/custom_effect.pte",
    delegate="coreml",       # "coreml", "mps", "xnnpack", or "portable"
    bend_mode="delegate",    # "delegate", "full", or "none"
    extra_metadata={"sr": 48000, "in_dim": 1, "out_dim": 1, "higher_ratio": 2048},
)
```

---

## Using the JUCE Plugin & Standalone (`NetBender`)

1. **Loading Models**:
   - Click **LOAD MODEL** in the top bar and select any **`.pte`**, **`.onnx`**, or **`.ts`** file.
   - The status bar displays the active model name, active backend badge (`[ExecuTorch (coreml)]`, `[ONNX Runtime]`, or `[TorchScript]`), and latency/ratio info.
2. **Interactive Weight Canvas & Triple-Display Architecture (`1D` / `2D` / `Latents`)**:
   - **1D Multi-Trace Oscilloscope**: Draw, invert, quantize, or scramble weights directly on the selected layer's parameter curve.
   - **2D Filter Matrix Heatmap**: Inspect and bend 2D convolution kernels and projection matrices with cross-channel trace bridges.
   - **Latent Topographic Vector Pad**: Navigate the autoencoder bottleneck (`encode` $\rightarrow$ Latent Terrain Hook $\rightarrow$ `decode`) using Random Fourier Feature orbits (`FourierOrbit`), latent slew limiting (`LatentSlew`), and step quantization (`Quantize`). Works identically across `.pte`, `.onnx`, and `.ts` models without any runtime dependency on LibTorch.
3. **Trace Isolation & Macro Controls**:
   - Isolate specific architectural groups (**Norm / Gain ($\gamma, \beta$)**, **Early Transients**, **Internal Latents**, **Output Heads**) to bend dynamics, pitch/formants, or texture independently.
   - Trigger momentary **SHORT [!]** bursts with Attack/Hold/Release envelopes or audio sidechain transient detection.
   - Enable **Thermal Drift** for continuous, bounded Ornstein-Uhlenbeck parameter evolution.
4. **Saving Bent Models**:
   - Click **SAVE MODEL** to snapshot your current bent weights into a new `.pte` / `.onnx` (with updated `NBND` `.bin` sidecar) or `.ts` file.

---

## Using `netBender~` in Max/MSP & PureData

All Max/MSP (`netBender~`, `mc.netBender~`, `mcs.netBender~`, `netBender.info`) and PureData (`netBender~`, also aliased to `nn~` for backward compatibility) externals share the same multi-backend engine as the JUCE plugin.

### Instantiation

```text
netBender~ <model_path> [method] [buffer_size]
```

- **`<model_path>`**: Path or filename of a `.pte`, `.onnx`, or `.ts` model (if you omit the extension, `netBender~` automatically searches for `<name>.pte`, `<name>.onnx`, and `<name>.ts` in order).
- **`[method]`**: Synthesis method to run (`forward` by default; `encode`, `decode`, or `prior` when companion graphs/methods are available).
- **`[buffer_size]`**: Internal circular buffer size in samples (must be $\ge$ model's `higher_ratio`; `0` disables the worker thread).

**Examples:**

- `[netBender~ vintage_bendable.onnx forward 2048]`
- `[netBender~ stable_audio_1_vae.pte forward 8192 @gpu 1]`
- `[mc.netBender~ rave_model.pte decode 2048]`

### Live Weight-Bending Messages

Connect a message box to the left inlet of `netBender~` and route the rightmost info outlet to inspect and mutate network layers in real time:

| Message | Arguments | Description |
| --- | --- | --- |
| `layers` | *(none)* | Outputs `layers <layer_0> <layer_1> ...` from the info outlet listing all bendable parameter tensors. |
| `get_weights` | `<layer_name>` | Outputs `layer <w_0> <w_1> ...` containing the current float weights of `<layer_name>`. |
| `set_weights` | `<layer_name> <f1> <f2> ...` | Replaces the weights of `<layer_name>` in real time (zero-allocation in `.pte` / `.onnx`). |
| `reset_weights` | `<layer_name>` | Restores `<layer_name>` to its original baseline weights from the `NBND` sidecar / snapshot. |
| `reset_all_weights` | *(none)* | Restores all bendable layers in the model to their original baseline weights. |
| `reload` | *(none)* | Reloads the model and resets all state buffers and weights from disk. |
| `load` | `<model_path>` | Dynamically loads a new `.pte`, `.onnx`, or `.ts` model. |
| `method` | `<method_name>` | Switches active synthesis method (`forward`, `encode`, `decode`, `prior`). |
| `dump` | *(none)* | Prints model path, active backend (`ExecuTorch`, `ONNX Runtime`, `TorchScript`), dimensions, ratios, methods, and attributes to the console. |
| `enable` | `0` or `1` | Enables or bypasses audio processing. |
| `gpu` / `@gpu` | `0` or `1` | Enables hardware acceleration (CoreML Execution Provider for `.onnx`,MPS/CUDA for `.ts`, or built-in delegate for `.pte`). |

---

## Building from Source

> [!IMPORTANT]
> **Source Directory Note**: The root CMake configuration file (`CMakeLists.txt`) is located in **`src/`**, not at the repository root. Always pass `-S src -B build` when configuring CMake.

### Prerequisites

- **CMake** (v3.15+) and a C++20 compiler (Xcode Command Line Tools on macOS, GCC 10+ on Linux, Visual Studio 2022 on Windows).
- **ONNX Runtime** (`src/cmake/add_onnxruntime.cmake` automatically downloads the matching ONNX Runtime release if `ort_dir` is not specified).
- **ExecuTorch**: Automatically discovered from the project's Python virtual environment (`.venv`) once `pip install -r requirements.txt` (`executorch>=1.0.0`) has been installed, or via `-DEXECUTORCH_ROOT=/path/to/executorch`.
- **LibTorch**: Download from [pytorch.org](https://pytorch.org/get-started/locally/) into `./libtorch` or pass `-DCMAKE_PREFIX_PATH=/path/to/libtorch`.

### 1. Configure CMake

```bash
cmake -S src -B build \
    -DCMAKE_PREFIX_PATH="$(pwd)/libtorch" \
    -DCMAKE_BUILD_TYPE=Release
```

### 2. Build Targets

```bash
# Build the JUCE VST3 and AU plugins:
cmake --build build --target plugin

# Build the JUCE Standalone desktop app:
cmake --build build --target standalone

# Build the Max/MSP externals (netBender~, mc.netBender~, mcs.netBender~, netBender.info):
cmake --build build --target maxmsp

# Build the PureData external (netBender~):
cmake --build build --target puredata

# Build all enabled targets:
cmake --build build --target all
```

### CMake Options Reference

| Option | Default | Description |
| --- | --- | --- |
| `-DBUILD_JUCE_PLUGIN=ON\|OFF` | `ON` | Build JUCE VST3, AU, and Standalone plugin targets. |
| `-DBUILD_MAXMSP=ON\|OFF` | `ON` | Build Max/MSP externals (`netBender~`, `mc.netBender~`, `mcs.netBender~`, `netBender.info`). |
| `-DBUILD_PUREDATA=ON\|OFF` | `ON` | Build PureData external (`netBender~`). |
| `-DENABLE_EXECUTORCH_BACKEND=ON\|OFF` | `ON` | Enable ExecuTorch `.pte` backend discovery and linking (`EXECUTORCH_ROOT`). |
| `-DBUNDLE_DEPENDENCIES=ON\|OFF` | `OFF` | Bundle and relocate shared libraries into `support/` for redistributable packages. |
| `-DSIGN_ID="..."` | `"-"` | macOS codesigning identity (can also be placed in `.env` at the repo root). |

### 3. Multi-Backend Verification & Sentry Suite (`test_backends`)

The C++ test harness (`build/tests/test_backends`) verifies loading, streaming `forward` passes, `decode gen`, `prior` synthesis, NaN/Inf/fuse sentry protection, and real-time weight bending across `.ts`, `.onnx`, and `.pte` models, writing a summary report to `test_report.md`:

```bash
# Run the full CPU test suite (.ts CPU, .onnx CPU, and *_xnnpack.pte):
./build/tests/test_backends

# Run the full GPU / Apple Neural Engine test suite (.ts MPS, .onnx CoreML EP, and *_coreml.pte):
./build/tests/test_backends --gpu

# Or test specific model files directly:
./build/tests/test_backends --gpu models/stable_audio_1_vae_coreml.pte models/encodec_coreml.pte
```
