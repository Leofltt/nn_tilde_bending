# nn_tilde_bending — Creative Neural Circuit-Bending Engine

`nn_tilde_bending` is a creative, performative neural circuit-bending fork of [IRCAM's `nn~`](https://github.com/acids-ircam/nn_tilde). It transforms deep learning audio models (such as [RAVE](https://github.com/acids-ircam/RAVE) and other 1D/2D neural audio architectures) into expressive, tactile instruments by enabling real-time weight mutation, layer-selective trace isolation, momentary short-circuiting, thermal drift, resonant cross-talk, and safety circuit breakers.

---

## Frontends & Build Targets

The project provides multiple frontends tailored for modern music production workflows, live performance, and modular patch environments:

| Target | Description | Output / Installation |
|---|---|---|
| `plugin` | **JUCE VST3 & AudioUnit (AU)** plugins (`nn_bending_plugin`) | Bundled dylibs, signed with `SIGN_ID` & entitlements, copied to `~/Library/Audio/Plug-Ins/VST3` and `~/Library/Audio/Plug-Ins/Components` (macOS). |
| `standalone` | **JUCE Standalone Application** (`nn_bending_standalone`) | Standalone desktop executable with direct audio/MIDI I/O, embedded interactive UI, and LibTorch backend. |
| `maxmsp` | **Max/MSP Externals** (`nn~`, `mc.nn~`, `mcs.nn~`, `nn.info`) | Bundles externals into `src/frontend/maxmsp` for Max 8+. |
| `puredata` | **PureData External** (`nn~`) | Built to `build/frontend/puredata/Release`. |
| `all` (default) | Builds all enabled frontend targets | Builds everything configured. |

---

## Key Neural Circuit-Bending Features

1. **Interactive Weight Canvas & Visualizer (JUCE Plugin & Standalone)**:
   - Direct 2D interactive canvas allowing you to draw, invert, scramble, and sculpt weights in real time.
   - Live spectrum, gain staging, and layer parameter inspector.
2. **Layer-Selective "Trace Isolation" & Categorization**:
   - Categorizes model layers into functional trace types: **Normalization / Dynamics ($\gamma, \beta$)**, **Early Transient Layers**, **Internal Latent Residuals / Temporal Kernels**, and **Output Projection Heads**.
   - Target specific network structures instead of corrupting random parameter blocks.
3. **Momentary Short-Circuits & Dynamic Envelopes**:
   - Performative "SHORT [!]" momentary trigger with variable Attack/Hold/Release (AHR) smoothing.
   - Audio transient follower (sidechain trigger) that bursts glitches on transients and relaxes on tails.
4. **Stochastic Thermal Drift**:
   - Simulates physical circuit warm-up, component aging, and battery brownout via bounded Brownian / Ornstein-Uhlenbeck drift.
5. **Safety Fuse & DC Breaker**:
   - Automatic protection circuitry (hard limiter, soft clipper, and DC blocker) preventing ear/speaker damage from runaway weight blowouts.

---

## Quickstart & Build Instructions

> [!IMPORTANT]
> **Source Directory Note**: The primary CMake configuration file (`CMakeLists.txt`) is located in the **`src/`** directory, not at the root of the repository. When configuring CMake, always point the source directory to `src` (e.g., `-S src -B build`).

### Prerequisites
- **CMake** (v3.15 or newer recommended)
- **LibTorch** (C++ PyTorch library, CPU version matching your OS/architecture). Download it from [pytorch.org](https://pytorch.org/get-started/locally/) and extract it into the repository root as `libtorch/`, or pass `-DCMAKE_PREFIX_PATH` pointing to your LibTorch folder.
- Modern C++ compiler (Clang on macOS with Xcode Command Line Tools, GCC 9+ on Linux, or Visual Studio 2022 on Windows).

### 1. Configure CMake

From the repository root:

```bash
# Recommended modern CMake style:
cmake -S src -B build -DCMAKE_PREFIX_PATH="$(pwd)/libtorch" -DCMAKE_BUILD_TYPE=Release
```

*Or traditional out-of-source directory approach:*
```bash
mkdir -p build && cd build
cmake ../src -DCMAKE_PREFIX_PATH=../libtorch -DCMAKE_BUILD_TYPE=Release
```

### 2. Build Specific Targets

You can build specific targets using `cmake --build`:

```bash
# Build the JUCE VST3 and AU plugins:
cmake --build build --target plugin

# Build the standalone desktop app:
cmake --build build --target standalone

# Build the Max/MSP externals:
cmake --build build --target maxmsp

# Build the PureData externals:
cmake --build build --target puredata

# Build all enabled targets:
cmake --build build --target all
```
*(If you are already inside the `build/` directory, replace `build` with `.` : `cmake --build . --target <target>`)*

### CMake Configuration Options

The build system can be customized with the following flags during configuration (`-D<OPTION>=<VALUE>`):

- `-DBUILD_JUCE_PLUGIN=ON|OFF` (default: `ON`): Enable/disable JUCE VST3/AU plugins and standalone targets.
- `-DBUILD_MAXMSP=ON|OFF` (default: `ON`): Enable/disable Max/MSP externals.
- `-DBUILD_PUREDATA=ON|OFF` (default: `ON`): Enable/disable PureData externals.
- `-DBUNDLE_DEPENDENCIES=ON|OFF` (default: `OFF`): When `ON`, copies shared libraries (~1.5 GB) into `support/` for standalone distribution. Leave `OFF` for fast native `@rpath` development.
- `-DSIGN_ID="..."` (default: `-`): Codesigning identity for macOS binaries, frameworks, and bundles.
- `-DCMAKE_POLICY_VERSION_MINIMUM=3.15`: Ensures compatibility with modern CMake policies.

#### Environment Variables & `.env`
You can create a `.env` file at the root of the repository to set build variables automatically (e.g., `SIGN_ID`):

```bash
SIGN_ID="Developer ID Application: Your Name (TEAM_ID)"
```

---

## Original `nn~` Documentation & Architecture Reference (IRCAM)

Below is the upstream documentation from [acids-ircam/nn~](https://github.com/acids-ircam/nn_tilde) detailing model loading, Max/MSP and PureData integration, pretrained models, and scripting.

### Installation (Pre-built Binaries)

Grab the [latest release of nn~](https://github.com/acids-ircam/nn_tilde/releases/latest)! Be sure to download the correct version for your installation.

#### MaxMSP

Uncompress the `.tar.gz` file in the Package folder of your Max installation, i.e. in `Documents/Max [your version]/Packages/`. You can then instantiate an `nn~` object! Alt-click the `nn~` object to open the help patch, or access the nn~ Overview patch in the Extras menu.

##### Mac alert : codesigned with IRCAM identity and not trigger MacOS quarantine ; if it does so, please launch in the terminal:

```bash
cd "~/Max X/Packages/nn_tilde"
sudo codesign --deep --force --sign - support/*.dylib
sudo codesign --deep --force --sign - externals/*/Contents/MacOS/*
xattr -r -d com.apple.quarantine externals/*/Contents/MacOS/*  
```

Alt+click on the `nn~` object to open the help patch, and follow the tabs to learn more about this project.

#### PureData

Uncompress the `.tar.gz` file in the Package folder of your Pd installation, i.e. in `Documents/Pd/externals/`. You can then add a new path in the `Pd/File/Preferences/Path` menu pointing to the `nn_tilde` folder.

Similarly, the external should not be blocked on recent MacOS systems. If it still is, `cd` to the `nn_tilde` folder and fix with:

```bash
xattr -r -d com.apple.quarantine Documents/Pd/externals/nn_tilde
sudo codesign --deep --force --sign - Documents/Pd/externals/nn_tilde/*.dylib
sudo codesign --deep --force --sign - Documents/Pd/externals/nn_tilde/nn\~.pd_darwin
```

---

### Usage

#### Pretrained Models

At its core, `nn~` is a translation layer between Max/MSP or PureData and the [libtorch C++ interface for deep learning](https://pytorch.org/). Alone, `nn~` is like an empty shell, and **requires pretrained models** to operate. Since v1.6.0, you can download them directly through the Forum IRCAM API. Alternatively, you can find a few [RAVE](https://github.com/acids-ircam/RAVE) models [here](https://acids-ircam.github.io/rave_models_download) or [here](https://huggingface.co/Intelligent-Instruments-Lab/rave-models). A few [vschaos2](https://github.com/acids-ircam/vschaos2) models are also available [here](https://www.dropbox.com/sh/avdeiza7c6bn2of/AAAGZsnRo9ZVMa0iFhouCBL-a?dl=0).

Pretrained models for `nn~` are **TorchScript files**, with a `.ts` extension. You can add these files to `nn_tilde/models` folders, or any place accessible through the Max / Pd filesystem (Max: `Options/File Preferences`, PureData: `File/Preferences/Path`).

Once this is done, you can load a model with `nn~` by providing its name as first argument (for example, `isis.ts` located inside `nn_tilde/models` for Max, or among the PureData patch):

<table>
  <tr>
    <th width="50%">Max / MSP</th>
    <th width="50%">PureData</th>
  </tr>
  <tr>
    <td><img width="100%" src="assets/max_instance.png" /></td>
    <td><img width="100%" src="assets/pd_instance.png" /></td>
  </tr>
</table>

#### Model Information Fetching

The `nn.info` object allows model inspection and fetching available models for download on the IRCAM-API. With this object, you can get available methods and attributes for a given model. For example, you can see below that a RAVE model has three different methods: `encode`, `decode`, and `forward`.

<center>
<img src="assets/max_nninfo.png"/>
</center>

##### Methods

Models can have several *methods* that correspond to several processing pipelines the model can achieve. Hence, each method can have a different number of inlets / outlets. The method is given as the third argument (for example, `decode` above), and equals `forward` by default.

##### Attributes

It is possible to inspect and change the internal state of the module through *attributes*, which are **model-dependent** and defined at exportation. Model attributes can be set using *messages*, with the following syntax:

```bash
set ATTRIBUTE_NAME ATTRIBUTE_VAL_1 ATTRIBUTE_VAL_2
```

Using Max/MSP and PureData graphical objects, this provides an intuitive way to modify model behavior, such as generation temperature and generation mode, and the special `enable` attribute.

<table>
  <tr>
    <th width="50%">Max / MSP</th>
    <th width="50%">PureData</th>
  </tr>
  <tr>
    <td><img width="100%" src="assets/max_attr.png" /></td>
    <td><img width="100%" src="assets/pd_attr.png" /></td>
  </tr>
</table>

**Since v1.6.0:**
- Buffers (Max) / Array (Pd) attribute setting allows the `.ts` model to access internal buffers / arrays.
- `torch.Tensor` attributes can be set through Max/MSP `[array]`, allowing attributes of unlimited size.

#### Buffer Configuration

Internally, `nn~` has a circular buffer mechanism that helps maintain a reasonable computational load, if the given buffer size is greater than 0. You can modify its size through the use of an additional integer after the method declaration.

**Important for Windows users**: The circular buffer is automatically disabled on Windows due to an upstream TorchScript threading issue.

<table>
  <tr>
    <th width="50%">Max / MSP</th>
    <th width="50%">PureData</th>
  </tr>
  <tr>
    <td><img width="100%" src="assets/max_buffer.png" /></td>
    <td><img width="100%" src="assets/pd_buffer.png" /></td>
  </tr>
</table>

#### Multichannel (Max/MSP)

The Max/MSP release of `nn~` includes additional externals, namely `mc.nn~` and `mcs.nn~`, allowing the use of the multichannel abilities of Max 8+ to simplify patching and decrease computational load:

- `mc.nn~`: Builds multichannel signals **over different batches**.
- `mcs.nn~`: Builds multichannel signals **over different dimensions** (e.g., 16 latent dimensions of a RAVE model).

<table>
  <tr>
    <th>Regular</th>
    <th>mc.nn~</th>
    <th>mcs.nn~</th>
  </tr>
  <tr>
    <td><img src="assets/max_regular.png" /></td>
    <td><img src="assets/max_mc.png" /></td>
    <td><img src="assets/max_mcs.png" /></td>
  </tr>
</table>

#### Lazy Mode (Max/MSP)

`nn~` includes a `void` mode that allows initializing with a fixed number of inlets / outlets, attaching a model afterwards:

<center>
<img src="assets/max_void.png" width="30%"/>
</center>

#### Special Messages

- `enable [0 / 1]`: Enable / disable computation to save CPU without deleting the model (bypass).
- `reload`: Dynamically reload the model.
- `dump`: Print methods and attributes of the loaded model.
- `print_available_models`: Print models downloadable through the API.
- `download`: Download a model from the API.
- `delete`: Delete a downloaded model.
- `load`: Dynamically change the active model.
- `method`: Dynamically change the active method.

---

### Scripting any PyTorch Model in `nn~`

In the [`scripting`](https://github.com/acids-ircam/nn_tilde/tree/master/scripting) subfolder, you can find a series of examples demonstrating how to export PyTorch models into TorchScript for `nn~`:

- `effects.py`: Apply simple effects to the input (identical input and output shapes).
- `features.py`: Compute spectral descriptors from the PyTorch audio library.
- `unmix.py`: Apply the unmix deep source separation model.

---

### Platform-Specific Legacy Build Recipes

<details>
<summary><b>macOS (with Conda environment)</b></summary>

```bash
git clone https://github.com/acids-ircam/nn_tilde --recurse-submodules
cd nn_tilde
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh > miniconda.sh
chmod +x ./miniconda.sh
bash ./miniconda.sh -b -u -p ./env
source ./env/bin/activate
pip install -r requirements.txt
conda install -c conda-forge curl
mkdir build && cd build
mkdir puredata_include
curl -L https://raw.githubusercontent.com/pure-data/pure-data/master/src/m_pd.h -o puredata_include/m_pd.h
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
cmake ../src -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_PREFIX_PATH=../env/lib/python3.12/site-packages/torch -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.15 -DPUREDATA_INCLUDE_DIR=../puredata_include -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build . --config Release
```
</details>

<details>
<summary><b>Windows (Visual Studio 2022)</b></summary>

```bash
git clone https://github.com/acids-ircam/nn_tilde --recurse-submodules
cd nn_tilde
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.6.0%2Bcpu.zip > "libtorch.zip"
unzip libtorch.zip
mkdir pd && cd pd
curl -L https://msp.ucsd.edu/Software/pd-0.55-2.msw.zip -o pd.zip
unzip pd.zip
mv pd*/src .
mv pd*/bin .
cd ..
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg 
./bootstrap-vcpkg.bat
./vcpkg.exe integrate install
./vcpkg.exe install curl
cd ..
mkdir build && cd build
mkdir puredata_include
curl -L https://raw.githubusercontent.com/pure-data/pure-data/master/src/m_pd.h -o puredata_include/m_pd.h
cmake ../src -G "Visual Studio 17 2022" -DTorch_DIR=../libtorch/share/cmake/Torch -DCMAKE_POLICY_VERSION_MINIMUM=3.15 -DPUREDATA_INCLUDE_DIR=../pd/src -DPUREDATA_BIN_DIR=../pd/bin -A x64
cmake --build . --config Release
```
</details>

<details>
<summary><b>Raspberry Pi (64-bit)</b></summary>

```bash
curl -s https://raw.githubusercontent.com/acids-ircam/nn_tilde/master/install/raspberrypi.sh | bash
```
</details>
