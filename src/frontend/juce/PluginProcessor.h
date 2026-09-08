#pragma once
#include <JuceHeader.h>
#include "backend.h"
#include <mutex>
#include <atomic>
#include <vector>
#include <algorithm>

class CircularBuffer {
public:
    CircularBuffer() = default;
    
    CircularBuffer(CircularBuffer&& other) noexcept
    {
        m_buffer = std::move(other.m_buffer);
        m_capacity = other.m_capacity;
        m_writeIndex = other.m_writeIndex;
        m_readIndex = other.m_readIndex;
        m_available.store(other.m_available.load());
    }
    
    CircularBuffer& operator=(CircularBuffer&& other) noexcept
    {
        if (this != &other)
        {
            m_buffer = std::move(other.m_buffer);
            m_capacity = other.m_capacity;
            m_writeIndex = other.m_writeIndex;
            m_readIndex = other.m_readIndex;
            m_available.store(other.m_available.load());
        }
        return *this;
    }
    
    CircularBuffer(const CircularBuffer&) = delete;
    CircularBuffer& operator=(const CircularBuffer&) = delete;

    void init(int size) {
        m_capacity = std::max(size * 4, 8192);
        m_buffer.assign(m_capacity, 0.0f);
        m_writeIndex = 0;
        m_readIndex = 0;
        m_available.store(0);
    }
    
    void put(const float* data, int numSamples) {
        if (m_capacity <= 0 || numSamples <= 0) return;
        for (int i = 0; i < numSamples; ++i) {
            m_buffer[m_writeIndex] = data ? data[i] : 0.0f;
            m_writeIndex = (m_writeIndex + 1) % m_capacity;
        }
        m_available.fetch_add(numSamples);
    }
    
    void get(float* dest, int numSamples) {
        if (m_capacity <= 0 || numSamples <= 0) return;
        int avail = m_available.load();
        int toRead = std::min(numSamples, avail);
        for (int i = 0; i < toRead; ++i) {
            dest[i] = m_buffer[m_readIndex];
            m_readIndex = (m_readIndex + 1) % m_capacity;
        }
        if (toRead < numSamples) {
            std::fill(dest + toRead, dest + numSamples, 0.0f);
        }
        m_available.fetch_sub(toRead);
    }
    
    int getAvailable() const { return m_available.load(); }
    
    void clear() {
        if (m_capacity > 0)
            std::fill(m_buffer.begin(), m_buffer.end(), 0.0f);
        m_writeIndex = 0;
        m_readIndex = 0;
        m_available.store(0);
    }
    
private:
    std::vector<float> m_buffer;
    int m_capacity { 0 };
    int m_writeIndex { 0 };
    int m_readIndex { 0 };
    std::atomic<int> m_available { 0 };
};

class NNBendingAudioProcessor;

class ModelThread : public juce::Thread
{
public:
    ModelThread(NNBendingAudioProcessor& processor);
    ~ModelThread() override;
    
    void run() override;
    void triggerCompute();
    void stop();
    bool isProcessing() const { return m_processing.load(); }
    
private:
    NNBendingAudioProcessor& m_processor;
    juce::WaitableEvent m_event;
    std::atomic<bool> m_processing { false };
};

class NNBendingAudioProcessor  : public juce::AudioProcessor
{
public:
    NNBendingAudioProcessor();
    ~NNBendingAudioProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "nn~ Bending"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int index) override {}
    const juce::String getProgramName (int index) override { return {}; }
    void changeProgramName (int index, const juce::String& newName) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Backend methods
    bool loadModel(const juce::File& file);
    Backend& getBackend() { return m_backend; }
    bool isModelLoaded() const { return m_modelLoaded.load(); }
    juce::String getModelPath() const { return m_modelPath; }
    juce::String getCurrentMethod() const { return m_currentMethod; }
    void setCurrentMethod(const juce::String& method);
    juce::String getCurrentMode() const { return m_currentMethod; }
    void setCurrentMode(const juce::String& mode) { setCurrentMethod(mode); }
    std::vector<juce::String> getAvailableModes() const;
    int getBufferSize() const { return m_bufferSize; }
    void setBufferSize(int size);

    // Latent bending hook (applied during autoencode mode)
    void setLatentHook(Backend::LatentHook hook) { m_latentHook = hook; }

    // Thread communication
    void runInference();

    // Dry / Wet control
    float getDryWet() const { return m_dryWetParam ? m_dryWetParam->get() : m_dryWet.load(); }
    void setDryWet(float value)
    {
        float v = juce::jlimit(0.0f, 1.0f, value);
        m_dryWet.store(v);
        if (m_dryWetParam)
            *m_dryWetParam = v;
    }

    // Layer classification for Trace Isolation
    enum class LayerCategory
    {
        All = 0,
        Norm,       // LayerNorm, BatchNorm, weight_g, scale/shift
        Conv,       // Convolution kernels
        Linear,     // Dense / linear weights
        Bias,       // All bias vectors
        Other       // Any remaining tensor parameters
    };

    static LayerCategory classifyLayer(const std::string& name)
    {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("norm") != std::string::npos || lower.find("weight_g") != std::string::npos)
            return LayerCategory::Norm;
        if (lower.find("bias") != std::string::npos)
            return LayerCategory::Bias;
        if (lower.find("conv") != std::string::npos)
            return LayerCategory::Conv;
        if (lower.find("linear") != std::string::npos || lower.find("dense") != std::string::npos || lower.find("fc") != std::string::npos)
            return LayerCategory::Linear;
        return LayerCategory::Other;
    }

    // Stochastic Drift Modes
    enum class DriftMode
    {
        Noise = 0,    // Classic independent uniform/Gaussian jitter per block
        ThermalOU,    // Ornstein-Uhlenbeck mean-reverting thermal drift
        RandomWalk    // Continuous wandering walk
    };

    // Per-Layer Bending State
    struct LayerBendingState
    {
        float scale { 1.0f };
        float offset { 0.0f };
        float heat { 0.0f };         // Thermal volatility / Jitter depth (sigma)
        float memory { 0.8f };       // Mean reversion drag / elastic pull (theta)
        DriftMode driftMode { DriftMode::ThermalOU };
        bool frozen { false };
        std::vector<float> originalWeights; // Unbent baseline weights W0
        std::vector<float> drawnWeights;    // User-drawn / bent target curve
        std::vector<float> driftOffsets;    // Current stochastic drift vector
    };

    LayerBendingState getLayerState(const std::string& layerName) const
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        auto it = m_layerStates.find(layerName);
        if (it != m_layerStates.end())
            return it->second;
        return LayerBendingState();
    }

    void setLayerState(const std::string& layerName, const LayerBendingState& state)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName] = state;
    }

    void setLayerScale(const std::string& layerName, float scale)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].scale = scale;
    }

    void setLayerOffset(const std::string& layerName, float offset)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].offset = offset;
    }

    void setLayerHeat(const std::string& layerName, float heat)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].heat = std::max(0.0f, heat);
    }
    void setLayerJitter(const std::string& layerName, float jitter) { setLayerHeat(layerName, jitter); }

    void setLayerMemory(const std::string& layerName, float memory)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].memory = juce::jlimit(0.0f, 1.0f, memory);
    }

    void setLayerDriftMode(const std::string& layerName, DriftMode mode)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].driftMode = mode;
    }

    void setLayerFrozen(const std::string& layerName, bool frozen)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].frozen = frozen;
    }

    void setLayerDrawnWeights(const std::string& layerName, const std::vector<float>& weights)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].drawnWeights = weights;
    }

    void setLayerBaseWeights(const std::string& layerName, const std::vector<float>& weights)
    {
        setLayerDrawnWeights(layerName, weights);
    }

    void setLayerOriginalWeights(const std::string& layerName, const std::vector<float>& weights)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].originalWeights = weights;
    }

    void clearLayerBending(const std::string& layerName)
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates.erase(layerName);
    }

    void clearAllLayerBending()
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates.clear();
    }

    bool hasActiveBending() const
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        if (m_layerStates.empty()) return false;
        for (const auto& pair : m_layerStates)
        {
            const auto& s = pair.second;
            if (std::abs(s.scale - 1.0f) > 0.001f || std::abs(s.offset) > 0.001f || s.heat > 0.001f || s.frozen)
                return true;
        }
        return false;
    }

    // Safety Sentry & Blown Fuse Control
    bool isFuseBlown() const { return m_blownFuse.load(); }
    void resetFuse()
    {
        m_blownFuse.store(false);
        clearAllLayerBending();
        m_backend.reset_all_layer_weights();
    }
    void setAutoResetFuse(bool enabled) { m_autoResetFuse.store(enabled); }
    bool getAutoResetFuse() const { return m_autoResetFuse.load(); }

    // Bending Trigger Modes
    enum class TriggerMode
    {
        Continuous = 0, // Default: Always active / continuous bending
        Momentary,      // UI "Short" Button / Parameter hold
        Transient       // Audio sidechain transient follower
    };

    // Momentary Short-Circuit Gate Control
    void triggerShortCircuit(bool active) { m_shortCircuitActive.store(active); }
    bool isShortCircuitActive() const { return m_shortCircuitActive.load(); }
    float getMomentaryEnvelope() const { return m_momentaryEnvelope.load(); }

    void setTriggerMode(TriggerMode mode) { m_triggerMode.store(mode); }
    TriggerMode getTriggerMode() const { return m_triggerMode.load(); }

    void setEnvelopeAttack(float att) { m_envAttack.store(juce::jlimit(0.01f, 1.0f, att)); }
    float getEnvelopeAttack() const { return m_envAttack.load(); }

    void setEnvelopeRelease(float rel) { m_envRelease.store(juce::jlimit(0.5f, 0.999f, rel)); }
    float getEnvelopeRelease() const { return m_envRelease.load(); }

    void setTransientThreshold(float th) { m_transientThreshold.store(juce::jlimit(0.01f, 1.0f, th)); }
    float getTransientThreshold() const { return m_transientThreshold.load(); }

    // Active UI layer target for DAW automation mapping
    void setActiveLayerName(const juce::String& name) { m_activeLayerName = name; }
    juce::String getActiveLayerName() const { return m_activeLayerName; }

    // Automatable parameters accessors
    juce::AudioParameterFloat* getDryWetParam() const { return m_dryWetParam; }
    juce::AudioParameterFloat* getScaleParam() const { return m_scaleParam; }
    juce::AudioParameterFloat* getOffsetParam() const { return m_offsetParam; }
    juce::AudioParameterFloat* getHeatParam() const { return m_heatParam; }
    juce::AudioParameterFloat* getJitterParam() const { return m_heatParam; } // alias
    juce::AudioParameterFloat* getMemoryParam() const { return m_memoryParam; }
    juce::AudioParameterBool*  getShortCircuitParam() const { return m_shortCircuitParam; }

private:
    friend class ModelThread;

    Backend m_backend;
    Backend::LatentHook m_latentHook { nullptr };
    std::atomic<bool> m_modelLoaded { false };
    juce::String m_modelPath;
    juce::String m_currentMethod;
    int m_bufferSize { 2048 };
    int m_model_in { 0 };
    int m_model_out { 0 };

    std::atomic<float> m_dryWet { 1.0f }; // 0.0 = Dry, 1.0 = Wet

    // Safety Sentry state
    std::atomic<bool> m_blownFuse { false };
    std::atomic<bool> m_autoResetFuse { false };
    std::atomic<int>  m_fuseCooldownBlocks { 0 };

    // Momentary Glitch state
    std::atomic<bool>        m_shortCircuitActive { false };
    std::atomic<float>       m_momentaryEnvelope { 0.0f };
    std::atomic<TriggerMode> m_triggerMode { TriggerMode::Continuous };
    std::atomic<float>       m_envAttack { 0.25f };   // Fast ramp-in step (0.01 - 1.0)
    std::atomic<float>       m_envRelease { 0.85f };  // Exponential relaxation multiplier (0.5 - 0.999)
    std::atomic<float>       m_transientThreshold { 0.15f }; // Sidechain audio envelope follower threshold
    std::atomic<bool>        m_midiGateActive { false };
    std::atomic<float>       m_audioEnvelope { 0.0f };

    // DAW Automatable parameters
    juce::AudioParameterFloat* m_dryWetParam { nullptr };
    juce::AudioParameterFloat* m_scaleParam { nullptr };
    juce::AudioParameterFloat* m_offsetParam { nullptr };
    juce::AudioParameterFloat* m_heatParam { nullptr };
    juce::AudioParameterFloat* m_memoryParam { nullptr };
    juce::AudioParameterBool*  m_shortCircuitParam { nullptr };

    // Per-layer states & threading
    juce::String m_activeLayerName;
    std::unordered_map<std::string, LayerBendingState> m_layerStates;
    mutable std::mutex m_layerStateMutex;
    juce::Random m_jitterRng;

    // Buffers and synchronization
    std::vector<CircularBuffer> m_in_buffers;
    std::vector<CircularBuffer> m_out_buffers;
    std::vector<CircularBuffer> m_dry_delay_buffers;
    
    std::vector<std::vector<float>> m_staging_in;
    std::vector<std::vector<float>> m_staging_out;
    
    std::mutex m_staging_mutex;
    std::atomic<bool> m_output_ready { false };

    ModelThread m_model_thread;

    void initBuffers();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NNBendingAudioProcessor)
};

