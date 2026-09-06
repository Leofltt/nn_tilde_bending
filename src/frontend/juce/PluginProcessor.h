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
    float getDryWet() const { return m_dryWet.load(); }
    void setDryWet(float value) { m_dryWet.store(juce::jlimit(0.0f, 1.0f, value)); }

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

