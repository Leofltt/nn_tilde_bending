#pragma once
#include <JuceHeader.h>
#include "backend.h"
#include <mutex>
#include <atomic>
#include <vector>

class CircularBuffer {
public:
    CircularBuffer() = default;
    
    CircularBuffer(CircularBuffer&& other) noexcept
    {
        buffer = std::move(other.buffer);
        writeIndex = other.writeIndex;
        readIndex = other.readIndex;
        numSamplesAvailable.store(other.numSamplesAvailable.load());
    }
    
    CircularBuffer& operator=(CircularBuffer&& other) noexcept
    {
        if (this != &other)
        {
            buffer = std::move(other.buffer);
            writeIndex = other.writeIndex;
            readIndex = other.readIndex;
            numSamplesAvailable.store(other.numSamplesAvailable.load());
        }
        return *this;
    }
    
    CircularBuffer(const CircularBuffer&) = delete;
    CircularBuffer& operator=(const CircularBuffer&) = delete;

    void init(int size) {
        buffer.assign(size * 4, 0.0f);
        writeIndex = 0;
        readIndex = 0;
        numSamplesAvailable.store(0);
    }
    
    void put(const float* data, int numSamples) {
        if (buffer.empty()) return;
        for (int i = 0; i < numSamples; ++i) {
            buffer[writeIndex] = data[i];
            writeIndex = (writeIndex + 1) % buffer.size();
        }
        numSamplesAvailable += numSamples;
    }
    
    void get(float* dest, int numSamples) {
        if (buffer.empty()) {
            std::fill(dest, dest + numSamples, 0.0f);
            return;
        }
        for (int i = 0; i < numSamples; ++i) {
            dest[i] = buffer[readIndex];
            readIndex = (readIndex + 1) % buffer.size();
        }
        numSamplesAvailable -= numSamples;
    }
    
    int getAvailable() const { return numSamplesAvailable.load(); }
    
    void clear() {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        writeIndex = 0;
        readIndex = 0;
        numSamplesAvailable.store(0);
    }
    
private:
    std::vector<float> buffer;
    int writeIndex = 0;
    int readIndex = 0;
    std::atomic<int> numSamplesAvailable { 0 };
};

class NNBendingAudioProcessor;

class ModelThread : public juce::Thread
{
public:
    ModelThread(NNBendingAudioProcessor& processor);
    ~ModelThread() override;
    
    void run() override;
    void triggerCompute();
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
    int getBufferSize() const { return m_bufferSize; }
    void setBufferSize(int size);

    // Thread communication
    void runInference();

private:
    friend class ModelThread;

    Backend m_backend;
    std::atomic<bool> m_modelLoaded { false };
    juce::String m_modelPath;
    juce::String m_currentMethod;
    int m_bufferSize { 2048 };
    int m_model_in { 0 };
    int m_model_out { 0 };

    // Buffers and synchronization
    std::vector<CircularBuffer> m_in_buffers;
    std::vector<CircularBuffer> m_out_buffers;
    std::vector<std::vector<float>> m_in_model_data;
    std::vector<std::vector<float>> m_out_model_data;
    
    std::vector<std::vector<float>> m_in_thread_data;
    std::vector<std::vector<float>> m_out_thread_data;
    
    std::mutex m_data_mutex;
    std::atomic<bool> m_output_ready { false };

    ModelThread m_model_thread;

    void initBuffers();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NNBendingAudioProcessor)
};
