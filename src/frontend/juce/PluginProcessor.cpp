#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <random>
#include <cmath>

//==============================================================================
ModelThread::ModelThread(NNBendingAudioProcessor& processor)
    : juce::Thread("Model Thread"), m_processor(processor)
{
}

ModelThread::~ModelThread()
{
    stop();
}

void ModelThread::stop()
{
    signalThreadShouldExit();
    m_event.signal();
    stopThread(2000);
}

void ModelThread::run()
{
    while (!threadShouldExit())
    {
        // Wait for trigger signal (timeout 20ms to allow responsive shutdown or idle checks)
        m_event.wait(20);
        
        if (threadShouldExit())
            break;
            
        if (m_processing.load())
        {
            m_processor.runInference();
            m_processing.store(false);
        }
    }
}

void ModelThread::triggerCompute()
{
    m_processing.store(true);
    m_event.signal();
}

//==============================================================================
NNBendingAudioProcessor::NNBendingAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
       m_model_thread(*this)
#endif
{
    addParameter (m_dryWetParam = new juce::AudioParameterFloat (
        juce::ParameterID ("dry_wet", 1), "Dry / Wet",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 1.0f
    ));

    addParameter (m_scaleParam = new juce::AudioParameterFloat (
        juce::ParameterID ("scale", 1), "Layer Scale",
        juce::NormalisableRange<float> (0.0f, 5.0f, 0.01f), 1.0f
    ));

    addParameter (m_offsetParam = new juce::AudioParameterFloat (
        juce::ParameterID ("offset", 1), "Layer Offset",
        juce::NormalisableRange<float> (-2.0f, 2.0f, 0.001f), 0.0f
    ));

    addParameter (m_heatParam = new juce::AudioParameterFloat (
        juce::ParameterID ("heat", 1), "Layer Heat / Volatility",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f), 0.0f
    ));

    addParameter (m_memoryParam = new juce::AudioParameterFloat (
        juce::ParameterID ("memory", 1), "Drift Memory / Drag",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f), 0.8f
    ));

    addParameter (m_shortCircuitParam = new juce::AudioParameterBool (
        juce::ParameterID ("short_circuit", 1), "Momentary Short Circuit", false
    ));

    addParameter (m_latentXParam = new juce::AudioParameterFloat (
        juce::ParameterID ("latent_x", 1), "Latent Terrain X",
        juce::NormalisableRange<float> (-1.0f, 1.0f, 0.001f), 0.0f
    ));

    addParameter (m_latentYParam = new juce::AudioParameterFloat (
        juce::ParameterID ("latent_y", 1), "Latent Terrain Y",
        juce::NormalisableRange<float> (-1.0f, 1.0f, 0.001f), 0.0f
    ));

    addParameter (m_latentDepthParam = new juce::AudioParameterFloat (
        juce::ParameterID ("latent_depth", 1), "Latent Depth",
        juce::NormalisableRange<float> (0.0f, 2.0f, 0.01f), 0.5f
    ));

    m_backend = BackendFactory::createDefault();
}

NNBendingAudioProcessor::~NNBendingAudioProcessor()
{
    m_model_thread.stop();
}

//==============================================================================
void NNBendingAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(samplesPerBlock);
    m_hostSampleRate.store(sampleRate);
    if (m_backend)
        m_backend->set_sample_rate(sampleRate);
    if (m_modelLoaded.load() && m_backend)
    {
        double modelSr = m_backend->get_sample_rate();
        m_modelSampleRate.store(modelSr);
        m_resamplingActive.store(std::abs(sampleRate - modelSr) > 1.0);
    }
    initBuffers();
}

void NNBendingAudioProcessor::releaseResources()
{
    m_model_thread.stop();
}

bool NNBendingAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}

void NNBendingAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    const juce::ScopedTryLock sl (getCallbackLock());
    if (!sl.isLocked())
    {
        buffer.clear();
        return;
    }

    // Process incoming MIDI triggers (Note On kicks momentary glitch; Note Off releases)
    for (const auto metadata : midiMessages)
    {
        auto msg = metadata.getMessage();
        if (msg.isNoteOn())
        {
            m_midiGateActive.store(true);
        }
        else if (msg.isNoteOff())
        {
            m_midiGateActive.store(false);
        }
    }

    // Clear unused output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    if (!m_modelLoaded.load() || m_model_out <= 0)
    {
        for (auto i = 0; i < buffer.getNumChannels(); ++i)
            buffer.clear(i, 0, buffer.getNumSamples());
        return;
    }

    auto numSamples = buffer.getNumSamples();

    // Sidechain Audio Envelope Follower for Transient Triggering
    if (totalNumInputChannels > 0 && numSamples > 0)
    {
        float maxPeak = 0.0f;
        for (int c = 0; c < totalNumInputChannels; ++c)
        {
            float peak = buffer.getMagnitude(c, 0, numSamples);
            if (peak > maxPeak) maxPeak = peak;
        }
        float prevEnv = m_audioEnvelope.load();
        float newEnv = std::max(maxPeak, prevEnv * 0.9f); // Fast attack, smooth decay
        m_audioEnvelope.store(newEnv);
    }
    
    // 1. If inference output is ready from background thread, transfer to output ring buffers
    if (m_output_ready.load())
    {
        std::unique_lock<std::mutex> lock(m_staging_mutex, std::try_to_lock);
        if (lock.owns_lock())
        {
            if (m_output_ready.load())
            {
                int n_outs = std::min((int)m_out_buffers.size(), (int)m_staging_out.size());
                for (int c = 0; c < n_outs; ++c)
                {
                    m_out_buffers[c].put(m_staging_out[c].data(), m_bufferSize);
                }
                m_output_ready.store(false);
            }
        }
    }
    
    // 2. Feed incoming audio into dry delay buffers (for all modes) and model inputs (for forward mode)
    int numDryChannels = (int)m_dry_delay_buffers.size();
    for (int c = 0; c < numDryChannels; ++c)
    {
        const float* inPtr = nullptr;
        if (c < totalNumInputChannels)
            inPtr = buffer.getReadPointer(c);
        else if (totalNumInputChannels > 0)
            inPtr = buffer.getReadPointer(0);

        m_dry_delay_buffers[c].put(inPtr, numSamples);
    }

    bool resampling = m_resamplingActive.load();
    double hostSr = m_hostSampleRate.load();
    double modelSr = m_modelSampleRate.load();

    if (m_model_in > 0)
    {
        if (resampling && hostSr > 0.0 && modelSr > 0.0)
        {
            // Resample host input -> model sample rate
            double inRatio = hostSr / modelSr;
            int numModelSamples = (int)std::ceil(numSamples / inRatio);
            if (numModelSamples > 0)
            {
                for (int c = 0; c < m_model_in; ++c)
                {
                    const float* inPtr = nullptr;
                    if (c < totalNumInputChannels)
                        inPtr = buffer.getReadPointer(c);
                    else if (totalNumInputChannels > 0)
                        inPtr = buffer.getReadPointer(0);

                    if (c < (int)m_input_resamplers.size() && c < (int)m_resampled_in_scratch.size())
                    {
                        int targetSamples = std::min(numModelSamples, (int)m_resampled_in_scratch[c].size());

                        m_input_resamplers[c].process(
                            inRatio,
                            inPtr,
                            m_resampled_in_scratch[c].data(),
                            targetSamples
                        );

                        m_in_buffers[c].put(m_resampled_in_scratch[c].data(), targetSamples);
                    }
                }
            }
        }
        else
        {
            // Direct transfer without resampling
            for (int c = 0; c < m_model_in; ++c)
            {
                const float* inPtr = nullptr;
                if (c < totalNumInputChannels)
                    inPtr = buffer.getReadPointer(c);
                else if (totalNumInputChannels > 0)
                    inPtr = buffer.getReadPointer(0);

                m_in_buffers[c].put(inPtr, numSamples);
            }
        }
        
        // Check if we have enough samples to trigger inference and thread is idle
        if (!m_model_thread.isProcessing() && m_in_buffers[0].getAvailable() >= m_bufferSize)
        {
            std::unique_lock<std::mutex> lock(m_staging_mutex, std::try_to_lock);
            if (lock.owns_lock())
            {
                for (int c = 0; c < m_model_in; ++c)
                {
                    m_in_buffers[c].get(m_staging_in[c].data(), m_bufferSize);
                }
                m_model_thread.triggerCompute();
            }
        }
    }
    else
    {
        // Generative model (0 audio inputs): trigger inference whenever output buffer needs filling
        if (!m_model_thread.isProcessing() && m_out_buffers[0].getAvailable() < m_bufferSize * 3)
        {
            m_model_thread.triggerCompute();
        }
    }
    
    // 3. Pull processed samples from output circular buffers (resampling model SR -> host SR if active)
    int n_outs = std::min((int)totalNumOutputChannels, m_model_out);
    if (resampling && hostSr > 0.0 && modelSr > 0.0)
    {
        double outRatio = modelSr / hostSr;
        int masterConsumed = 0;

        for (int c = 0; c < n_outs; ++c)
        {
            if (c < (int)m_output_resamplers.size() && c < (int)m_resampled_out_scratch.size())
            {
                int maxScratch = (int)m_resampled_out_scratch[c].size();
                m_out_buffers[c].peek(m_resampled_out_scratch[c].data(), maxScratch);

                int consumed = m_output_resamplers[c].process(
                    outRatio,
                    m_resampled_out_scratch[c].data(),
                    buffer.getWritePointer(c),
                    numSamples
                );

                if (c == 0) masterConsumed = consumed;
            }
            else
            {
                buffer.clear(c, 0, numSamples);
            }
        }

        // Lock sample consumption across all channels to maintain strict stereo phase alignment
        for (int c = 0; c < n_outs; ++c)
        {
            if (c < (int)m_output_resamplers.size() && c < (int)m_resampled_out_scratch.size())
            {
                m_out_buffers[c].drop(masterConsumed);
            }
        }
    }
    else
    {
        for (int c = 0; c < n_outs; ++c)
        {
            m_out_buffers[c].get(buffer.getWritePointer(c), numSamples);
        }
    }
    
    // If model is mono (1 out) and host is stereo (2 out), duplicate Left to Right channel
    if (m_model_out == 1 && totalNumOutputChannels >= 2)
    {
        buffer.copyFrom(1, 0, buffer, 0, 0, numSamples);
    }
    // Clear any extra output channels beyond model outputs
    for (int c = std::max(n_outs, (m_model_out == 1 ? 2 : 1)); c < totalNumOutputChannels; ++c)
    {
        buffer.clear(c, 0, numSamples);
    }

    // 4. Dry/Wet Blending & Safety Sentry
    float wetGain = m_dryWet.load();
    float dryGain = 1.0f - wetGain;
    bool fuseBlown = m_blownFuse.load();

    std::vector<float> dryBlock(numSamples, 0.0f);
    for (int c = 0; c < (int)totalNumOutputChannels; ++c)
    {
        int dryCh = std::min(c, (int)m_dry_delay_buffers.size() - 1);
        if (dryCh >= 0)
        {
            m_dry_delay_buffers[dryCh].get(dryBlock.data(), numSamples);
            float* outPtr = buffer.getWritePointer(c);
            
            if (fuseBlown)
            {
                // Blown fuse: bypass wet model output and pass pure dry audio to protect listeners
                for (int i = 0; i < numSamples; ++i)
                {
                    outPtr[i] = dryBlock[i];
                }
            }
            else
            {
                for (int i = 0; i < numSamples; ++i)
                {
                    float rawWet = outPtr[i];

                    // NaN & Inf Sentry Check
                    if (std::isnan(rawWet) || std::isinf(rawWet))
                    {
                        m_blownFuse.store(true);
                        outPtr[i] = 0.0f;
                        continue;
                    }

                    // Transparent soft headroom saturation on wet model output:
                    // Linear unity gain for |x| <= 0.6, smoothly compressing transient peaks to <= 1.3f
                    float satWet = (std::abs(rawWet) < 0.6f) ? rawWet : (std::tanh(rawWet * 0.77f) * 1.3f);
                    float sample = satWet * wetGain + dryBlock[i] * dryGain;

                    // Hard cutoff safety check for runaway signals
                    if (std::isnan(sample) || std::isinf(sample) || std::abs(sample) > 2.0f)
                    {
                        m_blownFuse.store(true);
                        sample = 0.0f; // Instantly silence runaway sample
                    }
                    else
                    {
                        // Soft clamp to +/- 1.5 to protect output converters
                        sample = juce::jlimit(-1.5f, 1.5f, sample);
                    }
                    outPtr[i] = sample;
                }
            }
        }
    }

    // Auto-recovery handling if enabled
    if (m_blownFuse.load() && m_autoResetFuse.load())
    {
        int cooldown = m_fuseCooldownBlocks.fetch_add(1);
        if (cooldown > 40) // ~0.5-1.0s depending on block size
        {
            m_fuseCooldownBlocks.store(0);
            resetFuse();
        }
    }
}

//==============================================================================
std::vector<juce::String> NNBendingAudioProcessor::getAvailableModes() const
{
    std::vector<juce::String> modes;
    if (!m_modelLoaded.load() || !m_backend)
        return modes;

    auto backendModes = m_backend->get_plugin_modes();
    for (const auto& m : backendModes)
        modes.emplace_back(juce::String(m));
    return modes;
}

bool NNBendingAudioProcessor::loadModel(const juce::File& file)
{
    m_modelLoaded.store(false);
    m_model_thread.stop();
    
    std::string path = file.getFullPathName().toStdString();
    
    auto newBackend = BackendFactory::createForFile(path);
    if (!newBackend)
        return false;

    if (file.getFileName().containsIgnoreCase("_coreml.pte") || file.getFileName().containsIgnoreCase("_mps.pte"))
        m_useGpu.store(true);
    else if (file.getFileName().containsIgnoreCase("_xnnpack.pte") || file.getFileName().containsIgnoreCase("_portable.pte"))
        m_useGpu.store(false);

    newBackend->use_gpu(m_useGpu.load());
    m_backend = std::move(newBackend);

    int err = m_backend->load(path, getSampleRate());
    if (err == 0)
    {
        m_modelPath = file.getFullPathName();
        
        // Find default mode: prefer "forward", else "prior", else first available
        auto modes = m_backend->get_plugin_modes();
        if (!modes.empty())
        {
            std::string defaultMode = modes[0];
            for (const auto& m : modes)
            {
                if (m == "forward")
                {
                    defaultMode = "forward";
                    break;
                }
            }
            m_currentMethod = defaultMode;
            
            auto params = m_backend->get_mode_params(defaultMode);
            if (params.size() >= 4)
            {
                m_model_in = params[0];
                m_model_out = params[2];
            }
        }

        // Auto-configure buffer size to model ratio.
        // For deep convolutional autoencoders (such as Stable Audio / Oobleck / DAC / RAVE 2048),
        // each chunk needs multiple latent frames (at least 2 frames, >= 4096 samples)
        // to satisfy convolutional receptive field and streaming padding.
        int higher_ratio = m_backend->get_higher_ratio();
        int reqSize = 1;
        while (reqSize < higher_ratio)
            reqSize <<= 1;
        int minBufferForRatio = (higher_ratio >= 2048) ? 4096 : 2048;
        m_bufferSize = std::max(minBufferForRatio, reqSize * 2);
        
        // Reset latent terrain cached matrices on new model load
        m_rffMatrix.clear();
        m_prevLatentBuffer.clear();
        m_cachedLatentDim = 0;

        // Query model sample rate vs host sample rate
        double hostSr = getSampleRate();
        m_hostSampleRate.store(hostSr > 0.0 ? hostSr : 44100.0);
        double modelSr = m_backend->get_sample_rate();
        m_modelSampleRate.store(modelSr);
        m_resamplingActive.store(std::abs(m_hostSampleRate.load() - modelSr) > 1.0);

        m_modelLoaded.store(true);
        initBuffers();
        return true;
    }
    
    return false;
}

void NNBendingAudioProcessor::setCurrentMethod(const juce::String& method)
{
    if (!m_backend) return;
    std::string methodStr = method.toStdString();
    auto modes = m_backend->get_plugin_modes();
    bool isValidMode = (std::find(modes.begin(), modes.end(), methodStr) != modes.end())
                    || m_backend->has_method(methodStr);

    if (isValidMode)
    {
        m_model_thread.stop();

        m_currentMethod = method;
        auto params = m_backend->get_mode_params(methodStr);
        if (params.size() >= 4)
        {
            m_model_in = params[0];
            m_model_out = params[2];
        }
        initBuffers();
    }
}

void NNBendingAudioProcessor::setBufferSize(int size)
{
    if (size > 0 && (size & (size - 1)) == 0) // Power of two check
    {
        m_bufferSize = size;
        initBuffers();
    }
}

void NNBendingAudioProcessor::setUseGpu(bool enabled)
{
    if (m_useGpu.load() == enabled && m_modelLoaded.load())
        return;

    m_useGpu.store(enabled);
    if (!m_modelLoaded.load() || m_modelPath.isEmpty())
    {
        if (m_backend)
            m_backend->use_gpu(enabled);
        return;
    }

    juce::String savedMethod = m_currentMethod;
    juce::File targetFile(m_modelPath);

    // For .pte models, automatically swap between sibling *_xnnpack.pte (CPU) and *_coreml.pte (GPU/ANE) if present
    if (m_modelPath.endsWithIgnoreCase(".pte"))
    {
        juce::String candidatePath = m_modelPath;
        if (enabled && candidatePath.containsIgnoreCase("_xnnpack.pte"))
            candidatePath = candidatePath.replace("_xnnpack.pte", "_coreml.pte", true);
        else if (!enabled && candidatePath.containsIgnoreCase("_coreml.pte"))
            candidatePath = candidatePath.replace("_coreml.pte", "_xnnpack.pte", true);

        juce::File candidateFile(candidatePath);
        if (candidateFile.existsAsFile())
            targetFile = candidateFile;
    }

    if (loadModel(targetFile))
    {
        if (savedMethod.isNotEmpty())
            setCurrentMethod(savedMethod);
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        for (auto& pair : m_layerStates)
            pair.second.isApplied = false;
    }
}

void NNBendingAudioProcessor::initBuffers()
{
    m_model_thread.stop();
    
    const juce::ScopedLock sl (getCallbackLock());
    std::lock_guard<std::mutex> lock(m_staging_mutex);
    
    int n_in = std::max(1, m_model_in);
    int n_out = std::max(1, m_model_out);
    
    m_in_buffers.resize(n_in);
    m_out_buffers.resize(n_out);
    
    m_staging_in.resize(n_in);
    m_staging_out.resize(n_out);
    m_worker_in.resize(n_in);
    m_worker_out.resize(n_out);
    
    // In forward mode, delay is 2 * m_bufferSize (1 buffer input accumulation + 1 buffer model inference transfer)
    double hostSr = m_hostSampleRate.load();
    double modelSr = m_modelSampleRate.load();
    bool resampling = m_resamplingActive.load() && (hostSr > 0.0) && (modelSr > 0.0);
    
    // Scale latency to host sample rate if resampling is active
    int latencySamples = 0;
    if (m_model_in > 0)
    {
        double latencyModel = m_bufferSize * 2.0;
        latencySamples = resampling ? (int)std::round(latencyModel * (hostSr / modelSr)) : (m_bufferSize * 2);
    }
    setLatencySamples(latencySamples);

    int n_dry = std::max(2, std::max(n_in, n_out));
    m_dry_delay_buffers.resize(n_dry);

    // Initialize resamplers and scratch storage
    m_input_resamplers.resize(n_in);
    for (auto& r : m_input_resamplers) r.reset();

    m_output_resamplers.resize(n_out);
    for (auto& r : m_output_resamplers) r.reset();

    int maxScratchSize = std::max(8192, m_bufferSize * 4);
    m_resampled_in_scratch.resize(n_in);
    for (auto& buf : m_resampled_in_scratch)
        buf.assign(maxScratchSize, 0.0f);

    m_resampled_out_scratch.resize(n_out);
    for (auto& buf : m_resampled_out_scratch)
        buf.assign(maxScratchSize, 0.0f);

    for (int i = 0; i < n_in; ++i)
    {
        m_in_buffers[i].init(m_bufferSize);
        m_staging_in[i].assign(m_bufferSize, 0.0f);
        m_worker_in[i].assign(m_bufferSize, 0.0f);
    }

    for (int i = 0; i < n_dry; ++i)
    {
        int dryCap = std::max(m_bufferSize, latencySamples);
        m_dry_delay_buffers[i].init(dryCap);
        // Pre-fill dry delay buffer with latency zeros so dry aligns with wet output
        if (latencySamples > 0)
            m_dry_delay_buffers[i].put(nullptr, latencySamples);
    }
    
    for (int i = 0; i < n_out; ++i)
    {
        m_out_buffers[i].init(m_bufferSize);
        m_staging_out[i].assign(m_bufferSize, 0.0f);
        m_worker_out[i].assign(m_bufferSize, 0.0f);
        // Pre-fill output buffer with 1 buffer of latency cushion in forward mode
        // to prevent starvation during background model inference and align with dry delay buffer
        if (m_model_in > 0)
        {
            m_out_buffers[i].put(nullptr, m_bufferSize);
        }
    }
    
    m_output_ready.store(false);
    
    if (m_backend)
    {
        m_backend->clear_decode_gen_fifo();
    }
    
    if (m_modelLoaded.load())
    {
        m_model_thread.startThread(juce::Thread::Priority::high);
    }
}

void NNBendingAudioProcessor::runInference()
{
    if (!m_modelLoaded.load() || m_model_out <= 0)
        return;

    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        std::string activeLayer = m_activeLayerName.toStdString();
        if (!activeLayer.empty() && m_scaleParam && m_offsetParam && m_heatParam)
        {
            auto& state = m_layerStates[activeLayer];
            float newScale = m_scaleParam->get();
            float newOffset = m_offsetParam->get();
            float newHeat = m_heatParam->get();
            float newMemory = m_memoryParam ? m_memoryParam->get() : state.memory;

            if (std::abs(state.scale - newScale) > 0.0001f ||
                std::abs(state.offset - newOffset) > 0.0001f ||
                std::abs(state.heat - newHeat) > 0.0001f ||
                std::abs(state.memory - newMemory) > 0.0001f)
            {
                state.scale = newScale;
                state.offset = newOffset;
                state.heat = newHeat;
                state.memory = newMemory;
                state.isApplied = false; // Trigger re-application
            }
        }

        // Calculate momentary envelope progression based on TriggerMode:
        TriggerMode mode = m_triggerMode.load();
        bool shortActive = false;

        if (mode == TriggerMode::Momentary)
        {
            shortActive = m_shortCircuitActive.load() || (m_shortCircuitParam && m_shortCircuitParam->get());
        }
        else if (mode == TriggerMode::Transient)
        {
            shortActive = (m_audioEnvelope.load() >= m_transientThreshold.load());
        }

        float currentEnv = m_momentaryEnvelope.load();
        float att = m_envAttack.load();
        float rel = m_envRelease.load();

        if (mode == TriggerMode::Continuous)
        {
            // In continuous mode, bending is always 100% active
            currentEnv = 1.0f;
        }
        else
        {
            if (shortActive)
                currentEnv = std::min(1.0f, currentEnv + att);
            else
                currentEnv = std::max(0.0f, currentEnv * rel);
        }

        m_momentaryEnvelope.store(currentEnv);

        for (auto& pair : m_layerStates)
        {
            const std::string& layerName = pair.first;
            LayerBendingState& state = pair.second;
            if (state.originalWeights.empty())
            {
                state.originalWeights = m_backend->get_original_layer_weights(layerName);
            }
            if (state.originalWeights.empty())
                continue;

            size_t numWeights = state.originalWeights.size();

            if (state.drawnWeights.size() != numWeights)
            {
                state.drawnWeights = state.originalWeights;
            }

            if (state.driftOffsets.size() != numWeights)
            {
                state.driftOffsets.assign(numWeights, 0.0f);
            }

            // Cross-Talk (Trace Bridging): Blend target layer's base shape with a source layer
            if (state.scratchBridged.size() != numWeights)
                state.scratchBridged.resize(numWeights);
            std::memcpy(state.scratchBridged.data(), state.drawnWeights.data(), numWeights * sizeof(float));

            if (state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != layerName)
            {
                auto sourceWeights = m_backend->get_original_layer_weights(state.bridgeSourceLayer);
                if (!sourceWeights.empty())
                {
                    float alpha = juce::jlimit(0.0f, 1.0f, state.bridgeDepth);
                    size_t srcSize = sourceWeights.size();
                    for (size_t i = 0; i < numWeights; ++i)
                    {
                        float srcVal = sourceWeights[i % srcSize];
                        state.scratchBridged[i] = (1.0f - alpha) * state.drawnWeights[i] + alpha * srcVal;
                    }
                }
            }

            if (state.scratchFinal.size() != numWeights)
                state.scratchFinal.resize(numWeights);

            // Compute stochastic drift when heat > 0 and not frozen
            if (state.heat > 0.0001f && !state.frozen)
            {
                float heat = state.heat;
                float memory = juce::jlimit(0.0f, 0.999f, state.memory);

                for (size_t i = 0; i < numWeights; ++i)
                {
                    float noise = (m_jitterRng.nextFloat() * 2.0f - 1.0f) * heat;

                    if (state.driftMode == DriftMode::ThermalOU)
                    {
                        // Ornstein-Uhlenbeck: dW = -theta * W + sigma * dW_noise
                        state.driftOffsets[i] = (state.driftOffsets[i] * memory) + noise * (1.0f - memory);
                    }
                    else if (state.driftMode == DriftMode::RandomWalk)
                    {
                        // Continuous wandering walk with soft bounds
                        state.driftOffsets[i] = juce::jlimit(-1.0f, 1.0f, state.driftOffsets[i] + noise * 0.1f);
                    }
                    else // Classic Noise / Glitch
                    {
                        state.driftOffsets[i] = noise;
                    }

                    // Calculate bent target weight (Bridged base shape * Static Scale + Offset + Stochastic Drift)
                    float bentWeight = (state.scratchBridged[i] * state.scale + state.offset) + state.driftOffsets[i];

                    if (mode == TriggerMode::Continuous)
                    {
                        state.scratchFinal[i] = bentWeight;
                    }
                    else
                    {
                        // Crossfade from unbent baseline W0 (originalWeights) to target bent W based on envelope
                        state.scratchFinal[i] = state.originalWeights[i] + currentEnv * (bentWeight - state.originalWeights[i]);
                    }
                }
                m_backend->set_layer_weights(layerName, state.scratchFinal);
                state.isApplied = true;
                state.lastAppliedEnv = currentEnv;
            }
            else if (mode != TriggerMode::Continuous && currentEnv > 0.001f)
            {
                // Momentary gate active without heat: only recompute if envelope moved significantly
                if (std::abs(currentEnv - state.lastAppliedEnv) > 0.002f)
                {
                    // Vectorized: bentWeight = bridgedBase * scale + offset
                    juce::FloatVectorOperations::copy(state.scratchFinal.data(), state.scratchBridged.data(), (int)numWeights);
                    juce::FloatVectorOperations::multiply(state.scratchFinal.data(), state.scale, (int)numWeights);
                    juce::FloatVectorOperations::add(state.scratchFinal.data(), state.offset, (int)numWeights);

                    // Vectorized crossfade: finalWeights = originalWeights * (1 - env) + bentWeight * env
                    juce::FloatVectorOperations::multiply(state.scratchFinal.data(), currentEnv, (int)numWeights);
                    juce::FloatVectorOperations::addWithMultiply(state.scratchFinal.data(), state.originalWeights.data(), 1.0f - currentEnv, (int)numWeights);

                    m_backend->set_layer_weights(layerName, state.scratchFinal);
                    state.isApplied = true;
                    state.lastAppliedEnv = currentEnv;
                }
            }
            else if (mode == TriggerMode::Continuous)
            {
                // Continuous mode without drift: apply bent target once and skip redundant uploads
                if (!state.isApplied)
                {
                    // Vectorized: finalWeights = bridgedBase * scale + offset
                    juce::FloatVectorOperations::copy(state.scratchFinal.data(), state.scratchBridged.data(), (int)numWeights);
                    juce::FloatVectorOperations::multiply(state.scratchFinal.data(), state.scale, (int)numWeights);
                    juce::FloatVectorOperations::add(state.scratchFinal.data(), state.offset, (int)numWeights);

                    m_backend->set_layer_weights(layerName, state.scratchFinal);
                    state.isApplied = true;
                }
            }
            else if (mode != TriggerMode::Continuous && currentEnv <= 0.001f)
            {
                // Momentary gate inactive: restore unbent baseline weights once
                if (state.isApplied)
                {
                    m_backend->set_layer_weights(layerName, state.originalWeights);
                    state.isApplied = false;
                    state.lastAppliedEnv = 0.0f;
                }
            }
        }
    }
        
    // 3. Copy audio inputs from staging buffers to local worker buffers under a brief lock
    {
        std::lock_guard<std::mutex> lock(m_staging_mutex);
        for (int c = 0; c < m_model_in; ++c)
        {
            if (c < (int)m_staging_in.size() && c < (int)m_worker_in.size())
            {
                std::memcpy(m_worker_in[c].data(), m_staging_in[c].data(), m_bufferSize * sizeof(float));
            }
        }
    }
    std::vector<float*> in_ptrs;
    std::vector<float*> out_ptrs;
    
    int numIn = std::min(m_model_in, (int)m_worker_in.size());
    for (int c = 0; c < numIn; ++c)
        in_ptrs.emplace_back(m_worker_in[c].data());
        
    int numOut = std::min(m_model_out, (int)m_worker_out.size());
    for (int c = 0; c < numOut; ++c)
        out_ptrs.emplace_back(m_worker_out[c].data());
        
    std::string modeStr = m_currentMethod.toStdString();
    if (modeStr == "forward")
    {
        // Compose latent hook: if user hook is set, use it; otherwise if latent terrain is enabled, execute it!
        IBackend::LatentHook activeHook = m_latentHook;
        if (!activeHook && m_latentEnabled.load())
        {
            activeHook = [this](IBackend::LatentBuffer& buf) {
                applyLatentTerrainHook(buf);
            };
        }

        if (m_backend)
            m_backend->perform_forward(in_ptrs, out_ptrs, 1, m_model_out, m_bufferSize, activeHook);
    }
    else if (modeStr == "decode gen")
    {
        // Autonomous neural synthesis: generate latent trajectory with Pad/Drift/Harmonics and decode
        IBackend::LatentHook activeHook = m_latentHook;
        if (!activeHook)
        {
            activeHook = [this](IBackend::LatentBuffer& buf) {
                applyLatentTerrainHook(buf);
            };
        }

        if (m_backend)
            m_backend->perform_decode_gen(out_ptrs, 1, m_model_out, m_bufferSize, activeHook);
    }
    else if (modeStr == "prior" || modeStr == "generate")
    {
        if (m_backend)
            m_backend->perform_prior_decode(out_ptrs, 1, m_model_out, m_bufferSize);
    }
    else
    {
        if (m_backend)
            m_backend->perform(in_ptrs, out_ptrs, modeStr, 1, m_model_out, m_bufferSize);
    }

    // 5. Transfer computed output back to staging_out under a brief lock
    {
        std::lock_guard<std::mutex> lock(m_staging_mutex);
        for (int c = 0; c < m_model_out; ++c)
        {
            if (c < (int)m_staging_out.size() && c < (int)m_worker_out.size())
            {
                std::memcpy(m_staging_out[c].data(), m_worker_out[c].data(), m_bufferSize * sizeof(float));
            }
        }
    }
    
    m_output_ready.store(true);
}

void NNBendingAudioProcessor::applyLatentTerrainHook(IBackend::LatentBuffer& buf)
{
    if (buf.data == nullptr || buf.channels <= 0 || buf.length <= 0)
        return;

    float lx = m_latentXParam ? m_latentXParam->get() : m_latentX.load();
    float ly = m_latentYParam ? m_latentYParam->get() : m_latentY.load();
    float depth = m_latentDepthParam ? m_latentDepthParam->get() : m_latentDepth.load();
    auto mode = m_latentMode.load();

    int64_t latentDim = buf.channels;
    int64_t length = buf.length;
    size_t totalElems = (size_t)(latentDim * length);

    if (mode == LatentMode::FourierOrbit)
    {
        int64_t numHarmonics = std::max((int64_t)1, latentDim / 2);
        if (m_rffMatrix.size() != (size_t)(numHarmonics * 2) || m_cachedLatentDim != (int)latentDim)
        {
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 2.5f);
            m_rffMatrix.resize((size_t)(numHarmonics * 2));
            for (auto& val : m_rffMatrix)
                val = dist(rng);
            m_cachedLatentDim = (int)latentDim;
        }

        float normDepth = depth * (std::sqrt(16.0f) / std::sqrt((float)std::max((int64_t)1, latentDim)));

        for (int64_t h = 0; h < numHarmonics; ++h)
        {
            float proj = m_rffMatrix[(size_t)(h * 2)] * lx + m_rffMatrix[(size_t)(h * 2 + 1)] * ly;
            float cosOffset = normDepth * std::cos(proj);
            float sinOffset = normDepth * std::sin(proj);

            if (h < latentDim)
            {
                float* chPtr = buf.data + (h * length);
                for (int64_t t = 0; t < length; ++t)
                    chPtr[t] += cosOffset;
            }
            if ((numHarmonics + h) < latentDim)
            {
                float* chPtr = buf.data + ((numHarmonics + h) * length);
                for (int64_t t = 0; t < length; ++t)
                    chPtr[t] += sinOffset;
            }
        }
    }
    else if (mode == LatentMode::LatentSlew)
    {
        float slew = m_latentSlew.load();
        if (m_prevLatentBuffer.size() == totalElems)
        {
            for (size_t i = 0; i < totalElems; ++i)
            {
                buf.data[i] = (1.0f - slew) * buf.data[i] + slew * m_prevLatentBuffer[i];
            }
        }
        else
        {
            m_prevLatentBuffer.resize(totalElems);
        }
        std::memcpy(m_prevLatentBuffer.data(), buf.data, totalElems * sizeof(float));
    }
    else if (mode == LatentMode::Quantize)
    {
        float qStep = std::max(0.01f, depth * 0.5f);
        for (size_t i = 0; i < totalElems; ++i)
        {
            buf.data[i] = std::round(buf.data[i] / qStep) * qStep;
        }
    }
}

//==============================================================================
void NNBendingAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::XmlElement xmlState ("NNBendingSettings");
    xmlState.setAttribute ("modelPath", m_modelPath);
    xmlState.setAttribute ("currentMethod", m_currentMethod);
    xmlState.setAttribute ("bufferSize", m_bufferSize);
    xmlState.setAttribute ("activeLayer", m_activeLayerName);
    xmlState.setAttribute ("dryWet", (double)getDryWet());
    xmlState.setAttribute ("triggerMode", (int)m_triggerMode.load());
    xmlState.setAttribute ("envAttack", (double)m_envAttack.load());
    xmlState.setAttribute ("envRelease", (double)m_envRelease.load());
    xmlState.setAttribute ("transientThresh", (double)m_transientThreshold.load());

    std::lock_guard<std::mutex> lock(m_layerStateMutex);
    auto* layersElement = xmlState.createNewChildElement ("Layers");
    for (const auto& pair : m_layerStates)
    {
        auto* layerEl = layersElement->createNewChildElement ("Layer");
        layerEl->setAttribute ("name", pair.first);
        layerEl->setAttribute ("scale", (double)pair.second.scale);
        layerEl->setAttribute ("offset", (double)pair.second.offset);
        layerEl->setAttribute ("heat", (double)pair.second.heat);
        layerEl->setAttribute ("memory", (double)pair.second.memory);
        layerEl->setAttribute ("driftMode", (int)pair.second.driftMode);
        layerEl->setAttribute ("frozen", pair.second.frozen);
    }

    copyXmlToBinary (xmlState, destData);
}

void NNBendingAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState != nullptr)
    {
        if (xmlState->hasTagName ("NNBendingSettings"))
        {
            m_bufferSize = xmlState->getIntAttribute ("bufferSize", 2048);
            setDryWet ((float)xmlState->getDoubleAttribute ("dryWet", 1.0));
            m_activeLayerName = xmlState->getStringAttribute ("activeLayer");
            setTriggerMode ((TriggerMode)xmlState->getIntAttribute ("triggerMode", 0));
            setEnvelopeAttack ((float)xmlState->getDoubleAttribute ("envAttack", 0.25));
            setEnvelopeRelease ((float)xmlState->getDoubleAttribute ("envRelease", 0.85));
            setTransientThreshold ((float)xmlState->getDoubleAttribute ("transientThresh", 0.15));

            juce::String path = xmlState->getStringAttribute ("modelPath");
            if (path.isNotEmpty())
            {
                juce::File file(path);
                if (file.existsAsFile())
                    loadModel(file);
            }
            juce::String method = xmlState->getStringAttribute ("currentMethod");
            if (method.isNotEmpty())
                setCurrentMethod(method);

            // Restore per-layer bending states
            auto* layersElement = xmlState->getChildByName ("Layers");
            if (layersElement != nullptr)
            {
                std::lock_guard<std::mutex> lock(m_layerStateMutex);
                for (auto* layerEl : layersElement->getChildIterator())
                {
                    std::string name = layerEl->getStringAttribute ("name").toStdString();
                    if (!name.empty())
                    {
                        auto& state = m_layerStates[name];
                        state.scale = (float)layerEl->getDoubleAttribute ("scale", 1.0);
                        state.offset = (float)layerEl->getDoubleAttribute ("offset", 0.0);
                        // Read heat, fallback to jitter if older project
                        double heatVal = layerEl->getDoubleAttribute ("heat", -1.0);
                        if (heatVal < 0.0)
                            heatVal = layerEl->getDoubleAttribute ("jitter", 0.0);
                        state.heat = (float)heatVal;
                        state.memory = (float)layerEl->getDoubleAttribute ("memory", 0.8);
                        state.driftMode = (DriftMode)layerEl->getIntAttribute ("driftMode", 0);
                        state.frozen = layerEl->getBoolAttribute ("frozen", false);
                    }
                }
            }
        }
    }
}

//==============================================================================
juce::AudioProcessorEditor* NNBendingAudioProcessor::createEditor()
{
    return new NNBendingAudioProcessorEditor (*this);
}

//==============================================================================
// This creates the filter...
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NNBendingAudioProcessor();
}

