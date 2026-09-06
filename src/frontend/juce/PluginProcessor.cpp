#include "PluginProcessor.h"
#include "PluginEditor.h"

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

    addParameter (m_jitterParam = new juce::AudioParameterFloat (
        juce::ParameterID ("jitter", 1), "Layer Jitter",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f), 0.0f
    ));
}

NNBendingAudioProcessor::~NNBendingAudioProcessor()
{
    m_model_thread.stop();
}

//==============================================================================
void NNBendingAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(samplesPerBlock);
    m_backend.set_sample_rate(sampleRate);
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
    juce::ignoreUnused(midiMessages);
    
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

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

    if (m_model_in > 0)
    {
        for (int c = 0; c < m_model_in; ++c)
        {
            const float* inPtr = nullptr;
            if (c < totalNumInputChannels)
                inPtr = buffer.getReadPointer(c);
            else if (totalNumInputChannels > 0)
                inPtr = buffer.getReadPointer(0);

            m_in_buffers[c].put(inPtr, numSamples);
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
    
    // 3. Pull processed samples from output circular buffers
    int n_outs = std::min((int)totalNumOutputChannels, m_model_out);
    for (int c = 0; c < n_outs; ++c)
    {
        m_out_buffers[c].get(buffer.getWritePointer(c), numSamples);
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

    // 4. Dry/Wet Blending (equal-power / linear blend between dry input audio and model generated/processed audio)
    float wetGain = m_dryWet.load();
    float dryGain = 1.0f - wetGain;

    std::vector<float> dryBlock(numSamples, 0.0f);
    for (int c = 0; c < (int)totalNumOutputChannels; ++c)
    {
        int dryCh = std::min(c, (int)m_dry_delay_buffers.size() - 1);
        if (dryCh >= 0)
        {
            m_dry_delay_buffers[dryCh].get(dryBlock.data(), numSamples);
            float* outPtr = buffer.getWritePointer(c);
            for (int i = 0; i < numSamples; ++i)
            {
                outPtr[i] = outPtr[i] * wetGain + dryBlock[i] * dryGain;
            }
        }
    }
}

//==============================================================================
std::vector<juce::String> NNBendingAudioProcessor::getAvailableModes() const
{
    std::vector<juce::String> modes;
    if (!m_modelLoaded.load())
        return modes;

    auto backendModes = const_cast<Backend&>(m_backend).get_plugin_modes();
    for (const auto& m : backendModes)
        modes.push_back(juce::String(m));
    return modes;
}

bool NNBendingAudioProcessor::loadModel(const juce::File& file)
{
    m_modelLoaded.store(false);
    m_model_thread.stop();
    
    std::string path = file.getFullPathName().toStdString();
    
    int err = m_backend.load(path, getSampleRate());
    if (err == 0)
    {
        m_modelPath = file.getFullPathName();
        
        // Find default mode: prefer "forward", else "prior", else first available
        auto modes = m_backend.get_plugin_modes();
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
            
            auto params = m_backend.get_mode_params(defaultMode);
            if (params.size() >= 4)
            {
                m_model_in = params[0];
                m_model_out = params[2];
            }
        }

        // Auto-configure buffer size to model ratio
        int higher_ratio = m_backend.get_higher_ratio();
        int reqSize = 1;
        while (reqSize < higher_ratio)
            reqSize <<= 1;
        m_bufferSize = std::max(2048, reqSize);
        
        m_modelLoaded.store(true);
        initBuffers();
        return true;
    }
    
    return false;
}

void NNBendingAudioProcessor::setCurrentMethod(const juce::String& method)
{
    std::string methodStr = method.toStdString();
    auto modes = m_backend.get_plugin_modes();
    bool isValidMode = (std::find(modes.begin(), modes.end(), methodStr) != modes.end())
                    || m_backend.has_method(methodStr);

    if (isValidMode)
    {
        m_currentMethod = method;
        auto params = m_backend.get_mode_params(methodStr);
        if (params.size() >= 4)
        {
            m_model_in = params[0];
            m_model_out = params[2];
            initBuffers();
        }
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

void NNBendingAudioProcessor::initBuffers()
{
    m_model_thread.stop();
    
    std::lock_guard<std::mutex> lock(m_staging_mutex);
    
    int n_in = std::max(1, m_model_in);
    int n_out = std::max(1, m_model_out);
    
    m_in_buffers.resize(n_in);
    m_out_buffers.resize(n_out);
    
    m_staging_in.resize(n_in);
    m_staging_out.resize(n_out);
    
    // In forward mode, delay is 2 * m_bufferSize (1 buffer input accumulation + 1 buffer model inference transfer)
    int latencySamples = (m_model_in > 0) ? (m_bufferSize * 2) : 0;
    setLatencySamples(latencySamples);

    int n_dry = std::max(2, std::max(n_in, n_out));
    m_dry_delay_buffers.resize(n_dry);

    for (int i = 0; i < n_in; ++i)
    {
        m_in_buffers[i].init(m_bufferSize);
        m_staging_in[i].assign(m_bufferSize, 0.0f);
    }

    for (int i = 0; i < n_dry; ++i)
    {
        m_dry_delay_buffers[i].init(m_bufferSize);
        // Pre-fill dry delay buffer with latency zeros so dry aligns with wet output
        if (latencySamples > 0)
            m_dry_delay_buffers[i].put(nullptr, latencySamples);
    }
    
    for (int i = 0; i < n_out; ++i)
    {
        m_out_buffers[i].init(m_bufferSize);
        m_staging_out[i].assign(m_bufferSize, 0.0f);
    }
    
    m_output_ready.store(false);
    
    if (m_modelLoaded.load())
    {
        m_model_thread.startThread(juce::Thread::Priority::high);
    }
}

void NNBendingAudioProcessor::runInference()
{
    if (!m_modelLoaded.load() || m_model_out <= 0)
        return;

    // 1. Sync active DAW parameter values into the active layer state
    {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        std::string activeLayer = m_activeLayerName.toStdString();
        if (!activeLayer.empty() && m_scaleParam && m_offsetParam && m_jitterParam)
        {
            auto& state = m_layerStates[activeLayer];
            state.scale = m_scaleParam->get();
            state.offset = m_offsetParam->get();
            state.jitter = m_jitterParam->get();
        }

        // 2. For each configured layer, apply its own Scale, Offset, and Live Jitter
        for (auto& pair : m_layerStates)
        {
            const std::string& layerName = pair.first;
            LayerBendingState& state = pair.second;

            // If baseDrawnWeights is empty, fetch original layer weights as baseline
            if (state.baseDrawnWeights.empty())
            {
                state.baseDrawnWeights = m_backend.get_original_layer_weights(layerName);
            }

            if (state.baseDrawnWeights.empty())
                continue;

            // If jitter is active and not frozen, compute fresh stochastic weights per compute pass
            if (state.jitter > 0.0001f && !state.frozen)
            {
                std::vector<float> jittered(state.baseDrawnWeights.size());
                for (size_t i = 0; i < state.baseDrawnWeights.size(); ++i)
                {
                    float noise = (m_jitterRng.nextFloat() * 2.0f - 1.0f) * state.jitter;
                    jittered[i] = (state.baseDrawnWeights[i] * state.scale + state.offset) + noise;
                }
                m_backend.set_layer_weights(layerName, jittered);
            }
        }
    }
        
    std::vector<float*> in_ptrs;
    std::vector<float*> out_ptrs;
    
    {
        std::lock_guard<std::mutex> lock(m_staging_mutex);
        for (int c = 0; c < m_model_in; ++c)
            in_ptrs.push_back(m_staging_in[c].data());
            
        for (int c = 0; c < m_model_out; ++c)
            out_ptrs.push_back(m_staging_out[c].data());
            
        std::string modeStr = m_currentMethod.toStdString();
        if (modeStr == "forward")
        {
            m_backend.perform_forward(in_ptrs, out_ptrs, 1, m_model_out, m_bufferSize, m_latentHook);
        }
        else if (modeStr == "prior" || modeStr == "generate")
        {
            m_backend.perform_prior_decode(out_ptrs, 1, m_model_out, m_bufferSize);
        }
        else
        {
            m_backend.perform(in_ptrs, out_ptrs, modeStr, 1, m_model_out, m_bufferSize);
        }
    }
    
    m_output_ready.store(true);
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

    // Save per-layer bending states
    std::lock_guard<std::mutex> lock(m_layerStateMutex);
    auto* layersElement = xmlState.createNewChildElement ("Layers");
    for (const auto& pair : m_layerStates)
    {
        auto* layerEl = layersElement->createNewChildElement ("Layer");
        layerEl->setAttribute ("name", pair.first);
        layerEl->setAttribute ("scale", (double)pair.second.scale);
        layerEl->setAttribute ("offset", (double)pair.second.offset);
        layerEl->setAttribute ("jitter", (double)pair.second.jitter);
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
                        state.jitter = (float)layerEl->getDoubleAttribute ("jitter", 0.0);
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

