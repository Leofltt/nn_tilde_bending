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
        m_event.wait(50);
        
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
    
    // 2. Feed incoming audio into input circular buffers (if model takes audio input)
    if (m_model_in > 0)
    {
        for (int c = 0; c < m_model_in; ++c)
        {
            if (c < totalNumInputChannels)
                m_in_buffers[c].put(buffer.getReadPointer(c), numSamples);
            else if (totalNumInputChannels > 0)
                m_in_buffers[c].put(buffer.getReadPointer(0), numSamples);
            else
                m_in_buffers[c].put(nullptr, numSamples);
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
        if (!m_model_thread.isProcessing() && m_out_buffers[0].getAvailable() < m_bufferSize * 2)
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
        
        // Find default mode: prefer "autoencode", then "forward", else first available
        auto modes = m_backend.get_plugin_modes();
        if (!modes.empty())
        {
            std::string defaultMode = modes[0];
            if (m_backend.has_autoencode())
            {
                defaultMode = "autoencode";
            }
            else if (m_backend.has_method("forward"))
            {
                defaultMode = "forward";
            }
            m_currentMethod = defaultMode;
            
            auto params = m_backend.get_mode_params(defaultMode);
            if (params.size() >= 4)
            {
                m_model_in = params[0];
                m_model_out = params[2];
            }
        }
        
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
    
    for (int i = 0; i < n_in; ++i)
    {
        m_in_buffers[i].init(m_bufferSize);
        m_staging_in[i].assign(m_bufferSize, 0.0f);
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
        
    std::vector<float*> in_ptrs;
    std::vector<float*> out_ptrs;
    
    {
        std::lock_guard<std::mutex> lock(m_staging_mutex);
        for (int c = 0; c < m_model_in; ++c)
            in_ptrs.push_back(m_staging_in[c].data());
            
        for (int c = 0; c < m_model_out; ++c)
            out_ptrs.push_back(m_staging_out[c].data());
            
        std::string modeStr = m_currentMethod.toStdString();
        if (modeStr == "autoencode")
        {
            m_backend.perform_autoencode(in_ptrs, out_ptrs, 1, m_model_out, m_bufferSize, m_latentHook);
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

