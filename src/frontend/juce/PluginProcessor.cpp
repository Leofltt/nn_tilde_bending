#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
ModelThread::ModelThread(NNBendingAudioProcessor& processor)
    : juce::Thread("Model Thread"), m_processor(processor)
{
}

ModelThread::~ModelThread()
{
    stopThread(2000);
}

void ModelThread::run()
{
    while (!threadShouldExit())
    {
        m_event.wait(-1);
        
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
    m_model_thread.stopThread(2000);
}

//==============================================================================
void NNBendingAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    m_backend.set_sample_rate(sampleRate);
    initBuffers();
    
    if (!m_model_thread.isThreadRunning())
        m_model_thread.startThread(juce::Thread::Priority::high);
}

void NNBendingAudioProcessor::releaseResources()
{
    m_model_thread.stopThread(2000);
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

    if (!m_modelLoaded.load() || m_model_in <= 0 || m_model_out <= 0)
    {
        for (auto i = 0; i < buffer.getNumChannels(); ++i)
            buffer.clear(i, 0, buffer.getNumSamples());
        return;
    }

    auto numSamples = buffer.getNumSamples();
    
    // 1. Push incoming samples to input circular buffers
    int n_ins = std::min((int)totalNumInputChannels, m_model_in);
    for (int c = 0; c < n_ins; ++c)
    {
        m_in_buffers[c].put(buffer.getReadPointer(c), numSamples);
    }
    
    // 2. Check if we have enough samples to perform inference
    if (m_in_buffers[0].getAvailable() >= m_bufferSize)
    {
        // Extract block for model input
        for (int c = 0; c < m_model_in; ++c)
        {
            m_in_model_data[c].resize(m_bufferSize);
            m_in_buffers[c].get(m_in_model_data[c].data(), m_bufferSize);
        }
        
        // If thread is not currently processing, swap data and trigger
        if (!m_model_thread.isProcessing())
        {
            std::unique_lock<std::mutex> lock(m_data_mutex);
            
            // Swap input buffer with thread input buffer
            m_in_thread_data.swap(m_in_model_data);
            
            // If output from previous run is ready, retrieve it
            if (m_output_ready.load())
            {
                m_out_model_data.swap(m_out_thread_data);
                m_output_ready.store(false);
                
                int n_outs = std::min((int)totalNumOutputChannels, m_model_out);
                for (int c = 0; c < n_outs; ++c)
                {
                    m_out_buffers[c].put(m_out_model_data[c].data(), m_bufferSize);
                }
            }
            
            lock.unlock();
            m_model_thread.triggerCompute();
        }
    }
    
    // 3. Pull processed samples from output circular buffers
    int n_outs = std::min((int)totalNumOutputChannels, m_model_out);
    for (int c = 0; c < n_outs; ++c)
    {
        if (m_out_buffers[c].getAvailable() >= numSamples)
        {
            m_out_buffers[c].get(buffer.getWritePointer(c), numSamples);
        }
        else
        {
            buffer.clear(c, 0, numSamples);
        }
    }
}

//==============================================================================
bool NNBendingAudioProcessor::loadModel(const juce::File& file)
{
    m_modelLoaded.store(false);
    
    std::string path = file.getFullPathName().toStdString();
    
    int err = m_backend.load(path, getSampleRate());
    if (err == 0)
    {
        m_modelPath = file.getFullPathName();
        
        // Find default method
        auto methods = m_backend.get_available_methods();
        if (!methods.empty())
        {
            // Prefer "forward" if available, else take the first one
            std::string defaultMethod = methods[0];
            for (const auto& m : methods)
            {
                if (m == "forward")
                {
                    defaultMethod = m;
                    break;
                }
            }
            m_currentMethod = defaultMethod;
            
            auto params = m_backend.get_method_params(defaultMethod);
            if (params.size() >= 3)
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
    if (m_backend.has_method(method.toStdString()))
    {
        m_currentMethod = method;
        auto params = m_backend.get_method_params(method.toStdString());
        if (params.size() >= 3)
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
    std::unique_lock<std::mutex> lock(m_data_mutex);
    
    m_in_buffers.resize(std::max(1, m_model_in));
    m_out_buffers.resize(std::max(1, m_model_out));
    
    m_in_model_data.resize(std::max(1, m_model_in));
    m_out_model_data.resize(std::max(1, m_model_out));
    
    m_in_thread_data.resize(std::max(1, m_model_in));
    m_out_thread_data.resize(std::max(1, m_model_out));
    
    for (int i = 0; i < m_model_in; ++i)
    {
        m_in_buffers[i].init(m_bufferSize);
        m_in_model_data[i].assign(m_bufferSize, 0.0f);
        m_in_thread_data[i].assign(m_bufferSize, 0.0f);
    }
    
    for (int i = 0; i < m_model_out; ++i)
    {
        m_out_buffers[i].init(m_bufferSize);
        m_out_model_data[i].assign(m_bufferSize, 0.0f);
        m_out_thread_data[i].assign(m_bufferSize, 0.0f);
    }
    
    m_output_ready.store(false);
}

void NNBendingAudioProcessor::runInference()
{
    if (!m_modelLoaded.load() || m_model_in <= 0 || m_model_out <= 0)
        return;
        
    std::vector<float*> in_ptrs;
    std::vector<float*> out_ptrs;
    
    for (int c = 0; c < m_model_in; ++c)
        in_ptrs.push_back(m_in_thread_data[c].data());
        
    for (int c = 0; c < m_model_out; ++c)
        out_ptrs.push_back(m_out_thread_data[c].data());
        
    m_backend.perform(in_ptrs, out_ptrs, m_currentMethod.toStdString(), 1, m_model_out, m_bufferSize);
    
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
