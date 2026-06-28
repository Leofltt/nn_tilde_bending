#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
NNBendingAudioProcessorEditor::NNBendingAudioProcessorEditor (NNBendingAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Configure Title
    titleLabel.setText ("nn~ parameter bending", juce::dontSendNotification);
    titleLabel.setFont (juce::FontOptions (20.0f, juce::Font::bold));
    titleLabel.setJustificationType (juce::Justification::centred);
    titleLabel.setColour (juce::Label::textColourId, juce::Colours::lightcyan);
    addAndMakeVisible (titleLabel);

    // Configure Load Button
    loadButton.setButtonText ("Load local PyTorch Model (.ts)...");
    loadButton.addListener (this);
    loadButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a154b")); // Deep dark purple
    loadButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
    addAndMakeVisible (loadButton);

    // Configure Status Label
    statusLabel.setText ("No model loaded.", juce::dontSendNotification);
    statusLabel.setJustificationType (juce::Justification::centred);
    statusLabel.setColour (juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible (statusLabel);

    // Method Selector
    addAndMakeVisible (methodLabel);
    methodCombo.addListener (this);
    addAndMakeVisible (methodCombo);

    // Buffer Selector
    addAndMakeVisible (bufferLabel);
    bufferCombo.addItem ("512", 512);
    bufferCombo.addItem ("1024", 1024);
    bufferCombo.addItem ("2048", 2048);
    bufferCombo.addItem ("4096", 4096);
    bufferCombo.addItem ("8192", 8192);
    bufferCombo.setSelectedId (audioProcessor.getBufferSize(), juce::dontSendNotification);
    bufferCombo.addListener (this);
    addAndMakeVisible (bufferCombo);

    // Bending Section Components
    addAndMakeVisible (bendingGroup);
    
    addAndMakeVisible (layerLabel);
    layerCombo.addListener (this);
    addAndMakeVisible (layerCombo);

    // Scale Slider (0.0 to 10.0, default 1.0)
    addAndMakeVisible (scaleLabel);
    scaleSlider.setRange (0.0, 10.0, 0.01);
    scaleSlider.setValue (1.0);
    scaleSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    scaleSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 20);
    scaleSlider.setColour (juce::Slider::thumbColourId, juce::Colours::violet);
    scaleSlider.setColour (juce::Slider::rotarySliderOutlineColourId, juce::Colours::darkgrey);
    scaleSlider.addListener (this);
    addAndMakeVisible (scaleSlider);

    // Offset Slider (-1.0 to 1.0, default 0.0)
    addAndMakeVisible (offsetLabel);
    offsetSlider.setRange (-1.0, 1.0, 0.001);
    offsetSlider.setValue (0.0);
    offsetSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    offsetSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 20);
    offsetSlider.setColour (juce::Slider::thumbColourId, juce::Colours::turquoise);
    offsetSlider.setColour (juce::Slider::rotarySliderOutlineColourId, juce::Colours::darkgrey);
    offsetSlider.addListener (this);
    addAndMakeVisible (offsetSlider);

    // Reset Button
    resetButton.addListener (this);
    resetButton.setColour (juce::TextButton::buttonColourId, juce::Colours::darkred);
    addAndMakeVisible (resetButton);

    // Info Label
    infoBendingLabel.setText ("Select a layer to bend weights.", juce::dontSendNotification);
    infoBendingLabel.setJustificationType (juce::Justification::centred);
    infoBendingLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (infoBendingLabel);

    // Make window resizable or set fixed size
    setSize (500, 480);
    
    // Initial UI load
    updateModelInfo();
    
    // Start periodic UI timer
    startTimer (300);
}

NNBendingAudioProcessorEditor::~NNBendingAudioProcessorEditor()
{
    stopTimer();
}

//==============================================================================
void NNBendingAudioProcessorEditor::paint (juce::Graphics& g)
{
    // Premium dark gradient background
    juce::Colour color1 = juce::Colour::fromString ("#ff121216");
    juce::Colour color2 = juce::Colour::fromString ("#ff1e1c24");
    juce::ColourGradient gradient (color1, 0, 0, color2, 0, (float)getHeight(), false);
    g.setGradientFill (gradient);
    g.fillAll();

    // Visual dividers
    g.setColour (juce::Colours::violet.withAlpha (0.4f));
    g.drawHorizontalLine (50, 20.0f, (float)getWidth() - 20.0f);
    g.drawHorizontalLine (150, 20.0f, (float)getWidth() - 20.0f);
}

void NNBendingAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced (20);
    
    titleLabel.setBounds (area.removeFromTop (30));
    area.removeFromTop (10);

    // Top Model Load row
    auto loadArea = area.removeFromTop (35);
    loadButton.setBounds (loadArea.removeFromLeft (area.getWidth() - 100));
    area.removeFromTop (5);
    statusLabel.setBounds (area.removeFromTop (20));
    area.removeFromTop (10);

    // Dropdown selectors row (Method & Buffer)
    auto selectRow = area.removeFromTop (45);
    
    auto methodArea = selectRow.removeFromLeft (selectRow.getWidth() / 2).reduced(5, 0);
    methodLabel.setBounds (methodArea.removeFromLeft (60).withHeight (25));
    methodCombo.setBounds (methodArea.withHeight (25));
    
    auto bufferArea = selectRow.reduced(5, 0);
    bufferLabel.setBounds (bufferArea.removeFromLeft (80).withHeight (25));
    bufferCombo.setBounds (bufferArea.withHeight (25));

    area.removeFromTop (15);

    // Bending group layout
    bendingGroup.setBounds (area.removeFromTop (230));
    auto groupArea = bendingGroup.getLocalBounds().reduced (15);
    groupArea.removeFromTop (15); // Title offset

    // Layer Selector Row
    auto layerRow = groupArea.removeFromTop (35);
    layerLabel.setBounds (layerRow.removeFromLeft (90).withHeight (25));
    layerCombo.setBounds (layerRow.withHeight (25));
    
    groupArea.removeFromTop (10);

    // Sliders row
    auto slidersRow = groupArea.removeFromTop (100);
    auto scaleArea = slidersRow.removeFromLeft (slidersRow.getWidth() / 2).reduced (10, 0);
    scaleLabel.setBounds (scaleArea.removeFromTop (20));
    scaleSlider.setBounds (scaleArea);

    auto offsetArea = slidersRow.reduced (10, 0);
    offsetLabel.setBounds (offsetArea.removeFromTop (20));
    offsetSlider.setBounds (offsetArea);

    groupArea.removeFromTop (10);

    // Info and Reset row
    auto resetRow = groupArea.removeFromTop (35);
    resetButton.setBounds (resetRow.removeFromRight (150).withHeight (30));
    infoBendingLabel.setBounds (resetRow.withHeight (30));
}

//==============================================================================
void NNBendingAudioProcessorEditor::comboBoxChanged (juce::ComboBox* comboBoxThatHasChanged)
{
    if (comboBoxThatHasChanged == &bufferCombo)
    {
        audioProcessor.setBufferSize (bufferCombo.getSelectedId());
    }
    else if (comboBoxThatHasChanged == &methodCombo)
    {
        audioProcessor.setCurrentMethod (methodCombo.getText());
        updateModelInfo();
    }
    else if (comboBoxThatHasChanged == &layerCombo)
    {
        selectLayer (layerCombo.getText());
    }
}

void NNBendingAudioProcessorEditor::sliderValueChanged (juce::Slider* slider)
{
    juce::ignoreUnused(slider);
    applyWeightBending();
}

void NNBendingAudioProcessorEditor::buttonClicked (juce::Button* button)
{
    if (button == &loadButton)
    {
        fileChooser = std::make_unique<juce::FileChooser> (
            "Select a PyTorch TorchScript model (.ts)...",
            juce::File::getSpecialLocation (juce::File::userHomeDirectory),
            "*.ts"
        );
        
        auto flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
        
        fileChooser->launchAsync (flags, [this] (const juce::FileChooser& fc)
        {
            auto file = fc.getResult();
            if (file.existsAsFile())
            {
                if (audioProcessor.loadModel (file))
                {
                    updateModelInfo();
                }
                else
                {
                    statusLabel.setText ("Error: Failed to load " + file.getFileName(), juce::dontSendNotification);
                    statusLabel.setColour (juce::Label::textColourId, juce::Colours::orangered);
                }
            }
        });
    }
    else if (button == &resetButton)
    {
        resetWeightBending();
    }
}

//==============================================================================
void NNBendingAudioProcessorEditor::timerCallback()
{
    // Periodically sync status in case load happened externally
    if (audioProcessor.isModelLoaded() && statusLabel.getText() == "No model loaded.")
    {
        updateModelInfo();
    }
}

void NNBendingAudioProcessorEditor::updateModelInfo()
{
    if (audioProcessor.isModelLoaded())
    {
        juce::File file (audioProcessor.getModelPath());
        statusLabel.setText ("Loaded: " + file.getFileName(), juce::dontSendNotification);
        statusLabel.setColour (juce::Label::textColourId, juce::Colours::lightgreen);

        // Update Method combo box
        methodCombo.clear (juce::dontSendNotification);
        auto methods = audioProcessor.getBackend().get_available_methods();
        int id = 1;
        for (const auto& m : methods)
        {
            methodCombo.addItem (m, id++);
        }
        methodCombo.setText (audioProcessor.getCurrentMethod(), juce::dontSendNotification);

        // Update Layer combo box
        layerCombo.clear (juce::dontSendNotification);
        auto layers = audioProcessor.getBackend().get_available_layers();
        id = 1;
        for (const auto& l : layers)
        {
            layerCombo.addItem (l, id++);
        }
        
        if (currentBendingLayer.isNotEmpty() && layers.size() > 0)
        {
            layerCombo.setText (currentBendingLayer, juce::dontSendNotification);
        }
        else if (layers.size() > 0)
        {
            layerCombo.setText (layers[0], juce::dontSendNotification);
            selectLayer (layers[0]);
        }
    }
    else
    {
        statusLabel.setText ("No model loaded.", juce::dontSendNotification);
        statusLabel.setColour (juce::Label::textColourId, juce::Colours::darkgrey);
        methodCombo.clear (juce::dontSendNotification);
        layerCombo.clear (juce::dontSendNotification);
        infoBendingLabel.setText ("Load a model to view weights.", juce::dontSendNotification);
        originalWeights.clear();
        currentBendingLayer = "";
    }
}

void NNBendingAudioProcessorEditor::selectLayer (const juce::String& layerName)
{
    if (layerName.isEmpty()) return;
    
    currentBendingLayer = layerName;
    originalWeights = audioProcessor.getBackend().get_layer_weights (layerName.toStdString());
    
    // Disable listener to set default values without triggering a loop recalculation
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    
    scaleSlider.setValue (1.0);
    offsetSlider.setValue (0.0);
    
    scaleSlider.addListener (this);
    offsetSlider.addListener (this);

    infoBendingLabel.setText (juce::String (originalWeights.size()) + " parameters", juce::dontSendNotification);
}

void NNBendingAudioProcessorEditor::applyWeightBending()
{
    if (currentBendingLayer.isEmpty() || originalWeights.empty()) return;
    
    float scale = (float)scaleSlider.getValue();
    float offset = (float)offsetSlider.getValue();
    
    std::vector<float> bendedWeights (originalWeights.size());
    for (size_t i = 0; i < originalWeights.size(); ++i)
    {
        bendedWeights[i] = originalWeights[i] * scale + offset;
    }
    
    audioProcessor.getBackend().set_layer_weights (currentBendingLayer.toStdString(), bendedWeights);
}

void NNBendingAudioProcessorEditor::resetWeightBending()
{
    if (currentBendingLayer.isEmpty() || originalWeights.empty()) return;
    
    scaleSlider.setValue (1.0);
    offsetSlider.setValue (0.0);
    
    audioProcessor.getBackend().set_layer_weights (currentBendingLayer.toStdString(), originalWeights);
}
