#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
NNBendingAudioProcessorEditor::NNBendingAudioProcessorEditor (NNBendingAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Configure Title
    titleLabel.setText ("nn~ parameter bending", juce::dontSendNotification);
    titleLabel.setFont (juce::FontOptions (18.0f, juce::Font::bold));
    titleLabel.setJustificationType (juce::Justification::centredLeft);
    titleLabel.setColour (juce::Label::textColourId, juce::Colours::lightcyan);
    addAndMakeVisible (titleLabel);

    // Configure Load Button
    loadButton.setButtonText ("Load Model (.ts)");
    loadButton.addListener (this);
    loadButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff581c87"));
    loadButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
    addAndMakeVisible (loadButton);

    // Configure Status Label
    statusLabel.setText ("No model loaded.", juce::dontSendNotification);
    statusLabel.setJustificationType (juce::Justification::centredLeft);
    statusLabel.setColour (juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible (statusLabel);

    // Mode Selector
    methodLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (methodLabel);
    methodCombo.addListener (this);
    addAndMakeVisible (methodCombo);

    // Buffer Readout Badge
    bufferLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (bufferLabel);
    bufferStatusLabel.setText (juce::String (audioProcessor.getBufferSize()), juce::dontSendNotification);
    bufferStatusLabel.setJustificationType (juce::Justification::centred);
    bufferStatusLabel.setColour (juce::Label::backgroundColourId, juce::Colour::fromString ("#ff1c1926"));
    bufferStatusLabel.setColour (juce::Label::outlineColourId, juce::Colour::fromString ("#ff3d3554"));
    bufferStatusLabel.setColour (juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible (bufferStatusLabel);

    // Dry / Wet Slider
    dryWetLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (dryWetLabel);

    dryWetSlider.setRange (0.0, 1.0, 0.01);
    dryWetSlider.setValue (audioProcessor.getDryWet());
    dryWetSlider.setSliderStyle (juce::Slider::LinearBar);
    dryWetSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 45, 18);
    dryWetSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff7c3aed"));
    dryWetSlider.addListener (this);
    addAndMakeVisible (dryWetSlider);

    // Save Model Button
    saveModelButton.addListener (this);
    saveModelButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1e3a5f"));
    saveModelButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
    addAndMakeVisible (saveModelButton);

    // Bending Group
    addAndMakeVisible (bendingGroup);

    layerLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (layerLabel);
    layerCombo.addListener (this);
    addAndMakeVisible (layerCombo);

    resetLayerButton.addListener (this);
    resetLayerButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff801e2a"));
    resetLayerButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
    addAndMakeVisible (resetLayerButton);

    resetAllButton.addListener (this);
    resetAllButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff5a1018"));
    resetAllButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
    addAndMakeVisible (resetAllButton);

    // Weight Canvas & Drawing Hook
    addAndMakeVisible (weightCanvas);
    weightCanvas.onWeightsModified = [this] (const std::vector<float>& modifiedWeights)
    {
        if (currentBendingLayer.isNotEmpty() && !modifiedWeights.empty())
        {
            currentWeights = modifiedWeights;

            // Keep scale and offset knobs where they are:
            // Inverse-transform the modified weights into baseDrawnWeights so subsequent
            // knob tweaks scale/offset from the new drawn curve smoothly!
            float scale = (float)scaleSlider.getValue();
            float offset = (float)offsetSlider.getValue();

            baseDrawnWeights.resize (modifiedWeights.size());
            if (std::abs (scale) > 0.00001f)
            {
                for (size_t i = 0; i < modifiedWeights.size(); ++i)
                    baseDrawnWeights[i] = (modifiedWeights[i] - offset) / scale;
            }
            else
            {
                baseDrawnWeights = modifiedWeights;
            }

            audioProcessor.getBackend().set_layer_weights (currentBendingLayer.toStdString(), currentWeights);
        }
    };

    // Compact Side Knobs (Scale & Offset)
    scaleLabel.setText ("Scale", juce::dontSendNotification);
    scaleLabel.setJustificationType (juce::Justification::centred);
    scaleLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (scaleLabel);

    scaleSlider.setRange (0.0, 5.0, 0.01);
    scaleSlider.setValue (1.0);
    scaleSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    scaleSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 18);
    scaleSlider.setColour (juce::Slider::thumbColourId, juce::Colours::violet);
    scaleSlider.setColour (juce::Slider::rotarySliderOutlineColourId, juce::Colours::darkgrey);
    scaleSlider.addListener (this);
    addAndMakeVisible (scaleSlider);

    offsetLabel.setText ("Offset", juce::dontSendNotification);
    offsetLabel.setJustificationType (juce::Justification::centred);
    offsetLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (offsetLabel);

    offsetSlider.setRange (-2.0, 2.0, 0.001);
    offsetSlider.setValue (0.0);
    offsetSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    offsetSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 18);
    offsetSlider.setColour (juce::Slider::thumbColourId, juce::Colours::turquoise);
    offsetSlider.setColour (juce::Slider::rotarySliderOutlineColourId, juce::Colours::darkgrey);
    offsetSlider.addListener (this);
    addAndMakeVisible (offsetSlider);

    // Info Label
    infoBendingLabel.setText ("Select a layer to bend weights.", juce::dontSendNotification);
    infoBendingLabel.setJustificationType (juce::Justification::centredLeft);
    infoBendingLabel.setColour (juce::Label::textColourId, juce::Colours::silver);
    addAndMakeVisible (infoBendingLabel);

    // Expanded window size (860 x 580)
    setSize (860, 580);
    setResizable (true, true);
    setResizeLimits (760, 480, 1400, 900);

    // Initial load
    updateModelInfo();
    startTimer (300);
}

NNBendingAudioProcessorEditor::~NNBendingAudioProcessorEditor()
{
    stopTimer();
}

//==============================================================================
void NNBendingAudioProcessorEditor::paint (juce::Graphics& g)
{
    // Sleek dark gradient background
    juce::Colour color1 = juce::Colour::fromString ("#ff0f0e13");
    juce::Colour color2 = juce::Colour::fromString ("#ff1a1724");
    juce::ColourGradient gradient (color1, 0, 0, color2, 0, (float)getHeight(), false);
    g.setGradientFill (gradient);
    g.fillAll();

    // Subtle divider under header
    g.setColour (juce::Colour::fromString ("#ff382f4c"));
    g.drawHorizontalLine (80, 15.0f, (float)getWidth() - 15.0f);
}

void NNBendingAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced (15);

    // Top Header: Row 1 (Title + Status)
    auto titleRow = area.removeFromTop (26);
    titleLabel.setBounds (titleRow.removeFromLeft (260));
    statusLabel.setBounds (titleRow);

    area.removeFromTop (6); // Spacer

    // Header Controls: Row 2 (Load, Mode, Buffer, Save)
    auto headerRow = area.removeFromTop (32);
    loadButton.setBounds (headerRow.removeFromLeft (130));
    headerRow.removeFromLeft (12); // Gap

    methodLabel.setBounds (headerRow.removeFromLeft (45));
    methodCombo.setBounds (headerRow.removeFromLeft (100));
    headerRow.removeFromLeft (12); // Gap

    bufferLabel.setBounds (headerRow.removeFromLeft (50));
    bufferStatusLabel.setBounds (headerRow.removeFromLeft (65));
    headerRow.removeFromLeft (14); // Gap

    dryWetLabel.setBounds (headerRow.removeFromLeft (60));
    dryWetSlider.setBounds (headerRow.removeFromLeft (110));

    saveModelButton.setBounds (headerRow.removeFromRight (140));

    area.removeFromTop (20); // Spacer clearing the divider line

    // Bending Group Section
    bendingGroup.setBounds (area);
    auto groupArea = bendingGroup.getBounds().reduced (14);
    groupArea.removeFromTop (12); // Group title offset

    // Layer Selection & Action Row
    auto layerRow = groupArea.removeFromTop (32);
    layerLabel.setBounds (layerRow.removeFromLeft (45));
    
    resetAllButton.setBounds (layerRow.removeFromRight (130));
    layerRow.removeFromRight (8); // Gap
    resetLayerButton.setBounds (layerRow.removeFromRight (110));
    layerRow.removeFromRight (12); // Gap
    
    layerCombo.setBounds (layerRow); // Takes remaining center width

    groupArea.removeFromTop (10); // Spacer

    // Bottom info readout row
    auto bottomRow = groupArea.removeFromBottom (20);
    infoBendingLabel.setBounds (bottomRow);

    groupArea.removeFromBottom (6); // Spacer

    // Main Bending Area: Center Canvas + Right Controls
    auto controlsWidth = 100;
    auto sideControls = groupArea.removeFromRight (controlsWidth);
    groupArea.removeFromRight (12); // Gap between canvas and side knobs

    weightCanvas.setBounds (groupArea);

    // Layout side controls
    scaleLabel.setBounds (sideControls.removeFromTop (18));
    scaleSlider.setBounds (sideControls.removeFromTop (80));
    sideControls.removeFromTop (15); // Spacer
    offsetLabel.setBounds (sideControls.removeFromTop (18));
    offsetSlider.setBounds (sideControls.removeFromTop (80));
}

//==============================================================================
void NNBendingAudioProcessorEditor::comboBoxChanged (juce::ComboBox* comboBoxThatHasChanged)
{
    if (comboBoxThatHasChanged == &methodCombo)
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
    if (slider == &dryWetSlider)
    {
        audioProcessor.setDryWet ((float)dryWetSlider.getValue());
    }
    else
    {
        applyKnobBending();
    }
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
                currentBendingLayer = "";
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
    else if (button == &resetLayerButton)
    {
        resetLayerWeights();
    }
    else if (button == &resetAllButton)
    {
        resetAllLayerWeights();
    }
    else if (button == &saveModelButton)
    {
        saveModelToFile();
    }
}

//==============================================================================
void NNBendingAudioProcessorEditor::timerCallback()
{
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

        // Update Buffer readout badge
        bufferStatusLabel.setText (juce::String (audioProcessor.getBufferSize()), juce::dontSendNotification);

        // Update Mode combo box
        methodCombo.clear (juce::dontSendNotification);
        auto modes = audioProcessor.getAvailableModes();
        int id = 1;
        for (const auto& m : modes)
        {
            methodCombo.addItem (m, id++);
        }
        methodCombo.setText (audioProcessor.getCurrentMode(), juce::dontSendNotification);

        // Update Layer combo box
        layerCombo.clear (juce::dontSendNotification);
        auto layers = audioProcessor.getBackend().get_available_layers();
        id = 1;
        for (const auto& l : layers)
        {
            layerCombo.addItem (l, id++);
        }
        
        bool layerFound = false;
        for (const auto& l : layers)
        {
            if (l == currentBendingLayer.toStdString())
            {
                layerFound = true;
                break;
            }
        }

        if (layerFound && currentBendingLayer.isNotEmpty())
        {
            layerCombo.setText (currentBendingLayer, juce::dontSendNotification);
            selectLayer (currentBendingLayer);
        }
        else if (!layers.empty())
        {
            layerCombo.setText (layers[0], juce::dontSendNotification);
            selectLayer (layers[0]);
        }
        else
        {
            originalWeights.clear();
            currentWeights.clear();
            currentBendingLayer = "";
            weightCanvas.setWeights ({}, {});
            infoBendingLabel.setText ("Model has no trainable parameters.", juce::dontSendNotification);
        }
    }
    else
    {
        statusLabel.setText ("No model loaded.", juce::dontSendNotification);
        statusLabel.setColour (juce::Label::textColourId, juce::Colours::darkgrey);
        bufferStatusLabel.setText ("-", juce::dontSendNotification);
        methodCombo.clear (juce::dontSendNotification);
        layerCombo.clear (juce::dontSendNotification);
        infoBendingLabel.setText ("Load a model to view weights.", juce::dontSendNotification);
        originalWeights.clear();
        currentWeights.clear();
        currentBendingLayer = "";
        weightCanvas.setWeights ({}, {});
    }
}

void NNBendingAudioProcessorEditor::selectLayer (const juce::String& layerName)
{
    if (layerName.isEmpty()) return;
    
    currentBendingLayer = layerName;
    originalWeights = audioProcessor.getBackend().get_original_layer_weights (layerName.toStdString());
    currentWeights = audioProcessor.getBackend().get_layer_weights (layerName.toStdString());
    baseDrawnWeights = currentWeights;
    
    // Reset knob listeners to prevent feedback loop
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    
    scaleSlider.setValue (1.0);
    offsetSlider.setValue (0.0);
    lastKnobScale = 1.0;
    lastKnobOffset = 0.0;
    
    scaleSlider.addListener (this);
    offsetSlider.addListener (this);

    weightCanvas.setWeights (originalWeights, currentWeights);
    infoBendingLabel.setText ("Layer: " + currentBendingLayer + "  |  " + juce::String (originalWeights.size()) + " parameters", juce::dontSendNotification);
}

void NNBendingAudioProcessorEditor::applyKnobBending()
{
    if (currentBendingLayer.isEmpty() || currentWeights.empty()) return;
    if (baseDrawnWeights.empty())
        baseDrawnWeights = currentWeights;
    
    float scale = (float)scaleSlider.getValue();
    float offset = (float)offsetSlider.getValue();
    
    currentWeights.resize (baseDrawnWeights.size());
    for (size_t i = 0; i < baseDrawnWeights.size(); ++i)
    {
        currentWeights[i] = baseDrawnWeights[i] * scale + offset;
    }
    
    audioProcessor.getBackend().set_layer_weights (currentBendingLayer.toStdString(), currentWeights);
    weightCanvas.updateCurrentWeights (currentWeights);
}

void NNBendingAudioProcessorEditor::resetLayerWeights()
{
    if (currentBendingLayer.isEmpty()) return;
    
    audioProcessor.getBackend().reset_layer_weights (currentBendingLayer.toStdString());
    
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    scaleSlider.setValue (1.0);
    offsetSlider.setValue (0.0);
    lastKnobScale = 1.0;
    lastKnobOffset = 0.0;
    scaleSlider.addListener (this);
    offsetSlider.addListener (this);

    originalWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
    currentWeights = originalWeights;
    baseDrawnWeights = originalWeights;
    weightCanvas.setWeights (originalWeights, currentWeights);
}

void NNBendingAudioProcessorEditor::resetAllLayerWeights()
{
    audioProcessor.getBackend().reset_all_layer_weights();
    
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    scaleSlider.setValue (1.0);
    offsetSlider.setValue (0.0);
    lastKnobScale = 1.0;
    lastKnobOffset = 0.0;
    scaleSlider.addListener (this);
    offsetSlider.addListener (this);

    if (currentBendingLayer.isNotEmpty())
    {
        originalWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
        currentWeights = originalWeights;
        baseDrawnWeights = originalWeights;
        weightCanvas.setWeights (originalWeights, currentWeights);
    }
}

void NNBendingAudioProcessorEditor::saveModelToFile()
{
    if (!audioProcessor.isModelLoaded())
    {
        statusLabel.setText ("Cannot save: No model loaded.", juce::dontSendNotification);
        statusLabel.setColour (juce::Label::textColourId, juce::Colours::orangered);
        return;
    }

    fileChooser = std::make_unique<juce::FileChooser> (
        "Save Bended TorchScript Model (.ts)...",
        juce::File::getSpecialLocation (juce::File::userHomeDirectory).getChildFile ("bended_model.ts"),
        "*.ts"
    );

    auto flags = juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting;
    fileChooser->launchAsync (flags, [this] (const juce::FileChooser& fc)
    {
        auto file = fc.getResult();
        if (file.getFullPathName().isNotEmpty())
        {
            int err = audioProcessor.getBackend().save_model (file.getFullPathName().toStdString());
            if (err == 0)
            {
                statusLabel.setText ("Saved bended model: " + file.getFileName(), juce::dontSendNotification);
                statusLabel.setColour (juce::Label::textColourId, juce::Colours::lightgreen);
            }
            else
            {
                statusLabel.setText ("Error: Failed to save model.", juce::dontSendNotification);
                statusLabel.setColour (juce::Label::textColourId, juce::Colours::orangered);
            }
        }
    });
}
