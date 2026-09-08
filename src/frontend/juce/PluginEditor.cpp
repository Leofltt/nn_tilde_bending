#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
NNBendingAudioProcessorEditor::NNBendingAudioProcessorEditor (NNBendingAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Configure Title
    titleLabel.setText ("nn~ bending", juce::dontSendNotification);
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

    // Configure Fuse Button
    fuseButton.setButtonText ("FUSE: OK");
    fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1b3a2a"));
    fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgreen);
    fuseButton.addListener (this);
    addAndMakeVisible (fuseButton);

    // Configure Momentary Short Circuit Button
    shortCircuitButton.setButtonText ("SHORT [!]");
    shortCircuitButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff8b2500"));
    shortCircuitButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
    shortCircuitButton.addListener (this);
    addAndMakeVisible (shortCircuitButton);

    // Trigger Mode Selector
    triggerModeLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (triggerModeLabel);
    triggerModeCombo.addItem ("Continuous", 1);
    triggerModeCombo.addItem ("Momentary", 2);
    triggerModeCombo.addItem ("MIDI Gate", 3);
    triggerModeCombo.addItem ("Transient", 4);
    triggerModeCombo.setSelectedId ((int)audioProcessor.getTriggerMode() + 1, juce::dontSendNotification);
    triggerModeCombo.addListener (this);
    addAndMakeVisible (triggerModeCombo);
    shortCircuitButton.setVisible (audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Momentary);

    // Bending Group
    addAndMakeVisible (bendingGroup);

    // Category Filter
    categoryLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (categoryLabel);
    categoryCombo.addItem ("All Traces", 1);
    categoryCombo.addItem ("Norm / Dynamics", 2);
    categoryCombo.addItem ("Conv Kernels", 3);
    categoryCombo.addItem ("Linear / Dense", 4);
    categoryCombo.addItem ("Biases", 5);
    categoryCombo.setSelectedId (1, juce::dontSendNotification);
    categoryCombo.addListener (this);
    addAndMakeVisible (categoryCombo);

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

            audioProcessor.setLayerBaseWeights (currentBendingLayer.toStdString(), baseDrawnWeights);
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

    heatLabel.setText ("Heat", juce::dontSendNotification);
    heatLabel.setJustificationType (juce::Justification::centred);
    heatLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (heatLabel);

    heatSlider.setRange (0.0, 1.0, 0.001);
    heatSlider.setValue (0.0);
    heatSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    heatSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 18);
    heatSlider.setColour (juce::Slider::thumbColourId, juce::Colour::fromString ("#ffff9933"));
    heatSlider.setColour (juce::Slider::rotarySliderOutlineColourId, juce::Colours::darkgrey);
    heatSlider.addListener (this);
    addAndMakeVisible (heatSlider);

    memoryLabel.setText ("Memory", juce::dontSendNotification);
    memoryLabel.setJustificationType (juce::Justification::centred);
    memoryLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (memoryLabel);

    memorySlider.setRange (0.0, 1.0, 0.001);
    memorySlider.setValue (0.8);
    memorySlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    memorySlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 18);
    memorySlider.setColour (juce::Slider::thumbColourId, juce::Colour::fromString ("#ff38bdf8"));
    memorySlider.setColour (juce::Slider::rotarySliderOutlineColourId, juce::Colours::darkgrey);
    memorySlider.addListener (this);
    addAndMakeVisible (memorySlider);

    driftModeCombo.addItem ("Thermal OU", 1);
    driftModeCombo.addItem ("Random Walk", 2);
    driftModeCombo.addItem ("Glitch Noise", 3);
    driftModeCombo.setSelectedId (1, juce::dontSendNotification);
    driftModeCombo.addListener (this);
    addAndMakeVisible (driftModeCombo);

    freezeButton.setButtonText ("Freeze");
    freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
    freezeButton.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffff9933"));
    freezeButton.addListener (this);
    addAndMakeVisible (freezeButton);

    // Info Label
    infoBendingLabel.setText ("Select a layer to bend weights.", juce::dontSendNotification);
    infoBendingLabel.setJustificationType (juce::Justification::centredLeft);
    infoBendingLabel.setColour (juce::Label::textColourId, juce::Colours::silver);
    addAndMakeVisible (infoBendingLabel);

    // Expanded window size (860 x 600)
    setSize (860, 600);
    setResizable (true, true);
    setResizeLimits (760, 500, 1400, 900);

    // Initial load
    updateModelInfo();
    startTimer (60); // 16 FPS for responsive live bending and jitter animation
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

    // Top Header: Row 1 (Title + Status + Fuse)
    auto titleRow = area.removeFromTop (26);
    titleLabel.setBounds (titleRow.removeFromLeft (240));
    fuseButton.setBounds (titleRow.removeFromRight (100));
    titleRow.removeFromRight (12);
    statusLabel.setBounds (titleRow);

    area.removeFromTop (6); // Spacer

    // Header Controls: Row 2 (Load, Mode, Buffer, Dry/Wet, Trigger, Short, Save)
    auto headerRow = area.removeFromTop (32);
    loadButton.setBounds (headerRow.removeFromLeft (125));
    headerRow.removeFromLeft (10);

    methodLabel.setBounds (headerRow.removeFromLeft (45));
    methodCombo.setBounds (headerRow.removeFromLeft (95));
    headerRow.removeFromLeft (10);

    bufferLabel.setBounds (headerRow.removeFromLeft (48));
    bufferStatusLabel.setBounds (headerRow.removeFromLeft (60));
    headerRow.removeFromLeft (10);

    dryWetLabel.setBounds (headerRow.removeFromLeft (55));
    dryWetSlider.setBounds (headerRow.removeFromLeft (100));
    headerRow.removeFromLeft (10);

    triggerModeLabel.setBounds (headerRow.removeFromLeft (50));
    triggerModeCombo.setBounds (headerRow.removeFromLeft (100));

    if (shortCircuitButton.isVisible())
    {
        headerRow.removeFromLeft (8);
        shortCircuitButton.setBounds (headerRow.removeFromLeft (85));
    }

    saveModelButton.setBounds (headerRow.removeFromRight (130));

    area.removeFromTop (18); // Spacer clearing the divider line

    // Bending Group Section
    bendingGroup.setBounds (area);
    auto groupArea = bendingGroup.getBounds().reduced (14);
    groupArea.removeFromTop (12); // Group title offset

    // Layer Selection & Action Row (Category filter + Layer Combo + Reset buttons)
    auto layerRow = groupArea.removeFromTop (32);
    categoryLabel.setBounds (layerRow.removeFromLeft (42));
    categoryCombo.setBounds (layerRow.removeFromLeft (120));
    layerRow.removeFromLeft (10);

    layerLabel.setBounds (layerRow.removeFromLeft (42));
    
    resetAllButton.setBounds (layerRow.removeFromRight (125));
    layerRow.removeFromRight (6);
    resetLayerButton.setBounds (layerRow.removeFromRight (105));
    layerRow.removeFromRight (10);
    
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

    // Layout side controls: Scale, Offset, Heat, Memory, Drift Mode, Freeze
    scaleLabel.setBounds (sideControls.removeFromTop (14));
    scaleSlider.setBounds (sideControls.removeFromTop (64));
    sideControls.removeFromTop (4);

    offsetLabel.setBounds (sideControls.removeFromTop (14));
    offsetSlider.setBounds (sideControls.removeFromTop (64));
    sideControls.removeFromTop (4);

    heatLabel.setBounds (sideControls.removeFromTop (14));
    heatSlider.setBounds (sideControls.removeFromTop (64));
    sideControls.removeFromTop (4);

    memoryLabel.setBounds (sideControls.removeFromTop (14));
    memorySlider.setBounds (sideControls.removeFromTop (64));
    sideControls.removeFromTop (6);

    driftModeCombo.setBounds (sideControls.removeFromTop (24));
    sideControls.removeFromTop (4);
    freezeButton.setBounds (sideControls.removeFromTop (22));
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
    else if (comboBoxThatHasChanged == &categoryCombo)
    {
        // Filter layerCombo based on selected trace category
        int catId = categoryCombo.getSelectedId();
        layerCombo.clear (juce::dontSendNotification);
        auto layers = audioProcessor.getBackend().get_available_layers();
        int id = 1;
        for (const auto& l : layers)
        {
            auto cat = NNBendingAudioProcessor::classifyLayer (l);
            bool include = false;
            if (catId == 1) include = true; // All
            else if (catId == 2 && cat == NNBendingAudioProcessor::LayerCategory::Norm) include = true;
            else if (catId == 3 && cat == NNBendingAudioProcessor::LayerCategory::Conv) include = true;
            else if (catId == 4 && cat == NNBendingAudioProcessor::LayerCategory::Linear) include = true;
            else if (catId == 5 && cat == NNBendingAudioProcessor::LayerCategory::Bias) include = true;

            if (include)
                layerCombo.addItem (l, id++);
        }
        if (layerCombo.getNumItems() > 0)
        {
            layerCombo.setSelectedId (1);
            selectLayer (layerCombo.getText());
        }
    }
    else if (comboBoxThatHasChanged == &triggerModeCombo)
    {
        int modeIdx = triggerModeCombo.getSelectedId() - 1;
        auto mode = (NNBendingAudioProcessor::TriggerMode)modeIdx;
        audioProcessor.setTriggerMode (mode);
        shortCircuitButton.setVisible (mode == NNBendingAudioProcessor::TriggerMode::Momentary);
        resized();
    }
    else if (comboBoxThatHasChanged == &driftModeCombo)
    {
        int modeIdx = driftModeCombo.getSelectedId() - 1;
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerDriftMode (currentBendingLayer.toStdString(), (NNBendingAudioProcessor::DriftMode)modeIdx);
    }
}

void NNBendingAudioProcessorEditor::sliderValueChanged (juce::Slider* slider)
{
    if (slider == &dryWetSlider)
    {
        audioProcessor.setDryWet ((float)dryWetSlider.getValue());
    }
    else if (slider == &heatSlider)
    {
        float val = (float)heatSlider.getValue();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerHeat (currentBendingLayer.toStdString(), val);
        if (auto* param = audioProcessor.getHeatParam())
            *param = val;
    }
    else if (slider == &memorySlider)
    {
        float val = (float)memorySlider.getValue();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerMemory (currentBendingLayer.toStdString(), val);
        if (auto* param = audioProcessor.getMemoryParam())
            *param = val;
    }
    else
    {
        applyKnobBending();
    }
}

void NNBendingAudioProcessorEditor::buttonClicked (juce::Button* button)
{
    if (button == &fuseButton)
    {
        audioProcessor.resetFuse();
        fuseButton.setButtonText ("FUSE: OK");
        fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1b3a2a"));
        fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgreen);
        if (currentBendingLayer.isNotEmpty())
            selectLayer (currentBendingLayer);
    }
    else if (button == &shortCircuitButton)
    {
        // Toggle or pulse momentary short-circuit gate
        bool active = !audioProcessor.isShortCircuitActive();
        audioProcessor.triggerShortCircuit (active);
        shortCircuitButton.setColour (juce::TextButton::buttonColourId,
                                      active ? juce::Colour::fromString ("#ffff3300") : juce::Colour::fromString ("#ff8b2500"));
    }
    else if (button == &freezeButton)
    {
        bool frozen = freezeButton.getToggleState();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerFrozen (currentBendingLayer.toStdString(), frozen);

        if (frozen)
        {
            freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffff9933"));
            // When freezing, update currentWeights and baseDrawnWeights with the frozen snapshot
            if (currentBendingLayer.isNotEmpty())
            {
                currentWeights = audioProcessor.getBackend().get_layer_weights (currentBendingLayer.toStdString());
                float scale = (float)scaleSlider.getValue();
                float offset = (float)offsetSlider.getValue();
                baseDrawnWeights.resize (currentWeights.size());
                if (std::abs (scale) > 0.00001f)
                {
                    for (size_t i = 0; i < currentWeights.size(); ++i)
                        baseDrawnWeights[i] = (currentWeights[i] - offset) / scale;
                }
                else
                {
                    baseDrawnWeights = currentWeights;
                }
                audioProcessor.setLayerBaseWeights (currentBendingLayer.toStdString(), baseDrawnWeights);
                weightCanvas.updateCurrentWeights (currentWeights);
            }
        }
        else
        {
            freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
        }
    }
    else if (button == &loadButton)
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

    // If DAW host is automating parameters, reflect changes into UI knobs
    if (auto* p = audioProcessor.getScaleParam())
    {
        if (std::abs (p->get() - (float)scaleSlider.getValue()) > 0.001f && !scaleSlider.isMouseButtonDown())
            scaleSlider.setValue (p->get(), juce::sendNotificationSync);
    }
    if (auto* p = audioProcessor.getOffsetParam())
    {
        if (std::abs (p->get() - (float)offsetSlider.getValue()) > 0.0001f && !offsetSlider.isMouseButtonDown())
            offsetSlider.setValue (p->get(), juce::sendNotificationSync);
    }
    if (auto* p = audioProcessor.getHeatParam())
    {
        if (std::abs (p->get() - (float)heatSlider.getValue()) > 0.0001f && !heatSlider.isMouseButtonDown())
            heatSlider.setValue (p->get(), juce::sendNotificationSync);
    }
    if (auto* p = audioProcessor.getMemoryParam())
    {
        if (std::abs (p->get() - (float)memorySlider.getValue()) > 0.0001f && !memorySlider.isMouseButtonDown())
            memorySlider.setValue (p->get(), juce::sendNotificationSync);
    }
    if (auto* p = audioProcessor.getDryWetParam())
    {
        if (std::abs (p->get() - (float)dryWetSlider.getValue()) > 0.001f && !dryWetSlider.isMouseButtonDown())
            dryWetSlider.setValue (p->get(), juce::dontSendNotification);
    }

    // Update Fuse Button State visually
    if (audioProcessor.isFuseBlown())
    {
        fuseButton.setButtonText ("FUSE: BLOWN!");
        fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff990000"));
        fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::yellow);
    }
    else
    {
        // Live bending status: in Continuous mode with bends active, or momentary/midi/transient active
        float env = audioProcessor.getMomentaryEnvelope();
        bool bendingActive = false;
        if (audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Continuous)
        {
            bendingActive = audioProcessor.hasActiveBending();
        }
        else
        {
            bendingActive = (env > 0.01f);
        }

        if (bendingActive)
        {
            fuseButton.setButtonText ("FUSE: ACTIVE");
            fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff006633")); // Vibrant glowing green
            fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        }
        else
        {
            fuseButton.setButtonText ("FUSE: OK");
            fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1b3a2a")); // Idle dark green
            fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgreen);
        }
    }

    // Update Short Button glow based on live momentary envelope
    float env = audioProcessor.getMomentaryEnvelope();
    if (env > 0.05f)
    {
        shortCircuitButton.setColour (juce::TextButton::buttonColourId,
            juce::Colour::fromString ("#ff8b2500").interpolatedWith (juce::Colour::fromString ("#ffff3300"), env));
    }
    else
    {
        shortCircuitButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff8b2500"));
    }

    // If live heat is active and not frozen, or momentary envelope is animating, visualize live weights on the canvas
    // (Only update when user is not actively drawing with the mouse)
    if (audioProcessor.isModelLoaded() && currentBendingLayer.isNotEmpty() && !weightCanvas.isCurrentlyDrawing())
    {
        auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());
        bool isAnimating = (state.heat > 0.0001f && !state.frozen)
                        || (audioProcessor.getTriggerMode() != NNBendingAudioProcessor::TriggerMode::Continuous && env > 0.01f);
        if (isAnimating)
        {
            auto liveWeights = audioProcessor.getBackend().get_layer_weights (currentBendingLayer.toStdString());
            if (!liveWeights.empty())
            {
                weightCanvas.updateCurrentWeights (liveWeights);
            }
        }
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

        // Update Layer combo box respecting selected Category
        layerCombo.clear (juce::dontSendNotification);
        auto layers = audioProcessor.getBackend().get_available_layers();
        int catId = categoryCombo.getSelectedId();
        id = 1;
        for (const auto& l : layers)
        {
            auto cat = NNBendingAudioProcessor::classifyLayer (l);
            bool include = false;
            if (catId <= 1) include = true;
            else if (catId == 2 && cat == NNBendingAudioProcessor::LayerCategory::Norm) include = true;
            else if (catId == 3 && cat == NNBendingAudioProcessor::LayerCategory::Conv) include = true;
            else if (catId == 4 && cat == NNBendingAudioProcessor::LayerCategory::Linear) include = true;
            else if (catId == 5 && cat == NNBendingAudioProcessor::LayerCategory::Bias) include = true;

            if (include)
                layerCombo.addItem (l, id++);
        }
        
        bool layerFound = false;
        for (int i = 0; i < layerCombo.getNumItems(); ++i)
        {
            if (layerCombo.getItemText(i) == currentBendingLayer)
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
        else if (layerCombo.getNumItems() > 0)
        {
            layerCombo.setSelectedId (1, juce::dontSendNotification);
            selectLayer (layerCombo.getItemText(0));
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
    audioProcessor.setActiveLayerName (currentBendingLayer);

    originalWeights = audioProcessor.getBackend().get_original_layer_weights (layerName.toStdString());
    currentWeights = audioProcessor.getBackend().get_layer_weights (layerName.toStdString());

    // Fetch this layer's stored state
    auto state = audioProcessor.getLayerState (layerName.toStdString());
    if (state.baseDrawnWeights.empty())
    {
        baseDrawnWeights = currentWeights;
        state.baseDrawnWeights = currentWeights;
        audioProcessor.setLayerBaseWeights (layerName.toStdString(), baseDrawnWeights);
    }
    else
    {
        baseDrawnWeights = state.baseDrawnWeights;
    }

    // Set knobs to this layer's saved values without triggering recursive listeners
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    heatSlider.removeListener (this);
    memorySlider.removeListener (this);
    driftModeCombo.removeListener (this);
    
    scaleSlider.setValue (state.scale, juce::dontSendNotification);
    offsetSlider.setValue (state.offset, juce::dontSendNotification);
    heatSlider.setValue (state.heat, juce::dontSendNotification);
    memorySlider.setValue (state.memory, juce::dontSendNotification);
    driftModeCombo.setSelectedId ((int)state.driftMode + 1, juce::dontSendNotification);

    freezeButton.setToggleState (state.frozen, juce::dontSendNotification);
    freezeButton.setColour (juce::ToggleButton::textColourId,
                            state.frozen ? juce::Colour::fromString ("#ffff9933") : juce::Colours::lightgrey);

    // Sync DAW host automatable parameters to active layer values
    if (auto* p = audioProcessor.getScaleParam())  *p = state.scale;
    if (auto* p = audioProcessor.getOffsetParam()) *p = state.offset;
    if (auto* p = audioProcessor.getHeatParam())   *p = state.heat;
    if (auto* p = audioProcessor.getMemoryParam()) *p = state.memory;

    lastKnobScale = state.scale;
    lastKnobOffset = state.offset;
    
    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    heatSlider.addListener (this);
    memorySlider.addListener (this);
    driftModeCombo.addListener (this);

    weightCanvas.setWeights (originalWeights, currentWeights);
    
    // Display layer shape / category info
    auto category = NNBendingAudioProcessor::classifyLayer (currentBendingLayer.toStdString());
    juce::String catName = "Other";
    if (category == NNBendingAudioProcessor::LayerCategory::Norm) catName = "Norm / Dynamics";
    else if (category == NNBendingAudioProcessor::LayerCategory::Conv) catName = "Conv Kernel";
    else if (category == NNBendingAudioProcessor::LayerCategory::Linear) catName = "Linear / Dense";
    else if (category == NNBendingAudioProcessor::LayerCategory::Bias) catName = "Bias";

    infoBendingLabel.setText ("Trace: [" + catName + "]  |  Layer: " + currentBendingLayer + "  |  " + juce::String (originalWeights.size()) + " params", juce::dontSendNotification);
}

void NNBendingAudioProcessorEditor::applyKnobBending()
{
    if (currentBendingLayer.isEmpty() || currentWeights.empty()) return;
    if (baseDrawnWeights.empty())
        baseDrawnWeights = currentWeights;
    
    float scale = (float)scaleSlider.getValue();
    float offset = (float)offsetSlider.getValue();

    // Store per-layer scale and offset
    audioProcessor.setLayerScale (currentBendingLayer.toStdString(), scale);
    audioProcessor.setLayerOffset (currentBendingLayer.toStdString(), offset);

    if (auto* p = audioProcessor.getScaleParam())  *p = scale;
    if (auto* p = audioProcessor.getOffsetParam()) *p = offset;
    
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
    audioProcessor.clearLayerBending (currentBendingLayer.toStdString());
    
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    heatSlider.removeListener (this);

    scaleSlider.setValue (1.0, juce::dontSendNotification);
    offsetSlider.setValue (0.0, juce::dontSendNotification);
    heatSlider.setValue (0.0, juce::dontSendNotification);
    freezeButton.setToggleState (false, juce::dontSendNotification);
    freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);

    if (auto* p = audioProcessor.getScaleParam())  *p = 1.0f;
    if (auto* p = audioProcessor.getOffsetParam()) *p = 0.0f;
    if (auto* p = audioProcessor.getHeatParam())   *p = 0.0f;

    lastKnobScale = 1.0;
    lastKnobOffset = 0.0;

    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    heatSlider.addListener (this);

    originalWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
    currentWeights = originalWeights;
    baseDrawnWeights = originalWeights;

    weightCanvas.setWeights (originalWeights, currentWeights);
}

void NNBendingAudioProcessorEditor::resetAllLayerWeights()
{
    audioProcessor.getBackend().reset_all_layer_weights();
    audioProcessor.clearAllLayerBending();
    
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    heatSlider.removeListener (this);

    scaleSlider.setValue (1.0, juce::dontSendNotification);
    offsetSlider.setValue (0.0, juce::dontSendNotification);
    heatSlider.setValue (0.0, juce::dontSendNotification);
    freezeButton.setToggleState (false, juce::dontSendNotification);
    freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);

    if (auto* p = audioProcessor.getScaleParam())  *p = 1.0f;
    if (auto* p = audioProcessor.getOffsetParam()) *p = 0.0f;
    if (auto* p = audioProcessor.getHeatParam())   *p = 0.0f;

    lastKnobScale = 1.0;
    lastKnobOffset = 0.0;

    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    heatSlider.addListener (this);

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
