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
    triggerModeCombo.addItem ("Transient", 3);
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
            // The drawn weights are stored as the layer's target drawn curve (drawnWeights)
            // Normalized inverse against scale/offset so subsequent knob tweaks scale from the drawn shape:
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

            audioProcessor.setLayerDrawnWeights (currentBendingLayer.toStdString(), baseDrawnWeights);

            auto mode = audioProcessor.getTriggerMode();
            if (mode == NNBendingAudioProcessor::TriggerMode::Continuous)
            {
                // In continuous mode, drawn weights apply directly to real-time model weights
                currentWeights = modifiedWeights;
                audioProcessor.getBackend().set_layer_weights (currentBendingLayer.toStdString(), currentWeights);
                weightCanvas.setTargetBentWeights ({});
            }
            else
            {
                // In momentary / midi / transient modes:
                // Keep live model at whatever backend is currently playing, and update pink target preview
                weightCanvas.setTargetBentWeights (modifiedWeights);
            }
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

    // Trace Bridging (Cross-Talk) Controls
    bridgeLabel.setText ("Bridge Wire", juce::dontSendNotification);
    bridgeLabel.setJustificationType (juce::Justification::centred);
    bridgeLabel.setColour (juce::Label::textColourId, juce::Colour::fromString ("#ffc26a38")); // Warm copper
    addAndMakeVisible (bridgeLabel);

    bridgeCombo.addListener (this);
    addAndMakeVisible (bridgeCombo);

    bridgeDepthLabel.setText ("Cross-Talk", juce::dontSendNotification);
    bridgeDepthLabel.setJustificationType (juce::Justification::centred);
    bridgeDepthLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (bridgeDepthLabel);

    bridgeDepthSlider.setRange (0.0, 1.0, 0.01);
    bridgeDepthSlider.setValue (0.0);
    bridgeDepthSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    bridgeDepthSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 18);
    bridgeDepthSlider.setColour (juce::Slider::thumbColourId, juce::Colour::fromString ("#ffd99b26")); // Warm golden ochre
    bridgeDepthSlider.setColour (juce::Slider::rotarySliderOutlineColourId, juce::Colours::darkgrey);
    bridgeDepthSlider.addListener (this);
    addAndMakeVisible (bridgeDepthSlider);

    // Info Label
    infoBendingLabel.setText ("Select a layer to bend weights.", juce::dontSendNotification);
    infoBendingLabel.setJustificationType (juce::Justification::centredLeft);
    infoBendingLabel.setColour (juce::Label::textColourId, juce::Colours::silver);
    addAndMakeVisible (infoBendingLabel);

    // Expanded window size (1100 x 640)
    setSize (1100, 640);
    setResizable (true, true);
    setResizeLimits (1060, 600, 1600, 1000);

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
    g.drawHorizontalLine (52, 0.0f, (float)getWidth());
}

void NNBendingAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced (12);

    // Header Controls Row (Height: 34px)
    auto headerRow = area.removeFromTop (34);

    // Pin critical action buttons from right first so they never get truncated
    saveModelButton.setBounds (headerRow.removeFromRight (125));
    headerRow.removeFromRight (10);

    fuseButton.setBounds (headerRow.removeFromRight (88));
    headerRow.removeFromRight (8);

    if (shortCircuitButton.isVisible())
    {
        shortCircuitButton.setBounds (headerRow.removeFromRight (82));
        headerRow.removeFromRight (8);
    }

    triggerModeCombo.setBounds (headerRow.removeFromRight (90));
    triggerModeLabel.setBounds (headerRow.removeFromRight (48));
    headerRow.removeFromRight (8);

    dryWetSlider.setBounds (headerRow.removeFromRight (85));
    dryWetLabel.setBounds (headerRow.removeFromRight (50));
    headerRow.removeFromRight (10);

    bufferStatusLabel.setBounds (headerRow.removeFromRight (44));
    bufferLabel.setBounds (headerRow.removeFromRight (40));
    headerRow.removeFromRight (10);

    // Now fill remaining left side
    loadButton.setBounds (headerRow.removeFromLeft (90));
    headerRow.removeFromLeft (8);

    methodLabel.setBounds (headerRow.removeFromLeft (40));
    methodCombo.setBounds (headerRow.removeFromLeft (90));
    headerRow.removeFromLeft (8);

    // statusLabel takes all remaining middle space
    statusLabel.setBounds (headerRow);

    area.removeFromTop (10); // Spacer clearing the divider line

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

    // Main Bending Area: Center Canvas + Right Dual-Column Controls (Mutate / Thermal & Bridge)
    auto controlsWidth = 195;
    auto sideControls = groupArea.removeFromRight (controlsWidth);
    groupArea.removeFromRight (12); // Gap between canvas and side knobs

    weightCanvas.setBounds (groupArea);

    // Split sideControls into Column 1 (Static & Drift: 92px) and Column 2 (Bridge: 92px)
    auto col1 = sideControls.removeFromLeft (92);
    sideControls.removeFromLeft (8); // Column gutter
    auto col2 = sideControls;

    // Column 1: Scale, Offset, Heat, Memory, Drift Mode, Freeze
    scaleLabel.setBounds (col1.removeFromTop (14));
    scaleSlider.setBounds (col1.removeFromTop (64));
    col1.removeFromTop (4);

    offsetLabel.setBounds (col1.removeFromTop (14));
    offsetSlider.setBounds (col1.removeFromTop (64));
    col1.removeFromTop (4);

    heatLabel.setBounds (col1.removeFromTop (14));
    heatSlider.setBounds (col1.removeFromTop (64));
    col1.removeFromTop (4);

    memoryLabel.setBounds (col1.removeFromTop (14));
    memorySlider.setBounds (col1.removeFromTop (64));
    col1.removeFromTop (6);

    driftModeCombo.setBounds (col1.removeFromTop (24));
    col1.removeFromTop (4);
    freezeButton.setBounds (col1.removeFromTop (22));

    // Column 2: Bridge Wire source selector & Cross-Talk depth
    bridgeLabel.setBounds (col2.removeFromTop (14));
    bridgeCombo.setBounds (col2.removeFromTop (26));
    col2.removeFromTop (8);

    bridgeDepthLabel.setBounds (col2.removeFromTop (14));
    bridgeDepthSlider.setBounds (col2.removeFromTop (64));
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

        // Update target overlay and bridge line style (dotted vs solid) immediately on mode change
        if (currentBendingLayer.isNotEmpty())
            applyKnobBending();
    }
    else if (comboBoxThatHasChanged == &driftModeCombo)
    {
        int modeIdx = driftModeCombo.getSelectedId() - 1;
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerDriftMode (currentBendingLayer.toStdString(), (NNBendingAudioProcessor::DriftMode)modeIdx);
    }
    else if (comboBoxThatHasChanged == &bridgeCombo)
    {
        if (currentBendingLayer.isNotEmpty())
        {
            juce::String src = (bridgeCombo.getSelectedId() > 1) ? bridgeCombo.getText() : "";
            audioProcessor.setLayerBridgeSource (currentBendingLayer.toStdString(), src.toStdString());
            applyKnobBending();
        }
    }
}

void NNBendingAudioProcessorEditor::sliderValueChanged (juce::Slider* slider)
{
    if (slider == &dryWetSlider)
    {
        float val = (float)dryWetSlider.getValue();
        audioProcessor.setDryWet (val);
        lastKnownDryWet = val;
    }
    else if (slider == &heatSlider)
    {
        float val = (float)heatSlider.getValue();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerHeat (currentBendingLayer.toStdString(), val);
        if (auto* param = audioProcessor.getHeatParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownHeat = val;
    }
    else if (slider == &memorySlider)
    {
        float val = (float)memorySlider.getValue();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerMemory (currentBendingLayer.toStdString(), val);
        if (auto* param = audioProcessor.getMemoryParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownMemory = val;
    }
    else if (slider == &scaleSlider)
    {
        float val = (float)scaleSlider.getValue();
        if (auto* param = audioProcessor.getScaleParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownScale = val;
        applyKnobBending();
    }
    else if (slider == &offsetSlider)
    {
        float val = (float)offsetSlider.getValue();
        if (auto* param = audioProcessor.getOffsetParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownOffset = val;
        applyKnobBending();
    }
    else if (slider == &bridgeDepthSlider)
    {
        float val = (float)bridgeDepthSlider.getValue();
        if (currentBendingLayer.isNotEmpty())
        {
            audioProcessor.setLayerBridgeDepth (currentBendingLayer.toStdString(), val);
            applyKnobBending();
        }
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

    // If DAW host is automating parameters, reflect changes into UI knobs (without feedback loops)
    if (auto* p = audioProcessor.getScaleParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownScale) > 0.005f && !scaleSlider.isMouseButtonDown())
        {
            lastKnownScale = pVal;
            scaleSlider.setValue (pVal, juce::dontSendNotification);
            applyKnobBending();
        }
    }
    if (auto* p = audioProcessor.getOffsetParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownOffset) > 0.001f && !offsetSlider.isMouseButtonDown())
        {
            lastKnownOffset = pVal;
            offsetSlider.setValue (pVal, juce::dontSendNotification);
            applyKnobBending();
        }
    }
    if (auto* p = audioProcessor.getHeatParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownHeat) > 0.001f && !heatSlider.isMouseButtonDown())
        {
            lastKnownHeat = pVal;
            heatSlider.setValue (pVal, juce::dontSendNotification);
            if (currentBendingLayer.isNotEmpty())
                audioProcessor.setLayerHeat (currentBendingLayer.toStdString(), pVal);
        }
    }
    if (auto* p = audioProcessor.getMemoryParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownMemory) > 0.001f && !memorySlider.isMouseButtonDown())
        {
            lastKnownMemory = pVal;
            memorySlider.setValue (pVal, juce::dontSendNotification);
            if (currentBendingLayer.isNotEmpty())
                audioProcessor.setLayerMemory (currentBendingLayer.toStdString(), pVal);
        }
    }
    if (auto* p = audioProcessor.getDryWetParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownDryWet) > 0.005f && !dryWetSlider.isMouseButtonDown())
        {
            lastKnownDryWet = pVal;
            dryWetSlider.setValue (pVal, juce::dontSendNotification);
            audioProcessor.setDryWet (pVal);
        }
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

    // Update canvas curves:
    // 1) In Momentary or Transient modes, display the target bent curve (neon pink dashed line)
    // 2) Display live weights currently active inside the model (purple curve with gradient fill)
    if (audioProcessor.isModelLoaded() && currentBendingLayer.isNotEmpty() && !weightCanvas.isCurrentlyDrawing())
    {
        auto triggerMode = audioProcessor.getTriggerMode();
        auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());

        if (triggerMode == NNBendingAudioProcessor::TriggerMode::Continuous)
        {
            // In continuous mode, target is identical to current live state, so no separate dashed ghost needed
            weightCanvas.setTargetBentWeights ({});
            
            // If heat/jitter is oscillating or parameters change, stream live weights to canvas
            if (state.heat > 0.0001f && !state.frozen)
            {
                auto liveWeights = audioProcessor.getBackend().get_layer_weights (currentBendingLayer.toStdString());
                if (!liveWeights.empty())
                    weightCanvas.updateCurrentWeights (liveWeights);
            }
        }
        else
        {
            // In Momentary / Transient modes:
            // Calculate target bent curve: (bridgedBase * scale + offset)
            if (!state.drawnWeights.empty())
            {
                std::vector<float> bridgedBase = state.drawnWeights;
                std::vector<float> srcTiled;
                if (state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != currentBendingLayer.toStdString())
                {
                    auto srcW = audioProcessor.getBackend().get_original_layer_weights (state.bridgeSourceLayer);
                    if (!srcW.empty())
                    {
                        float alpha = state.bridgeDepth;
                        size_t srcSize = srcW.size();
                        srcTiled.resize (bridgedBase.size());
                        for (size_t i = 0; i < bridgedBase.size(); ++i)
                        {
                            srcTiled[i] = srcW[i % srcSize];
                            bridgedBase[i] = (1.0f - alpha) * state.drawnWeights[i] + alpha * srcTiled[i];
                        }
                    }
                }
                weightCanvas.setBridgeWeights (srcTiled, bridgedBase, state.bridgeDepth, false);

                std::vector<float> targetBent(bridgedBase.size());
                for (size_t i = 0; i < bridgedBase.size(); ++i)
                    targetBent[i] = bridgedBase[i] * state.scale + state.offset;
                weightCanvas.setTargetBentWeights (targetBent);
            }
            else
            {
                weightCanvas.setTargetBentWeights ({});
            }

            // Always update live curve to show real-time model weights transitioning between baseline and bent
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

        // Populate Bridge Combo with [No Bridge] + all model layers
        bridgeCombo.clear (juce::dontSendNotification);
        bridgeCombo.addItem ("(Off)", 1);
        int bridgeId = 2;
        for (const auto& l : layers)
        {
            bridgeCombo.addItem (l, bridgeId++);
        }
        bridgeCombo.setSelectedId (1, juce::dontSendNotification);
        
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
    if (state.originalWeights.empty())
    {
        state.originalWeights = originalWeights;
        audioProcessor.setLayerOriginalWeights (layerName.toStdString(), originalWeights);
    }

    if (state.drawnWeights.empty())
    {
        baseDrawnWeights = originalWeights;
        state.drawnWeights = originalWeights;
        audioProcessor.setLayerDrawnWeights (layerName.toStdString(), baseDrawnWeights);
    }
    else
    {
        baseDrawnWeights = state.drawnWeights;
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
    if (auto* p = audioProcessor.getScaleParam())  p->setValueNotifyingHost (p->convertTo0to1 (state.scale));
    if (auto* p = audioProcessor.getOffsetParam()) p->setValueNotifyingHost (p->convertTo0to1 (state.offset));
    if (auto* p = audioProcessor.getHeatParam())   p->setValueNotifyingHost (p->convertTo0to1 (state.heat));
    if (auto* p = audioProcessor.getMemoryParam()) p->setValueNotifyingHost (p->convertTo0to1 (state.memory));

    // Sync bridge controls with this layer's state
    bridgeCombo.removeListener (this);
    bridgeDepthSlider.removeListener (this);

    if (state.bridgeSourceLayer.empty())
    {
        bridgeCombo.setSelectedId (1, juce::dontSendNotification);
    }
    else
    {
        int foundId = 1;
        for (int i = 0; i < bridgeCombo.getNumItems(); ++i)
        {
            if (bridgeCombo.getItemText(i).toStdString() == state.bridgeSourceLayer)
            {
                foundId = bridgeCombo.getItemId(i);
                break;
            }
        }
        bridgeCombo.setSelectedId (foundId, juce::dontSendNotification);
    }
    bridgeDepthSlider.setValue (state.bridgeDepth, juce::dontSendNotification);

    bridgeCombo.addListener (this);
    bridgeDepthSlider.addListener (this);

    weightCanvas.setWeights (originalWeights, currentWeights);

    // Compute and send bridge source & mix curves to canvas
    std::vector<float> bridgedBase = baseDrawnWeights;
    std::vector<float> srcTiled;
    if (state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != layerName.toStdString())
    {
        auto srcW = audioProcessor.getBackend().get_original_layer_weights (state.bridgeSourceLayer);
        if (!srcW.empty())
        {
            float alpha = state.bridgeDepth;
            size_t srcSize = srcW.size();
            srcTiled.resize (bridgedBase.size());
            for (size_t i = 0; i < bridgedBase.size(); ++i)
            {
                srcTiled[i] = srcW[i % srcSize];
                bridgedBase[i] = (1.0f - alpha) * baseDrawnWeights[i] + alpha * srcTiled[i];
            }
        }
    }
    bool isContinuous = (audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Continuous);
    weightCanvas.setBridgeWeights (srcTiled, bridgedBase, state.bridgeDepth, isContinuous);

    // If in momentary/transient modes, compute target bent overlay
    if (!isContinuous)
    {
        std::vector<float> targetBent(bridgedBase.size());
        for (size_t i = 0; i < bridgedBase.size(); ++i)
            targetBent[i] = bridgedBase[i] * state.scale + state.offset;
        weightCanvas.setTargetBentWeights (targetBent);
    }
    else
    {
        weightCanvas.setTargetBentWeights ({});
    }
    
    // Display layer shape / category info
    auto category = NNBendingAudioProcessor::classifyLayer (currentBendingLayer.toStdString());
    juce::String catName = "Other";
    if (category == NNBendingAudioProcessor::LayerCategory::Norm) catName = "Norm / Dynamics";
    else if (category == NNBendingAudioProcessor::LayerCategory::Conv) catName = "Conv Kernel";
    else if (category == NNBendingAudioProcessor::LayerCategory::Linear) catName = "Linear / Dense";
    else if (category == NNBendingAudioProcessor::LayerCategory::Bias) catName = "Bias";

    infoBendingLabel.setText ("Trace: [" + catName + "]  |  Layer: " + currentBendingLayer + "  |  " + juce::String (originalWeights.size()) + " params", juce::dontSendNotification);

    // Re-enable listeners so user interactions are captured
    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    heatSlider.addListener (this);
    memorySlider.addListener (this);
    driftModeCombo.addListener (this);

    // Update last known parameters to current layer's state so timer doesn't snap them back
    lastKnownScale = state.scale;
    lastKnownOffset = state.offset;
    lastKnownHeat = state.heat;
    lastKnownMemory = state.memory;
}

void NNBendingAudioProcessorEditor::applyKnobBending()
{
    if (currentBendingLayer.isEmpty()) return;
    if (baseDrawnWeights.empty())
        baseDrawnWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
    
    float scale = (float)scaleSlider.getValue();
    float offset = (float)offsetSlider.getValue();

    // Store per-layer scale and offset
    audioProcessor.setLayerScale (currentBendingLayer.toStdString(), scale);
    audioProcessor.setLayerOffset (currentBendingLayer.toStdString(), offset);

    // Fetch bridge state
    auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());
    std::vector<float> bridgedBase = baseDrawnWeights;
    std::vector<float> srcTiled;
    if (state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != currentBendingLayer.toStdString())
    {
        auto srcW = audioProcessor.getBackend().get_original_layer_weights (state.bridgeSourceLayer);
        if (!srcW.empty())
        {
            float alpha = state.bridgeDepth;
            size_t srcSize = srcW.size();
            srcTiled.resize (bridgedBase.size());
            for (size_t i = 0; i < bridgedBase.size(); ++i)
            {
                srcTiled[i] = srcW[i % srcSize];
                bridgedBase[i] = (1.0f - alpha) * baseDrawnWeights[i] + alpha * srcTiled[i];
            }
        }
    }

    auto mode = audioProcessor.getTriggerMode();
    bool isContinuous = (mode == NNBendingAudioProcessor::TriggerMode::Continuous);
    weightCanvas.setBridgeWeights (srcTiled, bridgedBase, state.bridgeDepth, isContinuous);

    std::vector<float> bentTarget(bridgedBase.size());
    for (size_t i = 0; i < bridgedBase.size(); ++i)
    {
        bentTarget[i] = bridgedBase[i] * scale + offset;
    }

    if (isContinuous)
    {
        currentWeights = bentTarget;
        audioProcessor.getBackend().set_layer_weights (currentBendingLayer.toStdString(), currentWeights);
        weightCanvas.updateCurrentWeights (currentWeights);
        weightCanvas.setTargetBentWeights ({});
    }
    else
    {
        // In momentary/transient mode, knobs & bridge shape the neon pink target bent curve!
        weightCanvas.setTargetBentWeights (bentTarget);
    }
}

void NNBendingAudioProcessorEditor::resetLayerWeights()
{
    if (currentBendingLayer.isEmpty()) return;
    
    audioProcessor.getBackend().reset_layer_weights (currentBendingLayer.toStdString());
    audioProcessor.clearLayerBending (currentBendingLayer.toStdString());
    
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    heatSlider.removeListener (this);
    memorySlider.removeListener (this);
    bridgeCombo.removeListener (this);
    bridgeDepthSlider.removeListener (this);

    scaleSlider.setValue (1.0, juce::dontSendNotification);
    offsetSlider.setValue (0.0, juce::dontSendNotification);
    heatSlider.setValue (0.0, juce::dontSendNotification);
    memorySlider.setValue (0.8, juce::dontSendNotification);
    bridgeCombo.setSelectedId (1, juce::dontSendNotification);
    bridgeDepthSlider.setValue (0.0, juce::dontSendNotification);
    freezeButton.setToggleState (false, juce::dontSendNotification);
    freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);

    if (auto* p = audioProcessor.getScaleParam())  p->setValueNotifyingHost (p->convertTo0to1 (1.0f));
    if (auto* p = audioProcessor.getOffsetParam()) p->setValueNotifyingHost (p->convertTo0to1 (0.0f));
    if (auto* p = audioProcessor.getHeatParam())   p->setValueNotifyingHost (p->convertTo0to1 (0.0f));
    if (auto* p = audioProcessor.getMemoryParam()) p->setValueNotifyingHost (p->convertTo0to1 (0.8f));

    lastKnownScale = 1.0f;
    lastKnownOffset = 0.0f;
    lastKnownHeat = 0.0f;
    lastKnownMemory = 0.8f;

    lastKnobScale = 1.0;
    lastKnobOffset = 0.0;

    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    heatSlider.addListener (this);
    memorySlider.addListener (this);
    bridgeCombo.addListener (this);
    bridgeDepthSlider.addListener (this);

    originalWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
    currentWeights = originalWeights;
    baseDrawnWeights = originalWeights;

    weightCanvas.setWeights (originalWeights, currentWeights);
    weightCanvas.setTargetBentWeights ({});
    weightCanvas.setBridgeWeights ({}, {}, 0.0f, audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Continuous);
}

void NNBendingAudioProcessorEditor::resetAllLayerWeights()
{
    audioProcessor.getBackend().reset_all_layer_weights();
    audioProcessor.clearAllLayerBending();
    
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    heatSlider.removeListener (this);
    memorySlider.removeListener (this);
    bridgeCombo.removeListener (this);
    bridgeDepthSlider.removeListener (this);

    scaleSlider.setValue (1.0, juce::dontSendNotification);
    offsetSlider.setValue (0.0, juce::dontSendNotification);
    heatSlider.setValue (0.0, juce::dontSendNotification);
    memorySlider.setValue (0.8, juce::dontSendNotification);
    bridgeCombo.setSelectedId (1, juce::dontSendNotification);
    bridgeDepthSlider.setValue (0.0, juce::dontSendNotification);
    freezeButton.setToggleState (false, juce::dontSendNotification);
    freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);

    if (auto* p = audioProcessor.getScaleParam())  p->setValueNotifyingHost (p->convertTo0to1 (1.0f));
    if (auto* p = audioProcessor.getOffsetParam()) p->setValueNotifyingHost (p->convertTo0to1 (0.0f));
    if (auto* p = audioProcessor.getHeatParam())   p->setValueNotifyingHost (p->convertTo0to1 (0.0f));
    if (auto* p = audioProcessor.getMemoryParam()) p->setValueNotifyingHost (p->convertTo0to1 (0.8f));

    lastKnownScale = 1.0f;
    lastKnownOffset = 0.0f;
    lastKnownHeat = 0.0f;
    lastKnownMemory = 0.8f;

    lastKnobScale = 1.0;
    lastKnobOffset = 0.0;

    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    heatSlider.addListener (this);
    memorySlider.addListener (this);
    bridgeCombo.addListener (this);
    bridgeDepthSlider.addListener (this);

    if (currentBendingLayer.isNotEmpty())
    {
        originalWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
        currentWeights = originalWeights;
        baseDrawnWeights = originalWeights;
        weightCanvas.setWeights (originalWeights, currentWeights);
        weightCanvas.setTargetBentWeights ({});
        weightCanvas.setBridgeWeights ({}, {}, 0.0f, audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Continuous);
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
