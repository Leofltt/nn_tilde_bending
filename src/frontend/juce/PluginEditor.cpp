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

    // Right-Side Tab Switcher Buttons (Weights View)
    tabMutateButton.addListener (this);
    tabHarmonicsButton.addListener (this);
    tabBridgeButton.addListener (this);
    tabMutateButton.setLookAndFeel (&flatTabLf);
    tabHarmonicsButton.setLookAndFeel (&flatTabLf);
    tabBridgeButton.setLookAndFeel (&flatTabLf);
    addAndMakeVisible (tabMutateButton);
    addAndMakeVisible (tabHarmonicsButton);
    addAndMakeVisible (tabBridgeButton);

    // =========================================================================
    // Tab 1: Mutate Controls (Modern LinearBar Sliders + Enable Toggles)
    // =========================================================================
    mutateEnableToggle.setToggleState (true, juce::dontSendNotification);
    mutateEnableToggle.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffc084fc"));
    mutateEnableToggle.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffc084fc"));
    mutateEnableToggle.addListener (this);
    addAndMakeVisible (mutateEnableToggle);

    scaleLabel.setText ("Scale", juce::dontSendNotification);
    scaleLabel.setJustificationType (juce::Justification::centredLeft);
    scaleLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (scaleLabel);

    scaleSlider.setRange (0.0, 5.0, 0.01);
    scaleSlider.setValue (1.0);
    scaleSlider.setSliderStyle (juce::Slider::LinearBar);
    scaleSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    scaleSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff7c3aed")); // Vibrant violet
    scaleSlider.addListener (this);
    addAndMakeVisible (scaleSlider);

    offsetLabel.setText ("Offset", juce::dontSendNotification);
    offsetLabel.setJustificationType (juce::Justification::centredLeft);
    offsetLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (offsetLabel);

    offsetSlider.setRange (-2.0, 2.0, 0.001);
    offsetSlider.setValue (0.0);
    offsetSlider.setSliderStyle (juce::Slider::LinearBar);
    offsetSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    offsetSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff06b6d4")); // Bright cyan
    offsetSlider.addListener (this);
    addAndMakeVisible (offsetSlider);

    driftEnableToggle.setToggleState (true, juce::dontSendNotification);
    driftEnableToggle.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffff9933"));
    driftEnableToggle.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffff9933"));
    driftEnableToggle.addListener (this);
    addAndMakeVisible (driftEnableToggle);

    heatLabel.setText ("Heat (Jitter)", juce::dontSendNotification);
    heatLabel.setJustificationType (juce::Justification::centredLeft);
    heatLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (heatLabel);

    heatSlider.setRange (0.0, 1.0, 0.001);
    heatSlider.setValue (0.0);
    heatSlider.setSliderStyle (juce::Slider::LinearBar);
    heatSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    heatSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffea580c")); // Flame orange
    heatSlider.addListener (this);
    addAndMakeVisible (heatSlider);

    memoryLabel.setText ("Memory (Drag)", juce::dontSendNotification);
    memoryLabel.setJustificationType (juce::Justification::centredLeft);
    memoryLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (memoryLabel);

    memorySlider.setRange (0.0, 1.0, 0.001);
    memorySlider.setValue (0.8);
    memorySlider.setSliderStyle (juce::Slider::LinearBar);
    memorySlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    memorySlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff0284c7")); // Deep sky blue
    memorySlider.addListener (this);
    addAndMakeVisible (memorySlider);

    driftModeLabel.setText ("Drift Algorithm:", juce::dontSendNotification);
    driftModeLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (driftModeLabel);

    driftModeCombo.addItem ("Thermal OU", 1);
    driftModeCombo.addItem ("Random Walk", 2);
    driftModeCombo.addItem ("Glitch Noise", 3);
    driftModeCombo.setSelectedId (1, juce::dontSendNotification);
    driftModeCombo.addListener (this);
    addAndMakeVisible (driftModeCombo);

    freezeButton.setButtonText ("Freeze Drift");
    freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
    freezeButton.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffff9933"));
    freezeButton.addListener (this);
    addAndMakeVisible (freezeButton);

    // =========================================================================
    // Tab 2: Harmonics Controls (LinearBar Sliders + Enable Toggle + Mode)
    // =========================================================================
    harmonicEnableToggle.setToggleState (true, juce::dontSendNotification);
    harmonicEnableToggle.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#fffbbf24"));
    harmonicEnableToggle.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#fffbbf24"));
    harmonicEnableToggle.addListener (this);
    addAndMakeVisible (harmonicEnableToggle);

    harmonicFreqLabel.setText ("Frequency", juce::dontSendNotification);
    harmonicFreqLabel.setJustificationType (juce::Justification::centredLeft);
    harmonicFreqLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (harmonicFreqLabel);

    harmonicFreqSlider.setRange (0.5, 32.0, 0.1);
    harmonicFreqSlider.setValue (2.0);
    harmonicFreqSlider.setSliderStyle (juce::Slider::LinearBar);
    harmonicFreqSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    harmonicFreqSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd97706")); // Warm amber
    harmonicFreqSlider.addListener (this);
    addAndMakeVisible (harmonicFreqSlider);

    harmonicPartialsLabel.setText ("Partials", juce::dontSendNotification);
    harmonicPartialsLabel.setJustificationType (juce::Justification::centredLeft);
    harmonicPartialsLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (harmonicPartialsLabel);

    harmonicPartialsSlider.setRange (1.0, 8.0, 1.0);
    harmonicPartialsSlider.setValue (1.0);
    harmonicPartialsSlider.setSliderStyle (juce::Slider::LinearBar);
    harmonicPartialsSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    harmonicPartialsSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd97706"));
    harmonicPartialsSlider.addListener (this);
    addAndMakeVisible (harmonicPartialsSlider);

    harmonicMorphLabel.setText ("Morph Phase", juce::dontSendNotification);
    harmonicMorphLabel.setJustificationType (juce::Justification::centredLeft);
    harmonicMorphLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (harmonicMorphLabel);

    harmonicMorphSlider.setRange (0.0, 6.2831853, 0.01);
    harmonicMorphSlider.setValue (0.0);
    harmonicMorphSlider.setSliderStyle (juce::Slider::LinearBar);
    harmonicMorphSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    harmonicMorphSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd97706"));
    harmonicMorphSlider.addListener (this);
    addAndMakeVisible (harmonicMorphSlider);

    harmonicDepthLabel.setText ("Wave Gain", juce::dontSendNotification);
    harmonicDepthLabel.setJustificationType (juce::Justification::centredLeft);
    harmonicDepthLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (harmonicDepthLabel);

    harmonicDepthSlider.setRange (0.0, 2.0, 0.01);
    harmonicDepthSlider.setValue (0.5);
    harmonicDepthSlider.setSliderStyle (juce::Slider::LinearBar);
    harmonicDepthSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    harmonicDepthSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd97706"));
    harmonicDepthSlider.addListener (this);
    addAndMakeVisible (harmonicDepthSlider);

    harmonicModeLabel.setText ("Combine Mode:", juce::dontSendNotification);
    harmonicModeLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (harmonicModeLabel);

    harmonicModeCombo.addItem ("Add (+)", 1);
    harmonicModeCombo.addItem ("Ring (*)", 2);
    harmonicModeCombo.addItem ("Replace", 3);
    harmonicModeCombo.setSelectedId (1, juce::dontSendNotification);
    harmonicModeCombo.addListener (this);
    addAndMakeVisible (harmonicModeCombo);

    harmonicApplyButton.setButtonText ("Stamp Wave");
    harmonicApplyButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a3712"));
    harmonicApplyButton.setColour (juce::TextButton::textColourOffId, juce::Colour::fromString ("#fffde047"));
    harmonicApplyButton.addListener (this);
    addAndMakeVisible (harmonicApplyButton);

    // =========================================================================
    // Tab 3: Trace Bridging (Cross-Talk) Controls
    // =========================================================================
    bridgeEnableToggle.setToggleState (true, juce::dontSendNotification);
    bridgeEnableToggle.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffc26a38"));
    bridgeEnableToggle.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffc26a38"));
    bridgeEnableToggle.addListener (this);
    addAndMakeVisible (bridgeEnableToggle);

    bridgeLabel.setText ("Bridge Wire Source:", juce::dontSendNotification);
    bridgeLabel.setJustificationType (juce::Justification::centredLeft);
    bridgeLabel.setColour (juce::Label::textColourId, juce::Colour::fromString ("#ffc26a38")); // Warm copper
    addAndMakeVisible (bridgeLabel);

    bridgeCombo.addListener (this);
    addAndMakeVisible (bridgeCombo);

    bridgeDepthLabel.setText ("Cross-Talk Depth", juce::dontSendNotification);
    bridgeDepthLabel.setJustificationType (juce::Justification::centredLeft);
    bridgeDepthLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (bridgeDepthLabel);

    bridgeDepthSlider.setRange (0.0, 1.0, 0.01);
    bridgeDepthSlider.setValue (0.0);
    bridgeDepthSlider.setSliderStyle (juce::Slider::LinearBar);
    bridgeDepthSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    bridgeDepthSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd99b26")); // Warm golden ochre
    bridgeDepthSlider.addListener (this);
    addAndMakeVisible (bridgeDepthSlider);

    // =========================================================================
    // Tab 4: Latent Hook Parameter Controls
    // =========================================================================
    latentEnableButton.setButtonText ("Enable Latent Hook");
    latentEnableButton.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffd946ef"));
    latentEnableButton.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffd946ef"));
    latentEnableButton.addListener (this);
    addAndMakeVisible (latentEnableButton);

    latentDepthLabel.setText ("Latent Depth", juce::dontSendNotification);
    latentDepthLabel.setJustificationType (juce::Justification::centredLeft);
    latentDepthLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (latentDepthLabel);

    latentDepthSlider.setRange (0.0, 2.0, 0.01);
    latentDepthSlider.setValue (audioProcessor.getLatentDepth());
    latentDepthSlider.setSliderStyle (juce::Slider::LinearBar);
    latentDepthSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    latentDepthSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd946ef"));
    latentDepthSlider.addListener (this);
    addAndMakeVisible (latentDepthSlider);

    latentModeLabel.setText ("Orbit Mode:", juce::dontSendNotification);
    latentModeLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (latentModeLabel);

    latentModeCombo.addItem ("Fourier Orbit", 1);
    latentModeCombo.addItem ("Latent Slew", 2);
    latentModeCombo.addItem ("Quantize", 3);
    latentModeCombo.setSelectedId ((int)audioProcessor.getLatentMode() + 1, juce::dontSendNotification);
    latentModeCombo.addListener (this);
    addAndMakeVisible (latentModeCombo);

    latentSlewLabel.setText ("Slew Speed", juce::dontSendNotification);
    latentSlewLabel.setJustificationType (juce::Justification::centredLeft);
    latentSlewLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (latentSlewLabel);

    latentSlewSlider.setRange (0.001, 1.0, 0.001);
    latentSlewSlider.setValue (audioProcessor.getLatentSlew());
    latentSlewSlider.setSliderStyle (juce::Slider::LinearBar);
    latentSlewSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
    latentSlewSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff38bdf8"));
    latentSlewSlider.addListener (this);
    addAndMakeVisible (latentSlewSlider);

    // Initial Side Tab: Mutate
    setSideTab (SideTab::Mutate);

    // View Switcher Buttons: "weights" and "latent"
    viewWeightsButton.addListener (this);
    viewLatentButton.addListener (this);
    viewWeightsButton.setLookAndFeel (&flatTabLf);
    viewLatentButton.setLookAndFeel (&flatTabLf);
    addAndMakeVisible (viewWeightsButton);
    addAndMakeVisible (viewLatentButton);

    // Weight Canvas Display Mode Switcher (1D Curve vs 2D Matrix)
    displayCurveButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a1d72"));
    displayCurveButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
    displayCurveButton.addListener (this);
    displayCurveButton.setLookAndFeel (&flatTabLf);
    displayMatrixButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
    displayMatrixButton.setColour (juce::TextButton::textColourOffId, juce::Colours::darkgrey);
    displayMatrixButton.addListener (this);
    displayMatrixButton.setLookAndFeel (&flatTabLf);
    addAndMakeVisible (displayCurveButton);
    addAndMakeVisible (displayMatrixButton);

    // Latent Terrain Pad
    addChildComponent (latentPad);
    latentPad.onCoordsChanged = [this] (float x, float y)
    {
        audioProcessor.setLatentCoords (x, y);
        if (auto* px = audioProcessor.getLatentXParam())
            px->setValueNotifyingHost (px->convertTo0to1 (x));
        if (auto* py = audioProcessor.getLatentYParam())
            py->setValueNotifyingHost (py->convertTo0to1 (y));
    };

    // Initial View Mode: weights
    setViewMode (ViewMode::Weights);

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
    tabMutateButton.setLookAndFeel (nullptr);
    tabHarmonicsButton.setLookAndFeel (nullptr);
    tabBridgeButton.setLookAndFeel (nullptr);
    viewWeightsButton.setLookAndFeel (nullptr);
    viewLatentButton.setLookAndFeel (nullptr);
    displayCurveButton.setLookAndFeel (nullptr);
    displayMatrixButton.setLookAndFeel (nullptr);
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

    // Layer Selection & Action Row (View buttons + [1D][2D] + Category filter + Layer Combo + Reset buttons)
    auto layerRow = groupArea.removeFromTop (32);

    // View Switcher Buttons on far left: [weights] [latent]
    viewWeightsButton.setBounds (layerRow.removeFromLeft (68));
    layerRow.removeFromLeft (4);
    viewLatentButton.setBounds (layerRow.removeFromLeft (64));
    layerRow.removeFromLeft (8);

    // Display mode buttons [1D] [2D] directly adjacent to view buttons (generous 44px so text never clips to '...')
    displayCurveButton.setBounds (layerRow.removeFromLeft (44));
    layerRow.removeFromLeft (3);
    displayMatrixButton.setBounds (layerRow.removeFromLeft (44));
    layerRow.removeFromLeft (10);

    categoryLabel.setBounds (layerRow.removeFromLeft (40));
    categoryCombo.setBounds (layerRow.removeFromLeft (110));
    layerRow.removeFromLeft (8);

    layerLabel.setBounds (layerRow.removeFromLeft (40));
    
    resetAllButton.setBounds (layerRow.removeFromRight (125));
    layerRow.removeFromRight (6);
    resetLayerButton.setBounds (layerRow.removeFromRight (105));
    layerRow.removeFromRight (8);
    
    layerCombo.setBounds (layerRow); // Takes remaining center width

    groupArea.removeFromTop (8); // Spacer directly above canvas & side panel

    // Bottom info readout row
    auto bottomRow = groupArea.removeFromBottom (20);
    infoBendingLabel.setBounds (bottomRow);

    groupArea.removeFromBottom (6); // Spacer above readout

    // Main Bending Area: Center Canvas + Right Controls Deck
    auto controlsWidth = 240;
    auto sideArea = groupArea.removeFromRight (controlsWidth);
    groupArea.removeFromRight (12); // Gap between canvas and side deck

    weightCanvas.setBounds (groupArea);
    latentPad.setBounds (groupArea);

    // Weights View: 3-Tab Header (Mutate, Harmonics, Bridge) (Height: 26px)
    auto tabHeader = sideArea.removeFromTop (26);
    int tabW = (sideArea.getWidth() - 6) / 3;
    tabMutateButton.setBounds (tabHeader.removeFromLeft (tabW));
    tabHeader.removeFromLeft (3);
    tabHarmonicsButton.setBounds (tabHeader.removeFromLeft (tabW));
    tabHeader.removeFromLeft (3);
    tabBridgeButton.setBounds (tabHeader); // Takes remainder

    sideArea.removeFromTop (10); // Spacer under tab header

    // Layout Tab 1: Mutate Controls (LinearBar sliders + Toggles)
    {
        auto area1 = sideArea;
        mutateEnableToggle.setBounds (area1.removeFromTop (24));
        area1.removeFromTop (6);

        scaleLabel.setBounds (area1.removeFromTop (16));
        scaleSlider.setBounds (area1.removeFromTop (24));
        area1.removeFromTop (8);

        offsetLabel.setBounds (area1.removeFromTop (16));
        offsetSlider.setBounds (area1.removeFromTop (24));
        area1.removeFromTop (14);

        driftEnableToggle.setBounds (area1.removeFromTop (24));
        area1.removeFromTop (6);

        heatLabel.setBounds (area1.removeFromTop (16));
        heatSlider.setBounds (area1.removeFromTop (24));
        area1.removeFromTop (8);

        memoryLabel.setBounds (area1.removeFromTop (16));
        memorySlider.setBounds (area1.removeFromTop (24));
        area1.removeFromTop (8);

        driftModeLabel.setBounds (area1.removeFromTop (16));
        driftModeCombo.setBounds (area1.removeFromTop (24));
        area1.removeFromTop (10);

        freezeButton.setBounds (area1.removeFromTop (24));
    }

    // Layout Tab 2: Harmonics Controls
    {
        auto area2 = sideArea;
        harmonicEnableToggle.setBounds (area2.removeFromTop (24));
        area2.removeFromTop (6);

        harmonicFreqLabel.setBounds (area2.removeFromTop (16));
        harmonicFreqSlider.setBounds (area2.removeFromTop (24));
        area2.removeFromTop (8);

        harmonicPartialsLabel.setBounds (area2.removeFromTop (16));
        harmonicPartialsSlider.setBounds (area2.removeFromTop (24));
        area2.removeFromTop (8);

        harmonicMorphLabel.setBounds (area2.removeFromTop (16));
        harmonicMorphSlider.setBounds (area2.removeFromTop (24));
        area2.removeFromTop (8);

        harmonicDepthLabel.setBounds (area2.removeFromTop (16));
        harmonicDepthSlider.setBounds (area2.removeFromTop (24));
        area2.removeFromTop (12);

        harmonicModeLabel.setBounds (area2.removeFromTop (16));
        harmonicModeCombo.setBounds (area2.removeFromTop (24));
        area2.removeFromTop (14);

        harmonicApplyButton.setBounds (area2.removeFromTop (28));
    }

    // Layout Tab 3: Trace Bridging (Cross-Talk) Controls
    {
        auto area3 = sideArea;
        bridgeEnableToggle.setBounds (area3.removeFromTop (24));
        area3.removeFromTop (8);

        bridgeLabel.setBounds (area3.removeFromTop (16));
        bridgeCombo.setBounds (area3.removeFromTop (26));
        area3.removeFromTop (12);

        bridgeDepthLabel.setBounds (area3.removeFromTop (16));
        bridgeDepthSlider.setBounds (area3.removeFromTop (24));
    }

    // Layout Latent View Controls (Active in Latent view)
    {
        auto area4 = sideArea;
        latentEnableButton.setBounds (area4.removeFromTop (24));
        area4.removeFromTop (10);

        latentDepthLabel.setBounds (area4.removeFromTop (16));
        latentDepthSlider.setBounds (area4.removeFromTop (24));
        area4.removeFromTop (12);

        latentModeLabel.setBounds (area4.removeFromTop (16));
        latentModeCombo.setBounds (area4.removeFromTop (26));
        area4.removeFromTop (12);

        latentSlewLabel.setBounds (area4.removeFromTop (16));
        latentSlewSlider.setBounds (area4.removeFromTop (24));
    }
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
    else if (comboBoxThatHasChanged == &harmonicModeCombo)
    {
        updateHarmonicGhostPreview();
    }
    else if (comboBoxThatHasChanged == &latentModeCombo)
    {
        int modeIdx = latentModeCombo.getSelectedId() - 1;
        audioProcessor.setLatentMode ((NNBendingAudioProcessor::LatentMode)modeIdx);
    }
}

void NNBendingAudioProcessorEditor::sliderValueChanged (juce::Slider* slider)
{
    if (slider == &harmonicFreqSlider || slider == &harmonicPartialsSlider
        || slider == &harmonicMorphSlider || slider == &harmonicDepthSlider)
    {
        updateHarmonicGhostPreview();
    }
    else if (slider == &dryWetSlider)
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
    else if (slider == &latentDepthSlider)
    {
        float val = (float)latentDepthSlider.getValue();
        audioProcessor.setLatentDepth (val);
        if (auto* param = audioProcessor.getLatentDepthParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
    }
    else if (slider == &latentSlewSlider)
    {
        float val = (float)latentSlewSlider.getValue();
        audioProcessor.setLatentSlew (val);
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
    else if (button == &harmonicApplyButton)
    {
        applyHarmonicWeights();
    }
    else if (button == &viewWeightsButton)
    {
        setViewMode (ViewMode::Weights);
    }
    else if (button == &viewLatentButton)
    {
        setViewMode (ViewMode::Latent);
    }
    else if (button == &latentEnableButton)
    {
        bool en = latentEnableButton.getToggleState();
        audioProcessor.setLatentEnabled (en);
        latentPad.setEnabledState (en, audioProcessor.hasAutoencode());
    }
    else if (button == &tabMutateButton)
    {
        setSideTab (SideTab::Mutate);
    }
    else if (button == &tabHarmonicsButton)
    {
        setSideTab (SideTab::Harmonics);
    }
    else if (button == &tabBridgeButton)
    {
        setSideTab (SideTab::Bridge);
    }
    else if (button == &mutateEnableToggle)
    {
        applyKnobBending();
    }
    else if (button == &driftEnableToggle)
    {
        // When drift is disabled, freeze live thermal jitter offsets
        if (!driftEnableToggle.getToggleState() && currentBendingLayer.isNotEmpty())
        {
            auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());
            state.driftOffsets.assign (state.driftOffsets.size(), 0.0f);
            audioProcessor.setLayerState (currentBendingLayer.toStdString(), state);
        }
    }
    else if (button == &bridgeEnableToggle)
    {
        bool en = bridgeEnableToggle.getToggleState();
        weightCanvas.setShowBridgeVisuals (en);
        applyKnobBending();
    }
    else if (button == &harmonicEnableToggle)
    {
        bool en = harmonicEnableToggle.getToggleState();
        weightCanvas.setShowHarmonics (en);
        updateHarmonicGhostPreview();
    }
    else if (button == &displayCurveButton)
    {
        weightCanvas.setRenderMode (WeightBendingComponent::RenderMode::Curve);
        displayCurveButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a1d72"));
        displayCurveButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        displayMatrixButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        displayMatrixButton.setColour (juce::TextButton::textColourOffId, juce::Colours::darkgrey);
    }
    else if (button == &displayMatrixButton)
    {
        weightCanvas.setRenderMode (WeightBendingComponent::RenderMode::Matrix);
        displayMatrixButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a1d72"));
        displayMatrixButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        displayCurveButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        displayCurveButton.setColour (juce::TextButton::textColourOffId, juce::Colours::darkgrey);
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

    // Sync Latent XY and Depth parameters
    if (auto* p = audioProcessor.getLatentDepthParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - (float)latentDepthSlider.getValue()) > 0.005f && !latentDepthSlider.isMouseButtonDown())
        {
            latentDepthSlider.setValue (pVal, juce::dontSendNotification);
            audioProcessor.setLatentDepth (pVal);
        }
    }

    // Sync Latent Pad coordinates and enabled state
    if (!latentPad.isMouseButtonDownAnywhere())
    {
        latentPad.setCoords (audioProcessor.getLatentX(), audioProcessor.getLatentY());
    }
    latentPad.setEnabledState (audioProcessor.isLatentEnabled(), audioProcessor.hasAutoencode());
    if (latentEnableButton.getToggleState() != audioProcessor.isLatentEnabled())
    {
        latentEnableButton.setToggleState (audioProcessor.isLatentEnabled(), juce::dontSendNotification);
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
                bool bridgeEnabled = bridgeEnableToggle.getToggleState();
                if (bridgeEnabled && state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != currentBendingLayer.toStdString())
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
                weightCanvas.setShowBridgeVisuals (bridgeEnabled);
                weightCanvas.setBridgeWeights (srcTiled, bridgedBase, bridgeEnabled ? state.bridgeDepth : 0.0f, false);

                std::vector<float> targetBent(bridgedBase.size());
                bool mutateEnabled = mutateEnableToggle.getToggleState();
                float effScale = mutateEnabled ? state.scale : 1.0f;
                float effOffset = mutateEnabled ? state.offset : 0.0f;
                for (size_t i = 0; i < bridgedBase.size(); ++i)
                    targetBent[i] = bridgedBase[i] * effScale + effOffset;
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
        updateKnobContextLabels (NNBendingAudioProcessor::LayerCategory::Other);
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

    auto shape = audioProcessor.getBackend().get_layer_shape (layerName.toStdString());
    weightCanvas.setLayerShape (shape);
    weightCanvas.setWeights (originalWeights, currentWeights);

    // Compute and send bridge source & mix curves to canvas
    std::vector<float> bridgedBase = baseDrawnWeights;
    std::vector<float> srcTiled;
    bool bridgeEnabled = bridgeEnableToggle.getToggleState();
    if (bridgeEnabled && state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != layerName.toStdString())
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
    weightCanvas.setShowBridgeVisuals (bridgeEnabled);
    weightCanvas.setBridgeWeights (srcTiled, bridgedBase, bridgeEnabled ? state.bridgeDepth : 0.0f, isContinuous);

    // If in momentary/transient modes, compute target bent overlay
    if (!isContinuous)
    {
        std::vector<float> targetBent(bridgedBase.size());
        bool mutateEnabled = mutateEnableToggle.getToggleState();
        float effScale = mutateEnabled ? state.scale : 1.0f;
        float effOffset = mutateEnabled ? state.offset : 0.0f;
        for (size_t i = 0; i < bridgedBase.size(); ++i)
            targetBent[i] = bridgedBase[i] * effScale + effOffset;
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

    infoBendingLabel.setText ("Trace: [" + catName + "]  |  Layer: " + currentBendingLayer + "  |  " + juce::String (originalWeights.size()) + " params" + (audioProcessor.hasAutoencode() ? "  |  [Latent Hook Ready]" : ""), juce::dontSendNotification);

    // Update contextual knob labels (Gamma / Beta for Norm, Scale / Offset for other traces)
    updateKnobContextLabels (category);

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

    // Refresh harmonic preview for the newly selected layer
    updateHarmonicGhostPreview();
}

void NNBendingAudioProcessorEditor::applyKnobBending()
{
    if (currentBendingLayer.isEmpty()) return;
    if (baseDrawnWeights.empty())
        baseDrawnWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
    
    bool mutateEnabled = mutateEnableToggle.getToggleState();
    float scale = mutateEnabled ? (float)scaleSlider.getValue() : 1.0f;
    float offset = mutateEnabled ? (float)offsetSlider.getValue() : 0.0f;

    // Store per-layer scale and offset
    audioProcessor.setLayerScale (currentBendingLayer.toStdString(), scale);
    audioProcessor.setLayerOffset (currentBendingLayer.toStdString(), offset);

    // Fetch bridge state
    auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());
    std::vector<float> bridgedBase = baseDrawnWeights;
    std::vector<float> srcTiled;
    bool bridgeEnabled = bridgeEnableToggle.getToggleState();

    if (bridgeEnabled && state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != currentBendingLayer.toStdString())
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
    weightCanvas.setShowBridgeVisuals (bridgeEnabled);
    weightCanvas.setBridgeWeights (srcTiled, bridgedBase, bridgeEnabled ? state.bridgeDepth : 0.0f, isContinuous);

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
    updateHarmonicGhostPreview();
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
        updateHarmonicGhostPreview();
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

void NNBendingAudioProcessorEditor::updateKnobContextLabels (NNBendingAudioProcessor::LayerCategory category)
{
    if (category == NNBendingAudioProcessor::LayerCategory::Norm)
    {
        // Normalization / Dynamics context: Gamma (Gain) & Beta (Bias)
        scaleLabel.setText (juce::CharPointer_UTF8 ("Gamma (\xce\xb3)"), juce::dontSendNotification);
        scaleLabel.setColour (juce::Label::textColourId, juce::Colour::fromString ("#ffc084fc")); // Soft violet

        offsetLabel.setText (juce::CharPointer_UTF8 ("Beta (\xce\xb2)"), juce::dontSendNotification);
        offsetLabel.setColour (juce::Label::textColourId, juce::Colour::fromString ("#ff38bdf8")); // Sky blue
    }
    else
    {
        // Standard convolution / dense / bias context: Scale & Offset
        scaleLabel.setText ("Scale", juce::dontSendNotification);
        scaleLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);

        offsetLabel.setText ("Offset", juce::dontSendNotification);
        offsetLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    }
}

void NNBendingAudioProcessorEditor::updateHarmonicGhostPreview()
{
    if (!harmonicEnableToggle.getToggleState() || currentBendingLayer.isEmpty() || baseDrawnWeights.empty())
    {
        weightCanvas.setHarmonicGhostWeights ({}, false);
        return;
    }

    size_t N = baseDrawnWeights.size();
    std::vector<float> ghost (N);

    float f = (float)harmonicFreqSlider.getValue();
    int partials = (int)harmonicPartialsSlider.getValue();
    float morph = (float)harmonicMorphSlider.getValue();
    float gain = (float)harmonicDepthSlider.getValue();
    int mode = harmonicModeCombo.getSelectedId(); // 1 = Add, 2 = Ring, 3 = Replace

    // TwoPi constant
    const float twoPi = 6.28318530717958647692f;

    for (size_t i = 0; i < N; ++i)
    {
        float x = (N > 1) ? (float)i / (float)(N - 1) : 0.0f;
        float synthVal = 0.0f;

        for (int k = 1; k <= partials; ++k)
        {
            float harmonicAmp = 1.0f / std::sqrt ((float)k); // Soft 1/sqrt(k) harmonic decay
            synthVal += harmonicAmp * std::sin (twoPi * (float)k * f * x + morph);
        }
        synthVal *= gain;

        if (mode == 1) // Add
        {
            ghost[i] = baseDrawnWeights[i] + synthVal;
        }
        else if (mode == 2) // Ring (multiply)
        {
            ghost[i] = baseDrawnWeights[i] * (1.0f + synthVal);
        }
        else // Replace
        {
            ghost[i] = synthVal;
        }
    }

    weightCanvas.setHarmonicGhostWeights (ghost, true);
}

void NNBendingAudioProcessorEditor::applyHarmonicWeights()
{
    if (currentBendingLayer.isEmpty() || baseDrawnWeights.empty())
        return;

    size_t N = baseDrawnWeights.size();
    std::vector<float> stamped (N);

    float f = (float)harmonicFreqSlider.getValue();
    int partials = (int)harmonicPartialsSlider.getValue();
    float morph = (float)harmonicMorphSlider.getValue();
    float gain = (float)harmonicDepthSlider.getValue();
    int mode = harmonicModeCombo.getSelectedId();

    const float twoPi = 6.28318530717958647692f;

    for (size_t i = 0; i < N; ++i)
    {
        float x = (N > 1) ? (float)i / (float)(N - 1) : 0.0f;
        float synthVal = 0.0f;

        for (int k = 1; k <= partials; ++k)
        {
            float harmonicAmp = 1.0f / std::sqrt ((float)k);
            synthVal += harmonicAmp * std::sin (twoPi * (float)k * f * x + morph);
        }
        synthVal *= gain;

        if (mode == 1)
            stamped[i] = baseDrawnWeights[i] + synthVal;
        else if (mode == 2)
            stamped[i] = baseDrawnWeights[i] * (1.0f + synthVal);
        else
            stamped[i] = synthVal;
    }

    // Persist as new baseDrawnWeights
    baseDrawnWeights = stamped;
    audioProcessor.setLayerDrawnWeights (currentBendingLayer.toStdString(), baseDrawnWeights);

    // Hide ghost preview once stamped
    weightCanvas.setHarmonicGhostWeights ({}, false);

    // Apply via existing knob bending pipeline
    applyKnobBending();

    infoBendingLabel.setText ("Harmonic wave stamped to " + currentBendingLayer, juce::dontSendNotification);
}

void NNBendingAudioProcessorEditor::setSideTab (SideTab newTab)
{
    currentSideTab = newTab;

    bool isWeights = (currentViewMode == ViewMode::Weights);
    bool isMutate = isWeights && (currentSideTab == SideTab::Mutate);
    bool isHarmonics = isWeights && (currentSideTab == SideTab::Harmonics);
    bool isBridge = isWeights && (currentSideTab == SideTab::Bridge);

    juce::Colour activeBg = juce::Colour::fromString ("#ff4a1d72"); // Vivid violet
    juce::Colour inactiveBg = juce::Colour::fromString ("#ff1c1926"); // Dark charcoal
    juce::Colour activeText = juce::Colours::white;
    juce::Colour inactiveText = juce::Colours::darkgrey;

    tabMutateButton.setColour (juce::TextButton::buttonColourId, isMutate ? activeBg : inactiveBg);
    tabMutateButton.setColour (juce::TextButton::textColourOffId, isMutate ? activeText : inactiveText);

    tabHarmonicsButton.setColour (juce::TextButton::buttonColourId, isHarmonics ? activeBg : inactiveBg);
    tabHarmonicsButton.setColour (juce::TextButton::textColourOffId, isHarmonics ? activeText : inactiveText);

    tabBridgeButton.setColour (juce::TextButton::buttonColourId, isBridge ? activeBg : inactiveBg);
    tabBridgeButton.setColour (juce::TextButton::textColourOffId, isBridge ? activeText : inactiveText);

    // Tab 1: Mutate Visibility
    mutateEnableToggle.setVisible (isMutate);
    scaleLabel.setVisible (isMutate);
    scaleSlider.setVisible (isMutate);
    offsetLabel.setVisible (isMutate);
    offsetSlider.setVisible (isMutate);
    driftEnableToggle.setVisible (isMutate);
    heatLabel.setVisible (isMutate);
    heatSlider.setVisible (isMutate);
    memoryLabel.setVisible (isMutate);
    memorySlider.setVisible (isMutate);
    driftModeLabel.setVisible (isMutate);
    driftModeCombo.setVisible (isMutate);
    freezeButton.setVisible (isMutate);

    // Tab 2: Harmonics Visibility
    harmonicEnableToggle.setVisible (isHarmonics);
    harmonicFreqLabel.setVisible (isHarmonics);
    harmonicFreqSlider.setVisible (isHarmonics);
    harmonicPartialsLabel.setVisible (isHarmonics);
    harmonicPartialsSlider.setVisible (isHarmonics);
    harmonicMorphLabel.setVisible (isHarmonics);
    harmonicMorphSlider.setVisible (isHarmonics);
    harmonicDepthLabel.setVisible (isHarmonics);
    harmonicDepthSlider.setVisible (isHarmonics);
    harmonicModeLabel.setVisible (isHarmonics);
    harmonicModeCombo.setVisible (isHarmonics);
    harmonicApplyButton.setVisible (isHarmonics);

    // Tab 3: Bridge Visibility
    bridgeEnableToggle.setVisible (isBridge);
    bridgeLabel.setVisible (isBridge);
    bridgeCombo.setVisible (isBridge);
    bridgeDepthLabel.setVisible (isBridge);
    bridgeDepthSlider.setVisible (isBridge);

    repaint();
}

void NNBendingAudioProcessorEditor::setViewMode (ViewMode newMode)
{
    currentViewMode = newMode;

    bool isWeights = (currentViewMode == ViewMode::Weights);
    bool isLatent = (currentViewMode == ViewMode::Latent);

    // Style the toggle buttons
    juce::Colour activeBg = juce::Colour::fromString ("#ff4a1d72"); // Vivid violet
    juce::Colour inactiveBg = juce::Colour::fromString ("#ff1c1926"); // Dark charcoal
    juce::Colour activeText = juce::Colours::white;
    juce::Colour inactiveText = juce::Colours::darkgrey;

    viewWeightsButton.setColour (juce::TextButton::buttonColourId, isWeights ? activeBg : inactiveBg);
    viewWeightsButton.setColour (juce::TextButton::textColourOffId, isWeights ? activeText : inactiveText);

    viewLatentButton.setColour (juce::TextButton::buttonColourId, isLatent ? activeBg : inactiveBg);
    viewLatentButton.setColour (juce::TextButton::textColourOffId, isLatent ? activeText : inactiveText);

    // Switch main canvas visibility
    weightCanvas.setVisible (isWeights);
    latentPad.setVisible (isLatent);

    // Toggle weight-domain header controls visibility
    categoryLabel.setVisible (isWeights);
    categoryCombo.setVisible (isWeights);
    layerLabel.setVisible (isWeights);
    layerCombo.setVisible (isWeights);
    resetLayerButton.setVisible (isWeights);
    resetAllButton.setVisible (isWeights);
    displayCurveButton.setVisible (isWeights);
    displayMatrixButton.setVisible (isWeights);

    // Toggle weight-domain side tabs visibility
    tabMutateButton.setVisible (isWeights);
    tabHarmonicsButton.setVisible (isWeights);
    tabBridgeButton.setVisible (isWeights);

    // Toggle latent-domain controls visibility on right side
    latentEnableButton.setVisible (isLatent);
    latentDepthLabel.setVisible (isLatent);
    latentDepthSlider.setVisible (isLatent);
    latentModeLabel.setVisible (isLatent);
    latentModeCombo.setVisible (isLatent);
    latentSlewLabel.setVisible (isLatent);
    latentSlewSlider.setVisible (isLatent);

    // Refresh active weights tab visibility
    setSideTab (currentSideTab);

    // Update bottom readout status
    if (isLatent)
    {
        if (audioProcessor.hasAutoencode())
            infoBendingLabel.setText ("Latent Topographic Vector Pad: Drag cursor to modulate autoencoder bottleneck.", juce::dontSendNotification);
        else
            infoBendingLabel.setText ("Latent View: Standby. Loaded model does not expose paired encode + decode methods.", juce::dontSendNotification);
    }
    else
    {
        infoBendingLabel.setText (currentBendingLayer.isNotEmpty() ? ("Layer: " + currentBendingLayer) : "Select a layer to bend weights.", juce::dontSendNotification);
    }

    repaint();
}

