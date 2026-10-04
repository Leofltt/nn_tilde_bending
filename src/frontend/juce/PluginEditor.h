#pragma once
#include "PluginProcessor.h"
#include <JuceHeader.h>
#include "WeightBendingComponent.h"
#include "LatentTerrainComponent.h"
#include <vector>

class NNBendingAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                       public juce::ComboBox::Listener,
                                       public juce::Slider::Listener,
                                       public juce::Button::Listener,
                                       private juce::Timer
{
public:
    NNBendingAudioProcessorEditor (NNBendingAudioProcessor&);
    ~NNBendingAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

    void comboBoxChanged (juce::ComboBox* comboBoxThatHasChanged) override;
    void sliderValueChanged (juce::Slider* slider) override;
    void buttonClicked (juce::Button* button) override;

    enum class ViewMode { Weights, Latent };

private:
    void timerCallback() override;
    void updateModelInfo();
    void selectLayer (const juce::String& layerName);
    void applyKnobBending();
    void resetLayerWeights();
    void resetAllLayerWeights();
    void saveModelToFile();
    void updateKnobContextLabels (NNBendingAudioProcessor::LayerCategory category);
    void setViewMode (ViewMode newMode);

    NNBendingAudioProcessor& audioProcessor;

    // Header Components
    juce::Label titleLabel;
    juce::TextButton loadButton { "Load Model (.ts)" };
    juce::Label statusLabel;
    
    juce::Label methodLabel { {}, "Mode:" };
    juce::ComboBox methodCombo;
    
    juce::Label bufferLabel { {}, "Buffer:" };
    juce::Label bufferStatusLabel;

    // Output Mix Control
    juce::Label dryWetLabel { {}, "Dry/Wet:" };
    juce::Slider dryWetSlider;

    juce::TextButton saveModelButton { "Save Model (.ts)" };

    // View Switcher: "weights" and "latent"
    ViewMode currentViewMode { ViewMode::Weights };
    juce::TextButton viewWeightsButton { "weights" };
    juce::TextButton viewLatentButton { "latent" };

    // Bending Section
    juce::GroupComponent bendingGroup { "Bending", "Interactive Weight Bending" };
    
    juce::Label layerLabel { {}, "Layer:" };
    juce::ComboBox layerCombo;
    
    juce::TextButton resetLayerButton { "Reset Layer" };
    juce::TextButton resetAllButton { "Reset All Layers" };

    // Interactive canvases
    WeightBendingComponent weightCanvas;
    LatentTerrainComponent latentPad;

    // Weight canvas display mode toggle: "curve" vs "matrix"
    juce::TextButton displayCurveButton { "1D" };
    juce::TextButton displayMatrixButton { "2D" };

    // Latent Hook Controls (visible in latent view)
    juce::Label latentDepthLabel { {}, "Latent Depth" };
    juce::Slider latentDepthSlider;
    juce::Label latentModeLabel { {}, "Orbit Mode:" };
    juce::ComboBox latentModeCombo;
    juce::Label latentSlewLabel { {}, "Slew Speed" };
    juce::Slider latentSlewSlider;
    juce::ToggleButton latentEnableButton { "Enable Latent Hook" };

    // Category Filter for Trace Isolation
    juce::Label categoryLabel { {}, "Trace:" };
    juce::ComboBox categoryCombo;

    // Safety Sentry & Blown Fuse UI
    juce::TextButton fuseButton { "FUSE: OK" };

    // Momentary Short-Circuit Glitch UI
    juce::TextButton shortCircuitButton { "SHORT [!]" };
    juce::Label triggerModeLabel { {}, "Trigger:" };
    juce::ComboBox triggerModeCombo;

    // Side Controls
    juce::Label scaleLabel { {}, "Scale" };
    juce::Slider scaleSlider;
    
    juce::Label offsetLabel { {}, "Offset" };
    juce::Slider offsetSlider;

    juce::Label heatLabel { {}, "Heat" };
    juce::Slider heatSlider;

    juce::Label memoryLabel { {}, "Memory" };
    juce::Slider memorySlider;

    juce::ComboBox driftModeCombo;
    juce::ToggleButton freezeButton { "Freeze" };
    
    // Trace Bridging (Cross-Talk)
    juce::Label bridgeLabel { {}, "Bridge Wire:" };
    juce::ComboBox bridgeCombo;
    juce::Label bridgeDepthLabel { {}, "Cross-Talk" };
    juce::Slider bridgeDepthSlider;

    // Harmonic Weight Synthesizer (Fourier Generator)
    juce::Label harmonicTitleLabel { {}, "Harmonics:" };
    juce::Label harmonicFreqLabel { {}, "Freq" };
    juce::Slider harmonicFreqSlider;
    juce::Label harmonicPartialsLabel { {}, "Partials" };
    juce::Slider harmonicPartialsSlider;
    juce::Label harmonicMorphLabel { {}, "Morph" };
    juce::Slider harmonicMorphSlider;
    juce::Label harmonicDepthLabel { {}, "Gain" };
    juce::Slider harmonicDepthSlider;
    juce::ComboBox harmonicModeCombo;
    juce::TextButton harmonicApplyButton { "Stamp" };

    void updateHarmonicGhostPreview();
    void applyHarmonicWeights();

    juce::Label infoBendingLabel;

    // File Chooser
    std::unique_ptr<juce::FileChooser> fileChooser;

    // Active layer cache
    juce::String currentBendingLayer;
    std::vector<float> originalWeights;
    std::vector<float> currentWeights;
    std::vector<float> baseDrawnWeights;
    double lastKnobScale { 1.0 };
    double lastKnobOffset { 0.0 };
    float lastKnownScale { 1.0f };
    float lastKnownOffset { 0.0f };
    float lastKnownHeat { 0.0f };
    float lastKnownMemory { 0.8f };
    float lastKnownDryWet { 1.0f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NNBendingAudioProcessorEditor)
};
