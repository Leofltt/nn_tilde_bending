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
    enum class SideTab { Mutate, Harmonics, Bridge };

    struct FlatTabLookAndFeel : public juce::LookAndFeel_V4
    {
        void drawButtonBackground (juce::Graphics& g, juce::Button& button,
                                    const juce::Colour& backgroundColour,
                                    bool shouldDrawButtonAsHighlighted,
                                    bool shouldDrawButtonAsDown) override
        {
            auto bounds = button.getLocalBounds().toFloat();
            g.setColour (backgroundColour);
            g.fillRect (bounds);

            if (shouldDrawButtonAsDown)
            {
                g.setColour (juce::Colours::white.withAlpha (0.12f));
                g.fillRect (bounds);
            }
            else if (shouldDrawButtonAsHighlighted)
            {
                g.setColour (juce::Colours::white.withAlpha (0.06f));
                g.fillRect (bounds);
            }

            // Crisp 1px outline
            g.setColour (backgroundColour.brighter (0.18f).withAlpha (0.45f));
            g.drawRect (bounds, 1.0f);
        }
    };

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
    void setSideTab (SideTab newTab);

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

    // Category Filter for Trace Isolation
    juce::Label categoryLabel { {}, "Trace:" };
    juce::ComboBox categoryCombo;

    // Safety Sentry & Blown Fuse UI
    juce::TextButton fuseButton { "FUSE: OK" };

    // Momentary Short-Circuit Glitch UI
    juce::TextButton shortCircuitButton { "SHORT [!]" };
    juce::Label triggerModeLabel { {}, "Trigger:" };
    juce::ComboBox triggerModeCombo;

    // Right Side Tab System (Weights View)
    SideTab currentSideTab { SideTab::Mutate };
    juce::TextButton tabMutateButton { "Mutate" };
    juce::TextButton tabHarmonicsButton { "Harmonics" };
    juce::TextButton tabBridgeButton { "Bridge" };
    FlatTabLookAndFeel flatTabLf;

    // Tab 1: Mutate Controls (Modern LinearBar Sliders + Enable Toggles)
    juce::ToggleButton mutateEnableToggle { "Enable Transform" };
    juce::Label scaleLabel { {}, "Scale" };
    juce::Slider scaleSlider;
    juce::Label offsetLabel { {}, "Offset" };
    juce::Slider offsetSlider;

    juce::ToggleButton driftEnableToggle { "Enable Thermal Drift" };
    juce::Label heatLabel { {}, "Heat" };
    juce::Slider heatSlider;
    juce::Label memoryLabel { {}, "Memory" };
    juce::Slider memorySlider;
    juce::Label driftModeLabel { {}, "Drift Algorithm:" };
    juce::ComboBox driftModeCombo;
    juce::ToggleButton freezeButton { "Freeze Drift" };

    // Tab 2: Harmonics Controls
    juce::ToggleButton harmonicEnableToggle { "Enable Harmonics" };
    juce::Label harmonicFreqLabel { {}, "Frequency" };
    juce::Slider harmonicFreqSlider;
    juce::Label harmonicPartialsLabel { {}, "Partials" };
    juce::Slider harmonicPartialsSlider;
    juce::Label harmonicMorphLabel { {}, "Morph" };
    juce::Slider harmonicMorphSlider;
    juce::Label harmonicDepthLabel { {}, "Gain" };
    juce::Slider harmonicDepthSlider;
    juce::Label harmonicModeLabel { {}, "Combine Mode:" };
    juce::ComboBox harmonicModeCombo;
    juce::TextButton harmonicApplyButton { "Stamp Wave" };

    // Tab 3: Trace Bridging (Cross-Talk)
    juce::ToggleButton bridgeEnableToggle { "Enable Cross-Talk" };
    juce::Label bridgeLabel { {}, "Bridge Wire Source:" };
    juce::ComboBox bridgeCombo;
    juce::Label bridgeDepthLabel { {}, "Cross-Talk Depth" };
    juce::Slider bridgeDepthSlider;

    // Tab 4: Latent Hook Controls
    juce::ToggleButton latentEnableButton { "Enable Latent Hook" };
    juce::Label latentDepthLabel { {}, "Latent Depth" };
    juce::Slider latentDepthSlider;
    juce::Label latentModeLabel { {}, "Orbit Mode:" };
    juce::ComboBox latentModeCombo;
    juce::Label latentSlewLabel { {}, "Slew Speed" };
    juce::Slider latentSlewSlider;

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
