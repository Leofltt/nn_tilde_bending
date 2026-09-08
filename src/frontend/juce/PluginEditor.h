#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "WeightBendingComponent.h"
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

private:
    void timerCallback() override;
    void updateModelInfo();
    void selectLayer (const juce::String& layerName);
    void applyKnobBending();
    void resetLayerWeights();
    void resetAllLayerWeights();
    void saveModelToFile();

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

    // Bending Section
    juce::GroupComponent bendingGroup { "Bending", "Interactive Weight Bending" };
    
    juce::Label layerLabel { {}, "Layer:" };
    juce::ComboBox layerCombo;
    
    juce::TextButton resetLayerButton { "Reset Layer" };
    juce::TextButton resetAllButton { "Reset All Layers" };

    // Interactive canvas
    WeightBendingComponent weightCanvas;

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NNBendingAudioProcessorEditor)
};
