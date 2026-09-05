#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"
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
    void selectLayer(const juce::String& layerName);
    void applyWeightBending();
    void resetWeightBending();

    NNBendingAudioProcessor& audioProcessor;

    // GUI Components
    juce::Label titleLabel;
    juce::TextButton loadButton { "Load Local Model (.ts)" };
    juce::Label statusLabel;
    
    juce::Label methodLabel { {}, "Method:" };
    juce::ComboBox methodCombo;
    
    juce::Label bufferLabel { {}, "Buffer Size:" };
    juce::ComboBox bufferCombo;

    // Bending Section
    juce::GroupComponent bendingGroup { "Bending", "Weight Parameter Bending" };
    juce::Label layerLabel { {}, "Select Layer:" };
    juce::ComboBox layerCombo;
    
    juce::Label scaleLabel { {}, "Weight Scale (x)" };
    juce::Slider scaleSlider;
    
    juce::Label offsetLabel { {}, "Weight Offset (+)" };
    juce::Slider offsetSlider;
    
    juce::TextButton resetButton { "Reset Layer Weights" };
    juce::Label infoBendingLabel;

    // File Chooser
    std::unique_ptr<juce::FileChooser> fileChooser;

    // Cached weights for the currently selected bending layer
    juce::String currentBendingLayer;
    std::vector<float> originalWeights;
    bool isBendingActive { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NNBendingAudioProcessorEditor)
};
