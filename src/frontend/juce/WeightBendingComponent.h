#pragma once
#include <JuceHeader.h>
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>

class WeightBendingComponent : public juce::Component
{
public:
    WeightBendingComponent()
    {
        setOpaque(true);
    }

    ~WeightBendingComponent() override = default;

    std::function<void(const std::vector<float>&)> onWeightsModified;

    void setWeights(const std::vector<float>& original, const std::vector<float>& current)
    {
        m_originalWeights = original;
        m_currentWeights = current;
        recalculateStats();
        repaint();
    }

    void updateCurrentWeights(const std::vector<float>& current)
    {
        m_currentWeights = current;
        recalculateStats();
        repaint();
    }

    const std::vector<float>& getCurrentWeights() const { return m_currentWeights; }
    const std::vector<float>& getOriginalWeights() const { return m_originalWeights; }

    void paint(juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();

        // Canvas dark background
        g.setColour(juce::Colour::fromString("#ff0e0d13"));
        g.fillRect(bounds);

        // Subtle border
        g.setColour(juce::Colour::fromString("#ff3a344d"));
        g.drawRect(bounds, 1.0f);

        if (m_currentWeights.empty())
        {
            g.setColour(juce::Colours::grey.withAlpha(0.6f));
            g.setFont(juce::FontOptions(14.0f));
            g.drawText("No layer selected or layer has 0 parameters.",
                       getLocalBounds(), juce::Justification::centred, true);
            return;
        }

        // Draw grid and zero line
        float h = bounds.getHeight();
        float w = bounds.getWidth();
        float zeroY = valueToY(0.0f, h);

        g.setColour(juce::Colour::fromString("#ff242032"));
        // Draw horizontal grid lines
        for (float v : { -2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f })
        {
            float y = valueToY(v, h);
            if (y >= 0 && y <= h)
                g.drawHorizontalLine((int)y, 0.0f, w);
        }

        // Center zero line
        g.setColour(juce::Colour::fromString("#ff4c4363"));
        g.drawHorizontalLine((int)zeroY, 0.0f, w);

        // Render original weights curve (faint cyan/grey reference)
        if (!m_originalWeights.empty())
        {
            juce::Path origPath;
            buildPathForWeights(origPath, m_originalWeights, w, h);
            g.setColour(juce::Colours::cyan.withAlpha(0.25f));
            g.strokePath(origPath, juce::PathStrokeType(1.0f));
        }

        // Render current (bent) weights
        juce::Path bentPath;
        buildPathForWeights(bentPath, m_currentWeights, w, h);

        // Gradient fill under the curve
        juce::Path fillPath = bentPath;
        fillPath.lineTo(w, zeroY);
        fillPath.lineTo(0.0f, zeroY);
        fillPath.closeSubPath();

        juce::ColourGradient grad(
            juce::Colour::fromString("#88c084fc"), 0, 0,
            juce::Colour::fromString("#119333ea"), 0, h, false
        );
        g.setGradientFill(grad);
        g.fillPath(fillPath);

        // Vibrant neon purple/violet stroke
        g.setColour(juce::Colour::fromString("#ffc084fc"));
        g.strokePath(bentPath, juce::PathStrokeType(1.8f));

        // Stats readout on bottom right
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.setFont(juce::FontOptions(11.0f));
        juce::String stats = juce::String::formatted(
            "Weights: %d   Min: %.3f   Max: %.3f   Mean: %.4f",
            (int)m_currentWeights.size(), m_minVal, m_maxVal, m_meanVal
        );
        g.drawText(stats, getLocalBounds().reduced(10, 6), juce::Justification::bottomRight, true);

        // Label on top left
        g.setColour(juce::Colours::lightgrey.withAlpha(0.5f));
        g.drawText("Drag / Draw to bend weights  |  Cyan: Original  |  Violet: Bent",
                   getLocalBounds().reduced(10, 6), juce::Justification::topLeft, true);
    }

    void mouseDown(const juce::MouseEvent& e) override
    {
        applyDrawingAt(e);
    }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        applyDrawingAt(e);
    }

private:
    std::vector<float> m_originalWeights;
    std::vector<float> m_currentWeights;

    float m_minVal { -1.0f };
    float m_maxVal { 1.0f };
    float m_meanVal { 0.0f };
    float m_displayRange { 1.5f }; // Visual +/- ceiling

    void recalculateStats()
    {
        if (m_currentWeights.empty())
        {
            m_minVal = -1.0f;
            m_maxVal = 1.0f;
            m_meanVal = 0.0f;
            m_displayRange = 1.5f;
            return;
        }

        m_minVal = *std::min_element(m_currentWeights.begin(), m_currentWeights.end());
        m_maxVal = *std::max_element(m_currentWeights.begin(), m_currentWeights.end());
        
        double sum = 0.0;
        for (float w : m_currentWeights)
            sum += w;
        m_meanVal = (float)(sum / m_currentWeights.size());

        float maxAbs = std::max(std::abs(m_minVal), std::abs(m_maxVal));
        m_displayRange = std::max(1.0f, maxAbs * 1.2f);
    }

    float valueToY(float val, float height) const
    {
        float norm = (val + m_displayRange) / (2.0f * m_displayRange);
        norm = juce::jlimit(0.0f, 1.0f, norm);
        return (1.0f - norm) * (height - 16.0f) + 8.0f;
    }

    float yToValue(float y, float height) const
    {
        float norm = 1.0f - ((y - 8.0f) / (height - 16.0f));
        norm = juce::jlimit(0.0f, 1.0f, norm);
        return norm * (2.0f * m_displayRange) - m_displayRange;
    }

    void buildPathForWeights(juce::Path& p, const std::vector<float>& weights, float w, float h)
    {
        if (weights.empty() || w <= 0.0f) return;

        int numPoints = std::min((int)weights.size(), (int)w);
        if (numPoints <= 0) return;

        p.preallocateSpace(numPoints * 2);
        bool first = true;

        for (int x = 0; x < numPoints; ++x)
        {
            float normX = (float)x / (float)(numPoints - 1);
            size_t idx = (size_t)(normX * (weights.size() - 1));
            float y = valueToY(weights[idx], h);
            float screenX = normX * w;

            if (first)
            {
                p.startNewSubPath(screenX, y);
                first = false;
            }
            else
            {
                p.lineTo(screenX, y);
            }
        }
    }

    void applyDrawingAt(const juce::MouseEvent& e)
    {
        if (m_currentWeights.empty()) return;

        auto bounds = getLocalBounds().toFloat();
        float w = bounds.getWidth();
        float h = bounds.getHeight();
        if (w <= 0.0f || h <= 0.0f) return;

        float clickX = (float)e.x;
        float clickY = (float)e.y;

        float normX = juce::jlimit(0.0f, 1.0f, clickX / w);
        float targetVal = yToValue(clickY, h);

        size_t centerIdx = (size_t)(normX * (m_currentWeights.size() - 1));

        // Apply with a smooth brush radius across neighboring weights
        int totalWeights = (int)m_currentWeights.size();
        int brushRadius = std::max(1, totalWeights / 40); // 2.5% brush width

        for (int d = -brushRadius; d <= brushRadius; ++d)
        {
            int idx = (int)centerIdx + d;
            if (idx >= 0 && idx < totalWeights)
            {
                float dist = (float)std::abs(d) / (float)brushRadius;
                float falloff = 0.5f * (1.0f + std::cos(dist * 3.14159265f)); // Hann window
                m_currentWeights[idx] = m_currentWeights[idx] * (1.0f - falloff) + targetVal * falloff;
            }
        }

        recalculateStats();
        repaint();

        if (onWeightsModified)
            onWeightsModified(m_currentWeights);
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WeightBendingComponent)
};
