#pragma once
#include <JuceHeader.h>
#include <functional>
#include <cmath>

class LatentTerrainComponent : public juce::Component
{
public:
    LatentTerrainComponent()
    {
        setOpaque(true);
        setMouseCursor(juce::MouseCursor::CrosshairCursor);
    }

    ~LatentTerrainComponent() override = default;

    std::function<void(float, float)> onCoordsChanged;

    void setCoords(float x, float y)
    {
        m_x = juce::jlimit(-1.0f, 1.0f, x);
        m_y = juce::jlimit(-1.0f, 1.0f, y);
        repaint();
    }

    float getCoordX() const { return m_x; }
    float getCoordY() const { return m_y; }

    void setEnabledState(bool isEnabled, bool hasBottleneck)
    {
        m_isEnabled = isEnabled;
        m_hasBottleneck = hasBottleneck;
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();

        // Deep canvas background
        g.setColour(juce::Colour::fromString("#ff0e0d13"));
        g.fillRect(bounds);

        auto padArea = bounds.reduced(6.0f);
        float pw = padArea.getWidth();
        float ph = padArea.getHeight();

        // Inner pad background
        g.setColour(juce::Colour::fromString("#ff110f17"));
        g.fillRect(padArea);

        // Border
        g.setColour(juce::Colour::fromString("#ff2d253d"));
        g.drawRect(padArea, 1.0f);

        if (!m_hasBottleneck)
        {
            g.setColour(juce::Colours::grey.withAlpha(0.6f));
            g.setFont(juce::FontOptions(14.0f));
            g.drawText("Current model has no autoencode / latent bottleneck.\n(Requires paired encode + decode methods, e.g. RAVE)",
                       padArea.toNearestInt(), juce::Justification::centred, true);
            return;
        }

        // Center crosshairs (0, 0)
        float cx = padArea.getX() + pw * 0.5f;
        float cy = padArea.getY() + ph * 0.5f;

        g.setColour(juce::Colour::fromString("#ff252033"));
        g.drawHorizontalLine((int)cy, padArea.getX(), padArea.getRight());
        g.drawVerticalLine((int)cx, padArea.getY(), padArea.getBottom());

        // Topographic / radar contour rings
        float maxRadius = std::min(pw, ph) * 0.46f;
        for (int step = 1; step <= 4; ++step)
        {
            float r = maxRadius * (step / 4.0f);
            g.setColour(juce::Colour::fromString("#ff1b1726"));
            g.drawEllipse(cx - r, cy - r, r * 2.0f, r * 2.0f, 1.0f);

            // Subdued coordinate ring labels
            g.setColour(juce::Colours::white.withAlpha(0.18f));
            g.setFont(juce::FontOptions(9.0f));
            g.drawText(juce::String(step * 0.25f, 2),
                       (int)(cx + 4), (int)(cy - r - 6), 30, 12, juce::Justification::centredLeft);
        }

        // Fourier Orbit Contour Waves (Synthetic topographic contours in turquoise / purple)
        {
            juce::Graphics::ScopedSaveState sss(g);
            g.reduceClipRegion(padArea.toNearestInt());

            int gridSteps = 16;
            for (int i = 0; i <= gridSteps; ++i)
            {
                float frac = (float)i / (float)gridSteps;
                float xLine = padArea.getX() + frac * pw;
                float yLine = padArea.getY() + frac * ph;

                g.setColour(juce::Colour::fromString("#ff171322").withAlpha(0.45f));
                g.drawVerticalLine((int)xLine, padArea.getY(), padArea.getBottom());
                g.drawHorizontalLine((int)yLine, padArea.getX(), padArea.getRight());
            }
        }

        // Cursor Reticle Position
        // Map m_x [-1, 1] -> padArea X, m_y [-1, 1] -> padArea Y (inverted Y so up is positive)
        float cursorX = cx + m_x * (pw * 0.48f);
        float cursorY = cy - m_y * (ph * 0.48f);

        // Luminous radial glow around active reticle
        juce::Colour glowColour = m_isEnabled ? juce::Colour::fromString("#ffd946ef") // Vivid neon fuchsia
                                              : juce::Colour::fromString("#ff64748b"); // Slate standby
        
        juce::ColourGradient glowGrad(
            glowColour.withAlpha(0.35f), cursorX, cursorY,
            glowColour.withAlpha(0.0f), cursorX, cursorY + 28.0f, true
        );
        g.setGradientFill(glowGrad);
        g.fillEllipse(cursorX - 28.0f, cursorY - 28.0f, 56.0f, 56.0f);

        // Reticle target rings
        g.setColour(glowColour.withAlpha(0.9f));
        g.drawEllipse(cursorX - 8.0f, cursorY - 8.0f, 16.0f, 16.0f, 1.5f);
        g.drawEllipse(cursorX - 2.0f, cursorY - 2.0f, 4.0f, 4.0f, 2.0f);

        // Cross lines through reticle
        g.drawHorizontalLine((int)cursorY, cursorX - 14.0f, cursorX - 4.0f);
        g.drawHorizontalLine((int)cursorY, cursorX + 4.0f, cursorX + 14.0f);
        g.drawVerticalLine((int)cursorX, cursorY - 14.0f, cursorY - 4.0f);
        g.drawVerticalLine((int)cursorX, cursorY + 4.0f, cursorY + 14.0f);

        // Coordinate readout badge in bottom-left
        g.setFont(juce::FontOptions(11.0f));
        g.setColour(juce::Colours::white.withAlpha(0.75f));
        juce::String coordsText = juce::String::formatted("X: %+.3f   Y: %+.3f", m_x, m_y);
        g.drawText(coordsText, padArea.reduced(10.0f, 6.0f).toNearestInt(), juce::Justification::bottomLeft, true);

        // Top-right status readout badge
        juce::String modeText = m_isEnabled ? "LATENT HOOK: ACTIVE" : "STANDBY (Click to Drag)";
        g.setColour(m_isEnabled ? juce::Colour::fromString("#ffd946ef") : juce::Colours::silver.withAlpha(0.6f));
        g.drawText(modeText, padArea.reduced(10.0f, 6.0f).toNearestInt(), juce::Justification::topRight, true);
    }

    void mouseDown(const juce::MouseEvent& e) override
    {
        updateFromMouse(e);
    }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        updateFromMouse(e);
    }

    void mouseDoubleClick(const juce::MouseEvent& /*e*/) override
    {
        // Double-click snaps back to (0, 0)
        setCoords(0.0f, 0.0f);
        if (onCoordsChanged)
            onCoordsChanged(0.0f, 0.0f);
    }

private:
    float m_x { 0.0f };
    float m_y { 0.0f };
    bool m_isEnabled { false };
    bool m_hasBottleneck { true };

    void updateFromMouse(const juce::MouseEvent& e)
    {
        auto padArea = getLocalBounds().toFloat().reduced(6.0f);
        float pw = padArea.getWidth();
        float ph = padArea.getHeight();

        float cx = padArea.getX() + pw * 0.5f;
        float cy = padArea.getY() + ph * 0.5f;

        float normX = (e.position.x - cx) / (pw * 0.48f);
        float normY = -(e.position.y - cy) / (ph * 0.48f); // Inverted so up is +1

        m_x = juce::jlimit(-1.0f, 1.0f, normX);
        m_y = juce::jlimit(-1.0f, 1.0f, normY);

        repaint();

        if (onCoordsChanged)
            onCoordsChanged(m_x, m_y);
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatentTerrainComponent)
};
