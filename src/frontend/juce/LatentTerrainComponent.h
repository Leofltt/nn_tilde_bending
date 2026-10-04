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

        auto padArea = getLocalBounds().toFloat().reduced(6.0f);
        float pw = padArea.getWidth();
        float ph = padArea.getHeight();
        float cx = padArea.getX() + pw * 0.5f;
        float cy = padArea.getY() + ph * 0.5f;
        float cursorX = cx + m_x * (pw * 0.48f);
        float cursorY = cy - m_y * (ph * 0.48f);
        recordTrailPoint(cursorX, cursorY);

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

        // 1. Render multi-pole potential / topographic latent heat map
        // Generates an attractive background gradient field that visualizes the latent space topology
        {
            juce::Graphics::ScopedSaveState sss(g);
            g.reduceClipRegion(padArea.toNearestInt());

            int sampleSteps = 28;
            float stepW = pw / (float)sampleSteps;
            float stepH = ph / (float)sampleSteps;

            for (int sy = 0; sy < sampleSteps; ++sy)
            {
                float pyNorm = 1.0f - 2.0f * ((sy + 0.5f) / (float)sampleSteps);
                float pyPix = padArea.getY() + sy * stepH;

                for (int sx = 0; sx < sampleSteps; ++sx)
                {
                    float pxNorm = 2.0f * ((sx + 0.5f) / (float)sampleSteps) - 1.0f;
                    float pxPix = padArea.getX() + sx * stepW;

                    // Synthetic latent manifold potential: combined harmonic potential wells
                    float dOrigin = std::sqrt(pxNorm * pxNorm + pyNorm * pyNorm);
                    float dCursor = std::sqrt((pxNorm - m_x) * (pxNorm - m_x) + (pyNorm - m_y) * (pyNorm - m_y));

                    float field1 = std::sin(pxNorm * 3.14159f * 1.5f) * std::cos(pyNorm * 3.14159f * 1.5f);
                    float field2 = 0.5f * std::cos(dOrigin * 6.28318f);
                    float field = 0.5f * (field1 + field2); // [-0.75, +0.75]

                    // Dynamic influence of cursor position
                    float cursorProximity = std::exp(-dCursor * 2.5f);

                    float val = juce::jlimit(0.0f, 1.0f, 0.45f + field * 0.35f + cursorProximity * 0.35f);

                    // Deep midnight plum -> electric violet/cyan gradient
                    juce::Colour cellCol;
                    if (val < 0.45f)
                    {
                        float t = val / 0.45f;
                        cellCol = juce::Colour::fromString("#ff0b0a12").interpolatedWith(juce::Colour::fromString("#ff17122a"), t);
                    }
                    else if (val < 0.70f)
                    {
                        float t = (val - 0.45f) / 0.25f;
                        cellCol = juce::Colour::fromString("#ff17122a").interpolatedWith(juce::Colour::fromString("#ff2e1065"), t);
                    }
                    else
                    {
                        float t = (val - 0.70f) / 0.30f;
                        cellCol = juce::Colour::fromString("#ff2e1065").interpolatedWith(juce::Colour::fromString("#ff065f46"), t);
                    }

                    g.setColour(cellCol.withAlpha(0.70f));
                    g.fillRect(pxPix, pyPix, stepW + 0.5f, stepH + 0.5f);
                }
            }
        }

        // 2. Crisp precision coordinate grid
        {
            juce::Graphics::ScopedSaveState sss(g);
            g.reduceClipRegion(padArea.toNearestInt());

            int gridSteps = 16;
            for (int i = 0; i <= gridSteps; ++i)
            {
                float frac = (float)i / (float)gridSteps;
                float xLine = padArea.getX() + frac * pw;
                float yLine = padArea.getY() + frac * ph;

                g.setColour(juce::Colour::fromString("#ff382f4c").withAlpha(0.25f));
                g.drawVerticalLine((int)xLine, padArea.getY(), padArea.getBottom());
                g.drawHorizontalLine((int)yLine, padArea.getX(), padArea.getRight());
            }
        }

        // 3. Center crosshairs (0, 0)
        g.setColour(juce::Colour::fromString("#ff4c3d69").withAlpha(0.6f));
        g.drawHorizontalLine((int)cy, padArea.getX(), padArea.getRight());
        g.drawVerticalLine((int)cx, padArea.getY(), padArea.getBottom());

        // 4. Concentric topographic radar contour rings & radial spokes
        float maxRadius = std::min(pw, ph) * 0.46f;
        for (int step = 1; step <= 4; ++step)
        {
            float r = maxRadius * (step / 4.0f);
            g.setColour(juce::Colour::fromString("#ff58447d").withAlpha(step == 4 ? 0.55f : 0.30f));
            g.drawEllipse(cx - r, cy - r, r * 2.0f, r * 2.0f, step == 4 ? 1.2f : 0.8f);

            // Subdued coordinate ring labels
            g.setColour(juce::Colours::white.withAlpha(0.28f));
            g.setFont(juce::FontOptions(9.0f));
            g.drawText(juce::String(step * 0.25f, 2),
                       (int)(cx + 4), (int)(cy - r - 6), 30, 12, juce::Justification::centredLeft);
        }

        // Diagonal polar guidelines
        {
            g.setColour(juce::Colour::fromString("#ff3b3254").withAlpha(0.35f));
            float diagDist = maxRadius * 0.95f;
            float cos45 = 0.7071f * diagDist;
            g.drawLine(cx - cos45, cy - cos45, cx + cos45, cy + cos45, 0.8f);
            g.drawLine(cx - cos45, cy + cos45, cx + cos45, cy - cos45, 0.8f);
        }

        // 5. Cursor Reticle Position & Vector Ray
        float cursorX = cx + m_x * (pw * 0.48f);
        float cursorY = cy - m_y * (ph * 0.48f);

        // Vector ray from origin (0, 0) to cursor with distance magnitude
        {
            float distFromOrigin = std::sqrt(m_x * m_x + m_y * m_y);
            juce::Colour rayColour = m_isEnabled ? juce::Colour::fromString("#ff06b6d4") // Bright cyan
                                                 : juce::Colour::fromString("#ff64748b");
            g.setColour(rayColour.withAlpha(0.40f));
            g.drawLine(cx, cy, cursorX, cursorY, 1.0f);

            // Midpoint distance badge along vector ray if moved away from center
            if (distFromOrigin > 0.15f)
            {
                float midX = (cx + cursorX) * 0.5f;
                float midY = (cy + cursorY) * 0.5f;
                g.setColour(juce::Colours::cyan.withAlpha(0.85f));
                g.setFont(juce::FontOptions(9.0f));
                g.drawText(juce::String::formatted("r=%.2f", distFromOrigin),
                           (int)(midX + 4), (int)(midY - 6), 40, 12, juce::Justification::centredLeft);
            }
        }

        // Motion trail breadcrumbs (shows recent gesture path)
        if (m_trail.size() >= 2)
        {
            juce::Path trailPath;
            trailPath.startNewSubPath(m_trail[0].x, m_trail[0].y);
            for (size_t i = 1; i < m_trail.size(); ++i)
                trailPath.lineTo(m_trail[i].x, m_trail[i].y);

            juce::Colour trailColour = m_isEnabled ? juce::Colour::fromString("#ffd946ef") // Neon magenta
                                                   : juce::Colour::fromString("#ff64748b");
            g.setColour(trailColour.withAlpha(0.35f));
            g.strokePath(trailPath, juce::PathStrokeType(1.2f));

            // Fade trail points
            for (size_t i = 0; i < m_trail.size(); ++i)
            {
                float alpha = (float)(i + 1) / (float)m_trail.size() * 0.5f;
                g.setColour(trailColour.withAlpha(alpha));
                g.fillEllipse(m_trail[i].x - 2.0f, m_trail[i].y - 2.0f, 4.0f, 4.0f);
            }
        }

        // Luminous radial glow around active reticle
        juce::Colour glowColour = m_isEnabled ? juce::Colour::fromString("#ffd946ef") // Vivid neon fuchsia
                                              : juce::Colour::fromString("#ff64748b"); // Slate standby
        
        juce::ColourGradient glowGrad(
            glowColour.withAlpha(0.40f), cursorX, cursorY,
            glowColour.withAlpha(0.0f), cursorX, cursorY + 32.0f, true
        );
        g.setGradientFill(glowGrad);
        g.fillEllipse(cursorX - 32.0f, cursorY - 32.0f, 64.0f, 64.0f);

        // Reticle target concentric rings
        g.setColour(glowColour.withAlpha(0.95f));
        g.drawEllipse(cursorX - 10.0f, cursorY - 10.0f, 20.0f, 20.0f, 1.2f);
        g.drawEllipse(cursorX - 3.0f, cursorY - 3.0f, 6.0f, 6.0f, 1.8f);

        // Cross lines through reticle
        g.setColour(juce::Colours::white.withAlpha(0.85f));
        g.drawHorizontalLine((int)cursorY, cursorX - 16.0f, cursorX - 5.0f);
        g.drawHorizontalLine((int)cursorY, cursorX + 5.0f, cursorX + 16.0f);
        g.drawVerticalLine((int)cursorX, cursorY - 16.0f, cursorY - 5.0f);
        g.drawVerticalLine((int)cursorX, cursorY + 5.0f, cursorY + 16.0f);

        // Coordinate readout badge in bottom-left
        g.setFont(juce::FontOptions(11.0f));
        g.setColour(juce::Colours::white.withAlpha(0.85f));
        float mag = std::sqrt(m_x * m_x + m_y * m_y);
        float angleRad = std::atan2(m_y, m_x);
        float angleDeg = angleRad * (180.0f / 3.14159265f);
        if (angleDeg < 0.0f) angleDeg += 360.0f;

        juce::String coordsText = juce::String::formatted("X: %+.3f   Y: %+.3f   |   Mag: %.3f   θ: %.1f°", m_x, m_y, mag, angleDeg);
        g.drawText(coordsText, padArea.reduced(10.0f, 6.0f).toNearestInt(), juce::Justification::bottomLeft, true);

        // Top-right status readout badge
        juce::String modeText = m_isEnabled ? "LATENT HOOK: ACTIVE" : "STANDBY (Click to Modulate)";
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

    struct Point2D { float x; float y; };
    std::vector<Point2D> m_trail;

    void recordTrailPoint(float px, float py)
    {
        m_trail.push_back({ px, py });
        if (m_trail.size() > 24)
            m_trail.erase(m_trail.begin());
    }

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

        float cursorX = cx + m_x * (pw * 0.48f);
        float cursorY = cy - m_y * (ph * 0.48f);
        recordTrailPoint(cursorX, cursorY);

        repaint();

        if (onCoordsChanged)
            onCoordsChanged(m_x, m_y);
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatentTerrainComponent)
};
