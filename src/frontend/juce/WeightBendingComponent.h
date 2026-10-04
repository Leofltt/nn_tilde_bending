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
        setMouseCursor(juce::MouseCursor::CrosshairCursor);
    }

    ~WeightBendingComponent() override = default;

    std::function<void(const std::vector<float>&)> onWeightsModified;

    enum class RenderMode { Curve, Matrix };

    void setRenderMode(RenderMode mode)
    {
        if (m_renderMode != mode)
        {
            m_renderMode = mode;
            repaint();
        }
    }

    RenderMode getRenderMode() const { return m_renderMode; }

    void setLayerShape(const std::vector<int64_t>& shape)
    {
        m_layerShape = shape;
        calculateMatrixDimensions();
        repaint();
    }

    void setWeights(const std::vector<float>& original, const std::vector<float>& current)
    {
        m_originalWeights = original;
        m_currentWeights = current;
        calculateMatrixDimensions();
        recalculateStats();
        // Reset zoom on loading a new layer
        resetView();
        repaint();
    }

    void updateCurrentWeights(const std::vector<float>& current)
    {
        m_currentWeights = current;
        calculateMatrixDimensions();
        recalculateStats();
        repaint();
    }

    void setTargetBentWeights(const std::vector<float>& target)
    {
        m_targetBentWeights = target;
        repaint();
    }

    void setBridgeWeights(const std::vector<float>& sourceWeights, const std::vector<float>& mixWeights, float depth, bool isContinuous)
    {
        m_bridgeSourceWeights = sourceWeights;
        m_bridgeMixWeights = mixWeights;
        m_bridgeDepth = depth;
        m_isContinuousMode = isContinuous;
        repaint();
    }

    void setHarmonicGhostWeights(const std::vector<float>& ghostWeights, bool showGhost)
    {
        m_harmonicGhostWeights = ghostWeights;
        m_showHarmonicGhost = showGhost;
        repaint();
    }

    void setShowBridgeVisuals(bool show)
    {
        if (m_showBridgeVisuals != show)
        {
            m_showBridgeVisuals = show;
            repaint();
        }
    }

    void setShowHarmonics(bool show)
    {
        if (m_showHarmonics != show)
        {
            m_showHarmonics = show;
            repaint();
        }
    }

    bool isBridgeVisualsEnabled() const { return m_showBridgeVisuals; }
    bool isHarmonicsEnabled() const { return m_showHarmonics; }

    const std::vector<float>& getCurrentWeights() const { return m_currentWeights; }
    const std::vector<int64_t>& getLayerShape() const { return m_layerShape; }
    const std::vector<float>& getOriginalWeights() const { return m_originalWeights; }
    bool isCurrentlyDrawing() const { return m_dragMode == DragMode::Drawing; }

    void resetView()
    {
        m_viewStartX = 0.0f;
        m_viewSpanX = 1.0f;
        m_zoomY = 1.0f;
        m_panY = 0.0f;
        repaint();
    }

    //==============================================================================
    // Drawing & Layout
    //==============================================================================

    juce::Rectangle<float> getPlotArea() const
    {
        auto bounds = getLocalBounds().toFloat();
        float barSize = 14.0f;
        return juce::Rectangle<float>(bounds.getX(), bounds.getY(),
                                      std::max(10.0f, bounds.getWidth() - barSize),
                                      std::max(10.0f, bounds.getHeight() - barSize));
    }

    juce::Rectangle<float> getHScrollBarArea() const
    {
        auto bounds = getLocalBounds().toFloat();
        float barSize = 14.0f;
        return juce::Rectangle<float>(bounds.getX(), bounds.getBottom() - barSize,
                                      std::max(10.0f, bounds.getWidth() - barSize), barSize);
    }

    juce::Rectangle<float> getVScrollBarArea() const
    {
        auto bounds = getLocalBounds().toFloat();
        float barSize = 14.0f;
        return juce::Rectangle<float>(bounds.getRight() - barSize, bounds.getY(),
                                      barSize, std::max(10.0f, bounds.getHeight() - barSize));
    }

    juce::Rectangle<float> getCornerButtonArea() const
    {
        auto bounds = getLocalBounds().toFloat();
        float barSize = 14.0f;
        return juce::Rectangle<float>(bounds.getRight() - barSize, bounds.getBottom() - barSize, barSize, barSize);
    }

    void paint(juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();

        // Overall dark canvas background
        g.setColour(juce::Colour::fromString("#ff0e0d13"));
        g.fillRect(bounds);

        auto plotArea = getPlotArea();
        float pw = plotArea.getWidth();
        float ph = plotArea.getHeight();

        if (m_currentWeights.empty())
        {
            g.setColour(juce::Colours::grey.withAlpha(0.6f));
            g.setFont(juce::FontOptions(14.0f));
            g.drawText("No layer selected or layer has 0 parameters.",
                       plotArea.toNearestInt(), juce::Justification::centred, true);
            drawEdgeScrollBars(g);
            return;
        }

        // Draw plot area background & border
        g.setColour(juce::Colour::fromString("#ff110f17"));
        g.fillRect(plotArea);

        if (m_renderMode == RenderMode::Matrix)
        {
            paintMatrixView(g, plotArea);
        }
        else
        {
            paintCurveView(g, plotArea);
        }

        // Frame around plot area
        g.setColour(juce::Colour::fromString("#ff3a344d"));
        g.drawRect(plotArea, 1.0f);

        // Edge scroll and zoom bars (only relevant for Curve view)
        if (m_renderMode == RenderMode::Curve)
            drawEdgeScrollBars(g);
    }

    void paintCurveView(juce::Graphics& g, juce::Rectangle<float> plotArea)
    {
        float pw = plotArea.getWidth();
        float ph = plotArea.getHeight();

        // Clip drawing to plot area so curves and grid don't spill into scrollbars
        {
            juce::Graphics::ScopedSaveState sss(g);
            g.reduceClipRegion(plotArea.toNearestInt());

            // Zero line
            float zeroY = valueToY(0.0f, ph);
            g.setColour(juce::Colour::fromString("#ff262136"));

            // Horizontal grid lines
            for (float v : { -2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f })
            {
                float y = valueToY(v, ph);
                if (y >= 0 && y <= ph)
                    g.drawHorizontalLine((int)y, 0.0f, pw);
            }

            g.setColour(juce::Colour::fromString("#ff4c4363"));
            g.drawHorizontalLine((int)zeroY, 0.0f, pw);

            // Vertical grid lines based on parameter indices
            int totalWeights = (int)m_currentWeights.size();
            float idxStart = m_viewStartX * (totalWeights - 1);
            float idxEnd = (m_viewStartX + m_viewSpanX) * (totalWeights - 1);

            g.setColour(juce::Colour::fromString("#ff1a1727"));
            for (int step = 1; step <= 4; ++step)
            {
                float fraction = step / 5.0f;
                float x = fraction * pw;
                g.drawVerticalLine((int)x, 0.0f, ph);
            }

            // 1. Render gradient fill under current live weights (drawn early so curves overlay cleanly)
            juce::Path bentPath;
            buildPathForWeights(bentPath, m_currentWeights, pw, ph);

            if (!bentPath.isEmpty())
            {
                juce::Path fillPath = bentPath;
                fillPath.lineTo(pw, zeroY);
                fillPath.lineTo(0.0f, zeroY);
                fillPath.closeSubPath();

                juce::ColourGradient grad(
                    juce::Colour::fromString("#44c084fc"), 0, 0,
                    juce::Colour::fromString("#089333ea"), 0, ph, false
                );
                g.setGradientFill(grad);
                g.fillPath(fillPath);
            }

            // 2. Render original weights baseline curve (vivid emerald green, sleek 1.0f crisp stroke)
            if (!m_originalWeights.empty())
            {
                juce::Path origPath;
                buildPathForWeights(origPath, m_originalWeights, pw, ph);
                g.setColour(juce::Colour::fromString("#ff4ade80").withAlpha(0.85f)); // Vivid emerald green baseline
                g.strokePath(origPath, juce::PathStrokeType(1.0f));
            }

            // 3. Render bridged source layer curve (muted copper/terracotta reference if bridge active and enabled)
            if (m_showBridgeVisuals && m_bridgeDepth > 0.001f && !m_bridgeSourceWeights.empty())
            {
                juce::Path srcPath;
                buildPathForWeights(srcPath, m_bridgeSourceWeights, pw, ph);
                g.setColour(juce::Colour::fromString("#ffc26a38").withAlpha(0.55f)); // Muted copper
                g.strokePath(srcPath, juce::PathStrokeType(1.0f));
            }

            // 4. Render Cross-Talk composite blend curve (Warm golden amber: intermediate between emerald and copper)
            if (m_showBridgeVisuals && m_bridgeDepth > 0.001f && !m_bridgeMixWeights.empty())
            {
                juce::Path mixPath;
                buildPathForWeights(mixPath, m_bridgeMixWeights, pw, ph);
                g.setColour(juce::Colour::fromString("#ffd99b26").withAlpha(m_isContinuousMode ? 0.90f : 0.70f));
                g.strokePath(mixPath, juce::PathStrokeType(1.0f));
            }

            // 5. Render target bent curve (neon pink / magenta overlay showing target state in momentary)
            if (!m_targetBentWeights.empty())
            {
                juce::Path targetPath;
                buildPathForWeights(targetPath, m_targetBentWeights, pw, ph);
                g.setColour(juce::Colour::fromString("#ffec4899").withAlpha(0.85f)); // Vivid neon pink/magenta
                g.strokePath(targetPath, juce::PathStrokeType(1.0f));
            }

            // 6. Render harmonic synthesizer ghost preview (Glowing gold/amber dashed line)
            if (m_showHarmonics && m_showHarmonicGhost && !m_harmonicGhostWeights.empty())
            {
                juce::Path ghostPath;
                buildPathForWeights(ghostPath, m_harmonicGhostWeights, pw, ph);
                g.setColour(juce::Colour::fromString("#fffbbf24").withAlpha(0.95f)); // Luminous amber gold
                const float dashLengths[2] = { 5.0f, 3.0f };
                juce::Path dashedGhostStroke;
                juce::PathStrokeType(1.0f).createDashedStroke(dashedGhostStroke, ghostPath, dashLengths, 2);
                g.fillPath(dashedGhostStroke);
            }

            // 7. Render current live weights stroke (Neon purple/violet with crisp modern 1.2f line)
            if (!bentPath.isEmpty())
            {
                g.setColour(juce::Colour::fromString("#ffc084fc"));
                g.strokePath(bentPath, juce::PathStrokeType(1.2f));
            }

            // Dynamic Legend / Color key overlay in top right corner
            {
                int legendY = 8;
                g.setFont(juce::FontOptions(10.0f));

                // Calculate required width based on active curves
                bool showBridge = (m_showBridgeVisuals && m_bridgeDepth > 0.001f && !m_bridgeSourceWeights.empty());
                bool showTarget = (!m_targetBentWeights.empty());
                bool showHarmonic = (m_showHarmonics && m_showHarmonicGhost && !m_harmonicGhostWeights.empty());

                int totalItems = 2 + (showBridge ? 2 : 0) + (showTarget ? 1 : 0) + (showHarmonic ? 1 : 0);
                int itemW = 78;
                int legendW = totalItems * itemW;
                int legendX = (int)pw - legendW - 8;

                int curX = legendX;

                // Green: Baseline
                g.setColour(juce::Colour::fromString("#ff4ade80"));
                g.fillRect(curX, legendY + 3, 10, 3);
                g.drawText("Baseline W0", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
                curX += itemW;

                // Copper: Bridge Source
                if (showBridge)
                {
                    g.setColour(juce::Colour::fromString("#ffc26a38"));
                    g.fillRect(curX, legendY + 3, 10, 3);
                    g.drawText("Bridge Src", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
                    curX += itemW;

                    // Golden Amber: Cross-Talk Mix
                    g.setColour(juce::Colour::fromString("#ffd99b26"));
                    g.fillRect(curX, legendY + 3, 10, 3);
                    g.drawText(m_isContinuousMode ? "Cross-Talk" : "Cross-Talk (·)", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
                    curX += itemW;
                }

                // Pink: Target Bent (when present)
                if (showTarget)
                {
                    g.setColour(juce::Colour::fromString("#ffec4899"));
                    g.fillRect(curX, legendY + 3, 10, 3);
                    g.drawText("Target Bent", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
                    curX += itemW;
                }

                // Gold: Harmonic Preview (when present)
                if (showHarmonic)
                {
                    g.setColour(juce::Colour::fromString("#fffbbf24"));
                    g.fillRect(curX, legendY + 3, 10, 3);
                    g.drawText("Harmonic (·)", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
                    curX += itemW;
                }

                // Purple: Live Model
                g.setColour(juce::Colour::fromString("#ffc084fc"));
                g.fillRect(curX, legendY + 3, 10, 3);
                g.drawText("Live Model", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
            }

            // Stats & zoom readout on bottom right of plot area
            g.setColour(juce::Colours::white.withAlpha(0.65f));
            g.setFont(juce::FontOptions(10.5f));

            int startDisplayIdx = (int)std::round(idxStart);
            int endDisplayIdx = (int)std::round(idxEnd);
            juce::String stats = juce::String::formatted(
                "Idx: [%d..%d] / %d  |  Zoom: %.1fx (X) %.1fx (Y)  |  Mean: %.4f",
                startDisplayIdx, endDisplayIdx, totalWeights,
                1.0f / m_viewSpanX, m_zoomY, m_meanVal
            );
            g.drawText(stats, plotArea.reduced(8, 4).toNearestInt(), juce::Justification::bottomRight, true);

            // Hint label on top left
            g.setColour(juce::Colours::lightgrey.withAlpha(0.5f));
            juce::String hint = "Draw: Left Drag | Pan: 2-finger scroll / Right Drag | Zoom: Pinch / Cmd+Scroll | Double-Click: Reset";
            g.drawText(hint, plotArea.reduced(8, 4).toNearestInt(), juce::Justification::topLeft, true);
        }
    }

    juce::Colour getHeatmapColour(float val) const
    {
        // Perceptual diverging palette:
        // Deep Indigo/Purple (#ff4338ca) <--> Neutral Charcoal (#ff16131f) <--> Cyan/Emerald (#ff06b6d4 / #ff10b981)
        float maxAbs = std::max(0.001f, m_displayRange);
        float norm = juce::jlimit(-1.0f, 1.0f, val / maxAbs);

        if (norm < 0.0f)
        {
            // Negative: interpolate from charcoal to deep violet/indigo
            float t = -norm; // 0 to 1
            auto cMid = juce::Colour::fromString("#ff16131f");
            auto cNeg = juce::Colour::fromString("#ff6366f1"); // Electric indigo
            return cMid.interpolatedWith(cNeg, t);
        }
        else
        {
            // Positive: interpolate from charcoal to vibrant cyan/emerald
            float t = norm; // 0 to 1
            auto cMid = juce::Colour::fromString("#ff16131f");
            auto cPos = juce::Colour::fromString("#ff06b6d4"); // Bright neon cyan
            return cMid.interpolatedWith(cPos, t);
        }
    }

    void paintMatrixView(juce::Graphics& g, juce::Rectangle<float> plotArea)
    {
        float pw = plotArea.getWidth();
        float ph = plotArea.getHeight();

        juce::Graphics::ScopedSaveState sss(g);
        g.reduceClipRegion(plotArea.toNearestInt());

        int rows = m_matrixRows;
        int cols = m_matrixCols;
        if (rows <= 0 || cols <= 0 || m_currentWeights.empty())
            return;

        // Leave margin for channel indices and status
        float marginTop = 26.0f;
        float marginBottom = 24.0f;
        float marginLeft = 40.0f;
        float marginRight = 16.0f;

        float gridW = std::max(10.0f, pw - marginLeft - marginRight);
        float gridH = std::max(10.0f, ph - marginTop - marginBottom);

        float cellW = gridW / (float)cols;
        float cellH = gridH / (float)rows;

        // Draw cells
        size_t totalWeights = m_currentWeights.size();
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                size_t idx = (size_t)(r * cols + c);
                if (idx >= totalWeights)
                    break;

                float val = m_currentWeights[idx];
                auto cellRect = juce::Rectangle<float>(
                    plotArea.getX() + marginLeft + c * cellW,
                    plotArea.getY() + marginTop + r * cellH,
                    cellW, cellH
                );

                g.setColour(getHeatmapColour(val));
                g.fillRect(cellRect);

                // Subtle grid separator if cells are large enough
                if (cellW > 4.0f && cellH > 4.0f)
                {
                    g.setColour(juce::Colour::fromString("#ff0e0d13").withAlpha(0.6f));
                    g.drawRect(cellRect, 0.5f);
                }

                // If hovered, highlight cell
                if (r == m_hoveredRow && c == m_hoveredCol)
                {
                    g.setColour(juce::Colours::white.withAlpha(0.9f));
                    g.drawRect(cellRect, 1.5f);
                }
            }
        }

        // Bridge Wire cross-talk glowing indicator
        if (m_showBridgeVisuals && m_bridgeDepth > 0.001f && !m_bridgeSourceWeights.empty())
        {
            g.setColour(juce::Colour::fromString("#ffd99b26").withAlpha(0.7f)); // Warm golden amber
            float bridgeBarX = plotArea.getX() + marginLeft - 8.0f;
            g.fillRect(bridgeBarX, plotArea.getY() + marginTop, 4.0f, gridH);
        }

        // Axis Channel Labels
        g.setColour(juce::Colours::white.withAlpha(0.4f));
        g.setFont(juce::FontOptions(10.0f));

        // Row indices (C_out)
        int rowStep = std::max(1, rows / 8);
        for (int r = 0; r < rows; r += rowStep)
        {
            float y = plotArea.getY() + marginTop + r * cellH;
            g.drawText(juce::String(r), (int)plotArea.getX() + 2, (int)y, (int)marginLeft - 6, (int)cellH, juce::Justification::centredRight);
        }

        // Col indices (C_in * K)
        int colStep = std::max(1, cols / 8);
        for (int c = 0; c < cols; c += colStep)
        {
            float x = plotArea.getX() + marginLeft + c * cellW;
            g.drawText(juce::String(c), (int)x, (int)(plotArea.getY() + marginTop - 16.0f), (int)cellW, 14, juce::Justification::centred);
        }

        // Header Title / Shape readout
        juce::String shapeStr = "Shape: [";
        for (size_t i = 0; i < m_layerShape.size(); ++i)
        {
            shapeStr += juce::String(m_layerShape[i]);
            if (i + 1 < m_layerShape.size()) shapeStr += ", ";
        }
        shapeStr += juce::String::formatted("] -> Matrix: %d x %d (%d weights)", rows, cols, (int)totalWeights);

        g.setColour(juce::Colours::lightcyan.withAlpha(0.85f));
        g.setFont(juce::FontOptions(11.0f));
        g.drawText(shapeStr, (int)(plotArea.getX() + marginLeft), (int)plotArea.getY() + 4, (int)gridW, 16, juce::Justification::topLeft);

        // Hovered cell readout in bottom status bar
        if (m_hoveredRow >= 0 && m_hoveredRow < rows && m_hoveredCol >= 0 && m_hoveredCol < cols)
        {
            size_t idx = (size_t)(m_hoveredRow * cols + m_hoveredCol);
            if (idx < totalWeights)
            {
                float val = m_currentWeights[idx];
                float origVal = (idx < m_originalWeights.size()) ? m_originalWeights[idx] : val;
                juce::String cellInfo = juce::String::formatted(
                    "Cell [%d, %d] (Idx %d)  |  Current: %+.5f  |  Original: %+.5f  |  Diff: %+.5f",
                    m_hoveredRow, m_hoveredCol, (int)idx, val, origVal, (val - origVal)
                );

                g.setColour(juce::Colour::fromString("#ff06b6d4"));
                g.drawText(cellInfo, (int)(plotArea.getX() + marginLeft), (int)(plotArea.getBottom() - marginBottom + 4), (int)gridW, 16, juce::Justification::topLeft);
            }
        }
        else
        {
            g.setColour(juce::Colours::lightgrey.withAlpha(0.5f));
            juce::String hint = "Hover to inspect weights | Left Click & Drag: Paint / Bend weight values on 2D matrix";
            g.drawText(hint, (int)(plotArea.getX() + marginLeft), (int)(plotArea.getBottom() - marginBottom + 4), (int)gridW, 16, juce::Justification::topLeft);
        }

        // Diverging Colorbar Legend in top right
        float legW = 90.0f;
        float legH = 10.0f;
        float legX = plotArea.getRight() - marginRight - legW;
        float legY = plotArea.getY() + 6.0f;

        juce::ColourGradient legGrad(
            juce::Colour::fromString("#ff6366f1"), legX, legY,
            juce::Colour::fromString("#ff06b6d4"), legX + legW, legY, false
        );
        legGrad.addColour(0.5, juce::Colour::fromString("#ff16131f"));
        g.setGradientFill(legGrad);
        g.fillRect(legX, legY, legW, legH);
        g.setColour(juce::Colour::fromString("#ff4c4363"));
        g.drawRect(legX, legY, legW, legH, 1.0f);

        g.setFont(juce::FontOptions(9.0f));
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.drawText(juce::String(-m_displayRange, 1), (int)(legX - 24), (int)legY - 1, 22, (int)legH, juce::Justification::centredRight);
        g.drawText(juce::String(+m_displayRange, 1), (int)(legX + legW + 3), (int)legY - 1, 24, (int)legH, juce::Justification::centredLeft);
    }

    void drawEdgeScrollBars(juce::Graphics& g)
    {
        auto hArea = getHScrollBarArea();
        auto vArea = getVScrollBarArea();
        auto cornerArea = getCornerButtonArea();

        // Bar backgrounds
        g.setColour(juce::Colour::fromString("#ff14121b"));
        g.fillRect(hArea);
        g.fillRect(vArea);

        // Subtle borders
        g.setColour(juce::Colour::fromString("#ff282338"));
        g.drawRect(hArea, 1.0f);
        g.drawRect(vArea, 1.0f);

        // Horizontal thumb
        float hThumbX = hArea.getX() + m_viewStartX * hArea.getWidth();
        float hThumbW = std::max(12.0f, m_viewSpanX * hArea.getWidth());
        juce::Rectangle<float> hThumb(hThumbX, hArea.getY() + 2.0f, hThumbW, hArea.getHeight() - 4.0f);

        bool hHover = (m_dragMode == DragMode::HScrollThumb || m_dragMode == DragMode::HScrollLeftHandle || m_dragMode == DragMode::HScrollRightHandle);
        g.setColour(hHover ? juce::Colour::fromString("#ffa855f7") : juce::Colour::fromString("#ff635580"));
        g.fillRoundedRectangle(hThumb, 2.0f);

        // Vertical thumb (Y axis: top corresponds to positive max amplitude)
        // Normalized visible Y: from (normCenter - normSpan/2) to (normCenter + normSpan/2)
        float vSpanNorm = juce::jlimit(0.01f, 1.0f, 1.0f / m_zoomY);
        float vCenterNorm = juce::jlimit(vSpanNorm * 0.5f, 1.0f - vSpanNorm * 0.5f, 0.5f - (m_panY / (2.0f * m_displayRange)));
        float vStartNorm = vCenterNorm - vSpanNorm * 0.5f;

        float vThumbY = vArea.getY() + vStartNorm * vArea.getHeight();
        float vThumbH = std::max(12.0f, vSpanNorm * vArea.getHeight());
        juce::Rectangle<float> vThumb(vArea.getX() + 2.0f, vThumbY, vArea.getWidth() - 4.0f, vThumbH);

        bool vHover = (m_dragMode == DragMode::VScrollThumb || m_dragMode == DragMode::VScrollTopHandle || m_dragMode == DragMode::VScrollBottomHandle);
        g.setColour(vHover ? juce::Colour::fromString("#ffa855f7") : juce::Colour::fromString("#ff635580"));
        g.fillRoundedRectangle(vThumb, 2.0f);

        // Corner reset button (1:1 zoom reset)
        g.setColour(juce::Colour::fromString("#ff1d1929"));
        g.fillRect(cornerArea);
        g.setColour(juce::Colour::fromString("#ff4c4363"));
        g.drawRect(cornerArea, 1.0f);

        g.setColour(juce::Colours::lightgrey.withAlpha(0.7f));
        g.setFont(juce::FontOptions(8.5f));
        g.drawText("1:1", cornerArea.toNearestInt(), juce::Justification::centred, false);
    }

    //==============================================================================
    // Mouse & Gesture Handling
    //==============================================================================

    // Native macOS 2-finger trackpad pinch
    void mouseMagnify(const juce::MouseEvent& e, float scaleFactor) override
    {
        if (m_currentWeights.empty()) return;
        auto plotArea = getPlotArea();
        float mouseXNorm = juce::jlimit(0.0f, 1.0f, (e.position.x - plotArea.getX()) / plotArea.getWidth());

        // Zoom X: scaleFactor > 1 zooms in
        float newSpanX = juce::jlimit(0.005f, 1.0f, m_viewSpanX / scaleFactor);
        float pivotIdxNorm = m_viewStartX + mouseXNorm * m_viewSpanX;
        m_viewStartX = juce::jlimit(0.0f, 1.0f - newSpanX, pivotIdxNorm - mouseXNorm * newSpanX);
        m_viewSpanX = newSpanX;

        // Zoom Y
        m_zoomY = juce::jlimit(0.5f, 25.0f, m_zoomY * scaleFactor);

        repaint();
    }

    // 2-finger trackpad scroll & mouse wheel
    void mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override
    {
        if (m_currentWeights.empty()) return;
        auto plotArea = getPlotArea();

        bool isZoomModifier = e.mods.isAltDown() || e.mods.isCommandDown() || e.mods.isCtrlDown();

        if (isZoomModifier)
        {
            // Zoom horizontally with deltaX, vertically with deltaY
            float factorX = 1.0f + wheel.deltaX * 3.0f;
            float factorY = 1.0f + wheel.deltaY * 3.0f;

            if (std::abs(wheel.deltaX) < 0.0001f && std::abs(wheel.deltaY) > 0.0001f)
            {
                // If only vertical wheel is moved with Alt/Cmd, zoom both proportionally
                factorX = 1.0f + wheel.deltaY * 3.0f;
            }

            // X Zoom centered at mouse position
            float mouseXNorm = juce::jlimit(0.0f, 1.0f, (e.position.x - plotArea.getX()) / plotArea.getWidth());
            float pivotX = m_viewStartX + mouseXNorm * m_viewSpanX;
            float newSpanX = juce::jlimit(0.005f, 1.0f, m_viewSpanX / factorX);
            m_viewStartX = juce::jlimit(0.0f, 1.0f - newSpanX, pivotX - mouseXNorm * newSpanX);
            m_viewSpanX = newSpanX;

            // Y Zoom
            m_zoomY = juce::jlimit(0.5f, 25.0f, m_zoomY * factorY);
        }
        else
        {
            // Regular 2-finger scroll: PAN
            // Horizontal Pan
            float panStepX = -wheel.deltaX * m_viewSpanX * 0.7f;
            m_viewStartX = juce::jlimit(0.0f, 1.0f - m_viewSpanX, m_viewStartX + panStepX);

            // Vertical Pan
            float yVisibleSpan = (2.0f * m_displayRange) / m_zoomY;
            float panStepY = wheel.deltaY * yVisibleSpan * 0.7f;
            m_panY = juce::jlimit(-m_displayRange * 2.0f, m_displayRange * 2.0f, m_panY + panStepY);
        }

        repaint();
    }

    void mouseDoubleClick(const juce::MouseEvent& /*e*/) override
    {
        resetView();
    }

    void mouseMove(const juce::MouseEvent& e) override
    {
        if (m_renderMode == RenderMode::Matrix)
        {
            updateHoveredMatrixCell(e.position);
        }
    }

    void mouseExit(const juce::MouseEvent& /*e*/) override
    {
        if (m_renderMode == RenderMode::Matrix)
        {
            m_hoveredRow = -1;
            m_hoveredCol = -1;
            repaint();
        }
    }

    void mouseDown(const juce::MouseEvent& e) override
    {
        if (m_renderMode == RenderMode::Matrix)
        {
            m_dragMode = DragMode::Drawing;
            m_lastMousePos = e.position;
            applyMatrixDrawing(e.position, e.mods.isShiftDown() || e.mods.isRightButtonDown());
            return;
        }

        auto cornerArea = getCornerButtonArea();
        if (cornerArea.contains(e.position))
        {
            resetView();
            return;
        }

        auto hArea = getHScrollBarArea();
        if (hArea.contains(e.position))
        {
            startHScrollDrag(e);
            return;
        }

        auto vArea = getVScrollBarArea();
        if (vArea.contains(e.position))
        {
            startVScrollDrag(e);
            return;
        }

        // Inside plot area
        if (e.mods.isRightButtonDown() || e.mods.isMiddleButtonDown() || e.mods.isAltDown())
        {
            m_dragMode = DragMode::PanCanvas;
            m_lastMousePos = e.position;
            setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        }
        else
        {
            m_dragMode = DragMode::Drawing;
            m_lastMousePos = e.position;
            setMouseCursor(juce::MouseCursor::CrosshairCursor);
            applyDrawingSegment(e.position.x, e.position.y, e.position.x, e.position.y);
        }
    }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (m_renderMode == RenderMode::Matrix)
        {
            updateHoveredMatrixCell(e.position);
            applyMatrixDrawing(e.position, e.mods.isShiftDown() || e.mods.isRightButtonDown());
            m_lastMousePos = e.position;
            return;
        }

        if (m_dragMode == DragMode::PanCanvas)
        {
            auto delta = e.position - m_lastMousePos;
            m_lastMousePos = e.position;

            auto plotArea = getPlotArea();
            if (plotArea.getWidth() > 0 && plotArea.getHeight() > 0)
            {
                float normDeltaX = -delta.x / plotArea.getWidth() * m_viewSpanX;
                m_viewStartX = juce::jlimit(0.0f, 1.0f - m_viewSpanX, m_viewStartX + normDeltaX);

                float yVisibleSpan = (2.0f * m_displayRange) / m_zoomY;
                float normDeltaY = delta.y / plotArea.getHeight() * yVisibleSpan;
                m_panY = juce::jlimit(-m_displayRange * 2.0f, m_displayRange * 2.0f, m_panY + normDeltaY);

                repaint();
            }
        }
        else if (m_dragMode == DragMode::HScrollThumb)
        {
            auto hArea = getHScrollBarArea();
            float normDelta = (e.position.x - m_dragStartPos.x) / hArea.getWidth();
            m_viewStartX = juce::jlimit(0.0f, 1.0f - m_viewSpanX, m_dragInitialStartX + normDelta);
            repaint();
        }
        else if (m_dragMode == DragMode::HScrollLeftHandle)
        {
            auto hArea = getHScrollBarArea();
            float clickNorm = juce::jlimit(0.0f, 1.0f, (e.position.x - hArea.getX()) / hArea.getWidth());
            float currentEnd = m_dragInitialStartX + m_dragInitialSpanX;
            float newStart = std::min(clickNorm, currentEnd - 0.005f);
            m_viewStartX = newStart;
            m_viewSpanX = currentEnd - newStart;
            repaint();
        }
        else if (m_dragMode == DragMode::HScrollRightHandle)
        {
            auto hArea = getHScrollBarArea();
            float clickNorm = juce::jlimit(0.0f, 1.0f, (e.position.x - hArea.getX()) / hArea.getWidth());
            float newEnd = std::max(clickNorm, m_dragInitialStartX + 0.005f);
            m_viewSpanX = juce::jlimit(0.005f, 1.0f - m_dragInitialStartX, newEnd - m_dragInitialStartX);
            repaint();
        }
        else if (m_dragMode == DragMode::VScrollThumb)
        {
            auto vArea = getVScrollBarArea();
            float deltaNorm = (e.position.y - m_dragStartPos.y) / vArea.getHeight();
            float yVisibleSpan = (2.0f * m_displayRange) / m_zoomY;
            m_panY = juce::jlimit(-m_displayRange * 2.0f, m_displayRange * 2.0f, m_dragInitialPanY - deltaNorm * 2.0f * m_displayRange);
            repaint();
        }
        else if (m_dragMode == DragMode::VScrollTopHandle || m_dragMode == DragMode::VScrollBottomHandle)
        {
            auto vArea = getVScrollBarArea();
            float deltaY = (e.position.y - m_dragStartPos.y);
            float scale = 1.0f + (m_dragMode == DragMode::VScrollTopHandle ? -deltaY : deltaY) / (vArea.getHeight() * 0.5f);
            m_zoomY = juce::jlimit(0.5f, 25.0f, m_dragInitialZoomY * scale);
            repaint();
        }
        else if (m_dragMode == DragMode::Drawing)
        {
            applyDrawingSegment(m_lastMousePos.x, m_lastMousePos.y, e.position.x, e.position.y);
            m_lastMousePos = e.position;
        }
    }

    void mouseUp(const juce::MouseEvent& /*e*/) override
    {
        m_dragMode = DragMode::None;
        setMouseCursor(juce::MouseCursor::CrosshairCursor);
        repaint();
    }

    void updateHoveredMatrixCell(juce::Point<float> pos)
    {
        auto plotArea = getPlotArea();
        float marginTop = 26.0f;
        float marginBottom = 24.0f;
        float marginLeft = 40.0f;
        float marginRight = 16.0f;

        float gridX = plotArea.getX() + marginLeft;
        float gridY = plotArea.getY() + marginTop;
        float gridW = std::max(10.0f, plotArea.getWidth() - marginLeft - marginRight);
        float gridH = std::max(10.0f, plotArea.getHeight() - marginTop - marginBottom);

        if (pos.x >= gridX && pos.x <= gridX + gridW && pos.y >= gridY && pos.y <= gridY + gridH && m_matrixCols > 0 && m_matrixRows > 0)
        {
            int col = (int)((pos.x - gridX) / (gridW / (float)m_matrixCols));
            int row = (int)((pos.y - gridY) / (gridH / (float)m_matrixRows));

            col = juce::jlimit(0, m_matrixCols - 1, col);
            row = juce::jlimit(0, m_matrixRows - 1, row);

            if (col != m_hoveredCol || row != m_hoveredRow)
            {
                m_hoveredCol = col;
                m_hoveredRow = row;
                repaint();
            }
        }
        else
        {
            if (m_hoveredCol != -1 || m_hoveredRow != -1)
            {
                m_hoveredCol = -1;
                m_hoveredRow = -1;
                repaint();
            }
        }
    }

    void applyMatrixDrawing(juce::Point<float> pos, bool invertOrErase)
    {
        updateHoveredMatrixCell(pos);
        if (m_hoveredRow >= 0 && m_hoveredRow < m_matrixRows && m_hoveredCol >= 0 && m_hoveredCol < m_matrixCols)
        {
            size_t idx = (size_t)(m_hoveredRow * m_matrixCols + m_hoveredCol);
            if (idx < m_currentWeights.size())
            {
                float delta = invertOrErase ? -0.1f * m_displayRange : 0.1f * m_displayRange;
                m_currentWeights[idx] = juce::jlimit(-10.0f, 10.0f, m_currentWeights[idx] + delta);
                recalculateStats();
                repaint();

                if (onWeightsModified)
                    onWeightsModified(m_currentWeights);
            }
        }
    }

private:
    enum class DragMode
    {
        None,
        Drawing,
        PanCanvas,
        HScrollThumb,
        HScrollLeftHandle,
        HScrollRightHandle,
        VScrollThumb,
        VScrollTopHandle,
        VScrollBottomHandle
    };

    DragMode m_dragMode { DragMode::None };
    juce::Point<float> m_lastMousePos;
    juce::Point<float> m_dragStartPos;
    float m_dragInitialStartX { 0.0f };
    float m_dragInitialSpanX { 1.0f };
    float m_dragInitialPanY { 0.0f };
    float m_dragInitialZoomY { 1.0f };

    std::vector<float> m_originalWeights;
    std::vector<float> m_currentWeights;
    std::vector<float> m_targetBentWeights;
    std::vector<float> m_bridgeSourceWeights;
    std::vector<float> m_bridgeMixWeights;
    std::vector<float> m_harmonicGhostWeights;
    bool m_showHarmonicGhost { false };
    bool m_showBridgeVisuals { true };
    bool m_showHarmonics { true };
    float m_bridgeDepth { 0.0f };
    bool m_isContinuousMode { true };

    RenderMode m_renderMode { RenderMode::Curve };
    std::vector<int64_t> m_layerShape;
    int m_matrixRows { 1 };
    int m_matrixCols { 1 };
    int m_hoveredRow { -1 };
    int m_hoveredCol { -1 };

    void calculateMatrixDimensions()
    {
        int N = (int)m_currentWeights.size();
        if (N <= 0)
        {
            m_matrixRows = 1;
            m_matrixCols = 1;
            return;
        }

        if (m_layerShape.size() >= 2)
        {
            m_matrixRows = (int)m_layerShape[0];
            int64_t cols = 1;
            for (size_t i = 1; i < m_layerShape.size(); ++i)
                cols *= m_layerShape[i];
            m_matrixCols = (int)cols;
        }
        else if (m_layerShape.size() == 1)
        {
            // 1D tensor (e.g., Bias): pick aspect ratio close to 1:1 or 1:4
            int r = (int)std::floor(std::sqrt((float)N));
            while (r > 1 && (N % r != 0))
                --r;
            m_matrixRows = r;
            m_matrixCols = N / r;
        }
        else
        {
            // Fallback factorization
            int r = (int)std::floor(std::sqrt((float)N));
            while (r > 1 && (N % r != 0))
                --r;
            m_matrixRows = r;
            m_matrixCols = N / r;
        }

        m_matrixRows = std::max(1, m_matrixRows);
        m_matrixCols = std::max(1, m_matrixCols);
    }

    // Viewport transform
    float m_viewStartX { 0.0f }; // 0.0 to 1.0 (start index fraction)
    float m_viewSpanX { 1.0f };   // visible fraction of parameters (1.0 = full layer)
    float m_zoomY { 1.0f };       // Vertical amplitude zoom
    float m_panY { 0.0f };        // Vertical center offset in weight value units

    float m_minVal { -1.0f };
    float m_maxVal { 1.0f };
    float m_meanVal { 0.0f };
    float m_displayRange { 1.5f }; // Visual +/- base amplitude ceiling

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
        m_displayRange = std::max(1.0f, maxAbs * 1.25f);
    }

    float valueToY(float val, float height) const
    {
        // Compute relative to display range, pan, and vertical zoom
        float effectiveSpan = (2.0f * m_displayRange) / m_zoomY;
        float yMin = m_panY - effectiveSpan * 0.5f;
        float norm = (val - yMin) / effectiveSpan;
        norm = juce::jlimit(0.0f, 1.0f, norm);
        return (1.0f - norm) * (height - 16.0f) + 8.0f;
    }

    float yToValue(float y, float height) const
    {
        float norm = 1.0f - ((y - 8.0f) / (height - 16.0f));
        norm = juce::jlimit(0.0f, 1.0f, norm);
        float effectiveSpan = (2.0f * m_displayRange) / m_zoomY;
        float yMin = m_panY - effectiveSpan * 0.5f;
        return yMin + norm * effectiveSpan;
    }

    void buildPathForWeights(juce::Path& p, const std::vector<float>& weights, float w, float h)
    {
        if (weights.empty() || w <= 0.0f) return;

        int totalWeights = (int)weights.size();
        float startIdxF = m_viewStartX * (float)(totalWeights - 1);
        float endIdxF = (m_viewStartX + m_viewSpanX) * (float)(totalWeights - 1);
        int totalVisible = (int)std::ceil(endIdxF - startIdxF + 1);
        int numPixels = (int)w;
        if (numPixels <= 1 || totalVisible <= 1) return;

        // If weight density is high (> 2 weights per pixel), use min/max decimation
        // This avoids point aliasing, preserves visual peaks/transients, and reduces path overhead
        if (totalVisible > numPixels * 2)
        {
            p.preallocateSpace(numPixels * 4);
            bool first = true;

            for (int px = 0; px < numPixels; ++px)
            {
                float normX0 = (float)px / (float)numPixels;
                float normX1 = (float)(px + 1) / (float)numPixels;

                int i0 = juce::jlimit(0, totalWeights - 1, (int)std::floor((m_viewStartX + normX0 * m_viewSpanX) * (float)(totalWeights - 1)));
                int i1 = juce::jlimit(0, totalWeights - 1, (int)std::ceil((m_viewStartX + normX1 * m_viewSpanX) * (float)(totalWeights - 1)));
                if (i1 < i0) i1 = i0;

                float minVal = weights[(size_t)i0];
                float maxVal = minVal;
                for (int i = i0 + 1; i <= i1; ++i)
                {
                    float v = weights[(size_t)i];
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }

                float screenX = normX0 * w;
                float yMin = valueToY(minVal, h);
                float yMax = valueToY(maxVal, h);

                if (first)
                {
                    p.startNewSubPath(screenX, yMin);
                    if (std::abs(yMax - yMin) > 0.5f)
                        p.lineTo(screenX, yMax);
                    first = false;
                }
                else
                {
                    p.lineTo(screenX, yMin);
                    if (std::abs(yMax - yMin) > 0.5f)
                        p.lineTo(screenX, yMax);
                }
            }
            return;
        }

        int numPoints = std::min(totalVisible, numPixels);
        p.preallocateSpace(numPoints * 2);
        bool first = true;

        for (int x = 0; x < numPoints; ++x)
        {
            float normPlotX = (float)x / (float)(numPoints - 1);
            float weightIdxNorm = m_viewStartX + normPlotX * m_viewSpanX;
            size_t idx = (size_t)juce::jlimit(0, totalWeights - 1, (int)std::round(weightIdxNorm * (totalWeights - 1)));

            float y = valueToY(weights[idx], h);
            float screenX = normPlotX * w;

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

    void applyDrawingSegment(float x0, float y0, float x1, float y1)
    {
        if (m_currentWeights.empty()) return;

        auto plotArea = getPlotArea();
        float pw = plotArea.getWidth();
        float ph = plotArea.getHeight();
        if (pw <= 0.0f || ph <= 0.0f) return;

        int totalWeights = (int)m_currentWeights.size();

        // Convert screen coordinates to weight index and target amplitude
        auto screenToWeight = [&](float screenX, float screenY, float& outIdx, float& outVal) {
            float normPlotX = juce::jlimit(0.0f, 1.0f, (screenX - plotArea.getX()) / pw);
            float weightNormX = m_viewStartX + normPlotX * m_viewSpanX;
            outIdx = weightNormX * (float)(totalWeights - 1);
            outVal = yToValue(screenY - plotArea.getY(), ph);
        };

        float idx0, val0, idx1, val1;
        screenToWeight(x0, y0, idx0, val0);
        screenToWeight(x1, y1, idx1, val1);

        // Precise brush radius: exactly 0 (single point) or at most 1 neighbor for anti-aliasing
        // When zoomed out, brush affects a narrow pixel width instead of hundreds of weights
        float weightsPerPixel = (m_viewSpanX * (float)totalWeights) / pw;
        int brushRadius = (weightsPerPixel > 1.5f) ? (int)std::ceil(weightsPerPixel * 0.5f) : 1;
        brushRadius = juce::jlimit(1, 4, brushRadius);

        int minIdx = (int)std::floor(std::min(idx0, idx1)) - brushRadius;
        int maxIdx = (int)std::ceil(std::max(idx0, idx1)) + brushRadius;
        minIdx = juce::jlimit(0, totalWeights - 1, minIdx);
        maxIdx = juce::jlimit(0, totalWeights - 1, maxIdx);

        float dx = idx1 - idx0;
        float dy = val1 - val0;
        float lenSq = dx * dx;

        for (int idx = minIdx; idx <= maxIdx; ++idx)
        {
            float targetVal = val1;
            if (lenSq > 0.0001f)
            {
                float t = juce::jlimit(0.0f, 1.0f, (float)(idx - idx0) / dx);
                targetVal = val0 + t * dy;
            }

            // Falloff distance from the line stroke
            float distFromCenter = 0.0f;
            if (lenSq > 0.0001f)
            {
                if (idx < std::min(idx0, idx1))
                    distFromCenter = std::min(idx0, idx1) - (float)idx;
                else if (idx > std::max(idx0, idx1))
                    distFromCenter = (float)idx - std::max(idx0, idx1);
                else
                    distFromCenter = 0.0f;
            }
            else
            {
                distFromCenter = std::abs((float)idx - idx0);
            }

            if (distFromCenter <= (float)brushRadius)
            {
                // Sharp quadratic/linear falloff instead of broad bell curve
                float normDist = distFromCenter / (float)brushRadius;
                float falloff = (1.0f - normDist * normDist); // 1.0 at center, 0.0 at edge
                m_currentWeights[(size_t)idx] = m_currentWeights[(size_t)idx] * (1.0f - falloff) + targetVal * falloff;
            }
        }

        recalculateStats();
        repaint();

        if (onWeightsModified)
            onWeightsModified(m_currentWeights);
    }

    void startHScrollDrag(const juce::MouseEvent& e)
    {
        auto hArea = getHScrollBarArea();
        float thumbX = hArea.getX() + m_viewStartX * hArea.getWidth();
        float thumbW = std::max(12.0f, m_viewSpanX * hArea.getWidth());

        m_dragStartPos = e.position;
        m_dragInitialStartX = m_viewStartX;
        m_dragInitialSpanX = m_viewSpanX;

        float clickX = e.position.x;
        float handleSize = 6.0f;

        if (clickX >= thumbX && clickX <= thumbX + handleSize)
        {
            m_dragMode = DragMode::HScrollLeftHandle;
            setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
        }
        else if (clickX >= thumbX + thumbW - handleSize && clickX <= thumbX + thumbW)
        {
            m_dragMode = DragMode::HScrollRightHandle;
            setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
        }
        else if (clickX >= thumbX && clickX <= thumbX + thumbW)
        {
            m_dragMode = DragMode::HScrollThumb;
            setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        }
        else
        {
            // Click outside jumps thumb center to click location
            float clickNorm = (clickX - hArea.getX()) / hArea.getWidth();
            m_viewStartX = juce::jlimit(0.0f, 1.0f - m_viewSpanX, clickNorm - m_viewSpanX * 0.5f);
            m_dragInitialStartX = m_viewStartX;
            m_dragMode = DragMode::HScrollThumb;
            setMouseCursor(juce::MouseCursor::DraggingHandCursor);
            repaint();
        }
    }

    void startVScrollDrag(const juce::MouseEvent& e)
    {
        auto vArea = getVScrollBarArea();
        float vSpanNorm = juce::jlimit(0.01f, 1.0f, 1.0f / m_zoomY);
        float vCenterNorm = juce::jlimit(vSpanNorm * 0.5f, 1.0f - vSpanNorm * 0.5f, 0.5f - (m_panY / (2.0f * m_displayRange)));
        float vStartNorm = vCenterNorm - vSpanNorm * 0.5f;

        float thumbY = vArea.getY() + vStartNorm * vArea.getHeight();
        float thumbH = std::max(12.0f, vSpanNorm * vArea.getHeight());

        m_dragStartPos = e.position;
        m_dragInitialPanY = m_panY;
        m_dragInitialZoomY = m_zoomY;

        float clickY = e.position.y;
        float handleSize = 6.0f;

        if (clickY >= thumbY && clickY <= thumbY + handleSize)
        {
            m_dragMode = DragMode::VScrollTopHandle;
            setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
        }
        else if (clickY >= thumbY + thumbH - handleSize && clickY <= thumbY + thumbH)
        {
            m_dragMode = DragMode::VScrollBottomHandle;
            setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
        }
        else if (clickY >= thumbY && clickY <= thumbY + thumbH)
        {
            m_dragMode = DragMode::VScrollThumb;
            setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        }
        else
        {
            // Click outside jumps vertical center to click location
            float clickNorm = (clickY - vArea.getY()) / vArea.getHeight();
            m_panY = juce::jlimit(-m_displayRange * 2.0f, m_displayRange * 2.0f, (0.5f - clickNorm) * 2.0f * m_displayRange);
            m_dragInitialPanY = m_panY;
            m_dragMode = DragMode::VScrollThumb;
            setMouseCursor(juce::MouseCursor::DraggingHandCursor);
            repaint();
        }
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WeightBendingComponent)
};
