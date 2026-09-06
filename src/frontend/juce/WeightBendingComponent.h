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

    void setWeights(const std::vector<float>& original, const std::vector<float>& current)
    {
        m_originalWeights = original;
        m_currentWeights = current;
        recalculateStats();
        // Reset zoom on loading a new layer
        resetView();
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

            // Render original weights curve (faint cyan reference)
            if (!m_originalWeights.empty())
            {
                juce::Path origPath;
                buildPathForWeights(origPath, m_originalWeights, pw, ph);
                g.setColour(juce::Colours::cyan.withAlpha(0.28f));
                g.strokePath(origPath, juce::PathStrokeType(1.0f));
            }

            // Render current (bent) weights
            juce::Path bentPath;
            buildPathForWeights(bentPath, m_currentWeights, pw, ph);

            // Gradient fill under the curve
            if (!bentPath.isEmpty())
            {
                juce::Path fillPath = bentPath;
                fillPath.lineTo(pw, zeroY);
                fillPath.lineTo(0.0f, zeroY);
                fillPath.closeSubPath();

                juce::ColourGradient grad(
                    juce::Colour::fromString("#77c084fc"), 0, 0,
                    juce::Colour::fromString("#0d9333ea"), 0, ph, false
                );
                g.setGradientFill(grad);
                g.fillPath(fillPath);

                // Vibrant neon purple/violet stroke
                g.setColour(juce::Colour::fromString("#ffc084fc"));
                g.strokePath(bentPath, juce::PathStrokeType(1.8f));
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

        // Frame around plot area
        g.setColour(juce::Colour::fromString("#ff3a344d"));
        g.drawRect(plotArea, 1.0f);

        // Edge scroll and zoom bars
        drawEdgeScrollBars(g);
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

    void mouseDown(const juce::MouseEvent& e) override
    {
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
        float startIdx = m_viewStartX * (totalWeights - 1);
        float endIdx = (m_viewStartX + m_viewSpanX) * (totalWeights - 1);
        int numPoints = std::min((int)std::ceil(endIdx - startIdx + 1), (int)w);
        if (numPoints <= 1) return;

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
