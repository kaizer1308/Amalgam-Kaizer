// YawAccumulator.cpp

#include "YawAccumulator.h"
#include "MovementSimulation.h" // for MoveData, Math helpers, TIME_TO_TICKS, etc.

#include <algorithm>
#include <cmath>

void YawAccumulator::Begin(float straightFuzzyValue, int maxChanges, int maxChangeTimeTicks, float maxSpeedClamp)
{
    m_straightFuzzy = straightFuzzyValue;
    m_maxChanges = maxChanges;
    m_maxChangeTime = maxChangeTimeTicks;
    m_maxSpeed = maxSpeedClamp;
    m_started = false;
    m_changes = 0;
    m_startTick = 0;
    m_ticks = 0;
    m_tickSpan = 0.f;
    m_weightedYaw = 0.f;
    m_weightSum = 0.f;
    m_lastSign = 0;
    m_lastZero = false;
    m_clampEvents = 0;
    m_lowAmplitudeEvents = 0;
}

bool YawAccumulator::Step(MoveData& newer, MoveData& older)
{
    const float yawNew = Math::VectorAngles(newer.m_vDirection).y;
    const float yawOld = Math::VectorAngles(older.m_vDirection).y;
    const float simDelta = std::max(newer.m_flSimTime - older.m_flSimTime, TICK_INTERVAL);
    const float tickSpan = std::max(simDelta / TICK_INTERVAL, 1.f);
    const int   ticks    = std::max(1, static_cast<int>(std::lround(tickSpan)));

    float yawDelta = Math::NormalizeAngle(yawNew - yawOld);
    if (!std::isfinite(yawDelta))
        return false;

    float yawPerTick = yawDelta / tickSpan;
    float absYawPerTick = std::fabs(yawPerTick);

    constexpr float kMaxYawPerTick = 25.f; // ~1500 deg/s at 60hz
    if (absYawPerTick > kMaxYawPerTick)
    {
        float scale = kMaxYawPerTick / std::max(absYawPerTick, 1e-6f);
        yawPerTick *= scale;
        absYawPerTick = std::fabs(yawPerTick);
        ++m_clampEvents;
    }

    bool zeroed = false;
    const float straightThreshold = m_straightFuzzy;
    const float speed = newer.m_vVelocity.Length2D();
    const float speedFrac = (m_maxSpeed > 1.f)
        ? std::clamp(speed / std::max(m_maxSpeed, 1.f), 0.f, 1.2f)
        : 1.f;
    bool prevZero = m_lastZero;
    if (straightThreshold > 0.f)
    {
        if (absYawPerTick < straightThreshold)
        {
            float ratio = absYawPerTick / std::max(straightThreshold, 1e-4f);
            const float zeroRatio = 0.04f + 0.08f * speedFrac;      // tighter zeroing at high speed
            const float sustainRatio = 0.25f + 0.35f * speedFrac;   // wider ramp as speed increases

            if (ratio < zeroRatio)
            {
                yawPerTick = 0.f;
                zeroed = true;
            }
            else
            {
                float softRatio = std::clamp(ratio / std::max(sustainRatio, 1e-3f), 0.f, 1.f);
                float smooth = softRatio * softRatio * (3.f - 2.f * softRatio); // smoothstep
                float floorScale = 0.55f + 0.45f * speedFrac;
                yawPerTick = std::copysign(straightThreshold * smooth * floorScale, yawPerTick);
            }
            absYawPerTick = std::fabs(yawPerTick);
            ++m_lowAmplitudeEvents;
        }
    }

    if (absYawPerTick <= 1e-4f)
    {
        yawPerTick = 0.f;
        absYawPerTick = 0.f;
        zeroed = true;
    }

    int signNow = (yawPerTick > 0.f) ? 1 : (yawPerTick < 0.f ? -1 : 0);
    bool continuingStrafe = (signNow != 0 && signNow == m_lastSign && !prevZero && !zeroed);
    m_lastZero = zeroed || signNow == 0;

    if (!m_started)
    {
        m_started = true;
        m_startTick = TIME_TO_TICKS(newer.m_flSimTime);
        m_lastSign = signNow;
    }

    if (signNow != 0 && m_lastSign != 0 && signNow != m_lastSign)
    {
        ++m_changes;
        if (m_maxChanges > 0 && m_changes > m_maxChanges)
            return false;
    }

    m_ticks += ticks;
    m_tickSpan += tickSpan;
    if (m_maxChangeTime > 0 && m_ticks > m_maxChangeTime)
        return false;

    float weight = tickSpan;
    float clampedSpeedFrac = std::clamp(speedFrac, 0.f, 1.1f);
    weight *= (0.7f + 0.3f * clampedSpeedFrac);
    if (straightThreshold > 0.f)
    {
        float ampRatio = std::clamp(absYawPerTick / std::max(straightThreshold, 1e-3f), 0.f, 1.5f);
        weight *= (0.85f + 0.25f * ampRatio);
    }
    if (zeroed)
        weight *= 0.6f;
    else if (signNow == 0)
        weight *= 0.25f;
    else if (continuingStrafe)
        weight *= 1.15f;
    else
        weight *= 0.9f;

    m_weightedYaw += yawPerTick * weight;
    m_weightSum += weight;

    return true;
}

float YawAccumulator::Finalize(int minTicks, int dynamicMinTicks) const
{
    const int minTickCount = std::max(minTicks, 1);
    if (m_ticks < minTickCount || m_tickSpan <= TICK_INTERVAL || m_weightSum <= 1e-4f)
        return 0.f;

    const int spanRequirement = dynamicMinTicks > 0 ? dynamicMinTicks : minTickCount;
    if (m_tickSpan < static_cast<float>(spanRequirement))
        return 0.f;
    float avg = m_weightedYaw / m_weightSum;
    if (!std::isfinite(avg))
        return 0.f;
    return avg;
}