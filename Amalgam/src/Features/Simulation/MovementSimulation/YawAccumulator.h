// yawAccumulator.h
// deterministic, per-evaluation accumulator for computing average yaw change
// across movement records without relying on static state

#pragma once

#include <cstdint>

struct MoveData; // forward (defined in MovementSimulation structures)

class YawAccumulator
{
public:
    void Begin(float straightFuzzyValue, int maxChanges, int maxChangeTimeTicks, float maxSpeedClamp);
    // returns false if accumulation should terminate (invalid / exceeded constraints)
    bool Step(MoveData& newer, MoveData& older); // newer has more recent sim time
    float Finalize(int minTicks, int dynamicMinTicks) const; // returns 0.f if below thresholds
    int   AccumulatedTicks() const { return m_ticks; }
    float AccumulatedTickSpan() const { return m_tickSpan; }
    float WeightedSamples() const { return m_weightSum; }
    int   ChangeCount() const { return m_changes; }
    int   ClampEvents() const { return m_clampEvents; }
    int   LowAmplitudeEvents() const { return m_lowAmplitudeEvents; }

private:
    // configuration
    float m_straightFuzzy = 0.f;
    int   m_maxChanges = 0;
    int   m_maxChangeTime = 0; // in ticks
    float m_maxSpeed = 0.f;

    // state
    bool  m_started = false;
    int   m_changes = 0;
    int   m_startTick = 0;
    int   m_ticks = 0;
    float m_tickSpan = 0.f;
    float m_weightedYaw = 0.f;
    float m_weightSum = 0.f;
    int   m_lastSign = 0;
    bool  m_lastZero = false;
    int   m_clampEvents = 0;
    int   m_lowAmplitudeEvents = 0;
};