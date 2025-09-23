#include "MovementSimulation.h"
#include "YawAccumulator.h"

#include "../../EnginePrediction/EnginePrediction.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>
#include <vector>

static CUserCmd s_tDummyCmd = {};

void CMovementSimulation::ComputeYawResidualAndConfidence(const std::deque<MoveData>& recs, int usedTicks, float estYawPerTick, float& outResidualRMS, float& outConfidence) const
{
	outResidualRMS = 0.f;
	outConfidence = 0.f;
	if (recs.size() < 3 || usedTicks <= 0)
		return;

	const float targetTicks = std::max(1.f, static_cast<float>(usedTicks));
	float tickAccum = 0.f;
	float consumedSpan = 0.f;
	float weightSum = 0.f;
	float weightedSq = 0.f;
	int deviationPenalty = 0;
	bool anchorValid = false;
	float anchorYaw = 0.f;
	float anchorTick = 0.f;

	for (size_t i = 1; i < recs.size(); ++i)
	{
		const auto& newer = recs[i - 1];
		const auto& older = recs[i];
		if (newer.m_iMode != older.m_iMode)
			continue;

		float simDelta = std::max(newer.m_flSimTime - older.m_flSimTime, TICK_INTERVAL);
		float span = std::max(simDelta / TICK_INTERVAL, 1.f);

		float prevAccum = tickAccum;
		tickAccum += span;

		float cappedAccum = std::min(tickAccum, targetTicks);
		float effectiveSpan = cappedAccum - std::min(prevAccum, targetTicks);
		if (effectiveSpan <= 0.f)
		{
			if (tickAccum >= targetTicks)
				break;
			continue;
		}

		if (!anchorValid)
		{
			anchorYaw = Math::VectorAngles(older.m_vDirection).y;
			anchorTick = prevAccum;
			anchorValid = true;
		}

		const float yawObs = Math::VectorAngles(newer.m_vDirection).y;
		float yawPred = Math::NormalizeAngle(anchorYaw + estYawPerTick * (prevAccum + effectiveSpan - anchorTick));
		float diff = Math::NormalizeAngle(yawObs - yawPred);

		const Vec3 avgVel = (newer.m_vVelocity + older.m_vVelocity) * 0.5f;
		const float speed = avgVel.Length2D();
		const float baseRef = (newer.m_iMode == 1 ? 260.f : 320.f);
		float speedScale = baseRef > 1.f ? std::clamp(speed / baseRef, 0.f, 1.5f) : 1.f;

		float weight = effectiveSpan * (0.35f + 0.65f * speedScale);
		if (!std::isfinite(weight))
			weight = effectiveSpan;

		weightedSq += diff * diff * weight;
		weightSum += weight;
		consumedSpan += effectiveSpan;

		if (fabsf(diff) > 25.f)
			++deviationPenalty;

		if (tickAccum >= targetTicks)
			break;
	}

	if (weightSum <= 0.f || consumedSpan <= 0.f || !anchorValid)
		return;

	outResidualRMS = sqrtf(weightedSq / weightSum);

	const float coverage = std::clamp(consumedSpan / targetTicks, 0.f, 1.f);
	const float baseConf = std::clamp(1.f - (outResidualRMS / 12.f), 0.f, 1.f);
	const float penaltyFactor = 1.f / (1.f + 0.4f * static_cast<float>(deviationPenalty));
	outConfidence = std::clamp(baseConf * (0.65f + 0.35f * coverage) * penaltyFactor, 0.f, 1.f);
}

int CMovementSimulation::ComputeStabilityScore(const std::deque<MoveData>& recs, int window) const
{
	if ((int)recs.size() < window + 1) return 0;
	window = std::min(window, (int)recs.size() - 1);
	float lastYawDelta = 0.f; float jerkSum = 0.f;
	std::vector<float> speeds; speeds.reserve(window);
	for (int i = 1; i <= window; ++i)
	{
		const float yaw1 = Math::VectorAngles(recs[i-1].m_vDirection).y;
		const float yaw2 = Math::VectorAngles(recs[i].m_vDirection).y;
		float dyaw = Math::NormalizeAngle(yaw1 - yaw2);
		float jerk = fabsf(dyaw - lastYawDelta);
		jerkSum += jerk; lastYawDelta = dyaw;
		speeds.push_back(recs[i-1].m_vVelocity.Length2D());
	}

	float mean = std::accumulate(speeds.begin(), speeds.end(), 0.f) / speeds.size();
    float var = 0.f; for (float s : speeds) { float d = s - mean; var += d * d; }
    var /= std::max<size_t>(1, speeds.size());
    int score = (int)std::round(jerkSum * 0.25f + var * 0.002f);
    return std::max(0, score);
}

// kasa fit
float CMovementSimulation::EstimateCurvatureYawPerTick(const std::deque<MoveData>& recs, int maxSamples, int& outUsedTicks) const
{
	outUsedTicks = 0;
	if ((int)recs.size() < 3) return 0.f;

	// collect a contiguous window of points from the same movement mode starting at the newest sample
	const int targetMode = recs[0].m_iMode;
	std::vector<Vec3> pts; pts.reserve(std::min(maxSamples, (int)recs.size()));
	std::vector<int> idx; idx.reserve(pts.capacity());
	for (int i = 0; i < (int)recs.size() && (int)pts.size() < maxSamples; ++i)
	{
		if (recs[i].m_iMode != targetMode) break;
		pts.push_back(recs[i].m_vOrigin);
		idx.push_back(i);
	}
	if ((int)pts.size() < 3) return 0.f;

	std::reverse(pts.begin(), pts.end());
	std::reverse(idx.begin(), idx.end());

	// compute total ticks and distance across the selected contiguous window
	float totalSpan = 0.f; float dist = 0.f;
	for (size_t k = 1; k < idx.size(); ++k)
	{
		int idxOlder = idx[k - 1];
		int idxNewer = idx[k];
		float simOlder = recs[idxOlder].m_flSimTime;
		float simNewer = recs[idxNewer].m_flSimTime;
		float span = std::max(std::fabs(simNewer - simOlder) / TICK_INTERVAL, 1.f);
		totalSpan += span;
		dist += (pts[k] - pts[k - 1]).Length2D();
	}
	if (totalSpan <= 0.f || dist <= 0.f) return 0.f;
	outUsedTicks = std::max(1, (int)std::round(totalSpan));

	double mx = 0.0, my = 0.0;
	for (const auto& p : pts) { mx += p.x; my += p.y; }
	mx /= (double)pts.size(); my /= (double)pts.size();

	double Suu = 0.0, Suv = 0.0, Svv = 0.0, Suuu = 0.0, Suvv = 0.0, Svvv = 0.0, Svuu = 0.0;
	for (const auto& p : pts)
	{
		double u = (double)p.x - mx; double v = (double)p.y - my;
		double uu = u * u, vv = v * v;
		Suu += uu; Svv += vv; Suv += u * v;
		Suuu += uu * u; Svvv += vv * v; Suvv += u * vv; Svuu += v * uu;
	}

	double det = 2.0 * (Suu * Svv - Suv * Suv);
	if (fabs(det) < 1e-6) return 0.f;

	double uc = (Svv * (Suuu + Suvv) - Suv * (Svvv + Svuu)) / det;
	double vc = (Suu * (Svvv + Svuu) - Suv * (Suuu + Suvv)) / det;
	double R2 = uc * uc + vc * vc + (Suu + Svv) / (double)pts.size();
	if (!(R2 > 0.0 && std::isfinite(R2))) return 0.f;
	float R = sqrtf((float)R2);
	if (R < 1.f) return 0.f;

	// aggregate cross products over segments (chronological order)
	double crossSum = 0.0;
	for (size_t k = 0; k + 2 < pts.size(); ++k)
	{
		Vec3 b = pts[k + 1] - pts[k];
		Vec3 a = pts[k + 2] - pts[k + 1];
		crossSum += (double)(a.x * b.y - a.y * b.x);
	}
	float signDir = crossSum >= 0.0 ? 1.f : -1.f;

	// mean speed over the window and corresponding yaw rate
	float v = dist / (totalSpan * TICK_INTERVAL);
	constexpr float kPi = 3.14159265358979323846f;
	float yawPerSec = (v / R) * signDir * 180.f / kPi; // deg/sec
	float yawPerTick = yawPerSec * TICK_INTERVAL;
	return std::clamp(yawPerTick, -10.f, 10.f);
}

void CMovementSimulation::GetAverageYaw(PlayerStorage& tStorage, int iSamples)
{
	auto pPlayer = tStorage.m_pPlayer;
	auto& vRecords = m_mRecords[pPlayer->entindex()];
	if (vRecords.empty()) return;

	float flMaxSpeed = SDK::MaxSpeed(tStorage.m_pPlayer, false, true);

	const float groundLowMinDist = Vars::Aimbot::Projectile::GroundLowMinimumDistance.Value;
	const int groundLowMinSamples = Vars::Aimbot::Projectile::GroundLowMinimumSamples.Value;
	const float groundHighMinDist = Vars::Aimbot::Projectile::GroundHighMinimumDistance.Value;
	const int groundHighMinSamples = Vars::Aimbot::Projectile::GroundHighMinimumSamples.Value;

	const float airLowMinDist = Vars::Aimbot::Projectile::AirLowMinimumDistance.Value;
	const int airLowMinSamples = Vars::Aimbot::Projectile::AirLowMinimumSamples.Value;
	const float airHighMinDist = Vars::Aimbot::Projectile::AirHighMinimumDistance.Value;
	const int airHighMinSamples = Vars::Aimbot::Projectile::AirHighMinimumSamples.Value;

	bool bGroundInitial = tStorage.m_bDirectMove;

	iSamples = std::min(iSamples, int(vRecords.size()));
	if (iSamples < 2) return;

	int modeSkips = 0;
	YawAccumulator acc;
	bool accConfigured = false;
	bool modeLocked = false;
	bool targetGround = bGroundInitial;
	float spanGroundTicks = 0.f;
	float spanAirTicks = 0.f;
	int segGround = 0;
	int segAir = 0;
	size_t i = 1; for (; i < (size_t)iSamples; ++i)
	{
		auto& newer = vRecords[i - 1];
		auto& older = vRecords[i];
		if (newer.m_iMode != older.m_iMode)
		{
			modeSkips++;
			continue;
		}
		// static inline bool GetYawDifference(MoveData& tRecord1, MoveData& tRecord2, bool bStart, float* pYaw, float flStraightFuzzyValue, int iMaxChanges = 0, int iMaxChangeTime = 0, float flMaxSpeed = 0.f)
		bool bGround = newer.m_iMode != 1;
		float straightFuzzy = bGround ? Vars::Aimbot::Projectile::GroundStraightFuzzyValue.Value : Vars::Aimbot::Projectile::AirStraightFuzzyValue.Value;
		int maxChanges = bGround ? Vars::Aimbot::Projectile::GroundMaxChanges.Value : Vars::Aimbot::Projectile::AirMaxChanges.Value;
		int maxChangeTime = bGround ? Vars::Aimbot::Projectile::GroundMaxChangeTime.Value : Vars::Aimbot::Projectile::AirMaxChangeTime.Value;
		if (!modeLocked)
		{
			targetGround = bGround;
			modeLocked = true;
		}
		else if (targetGround != bGround)
		{
			break;
		}
		if (!accConfigured)
		{
			acc.Begin(straightFuzzy, maxChanges, maxChangeTime, flMaxSpeed);
			accConfigured = true;
		}

		float simDelta = std::max(newer.m_flSimTime - older.m_flSimTime, TICK_INTERVAL);
		float segTicks = std::max(simDelta / TICK_INTERVAL, 1.f);
		if (bGround)
		{
			spanGroundTicks += segTicks;
			++segGround;
		}
		else
		{
			spanAirTicks += segTicks;
			++segAir;
		}

		if (!acc.Step(newer, older))
			break;
	}

	if (!modeLocked || !accConfigured)
		return;

	const bool usingGround = targetGround;
	const int processedSegments = usingGround ? segGround : segAir;
	const float processedSpan = usingGround ? spanGroundTicks : spanAirTicks;

	const int minStrafesGround = 4 + Vars::Aimbot::Projectile::GroundMaxChanges.Value;
	const int minStrafesAir = 2 + std::max(1, Vars::Aimbot::Projectile::AirMaxChanges.Value);
	const int requiredStrafes = usingGround ? minStrafesGround : minStrafesAir;

	if (processedSegments < requiredStrafes)
		return;

	int modeLowMinSamples = usingGround ? groundLowMinSamples : airLowMinSamples;
	int modeHighMinSamples = usingGround ? groundHighMinSamples : airHighMinSamples;
	float modeLowMinDist = usingGround ? groundLowMinDist : airLowMinDist;
	float modeHighMinDist = usingGround ? groundHighMinDist : airHighMinDist;

	int dynamicMin = modeLowMinSamples;
	if (pPlayer->entindex() != I::EngineClient->GetLocalPlayer())
	{
		float flDistance = 0.f;
		if (auto pLocal = H::Entities.GetLocal())
			flDistance = pLocal->m_vecOrigin().DistTo(tStorage.m_pPlayer->m_vecOrigin());
		dynamicMin = static_cast<int>(std::round(Math::RemapVal(flDistance, modeLowMinDist, modeHighMinDist, float(modeLowMinSamples), float(modeHighMinSamples))));
	}
	dynamicMin = std::clamp(dynamicMin, modeLowMinSamples, modeHighMinSamples);
	dynamicMin = std::max(dynamicMin, requiredStrafes);
	dynamicMin = std::min(iSamples, dynamicMin);

	int stabilityScore = ComputeStabilityScore(vRecords, std::min(iSamples, 12));
	tStorage.m_flStability = (float)stabilityScore;
	if (Vars::Aimbot::Projectile::UseStabilityMinSamples.Value)
	{
		int stability = stabilityScore;
		dynamicMin = std::min(iSamples, dynamicMin + std::min(stability, 8));
	}
	dynamicMin = std::max(dynamicMin, requiredStrafes);
	dynamicMin = std::min(iSamples, dynamicMin);

	int minTickRequirement = std::max(modeLowMinSamples, requiredStrafes);
	int spanRequirement = std::max(minTickRequirement, dynamicMin);
	int processedSpanInt = static_cast<int>(std::round(processedSpan));
	spanRequirement = std::max(spanRequirement, processedSpanInt);

	float avgYaw = acc.Finalize(minTickRequirement, spanRequirement);
	if (!avgYaw)
	{
		if (Vars::Debug::Logging.Value)
		{
			std::ostringstream oss;
			oss << "flAverageYaw(det) rejected span=" << acc.AccumulatedTickSpan()
				<< " w=" << acc.WeightedSamples() << " min=" << dynamicMin;
			SDK::Output("MovementSimulation", oss.str().c_str(), { 140, 100, 120 }, Vars::Debug::Logging.Value);
		}
		return;
	}

	const float sampleWeight = acc.WeightedSamples();
	const float tickSpan = acc.AccumulatedTickSpan();
	const int changeCount = acc.ChangeCount();
	const int clampEvents = acc.ClampEvents();
	const int lowAmpEvents = acc.LowAmplitudeEvents();
	float coverageFactor = 0.f;
	if (tickSpan > 0.f)
	{
		float effectiveSpan = processedSpan > 0.f ? processedSpan : tickSpan;
		float coverageDenom = std::max(1.f, float(spanRequirement));
		coverageFactor = std::clamp(effectiveSpan / coverageDenom, 0.f, 1.5f);
	}
	float changePenalty = 1.f;
	if (changeCount > 1)
		changePenalty = std::clamp(1.f - 0.08f * float(changeCount - 1), 0.6f, 1.f);
	float clampPenalty = std::clamp(1.f - 0.05f * float(clampEvents), 0.6f, 1.f);
	float lowAmpPenalty = std::clamp(1.f - 0.02f * float(lowAmpEvents), 0.7f, 1.f);

	// if curvature fit is enabled, estimate and pick lower residual
	float chosenYaw = avgYaw; float conf = 0.f; float residual = 0.f;
	int evalTicks = std::clamp(static_cast<int>(std::round(tickSpan)), spanRequirement / 2, spanRequirement);
	ComputeYawResidualAndConfidence(vRecords, std::max(evalTicks, 1), avgYaw, residual, conf);
	float bestResidual = residual; float bestConf = conf;
	if (Vars::Aimbot::Projectile::UseCurvatureFit.Value)
	{
		int usedTicksCurv = 0;
		float curvYaw = EstimateCurvatureYawPerTick(vRecords, iSamples, usedTicksCurv);
		if (curvYaw != 0.f && usedTicksCurv > 0)
		{
			float r2=0.f, c2=0.f; ComputeYawResidualAndConfidence(vRecords, usedTicksCurv, curvYaw, r2, c2);
			if (r2 < bestResidual)
			{
				chosenYaw = curvYaw; bestResidual = r2; bestConf = c2;
			}
		}
	}

	if (tickSpan <= 0.f || sampleWeight <= 0.f)
		bestConf = 0.f;
	else
	{
		bestConf *= (0.65f + 0.35f * coverageFactor);
		bestConf *= changePenalty;
		bestConf *= clampPenalty;
		bestConf *= lowAmpPenalty;
		bestConf = std::clamp(bestConf, 0.f, 1.f);
	}

	tStorage.m_flAverageYaw = chosenYaw;
	tStorage.m_flAverageYawConfidence = bestConf;
	{
		std::ostringstream oss;
		oss << "flAverageYaw(det) " << chosenYaw
			<< " ticks=" << acc.AccumulatedTicks()
			<< " span=" << tickSpan
			<< " min=" << dynamicMin
			<< " cov=" << coverageFactor
			<< " flips=" << changeCount
			<< " clamp=" << clampEvents
			<< " low=" << lowAmpEvents
			<< " conf=" << bestConf;
		if (pPlayer->entindex() == I::EngineClient->GetLocalPlayer())
			oss << " (local)";
		SDK::Output("MovementSimulation", oss.str().c_str(), { 100, 200, 150 }, Vars::Debug::Logging.Value);
	}
}

void CMovementSimulation::Store()
{
	for (auto pEntity : H::Entities.GetGroup(EGroupType::PLAYERS_ALL))
	{
		auto pPlayer = pEntity->As<CTFPlayer>();
		auto& vRecords = m_mRecords[pPlayer->entindex()];

		if (!pPlayer->IsAlive() || pPlayer->IsAGhost() || pPlayer->m_vecVelocity().IsZero())
		{
			vRecords.clear();
			continue;
		}
		else if (!H::Entities.GetDeltaTime(pPlayer->entindex()))
			continue;

		bool bLocal = pPlayer->entindex() == I::EngineClient->GetLocalPlayer() && !I::EngineClient->IsPlayingDemo();
		Vec3 vPredVelocity = bLocal ? F::EnginePrediction.m_vVelocity : pPlayer->m_vecVelocity();
		Vec3 vServerVelocity = pPlayer->m_vecVelocity();
		Vec3 vVelocity = vPredVelocity;
		Vec3 vOrigin = bLocal ? F::EnginePrediction.m_vOrigin : pPlayer->m_vecOrigin();
		Vec3 vDirection = vVelocity.To2D();
		if (bLocal)
		{
			Vec3 predictedDir = Math::RotatePoint(F::EnginePrediction.m_vDirection, {}, { 0, F::EnginePrediction.m_vAngles.y, 0 });
			if (!predictedDir.To2D().IsZero())
				vDirection = predictedDir;
		}
		else if (!vServerVelocity.To2D().IsZero())
		{
			float serverSpeed = vServerVelocity.Length2D();
			float predSpeed = vPredVelocity.Length2D();
			if (serverSpeed > 1.f && predSpeed > 1.f)
			{
				float blend = std::clamp(serverSpeed / std::max(predSpeed, 1.f), 0.2f, 1.8f);
				vVelocity = vPredVelocity * 0.5f + vServerVelocity * 0.5f * blend;
				vDirection = vVelocity.To2D();
			}
		}

		MoveData* pLastRecord = !vRecords.empty() ? &vRecords.front() : nullptr;
		vRecords.emplace_front(
			vDirection,
			pPlayer->m_flSimulationTime(),
			pPlayer->IsSwimming() ? 2 : pPlayer->IsOnGround() ? 0 : 1,
			vVelocity,
			vOrigin
		);
		MoveData& tCurRecord = vRecords.front();
		if (vRecords.size() > 66)
			vRecords.pop_back();

		float flMaxSpeed = SDK::MaxSpeed(pPlayer);
		if (pLastRecord)
		{
			/*
			if (tRecord.m_iMode != pLastRecord->m_iMode)
				vRecords.clear();
			else // does this eat up fps? i can't tell currently
			*/
			{
				CGameTrace trace = {};
				CTraceFilterWorldAndPropsOnly filter = {};
				SDK::TraceHull(pLastRecord->m_vOrigin, pLastRecord->m_vOrigin + pLastRecord->m_vVelocity * TICK_INTERVAL, pPlayer->m_vecMins() + 0.125f, pPlayer->m_vecMaxs() - 0.125f, pPlayer->SolidMask(), &filter, &trace);
				if (trace.DidHit() && trace.plane.normal.z < 0.707f)
					vRecords.clear();
			}
		}
		if (pPlayer->InCond(TF_COND_SHIELD_CHARGE))
		{
			s_tDummyCmd.forwardmove = 450.f;
			s_tDummyCmd.sidemove = 0.f;
			SDK::FixMovement(&s_tDummyCmd, bLocal ? F::EnginePrediction.m_vAngles : pPlayer->GetEyeAngles(), {});
			tCurRecord.m_vDirection.x = s_tDummyCmd.forwardmove;
			tCurRecord.m_vDirection.y = -s_tDummyCmd.sidemove;
		}
		else
		{
			switch (tCurRecord.m_iMode)
			{
			case 0:
				if (bLocal && Vars::Misc::Movement::Bunnyhop.Value && G::OriginalCmd.buttons & IN_JUMP)
					tCurRecord.m_vDirection = vVelocity.Normalized2D() * flMaxSpeed;
				break;
			case 1:
				tCurRecord.m_vDirection = vVelocity.Normalized2D() * flMaxSpeed;
				break;
			case 2:
				tCurRecord.m_vDirection *= 2;
			}
		}
	}

	for (auto pEntity : H::Entities.GetGroup(EGroupType::PLAYERS_ALL))
	{
		auto pPlayer = pEntity->As<CTFPlayer>();
		auto& vSimTimes = m_mSimTimes[pPlayer->entindex()];

		if (pEntity->entindex() == I::EngineClient->GetLocalPlayer() || !pPlayer->IsAlive() || pPlayer->IsAGhost())
		{
			vSimTimes.clear();
			continue;
		}

		float flDeltaTime = H::Entities.GetDeltaTime(pPlayer->entindex());
		if (!flDeltaTime)
			continue;

		vSimTimes.push_front(flDeltaTime);
		if (vSimTimes.size() > Vars::Aimbot::Projectile::DeltaCount.Value)
			vSimTimes.pop_back();
	}
}



bool CMovementSimulation::Initialize(CBaseEntity* pEntity, PlayerStorage& tStorage, bool bHitchance, bool bStrafe)
{
	if (!pEntity || !pEntity->IsPlayer() || !pEntity->As<CTFPlayer>()->IsAlive())
	{
		tStorage.m_bInitFailed = tStorage.m_bFailed = true;
		return false;
	}

	auto pPlayer = pEntity->As<CTFPlayer>();
	tStorage.m_pPlayer = pPlayer;

	I::MoveHelper->SetHost(pPlayer);
	pPlayer->m_pCurrentCommand() = &s_tDummyCmd;

	// store player restore data
	Store(tStorage);

	// store vars
	m_bOldInPrediction = I::Prediction->m_bInPrediction;
	m_bOldFirstTimePredicted = I::Prediction->m_bFirstTimePredicted;
	m_flOldFrametime = I::GlobalVars->frametime;

	// the hacks that make it work
	{
		// use raw vel

		if (pPlayer->m_bDucked() = pPlayer->IsDucking())
		{
			pPlayer->m_fFlags() &= ~FL_DUCKING; // breaks origin's z if FL_DUCKING is not removed
			pPlayer->m_flDucktime() = 0.f;
			pPlayer->m_flDuckJumpTime() = 0.f;
			pPlayer->m_bDucking() = false;
			pPlayer->m_bInDuckJump() = false;
		}

		if (pPlayer != H::Entities.GetLocal())
		{
			pPlayer->m_vecBaseVelocity() = Vec3(); // residual basevelocity causes issues
			if (pPlayer->IsOnGround())
				pPlayer->m_vecVelocity().z = std::min(pPlayer->m_vecVelocity().z, 0.f); // step fix
			else
				pPlayer->m_hGroundEntity() = nullptr; // fix for velocity.z being set to 0 even if in air
		}
		else if (Vars::Misc::Movement::Bunnyhop.Value && G::OriginalCmd.buttons & IN_JUMP)
			tStorage.m_bBunnyHop = true;
	}

	// setup move data
	if (!SetupMoveData(tStorage))
	{
		tStorage.m_bFailed = true;
		return false;
	}

	const int iStrafeSamples = tStorage.m_bDirectMove
		? Vars::Aimbot::Projectile::GroundSamples.Value
		: Vars::Aimbot::Projectile::AirSamples.Value;

	// calculate strafe if desired
	bool bCalculated = bStrafe ? StrafePrediction(tStorage, iStrafeSamples) : false;

	// really hope this doesn't work like shit
	if (bHitchance && bCalculated && !pPlayer->m_vecVelocity().IsZero())
	{
		const auto& vRecords = m_mRecords[pPlayer->entindex()];
		const auto iSamples = vRecords.size();

		float flCurrentChance = 1.f, flAverageYaw = 0.f;
		for (size_t i = 0; i < iSamples; i++)
		{
			if (vRecords.size() <= i + 2)
				break;

			const auto& pRecord1 = vRecords[i], &pRecord2 = vRecords[i + 1];
			const float flYaw1 = Math::VectorAngles(pRecord1.m_vDirection).y, flYaw2 = Math::VectorAngles(pRecord2.m_vDirection).y;
			const float flTime1 = pRecord1.m_flSimTime, flTime2 = pRecord2.m_flSimTime;
			const int iTicks = std::max(TIME_TO_TICKS(flTime1 - flTime2), 1);

			float flYaw = Math::NormalizeAngle(flYaw1 - flYaw2) / iTicks;
			flAverageYaw += flYaw;
			if (tStorage.m_MoveData.m_flMaxSpeed)
				flYaw *= std::clamp(pRecord1.m_vVelocity.Length2D() / tStorage.m_MoveData.m_flMaxSpeed, 0.f, 1.f);

			if ((i + 1) % iStrafeSamples == 0 || i == iSamples - 1)
			{
				flAverageYaw /= i % iStrafeSamples + 1;
				if (fabsf(tStorage.m_flAverageYaw - flAverageYaw) > 0.5f)
					flCurrentChance -= 1.f / ((iSamples - 1) / float(iStrafeSamples) + 1);
				flAverageYaw = 0.f;
			}
		}

		float conf = std::clamp(tStorage.m_flAverageYawConfidence, 0.f, 1.f);
		float required = 0.f;
		if (Vars::Aimbot::Projectile::HitChance.Value > 0.f)
		{
			float base = Vars::Aimbot::Projectile::HitChance.Value / 100.f;
			float maxSpeed = std::max(tStorage.m_MoveData.m_flMaxSpeed, 1.f);
			float speedFrac = std::clamp(tStorage.m_MoveData.m_vecVelocity.Length2D() / maxSpeed, 0.f, 1.f);
			float instability = std::clamp(tStorage.m_flStability / 8.f, 0.f, 1.f);
			float coverage = iStrafeSamples > 0 ? std::clamp((float(std::max<size_t>(iSamples, size_t(1)) - 1) / float(iStrafeSamples)), 0.f, 1.f) : 1.f;
			float adjust = 1.f;
			adjust *= (1.f + 0.25f * speedFrac);
			adjust *= (1.f + 0.25f * instability);
			adjust *= (1.f - 0.20f * conf);
			adjust *= (1.f + 0.15f * (1.f - coverage));
			adjust = std::clamp(adjust, 0.75f, 1.35f);
			required = std::clamp(base * adjust, 0.05f, 0.99f);
		}
		else
		{
			float maxSpeed = std::max(tStorage.m_MoveData.m_flMaxSpeed, 1.f);
			float speedFrac = std::clamp(tStorage.m_MoveData.m_vecVelocity.Length2D() / maxSpeed, 0.f, 1.f);
			float instability = std::clamp(tStorage.m_flStability / 8.f, 0.f, 1.f);
			float coverage = iStrafeSamples > 0 ? std::clamp((float(std::max<size_t>(iSamples, size_t(1)) - 1) / float(iStrafeSamples)), 0.f, 1.f) : 1.f;

			float flDistance = 0.f;
			if (auto pLocal = H::Entities.GetLocal())
				flDistance = pLocal->m_vecOrigin().DistTo(pPlayer->m_vecOrigin());

			float base = 0.62f;
			if (flDistance <= 800.f)
				base = Math::RemapVal(flDistance, 0.f, 800.f, 0.78f, 0.65f);
			else
				base = Math::RemapVal(flDistance, 800.f, 1800.f, 0.65f, 0.52f);
			base = std::clamp(base, 0.50f, 0.80f);

			float adjust = 1.f;
			adjust *= (1.f + 0.20f * speedFrac);
			adjust *= (1.f + 0.15f * instability);
			adjust *= (1.f - 0.30f * conf);
			adjust *= (1.f + 0.10f * (1.f - coverage));
			adjust = std::clamp(adjust, 0.70f, 1.30f);

			required = std::clamp(base * adjust, 0.30f, 0.90f);
			required *= (1.f - 0.25f * conf);
			required = std::clamp(required, 0.10f, 0.90f);
		}

		if (flCurrentChance < required)
		{
			{
		std::ostringstream oss; oss << "Hitchance (" << flCurrentChance * 100 << "% < " << (required * 100.f) << "%)";
				SDK::Output("MovementSimulation", oss.str().c_str(), { 80, 200, 120 }, Vars::Debug::Logging.Value);
			}

			tStorage.m_bFailed = true;
			return false;
		}
	}

	for (int i = 0; i < H::Entities.GetChoke(pPlayer->entindex()); i++)
		RunTick(tStorage);

	return true;
}

bool CMovementSimulation::SetupMoveData(PlayerStorage& tStorage)
{
	if (!tStorage.m_pPlayer)
		return false;

	tStorage.m_MoveData.m_bFirstRunOfFunctions = false;
	tStorage.m_MoveData.m_bGameCodeMovedPlayer = false;
	tStorage.m_MoveData.m_nPlayerHandle = reinterpret_cast<IHandleEntity*>(tStorage.m_pPlayer)->GetRefEHandle();

	tStorage.m_MoveData.m_vecAbsOrigin = tStorage.m_pPlayer->m_vecOrigin();
	tStorage.m_MoveData.m_vecVelocity = tStorage.m_pPlayer->m_vecVelocity();
	tStorage.m_MoveData.m_flMaxSpeed = SDK::MaxSpeed(tStorage.m_pPlayer);
	tStorage.m_MoveData.m_flClientMaxSpeed = tStorage.m_MoveData.m_flMaxSpeed;

	if (!tStorage.m_MoveData.m_vecVelocity.To2D().IsZero())
	{
		int iIndex = tStorage.m_pPlayer->entindex();
		if (iIndex == I::EngineClient->GetLocalPlayer() && G::CurrentUserCmd)
			tStorage.m_MoveData.m_vecViewAngles = G::CurrentUserCmd->viewangles;
		else
		{
			if (!tStorage.m_pPlayer->InCond(TF_COND_SHIELD_CHARGE))
				tStorage.m_MoveData.m_vecViewAngles = { 0.f, Math::VectorAngles(tStorage.m_MoveData.m_vecVelocity).y, 0.f };
			else
				tStorage.m_MoveData.m_vecViewAngles = H::Entities.GetEyeAngles(iIndex);
		}

		const auto& vRecords = m_mRecords[tStorage.m_pPlayer->entindex()];
		if (!vRecords.empty())
		{
			auto& tRecord = vRecords.front();
			if (!tRecord.m_vDirection.IsZero())
			{
				s_tDummyCmd.forwardmove = tRecord.m_vDirection.x;
				s_tDummyCmd.sidemove = -tRecord.m_vDirection.y;
				s_tDummyCmd.upmove = tRecord.m_vDirection.z;
				SDK::FixMovement(&s_tDummyCmd, {}, tStorage.m_MoveData.m_vecViewAngles);
				tStorage.m_MoveData.m_flForwardMove = s_tDummyCmd.forwardmove;
				tStorage.m_MoveData.m_flSideMove = s_tDummyCmd.sidemove;
				tStorage.m_MoveData.m_flUpMove = s_tDummyCmd.upmove;
			}
		}

		if ((tStorage.m_MoveData.m_flForwardMove == 0.f && tStorage.m_MoveData.m_flSideMove == 0.f)
			&& !tStorage.m_MoveData.m_vecVelocity.To2D().IsZero())
		{
			Vec3 vDir = tStorage.m_MoveData.m_vecVelocity.Normalized2D() * tStorage.m_MoveData.m_flMaxSpeed;
			s_tDummyCmd.forwardmove = vDir.x;
			s_tDummyCmd.sidemove = -vDir.y;
			s_tDummyCmd.upmove = 0.f;
			SDK::FixMovement(&s_tDummyCmd, {}, tStorage.m_MoveData.m_vecViewAngles);
			tStorage.m_MoveData.m_flForwardMove = s_tDummyCmd.forwardmove;
			tStorage.m_MoveData.m_flSideMove = s_tDummyCmd.sidemove;
			tStorage.m_MoveData.m_flUpMove = s_tDummyCmd.upmove;
		}
	}

	tStorage.m_MoveData.m_vecAngles = tStorage.m_MoveData.m_vecOldAngles = tStorage.m_MoveData.m_vecViewAngles;
	if (auto pConstraintEntity = tStorage.m_pPlayer->m_hConstraintEntity().Get())
		tStorage.m_MoveData.m_vecConstraintCenter = pConstraintEntity->GetAbsOrigin();
	else
		tStorage.m_MoveData.m_vecConstraintCenter = tStorage.m_pPlayer->m_vecConstraintCenter();
	tStorage.m_MoveData.m_flConstraintRadius = tStorage.m_pPlayer->m_flConstraintRadius();
	tStorage.m_MoveData.m_flConstraintWidth = tStorage.m_pPlayer->m_flConstraintWidth();
	tStorage.m_MoveData.m_flConstraintSpeedFactor = tStorage.m_pPlayer->m_flConstraintSpeedFactor();

	tStorage.m_flPredictedDelta = GetPredictedDelta(tStorage.m_pPlayer);
	tStorage.m_flSimTime = tStorage.m_pPlayer->m_flSimulationTime();
	tStorage.m_flPredictedSimTime = tStorage.m_flSimTime + tStorage.m_flPredictedDelta;
	tStorage.m_vPredictedOrigin = tStorage.m_MoveData.m_vecAbsOrigin;
	tStorage.m_bDirectMove = tStorage.m_pPlayer->IsOnGround() || tStorage.m_pPlayer->IsSwimming();

    return true;
}

static inline float GetGravity()
{
    static auto sv_gravity = U::ConVars.FindVar("sv_gravity");

    return sv_gravity->GetFloat();
}

bool CMovementSimulation::DetectLedgeAndClamp(PlayerStorage& tStorage, float yawStep)
{
    if (!tStorage.m_pPlayer)
        return false;

    auto pPlayer = tStorage.m_pPlayer;

    if (!pPlayer->IsOnGround() || pPlayer->IsSwimming() || pPlayer->InCond(TF_COND_SHIELD_CHARGE))
        return false;

    float cmdF = tStorage.m_MoveData.m_flForwardMove;
    float cmdS = tStorage.m_MoveData.m_flSideMove;
    float cmdMag = sqrtf(cmdF * cmdF + cmdS * cmdS);
    float velMag = tStorage.m_MoveData.m_vecVelocity.Length2D();

    float yaw = tStorage.m_MoveData.m_vecViewAngles.y; // already includes yawStep if applied before
    constexpr float kPi = 3.14159265358979323846f;
    float rad = yaw * (kPi / 180.f);
    Vec3 fwd(cosf(rad), sinf(rad), 0.f);
    Vec3 right(-sinf(rad), cosf(rad), 0.f);
    Vec3 wish = fwd * cmdF + right * cmdS;
    if (wish.To2D().IsZero())
    {
        if (velMag < 1.f)
            return false;
        wish = tStorage.m_MoveData.m_vecVelocity; wish.z = 0.f;
    }
    Vec3 wishDir = wish.Normalized2D();
    Vec3 mins = pPlayer->m_vecMins() + 0.125f;
    Vec3 maxs = pPlayer->m_vecMaxs() - 0.125f;
    int nMask = pPlayer->SolidMask();
    CTraceFilterWorldAndPropsOnly filter = {};

    CGameTrace trFwd = {};
    Vec3 start = tStorage.m_MoveData.m_vecAbsOrigin + Vec3(0, 0, 2);
    float aheadDist = std::clamp(velMag * TICK_INTERVAL, 8.f, 24.f);
    Vec3 end = start + wishDir * aheadDist;
    SDK::TraceHull(start, end, mins, maxs, nMask, &filter, &trFwd);

    Vec3 edgePos = trFwd.endpos;
    CGameTrace trDown = {};
    SDK::Trace(edgePos + Vec3(0, 0, 2), edgePos + Vec3(0, 0, -64), nMask, &filter, &trDown);

    bool noGround = !trDown.DidHit();
    float drop = noGround ? 9999.f : (edgePos.z - trDown.endpos.z);
    bool bigDrop = drop > 20.f;

    if (noGround || bigDrop)
    {
        bool likelyJump = false;
        auto& recs = m_mRecords[pPlayer->entindex()];
        if (recs.size() >= 2)
        {
            const auto& r0 = recs[0];
            const auto& r1 = recs[1];
            float dvz = r0.m_vVelocity.z - r1.m_vVelocity.z;
            if (r0.m_iMode == 1 || dvz > 60.f)
                likelyJump = true;
        }

        if (!likelyJump)
        {
            tStorage.m_MoveData.m_flForwardMove = 0.f;
            if (cmdMag < 50.f)
                tStorage.m_MoveData.m_flSideMove = 0.f;
            return true;
        }
    }
    return false;
}

bool CMovementSimulation::StrafePrediction(PlayerStorage& tStorage, int iSamples)
{
    if (tStorage.m_bDirectMove
        ? !(Vars::Aimbot::Projectile::StrafePrediction.Value & Vars::Aimbot::Projectile::StrafePredictionEnum::Ground)
        : !(Vars::Aimbot::Projectile::StrafePrediction.Value & Vars::Aimbot::Projectile::StrafePredictionEnum::Air))
        return false;

    GetAverageYaw(tStorage, iSamples);
    const bool bCounterStrafe = (Vars::Aimbot::Projectile::StrafePrediction.Value & Vars::Aimbot::Projectile::StrafePredictionEnum::CounterStrafe);
    if (bCounterStrafe)
    {
        CounterStrafePrediction(tStorage, iSamples);
    }
    else
    {
        // clear when disabled to avoid stale state
        tStorage.m_iYawSign = 0;
        tStorage.m_iStrafePeriod = 0;
        tStorage.m_iTicksToFlip = -1;
        tStorage.m_flCounterStrafeConfidence = 0.f;
        tStorage.m_flYawAbs = fabsf(tStorage.m_flAverageYaw);
    }
    return true;
}

bool CMovementSimulation::SetDuck(PlayerStorage& tStorage, bool bDuck) // this only touches origin, bounds
{
    if (bDuck == tStorage.m_pPlayer->m_bDucked())
        return true;

    auto pGameRules = I::TFGameRules();
    auto pViewVectors = pGameRules ? pGameRules->GetViewVectors() : nullptr;
    float flScale = tStorage.m_pPlayer->m_flModelScale();

    if (!tStorage.m_pPlayer->IsOnGround())
    {
        Vec3 vHullMins = (pViewVectors ? pViewVectors->m_vHullMin : Vec3(-24, -24, 0)) * flScale;
        Vec3 vHullMaxs = (pViewVectors ? pViewVectors->m_vHullMax : Vec3(24, 24, 82)) * flScale;
        Vec3 vDuckHullMins = (pViewVectors ? pViewVectors->m_vDuckHullMin : Vec3(-24, -24, 0)) * flScale;
        Vec3 vDuckHullMaxs = (pViewVectors ? pViewVectors->m_vDuckHullMax : Vec3(24, 24, 62)) * flScale;

        if (bDuck)
            tStorage.m_MoveData.m_vecAbsOrigin += (vHullMaxs - vHullMins) - (vDuckHullMaxs - vDuckHullMins);
        else
        {
            Vec3 vOrigin = tStorage.m_MoveData.m_vecAbsOrigin - ((vHullMaxs - vHullMins) - (vDuckHullMaxs - vDuckHullMins));

            CGameTrace trace = {};
            CTraceFilterWorldAndPropsOnly filter = {};
            SDK::TraceHull(vOrigin, vOrigin, vHullMins, vHullMaxs, tStorage.m_pPlayer->SolidMask(), &filter, &trace);
            if (trace.DidHit())
                return false;

            tStorage.m_MoveData.m_vecAbsOrigin = vOrigin;
        }
    }
    tStorage.m_pPlayer->m_bDucked() = bDuck;

    return true;
}

void CMovementSimulation::SetBounds(CTFPlayer* pPlayer)
{
	if (pPlayer->entindex() == I::EngineClient->GetLocalPlayer())
		return;

	// fixes issues with origin compression
	if (auto pGameRules = I::TFGameRules())
	{
		if (auto pViewVectors = pGameRules->GetViewVectors())
		{
			pViewVectors->m_vHullMin = Vec3(-24, -24, 0) + 0.125f;
			pViewVectors->m_vHullMax = Vec3(24, 24, 82) - 0.125f;
			pViewVectors->m_vDuckHullMin = Vec3(-24, -24, 0) + 0.125f;
			pViewVectors->m_vDuckHullMax = Vec3(24, 24, 62) - 0.125f;
		}
	}
}

void CMovementSimulation::RestoreBounds(CTFPlayer* pPlayer)
{
	if (pPlayer->entindex() == I::EngineClient->GetLocalPlayer())
		return;

	if (auto pGameRules = I::TFGameRules())
	{
		if (auto pViewVectors = pGameRules->GetViewVectors())
		{
			pViewVectors->m_vHullMin = Vec3(-24, -24, 0);
			pViewVectors->m_vHullMax = Vec3(24, 24, 82);
			pViewVectors->m_vDuckHullMin = Vec3(-24, -24, 0);
			pViewVectors->m_vDuckHullMax = Vec3(24, 24, 62);
		}
	}
}

void CMovementSimulation::RunTick(PlayerStorage& tStorage, bool bPath, std::function<void(CMoveData&)>* pCallback)
{
	if (tStorage.m_bFailed || !tStorage.m_pPlayer || !tStorage.m_pPlayer->IsPlayer())
		return;

	if (bPath)
		tStorage.m_vPath.push_back(tStorage.m_MoveData.m_vecAbsOrigin);

	I::Prediction->m_bInPrediction = true;
	I::Prediction->m_bFirstTimePredicted = false;
	I::GlobalVars->frametime = I::Prediction->m_bEnginePaused ? 0.f : TICK_INTERVAL;
	SetBounds(tStorage.m_pPlayer);

	    if (tStorage.m_flAverageYaw)
	{
		float yawStep = 0.f;
		bool bScheduledThisTick = false; 
		const bool bCharging = tStorage.m_pPlayer->InCond(TF_COND_SHIELD_CHARGE);
		const bool bSwimming = tStorage.m_pPlayer->IsSwimming();
		if (!bCharging && !bSwimming)
		{
			const bool bCounterStrafe = (Vars::Aimbot::Projectile::StrafePrediction.Value & Vars::Aimbot::Projectile::StrafePredictionEnum::CounterStrafe);
			const bool bUseSchedule = bCounterStrafe
				&& (tStorage.m_flCounterStrafeConfidence > 0.6f)
				&& (tStorage.m_iStrafePeriod > 1)
				&& (tStorage.m_flYawAbs >= 0.08f)
				&& (tStorage.m_flAverageYawConfidence > 0.35f);
			if (bUseSchedule)
			{
				if (tStorage.m_iYawSign == 0)
										tStorage.m_iYawSign = (tStorage.m_flAverageYaw >= 0.f ? +1 : -1);

				yawStep = tStorage.m_flYawAbs * float(tStorage.m_iYawSign);
				bScheduledThisTick = true;

				if (tStorage.m_iTicksToFlip > 0)
					--tStorage.m_iTicksToFlip;
				if (tStorage.m_iTicksToFlip == 0)
				{
					tStorage.m_iYawSign = -tStorage.m_iYawSign;
					tStorage.m_iTicksToFlip = std::max(1, tStorage.m_iStrafePeriod);
				}
			}
			else
			{
				yawStep = tStorage.m_flAverageYaw;
			}
		}
		else
		{
			yawStep = tStorage.m_flAverageYaw;
		}

		// Suppress tiny or low-confidence yaw to prevent left/right drift on straight movement
		const float kMinYawApply = 0.06f; // deg/tick (~4 deg/sec)
		if (fabsf(yawStep) < kMinYawApply || tStorage.m_flAverageYawConfidence < 0.35f)
			yawStep = 0.f;

		if (yawStep)
		{
			// Apply raw yaw per tick (no clamping or scaling) for maximum fidelity
			tStorage.m_MoveData.m_vecViewAngles.y += yawStep;

			auto applyAdaptiveSidemove = [&](float desiredSign, bool groundMode, bool scheduled)
			{
				Vec3 vel2D = tStorage.m_MoveData.m_vecVelocity;
				vel2D.z = 0.f;
				float speed = vel2D.Length2D();
				float maxSpeed = std::max(tStorage.m_MoveData.m_flMaxSpeed, 1.f);
				float speedScale = std::clamp(speed / maxSpeed, 0.f, 1.25f);

				float yawMag = fabsf(yawStep);
				float refYaw = std::max(tStorage.m_flYawAbs, 0.08f);
				float yawScale = std::clamp(yawMag / refYaw, 0.35f, 1.35f);

				float baseMagnitude = 450.f * yawScale * (0.55f + 0.45f * speedScale);
				if (!groundMode)
					baseMagnitude = std::min(baseMagnitude, 380.f);
				if (scheduled)
					baseMagnitude *= 1.05f;
				else if (groundMode)
					baseMagnitude *= 0.90f;

				float minMagnitude = groundMode ? 140.f : 110.f;
				baseMagnitude = std::clamp(baseMagnitude, minMagnitude, groundMode ? 450.f : 380.f);

				float targetSide = baseMagnitude * (desiredSign >= 0.f ? 1.f : -1.f);

				if (fabsf(tStorage.m_MoveData.m_flSideMove) < minMagnitude * 0.5f)
					tStorage.m_MoveData.m_flSideMove = targetSide;
				else
					tStorage.m_MoveData.m_flSideMove = Math::Lerp(tStorage.m_MoveData.m_flSideMove, targetSide, 0.5f);
			};

			if (tStorage.m_bDirectMove)
			{
				if (bScheduledThisTick)
				{
					float sign = (tStorage.m_iYawSign >= 0 ? 1.f : -1.f);
					applyAdaptiveSidemove(sign, true, true);
				}
				else if (fabsf(yawStep) >= 0.04f)
				{
					float sign = (yawStep >= 0.f ? 1.f : -1.f);
					applyAdaptiveSidemove(sign, true, false);
				}
			}
			else if (fabsf(yawStep) >= 0.04f)
			{
				float sign = (yawStep >= 0.f ? 1.f : -1.f);
				applyAdaptiveSidemove(sign, false, bScheduledThisTick);
			}

			DetectLedgeAndClamp(tStorage, yawStep);
		}
	}

    DetectLedgeAndClamp(tStorage, 0.f);

	float flOldSpeed = tStorage.m_MoveData.m_flClientMaxSpeed;
	if (tStorage.m_pPlayer->m_bDucked() && tStorage.m_pPlayer->IsOnGround() && !tStorage.m_pPlayer->IsSwimming())
		tStorage.m_MoveData.m_flClientMaxSpeed /= 3;

	if (tStorage.m_bBunnyHop && tStorage.m_pPlayer->IsOnGround() && !tStorage.m_pPlayer->m_bDucked())
	{
		tStorage.m_MoveData.m_nOldButtons = 0;
		tStorage.m_MoveData.m_nButtons |= IN_JUMP;
	}

	I::GameMovement->ProcessMovement(tStorage.m_pPlayer, &tStorage.m_MoveData);
	if (pCallback)
		(*pCallback)(tStorage.m_MoveData);

	tStorage.m_MoveData.m_flClientMaxSpeed = flOldSpeed;

	tStorage.m_flSimTime += TICK_INTERVAL;
	tStorage.m_bPredictNetworked = tStorage.m_flSimTime >= tStorage.m_flPredictedSimTime;
	if (tStorage.m_bPredictNetworked)
	{
		tStorage.m_vPredictedOrigin = tStorage.m_MoveData.m_vecAbsOrigin;
		    tStorage.m_flPredictedSimTime += tStorage.m_flPredictedDelta;
    }
    bool bLastbDirectMove = tStorage.m_bDirectMove;
    tStorage.m_bDirectMove = tStorage.m_pPlayer->IsOnGround() || tStorage.m_pPlayer->IsSwimming();

    if (!tStorage.m_flAverageYaw
        && tStorage.m_bDirectMove && !bLastbDirectMove
        && !tStorage.m_MoveData.m_flForwardMove && !tStorage.m_MoveData.m_flSideMove
        && tStorage.m_MoveData.m_vecVelocity.Length2D() > tStorage.m_MoveData.m_flMaxSpeed * 0.015f)
    {
        Vec3 vDirection = tStorage.m_MoveData.m_vecVelocity.Normalized2D() * 450.f;
        s_tDummyCmd.forwardmove = vDirection.x, s_tDummyCmd.sidemove = -vDirection.y;
        SDK::FixMovement(&s_tDummyCmd, {}, tStorage.m_MoveData.m_vecViewAngles);
        tStorage.m_MoveData.m_flForwardMove = s_tDummyCmd.forwardmove, tStorage.m_MoveData.m_flSideMove = s_tDummyCmd.sidemove;
    }

    RestoreBounds(tStorage.m_pPlayer);
}

bool CMovementSimulation::CounterStrafePrediction(PlayerStorage& tStorage, int iSamples)
{
    if (!tStorage.m_pPlayer)
        return false;

    if (tStorage.m_pPlayer->InCond(TF_COND_SHIELD_CHARGE))
    {
        tStorage.m_iYawSign = 0;
        tStorage.m_iStrafePeriod = 0;
        tStorage.m_iTicksToFlip = -1;
        tStorage.m_flCounterStrafeConfidence = 0.f;
        tStorage.m_flYawAbs = fabsf(tStorage.m_flAverageYaw);
        return false;
    }

    auto pPlayer = tStorage.m_pPlayer;
    auto& recs = m_mRecords[pPlayer->entindex()];
    if ((int)recs.size() < 3)
        return false;

    iSamples = std::min(iSamples, (int)recs.size());

    int targetMode = recs[0].m_iMode;
    if (targetMode != 0 && targetMode != 1)
    {
        tStorage.m_iYawSign = 0;
        tStorage.m_iStrafePeriod = 0;
        tStorage.m_iTicksToFlip = -1;
        tStorage.m_flCounterStrafeConfidence = 0.f;
        tStorage.m_flYawAbs = fabsf(tStorage.m_flAverageYaw);
        return false;
    }

    	struct Seg
	{
		int   sign;
		float span;
		float magAvg;
		float weight;
		float speedAvg;
	};
	std::vector<Seg> segs;
	segs.reserve(8);

	int curSign = 0;
	float curSpan = 0.f;
	float magWeighted = 0.f;
	float magWeight = 0.f;
	float speedWeighted = 0.f;
	const float kMinYawPerTick = 0.085f; // ignore micro jitter but stay sensitive
	float idleSpan = 0.f;
	float zeroFillSpan = 0.f;
	float totalSpan = 0.f;

	for (int i = 1; i < iSamples; ++i)
	{
		const auto& newer = recs[i - 1];
		const auto& older = recs[i];
		if (newer.m_iMode != older.m_iMode || newer.m_iMode != targetMode)
		{
			break;
		}

		float simDelta = std::max(newer.m_flSimTime - older.m_flSimTime, TICK_INTERVAL);
		float span = std::max(simDelta / TICK_INTERVAL, 1.f);
		totalSpan += span;

		float yaw1 = Math::VectorAngles(newer.m_vDirection).y;
		float yaw2 = Math::VectorAngles(older.m_vDirection).y;
		float dyaw = Math::NormalizeAngle(yaw1 - yaw2);
		float perTick = dyaw / span;
		float absPerTick = fabsf(perTick);
		int s = (absPerTick >= kMinYawPerTick) ? (perTick >= 0.f ? +1 : -1) : 0;

		const Vec3 avgVel = (newer.m_vVelocity + older.m_vVelocity) * 0.5f;
		float speed = avgVel.Length2D();

		if (curSign == 0)
		{
			if (s == 0)
			{
				idleSpan += span;
				continue;
			}
			curSign = s;
			curSpan = span;
			magWeighted = absPerTick * span;
			magWeight = span;
			speedWeighted = speed * span;
			continue;
		}

		if (s == 0)
		{
			curSpan += span;
			magWeight += span;
			speedWeighted += speed * span;
			zeroFillSpan += span;
			continue;
		}

		if (s == curSign)
		{
			curSpan += span;
			magWeighted += absPerTick * span;
			magWeight += span;
			speedWeighted += speed * span;
			continue;
		}

		if (curSpan > 0.f)
		{
			float avgMag = magWeight > 1e-4f ? (magWeighted / magWeight) : 0.f;
			float avgSpeed = magWeight > 1e-4f ? (speedWeighted / magWeight) : 0.f;
			segs.push_back({ curSign, curSpan, avgMag, magWeight, avgSpeed });
		}
		curSign = s;
		curSpan = span;
		magWeighted = absPerTick * span;
		magWeight = span;
		speedWeighted = speed * span;
	}

	if (curSign != 0 && curSpan > 0.f)
	{
		float avgMag = magWeight > 1e-4f ? (magWeighted / magWeight) : 0.f;
		float avgSpeed = magWeight > 1e-4f ? (speedWeighted / magWeight) : 0.f;
		segs.push_back({ curSign, curSpan, avgMag, magWeight, avgSpeed });
	}

	if (segs.size() < 3)
		return false;

	float recentSpan = segs[0].span;

	int useCount = std::min<int>((int)segs.size() - 1, 5);
	if (useCount <= 0)
		return false;

	float sumSpan = 0.f, sumSpan2 = 0.f;
	float sumMag = 0.f, sumMag2 = 0.f, sumMagWeight = 0.f;
	float sumSpeed = 0.f, sumSpeed2 = 0.f, sumSpeedWeight = 0.f;
	for (int k = 1; k <= useCount; ++k)
	{
		const auto& seg = segs[k];
		sumSpan += seg.span;
		sumSpan2 += seg.span * seg.span;
		sumMag += seg.magAvg * seg.weight;
		sumMag2 += seg.magAvg * seg.magAvg * seg.weight;
		sumMagWeight += seg.weight;
		sumSpeed += seg.speedAvg * seg.weight;
		sumSpeed2 += seg.speedAvg * seg.speedAvg * seg.weight;
		sumSpeedWeight += seg.weight;
	}

	float avgSpan = sumSpan / useCount;
	float varSpan = std::max(0.f, (sumSpan2 / useCount) - (avgSpan * avgSpan));
	float sdSpan = sqrtf(varSpan);

	float avgMag = sumMagWeight > 1e-4f ? (sumMag / sumMagWeight) : 0.f;
	float varMag = sumMagWeight > 1e-4f ? std::max(0.f, (sumMag2 / sumMagWeight) - (avgMag * avgMag)) : 0.f;
	float sdMag = sqrtf(varMag);

	float avgSpeed = sumSpeedWeight > 1e-4f ? (sumSpeed / sumSpeedWeight) : 0.f;
	float varSpeed = sumSpeedWeight > 1e-4f ? std::max(0.f, (sumSpeed2 / sumSpeedWeight) - (avgSpeed * avgSpeed)) : 0.f;
	float sdSpeed = sqrtf(varSpeed);

	float periodSpan = std::clamp(avgSpan, 2.f, 48.f);
	int period = std::clamp((int)std::round(periodSpan), 2, 32);
	float remainingSpan = std::max(periodSpan - recentSpan, 0.5f);
	int ticksToFlip = std::clamp((int)std::round(remainingSpan), 1, 32);

	float fillRatio = 1.f - std::clamp(zeroFillSpan / std::max(totalSpan, 1.f), 0.f, 1.f);
	float idleRatio = 1.f - std::clamp(idleSpan / std::max(totalSpan, 1.f), 0.f, 1.f);
	float speedFrac = 0.f;
	if (tStorage.m_MoveData.m_flMaxSpeed > 1.f)
		speedFrac = std::clamp(avgSpeed / tStorage.m_MoveData.m_flMaxSpeed, 0.f, 1.3f);

	float cDur = 1.f - std::clamp(sdSpan / std::max(avgSpan, 1.f), 0.f, 1.f);
	float cMag = 1.f - std::clamp(sdMag / std::max(avgMag, 0.05f), 0.f, 1.f);
	float cSpeed = 1.f - std::clamp(sdSpeed / std::max(avgSpeed, 1.f), 0.f, 1.f);
	float cFill = fillRatio;
	float cIdle = idleRatio;
	float baseConf = 0.32f * cDur + 0.24f * cMag + 0.16f * cSpeed + 0.16f * cFill + 0.12f * cIdle;
	baseConf *= std::clamp(0.7f + 0.3f * tStorage.m_flAverageYawConfidence, 0.55f, 1.1f);
	if (avgMag < 0.06f)
		baseConf *= std::clamp(avgMag / 0.06f, 0.f, 1.f);
	baseConf *= std::clamp(0.65f + 0.35f * speedFrac, 0.6f, 1.2f);
	baseConf = std::clamp(baseConf, 0.f, 1.f);

	float yawAbsCandidate = segs[0].magAvg > 1e-4f ? segs[0].magAvg : avgMag;
	yawAbsCandidate = std::max(yawAbsCandidate, fabsf(tStorage.m_flAverageYaw));
	yawAbsCandidate = std::clamp(yawAbsCandidate, 0.f, 12.f);
	tStorage.m_flYawAbs = yawAbsCandidate;
	int yawSign = segs[0].sign != 0 ? segs[0].sign : (tStorage.m_flAverageYaw >= 0.f ? +1 : -1);
	tStorage.m_iYawSign = yawSign;
	tStorage.m_iStrafePeriod = period;
	tStorage.m_iTicksToFlip = ticksToFlip;
	tStorage.m_flCounterStrafeConfidence = baseConf;

	if (Vars::Debug::Logging.Value)
	{
		std::ostringstream oss;
		oss << "CounterStrafe period=" << periodSpan
			<< " recent=" << recentSpan
			<< " avgMag=" << avgMag
			<< " fill=" << (1.f - fillRatio)
			<< " idle=" << (1.f - idleRatio)
			<< " conf=" << baseConf;
		SDK::Output("MovementSimulation", oss.str().c_str(), { 120, 170, 220 }, Vars::Debug::Logging.Value);
	}

	return baseConf > 0.5f;
}

void CMovementSimulation::RunTick(PlayerStorage& tStorage, bool bPath, std::function<void(CMoveData&)> fCallback)
{
	RunTick(tStorage, bPath, &fCallback);
}

void CMovementSimulation::Restore(PlayerStorage& tStorage)
{
	if (tStorage.m_bInitFailed || !tStorage.m_pPlayer)
		return;

	I::MoveHelper->SetHost(nullptr);
	tStorage.m_pPlayer->m_pCurrentCommand() = nullptr;

	Reset(tStorage);

	I::Prediction->m_bInPrediction = m_bOldInPrediction;
	I::Prediction->m_bFirstTimePredicted = m_bOldFirstTimePredicted;
	I::GlobalVars->frametime = m_flOldFrametime;

	/*
	const bool bInitFailed = tStorage.m_bInitFailed, bFailed = tStorage.m_bFailed;
	memset(&tStorage, 0, sizeof(PlayerStorage));
	tStorage.m_bInitFailed = bInitFailed, tStorage.m_bFailed = bFailed;
	*/
}

float CMovementSimulation::GetPredictedDelta(CBaseEntity* pEntity)
{
    auto& vSimTimes = m_mSimTimes[pEntity->entindex()];
    // use latest observed network delta
    float raw = vSimTimes.empty() ? TICK_INTERVAL : vSimTimes.front();
    // do not upper-clamp; honor actual observed delta (minimum one tick)
    return std::max(raw, TICK_INTERVAL);
}

// store per-player state so we can non-destructively simulate and then restore
void CMovementSimulation::Store(PlayerStorage& tStorage)
{
	if (!tStorage.m_pPlayer) return;
	auto p = tStorage.m_pPlayer;
	auto& d = tStorage.m_PlayerData;
	d.m_vecOrigin = p->m_vecOrigin();
	d.m_vecVelocity = p->m_vecVelocity();
	d.m_vecBaseVelocity = p->m_vecBaseVelocity();
	d.m_vecViewOffset = p->m_vecViewOffset();
	d.m_hGroundEntity = p->m_hGroundEntity();
	d.m_fFlags = p->m_fFlags();
	d.m_flDucktime = p->m_flDucktime();
	d.m_flDuckJumpTime = p->m_flDuckJumpTime();
	d.m_bDucked = p->m_bDucked();
	d.m_bDucking = p->m_bDucking();
	d.m_bInDuckJump = p->m_bInDuckJump();
	d.m_flModelScale = p->m_flModelScale();
	d.m_nButtons = p->m_nButtons();
	d.m_flMaxspeed = p->m_flMaxspeed();
	d.m_flFallVelocity = p->m_flFallVelocity();
	d.m_flGravity = p->m_flGravity();
	d.m_nWaterLevel = p->m_nWaterLevel();
	d.m_nWaterType = p->m_nWaterType();
}

// restore stored state
void CMovementSimulation::Reset(PlayerStorage& tStorage)
{
	if (!tStorage.m_pPlayer) return;
	auto p = tStorage.m_pPlayer;
	const auto& d = tStorage.m_PlayerData;
	p->m_vecOrigin() = d.m_vecOrigin;
	p->m_vecVelocity() = d.m_vecVelocity;
	p->m_vecBaseVelocity() = d.m_vecBaseVelocity;
	p->m_vecViewOffset() = d.m_vecViewOffset;
	p->m_hGroundEntity() = d.m_hGroundEntity;
	p->m_fFlags() = d.m_fFlags;
	p->m_flDucktime() = d.m_flDucktime;
	p->m_flDuckJumpTime() = d.m_flDuckJumpTime;
	p->m_bDucked() = d.m_bDucked;
	p->m_bDucking() = d.m_bDucking;
	p->m_bInDuckJump() = d.m_bInDuckJump;
	p->m_flModelScale() = d.m_flModelScale;
	p->m_nButtons() = d.m_nButtons;
	p->m_flMaxspeed() = d.m_flMaxspeed;
	p->m_flFallVelocity() = d.m_flFallVelocity;
	p->m_flGravity() = d.m_flGravity;
	p->m_nWaterLevel() = d.m_nWaterLevel;
	p->m_nWaterType() = d.m_nWaterType;
}