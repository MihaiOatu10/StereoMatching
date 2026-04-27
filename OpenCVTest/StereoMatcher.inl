#include "StereoMatcher.h"
inline float StereoMatcher::calculateFitness(LinearQuadtree& qt)
{
	long long totalDataError = computeDataTerm(qt, 0, 0, 0, 0, 0, Config::WIDTH, Config::HEIGHT);
	long long totalBoundaryLength = computeSmoothnessTerm(qt, 0, 0, 0);

	float fitness = (float)totalDataError + (Config::SMOOTHNESS_LAMBDA * (float)totalBoundaryLength);

	qt.setFitness(fitness);
	return fitness;
}

inline float StereoMatcher::calculateFitness(LinearQuadtree& qt, NodeLocation k)
{
	int nodeW = Config::WIDTH / (1 << k.depth);
	int nodeH = Config::HEIGHT / (1 << k.depth);

	long long dataCost = computeDataTerm(qt, k.x, k.y, k.depth, k.x * nodeW, k.y * nodeH, nodeW, nodeH);

	long long boundaryCost = 0;

	int px = k.x * nodeW;
	int py = k.y * nodeH;

	if (py > 0)
	{
		NodeLocation up = qt.findLeafAt(px + nodeW / 2, py - 1);
		boundaryCost += calculateBorderPenalty(qt, k.depth, k.x, k.y, up.depth, up.x, up.y, 1);
	}
	if (py + nodeH < Config::HEIGHT) {
		NodeLocation down = qt.findLeafAt(px + nodeW / 2, py + nodeH);
		boundaryCost += calculateBorderPenalty(qt, k.depth, k.x, k.y, down.depth, down.x, down.y, 1);
	}
	if (px > 0) {
		NodeLocation left = qt.findLeafAt(px - 1, py + nodeH / 2);
		boundaryCost += calculateBorderPenalty(qt, k.depth, k.x, k.y, left.depth, left.x, left.y, 0);
	}
	if (px + nodeW < Config::WIDTH) {
		NodeLocation right = qt.findLeafAt(px + nodeW, py + nodeH / 2);
		boundaryCost += calculateBorderPenalty(qt, k.depth, k.x, k.y, right.depth, right.x, right.y, 0);
	}
	boundaryCost += computeInternalSmoothness(qt, k.depth, k.x, k.y);

	return (float)dataCost + (Config::SMOOTHNESS_LAMBDA * (float)boundaryCost);
}

inline long long StereoMatcher::computeInternalSmoothness(LinearQuadtree& qt, int depth, int x, int y)
{
	if (depth >= qt.getMaxDepth()) return 0LL;

	int k = qt.getIndex(depth, x, y);

	if (qt.getGene(k) != -1) return 0LL;

	long long penalty = 0LL;

	penalty += calculateBorderPenalty(qt, depth + 1, x * 2, y * 2, depth + 1, x * 2 + 1, y * 2, 0);
	penalty += calculateBorderPenalty(qt, depth + 1, x * 2, y * 2 + 1, depth + 1, x * 2 + 1, y * 2 + 1, 0);
	penalty += calculateBorderPenalty(qt, depth + 1, x * 2, y * 2, depth + 1, x * 2, y * 2 + 1, 1);
	penalty += calculateBorderPenalty(qt, depth + 1, x * 2 + 1, y * 2, depth + 1, x * 2 + 1, y * 2 + 1, 1);

	penalty += computeInternalSmoothness(qt, depth + 1, x * 2, y * 2);
	penalty += computeInternalSmoothness(qt, depth + 1, x * 2 + 1, y * 2);
	penalty += computeInternalSmoothness(qt, depth + 1, x * 2, y * 2 + 1);
	penalty += computeInternalSmoothness(qt, depth + 1, x * 2 + 1, y * 2 + 1);

	return penalty;
}

inline long long StereoMatcher::calculateBorderPenalty(LinearQuadtree& qt, int d1, int x1, int y1, int d2, int x2, int y2, int direction) {

	if (d1 > qt.getMaxDepth() || d2 > qt.getMaxDepth()) return 0LL;

	int idx1 = qt.getIndex(d1, x1, y1);
	int idx2 = qt.getIndex(d2, x2, y2);

	if (idx1 >= qt.getD().size() || idx2 >= qt.getD().size()) return 0LL;

	int v1 = qt.getGene(qt.getIndex(d1, x1, y1));
	int v2 = qt.getGene(qt.getIndex(d2, x2, y2));

	if (v1 != -1 && v2 != -1) {
		if (v1 != v2) {
			long long basePenalty = (direction == 0) ?
				penaltyTableHoriz[std::min(d1, d2)] :
				penaltyTableVert[std::min(d1, d2)];
			return basePenalty;
		}
		return 0LL;
	}

	if (v1 == -1) {
		if (direction == 0) {
			return calculateBorderPenalty(qt, d1 + 1, x1 * 2 + 1, y1 * 2, d2, x2, y2, direction) +
				calculateBorderPenalty(qt, d1 + 1, x1 * 2 + 1, y1 * 2 + 1, d2, x2, y2, direction);
		}
		else {
			return calculateBorderPenalty(qt, d1 + 1, x1 * 2, y1 * 2 + 1, d2, x2, y2, direction) +
				calculateBorderPenalty(qt, d1 + 1, x1 * 2 + 1, y1 * 2 + 1, d2, x2, y2, direction);
		}
	}

	if (v2 == -1) {
		if (direction == 0) {
			return calculateBorderPenalty(qt, d1, x1, y1, d2 + 1, x2 * 2, y2 * 2, direction) +
				calculateBorderPenalty(qt, d1, x1, y1, d2 + 1, x2 * 2, y2 * 2 + 1, direction);
		}
		else {
			return calculateBorderPenalty(qt, d1, x1, y1, d2 + 1, x2 * 2, y2 * 2, direction) +
				calculateBorderPenalty(qt, d1, x1, y1, d2 + 1, x2 * 2 + 1, y2 * 2, direction);
		}
	}

	return 0LL; 
}

inline long long StereoMatcher::computeSmoothnessTerm(LinearQuadtree& qt, int depth, int x, int y)
{
	int k = qt.getIndex(depth, x, y);
	if (qt.getGene(k) != -1) return 0LL;

	long long penalty = 0LL;

	penalty += computeSmoothnessTerm(qt, depth + 1, x * 2, y * 2);
	penalty += computeSmoothnessTerm(qt, depth + 1, x * 2 + 1, y * 2);
	penalty += computeSmoothnessTerm(qt, depth + 1, x * 2, y * 2 + 1);
	penalty += computeSmoothnessTerm(qt, depth + 1, x * 2 + 1, y * 2 + 1);

	penalty += calculateBorderPenalty(qt, depth + 1, x * 2, y * 2, depth + 1, x * 2 + 1, y * 2, 0);
	penalty += calculateBorderPenalty(qt, depth + 1, x * 2, y * 2 + 1, depth + 1, x * 2 + 1, y * 2 + 1, 0);

	penalty += calculateBorderPenalty(qt, depth + 1, x * 2, y * 2, depth + 1, x * 2, y * 2 + 1, 1);
	penalty += calculateBorderPenalty(qt, depth + 1, x * 2 + 1, y * 2, depth + 1, x * 2 + 1, y * 2 + 1, 1);

	return penalty;
}

inline long long StereoMatcher::computeDataTerm(LinearQuadtree& qt, int x, int y, int depth, int px, int py, int width, int height)
{
	int k = qt.getIndex(depth, x, y);
	int d = qt.getGene(k);

	if (d == -1 && depth < qt.getMaxDepth())
	{
		int halfW = width >> 1;
		int halfH = height >> 1;

		return computeDataTerm(qt, x * 2, y * 2, depth + 1, px, py, halfW, halfH) +
			computeDataTerm(qt, x * 2 + 1, y * 2, depth + 1, px + halfW, py, halfW, halfH) +
			computeDataTerm(qt, x * 2, y * 2 + 1, depth + 1, px, py + halfH, halfW, halfH) +
			computeDataTerm(qt, x * 2 + 1, y * 2 + 1, depth + 1, px + halfW, py + halfH, halfW, halfH);
	}
	else
	{
		int disp = std::max(Config::MIN_DISPARITY, std::min(d, Config::MAX_DISPARITY - 1));
		int idx = disp - Config::MIN_DISPARITY;
		const cv::Mat& sat = integralCosts[idx];
		const int* ptr = sat.ptr<int>();
		int step = sat.step / sizeof(int);
		int x1 = px;
		int y1 = py;
		int x2 = px + width;
		int y2 = py + height;
		long long sum = (long long)ptr[y2 * step + x2]
			- (long long)ptr[y1 * step + x2]
			- (long long)ptr[y2 * step + x1]
			+ (long long)ptr[y1 * step + x1];
		return sum;
	}
}

inline int StereoMatcher::getBestLocalDisparity(int depth, int x, int y)
{
	int nodeW = Config::WIDTH / (1 << depth);
	int nodeH = Config::HEIGHT / (1 << depth);
	int x0 = x * nodeW;
	int y0 = y * nodeH;
	int x1 = x0 + nodeW;
	int y1 = y0 + nodeH;

	int bestD = Config::MIN_DISPARITY;
	long minCost = LONG_MAX;

	for (int d = Config::MIN_DISPARITY; d < Config::MAX_DISPARITY; d++) {
		int idx = d - Config::MIN_DISPARITY;
		const cv::Mat& sumMat = integralCosts[idx];
		long currentCost = sumMat.at<int>(y1, x1)
			- sumMat.at<int>(y1, x0)
			- sumMat.at<int>(y0, x1)
			+ sumMat.at<int>(y0, x0);
		if (currentCost < minCost) {
			minCost = currentCost;
			bestD = d;
		}
	}
	return bestD;
}