inline float StereoMatcher::calculateFitness(LinearQuadtree& qt)
{
	long long lambda = static_cast<long long>(Config::SMOOTHNESS_LAMBDA * 1000.0f);

	float totalDataError = computeDataTerm(qt, 0, 0, 0);
	float totalBoundaryLength = computeSmoothnessTerm(qt, 0, 0, 0);

	float fitness = totalDataError + (Config::SMOOTHNESS_LAMBDA * totalBoundaryLength);

	qt.setFitness(fitness);
	return fitness;
}

inline float StereoMatcher::calculateBorderPenalty(LinearQuadtree& qt, int d1, int x1, int y1, int d2, int x2, int y2, int direction) {
	int v1 = qt.getGene(qt.getIndex(d1, x1, y1));
	int v2 = qt.getGene(qt.getIndex(d2, x2, y2));

	if (v1 != -1 && v2 != -1) {
		if (v1 != v2) {
			return (direction == 0) ? penaltyTableVert[std::max(d1, d2)]
				: penaltyTableHoriz[std::max(d1, d2)];
		}
		return 0.0f;
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

	return 0.0f;
}

inline float StereoMatcher::computeSmoothnessTerm(LinearQuadtree& qt, int depth, int x, int y)
{
	int k = qt.getIndex(depth, x, y);
	if (qt.getGene(k) != -1) return 0.0f;

	float penalty = 0.0f;

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

inline float StereoMatcher::computeDataTerm(LinearQuadtree& qt, int x, int y, int depth)
{
	int k = qt.getIndex(depth, x, y);
	int d = qt.getGene(k);

	if (d == -1 && depth < qt.getMaxDepth())
	{
		return computeDataTerm(qt, x * 2, y * 2, depth + 1) +
			computeDataTerm(qt, x * 2 + 1, y * 2, depth + 1) +
			computeDataTerm(qt, x * 2, y * 2 + 1, depth + 1) +
			computeDataTerm(qt, x * 2 + 1, y * 2 + 1, depth + 1);
	}
	else
	{
		cv::Rect r = qt.getRect(depth, x, y);
		int disp = std::max(0, std::min(d, d_max - 1));

		const cv::Mat& sat = integralCosts[disp];

		int x1 = r.x;
		int y1 = r.y;
		int x2 = r.x + r.width;
		int y2 = r.y + r.height;

		float sum = sat.at<float>(y2, x2) - sat.at<float>(y1, x2) - sat.at<float>(y2, x1) + sat.at<float>(y1, x1);

		return sum;
	}
}