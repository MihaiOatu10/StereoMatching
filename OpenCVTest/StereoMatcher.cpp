#include "StereoMatcher.h"
#include "LinearQuadtree.h"

void StereoMatcher::populateSpace(const cv::Mat& ImgL, const cv::Mat& ImgR)
{
	costVolume.clear();
	costVolume.resize(d_max);
	int w = Config::WINDOW_RADIUS;
	float OCCLUSION_COST = (float)((2 * w + 1) * (2 * w + 1) * 255.0f);

	for (int d = 0; d < d_max; d++) {
		costVolume[d] = cv::Mat(ImgL.size(), CV_32F, cv::Scalar(OCCLUSION_COST));

		for (int y = w; y < ImgL.rows - w; y++) {
			const uchar* rowL = ImgL.ptr<uchar>(y);
			for (int x = w; x < ImgL.cols - w; x++) {

				if (x - d - w < 0) continue;

				float windowSum = 0.0f;
				for (int i = -w; i <= w; i++) {
					const uchar* pL = ImgL.ptr<uchar>(y + i) + x;
					const uchar* pR = ImgR.ptr<uchar>(y + i) + (x - d);
					for (int j = -w; j <= w; j++) {
						windowSum += std::abs(pL[j] - pR[j]);
					}
				}
				costVolume[d].at<float>(y, x) = windowSum;
			}
		}
	}
}

float StereoMatcher::calculateFitness(LinearQuadtree& qt)
{
	float totalDataError = computeDataTerm(qt, 0, 0, 0);
	float totalBoundaryLength = computeSmoothnessTerm(qt, 0, 0, 0);

	float fitness = totalDataError + (Config::SMOOTHNESS_LAMBDA * totalBoundaryLength * Config::MRF_SYMMETRY_FACTOR);

	qt.setFitness(fitness);
	return fitness;
}

float StereoMatcher::calculateBorderPenalty(LinearQuadtree& qt, int d1, int x1, int y1, int d2, int x2, int y2, int direction) {
	int k1 = qt.getIndex(d1, x1, y1);
	int k2 = qt.getIndex(d2, x2, y2);
	int v1 = qt.getGene(k1);
	int v2 = qt.getGene(k2);

	if (v1 != -1 && v2 != -1) {
		if (v1 != v2) {
			int depth = std::max(d1, d2);
			if (direction == 0) {
				return (float)(Config::HEIGHT / (1 << depth));
			}
			else {
				return (float)(Config::WIDTH / (1 << depth));
			}
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
			return calculateBorderPenalty(qt, d1, x1, y1, d2 + 1, x2 * 2 + 1, y2 * 2, direction) +
				calculateBorderPenalty(qt, d1, x1, y1, d2 + 1, x2 * 2 + 1, y2 * 2 + 1, direction);
		}
	}
	return 0.0f;
}

float StereoMatcher::computeSmoothnessTerm(LinearQuadtree& qt, int depth, int x, int y)
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

float StereoMatcher::computeDataTerm(LinearQuadtree& qt, int x, int y, int depth)
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
		cv::Rect region = qt.getRect(depth, x, y);
		int disparity = std::max(0, std::min(d, d_max - 1));
		cv::Scalar sumCost = cv::sum(costVolume[disparity](region));
		return (float)sumCost[0];
	}
}
