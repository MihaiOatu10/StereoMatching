#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "Common.h"
#include "LinearQuadtree.h"

class LinearQuadtree;

class StereoMatcher
{
private:
	int d_max;
	std::vector<cv::Mat> costVolume;
	std::vector<cv::Mat> integralCosts;
	std::vector<long long> edgeLengths;
	std::array<long long, 20> penaltyTableVert;
	std::array<long long, 20> penaltyTableHoriz;
public:
	struct Individual {
		std::vector<int> genes;
		float fitness = 0.0f;
	};

	StereoMatcher(int maxDisparity);

	void populateSpace(const cv::Mat& ImgL, const cv::Mat& ImgR);

	void populateSpace(const std::vector<cv::Mat>& images, int centerIdx);

	float calculateFitness(LinearQuadtree& qt);

	float calculateFitness(LinearQuadtree& qt, NodeLocation k);

	long long computeInternalSmoothness(LinearQuadtree& qt, int depth, int x, int y);

	long long calculateBorderPenalty(LinearQuadtree& qt, int d1, int x1, int y1, int d2, int x2, int y2, int direction);

	long long computeSmoothnessTerm(LinearQuadtree& qt, int depth, int x, int y);

	long long computeDataTerm(LinearQuadtree& qt, int x, int y, int depth, int px, int py, int width, int height);

	int getBestLocalDisparity(int depth, int x, int y);
	
	std::vector<cv::Mat>& getCostVolume() { return costVolume; }
};

#include "StereoMatcher.inl"
