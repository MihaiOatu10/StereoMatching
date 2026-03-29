#pragma once
#include<opencv2/opencv.hpp>
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
	std::vector<float> edgeLengths;
	std::array<float, 20> penaltyTableVert;
	std::array<float, 20> penaltyTableHoriz;
	const float lambda = Config::SMOOTHNESS_LAMBDA;

public:
	struct Individual {
		std::vector<int> genes;
		float fitness = 0.0f;
	};

	StereoMatcher(int maxDisparity);

	void populateSpace(const cv::Mat& ImgL, const cv::Mat& ImgR);

	float calculateFitness(LinearQuadtree& qt);

	float calculateBorderPenalty(LinearQuadtree& qt, int d1, int x1, int y1, int d2, int x2, int y2, int direction);

	float computeSmoothnessTerm(LinearQuadtree& qt, int depth, int x, int y);

	float computeDataTerm(LinearQuadtree& qt, int x, int y, int depth);
	
	std::vector<cv::Mat>& getCostVolume() { return costVolume; }
};

#include "StereoMatcher.inl"
