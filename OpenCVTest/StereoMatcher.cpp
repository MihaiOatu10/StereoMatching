#include "StereoMatcher.h"

StereoMatcher::StereoMatcher(int maxDisparity) : d_max(maxDisparity)
{
	for (int d = 0; d <= Config::MAX_DEPTH; d++) {
		edgeLengths.push_back((float)Config::HEIGHT / (1 << d));
	}
	for (int d = 0; d < 20; d++) {
		penaltyTableVert[d] = (float)Config::SMOOTHNESS_LAMBDA * (Config::HEIGHT / (float)(1 << d));
		penaltyTableHoriz[d] = (float)Config::SMOOTHNESS_LAMBDA * (Config::WIDTH / (float)(1 << d));
	}
}

void StereoMatcher::populateSpace(const cv::Mat& ImgL, const cv::Mat& ImgR)
{
	costVolume.clear();
	costVolume.resize(d_max);
	integralCosts.clear();
	integralCosts.resize(d_max);

	int w = Config::WINDOW_RADIUS;
	float OCCLUSION_COST = (float)((2 * w + 1) * (2 * w + 1) * 255.0f);

	for (int d = 0; d < d_max; d++) {
		cv::Mat cost = cv::Mat(ImgL.size(), CV_32F, cv::Scalar(OCCLUSION_COST));

		for (int y = w; y < ImgL.rows - w; y++) {
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
				cost.at<float>(y, x) = windowSum;
			}
		}

		costVolume[d] = cost;

		cv::Mat integral;
		cv::integral(cost, integral, CV_32F);
		integralCosts[d] = integral;
	}
}

