#include "StereoMatcher.h"

StereoMatcher::StereoMatcher(int maxDisparity) : d_max(maxDisparity)
{
	for (int d = 0; d <= Config::MAX_DEPTH; d++) {
        edgeLengths.push_back((float)Config::HEIGHT / (1 << d));
	}
	for (int d = 0; d < 20; d++) {
		penaltyTableVert[d] = (Config::HEIGHT / (float)(1 << d));
		penaltyTableHoriz[d] = (Config::WIDTH / (float)(1 << d));
	}
}

void StereoMatcher::populateSpace(const cv::Mat& ImgL, const cv::Mat& ImgR)
{
    int minDisp = Config::MIN_DISPARITY;
    int maxDisp = Config::MAX_DISPARITY;
    int numDisp = maxDisp - minDisp;

    costVolume.clear();
    costVolume.resize(numDisp);
    integralCosts.clear();
    integralCosts.resize(numDisp);
    int w = Config::WINDOW_RADIUS;

    for (int d = minDisp; d < maxDisp; d++) {
        int idx = d - minDisp;
        cv::Mat cost = cv::Mat::zeros(ImgR.size(), CV_32F);

        for (int y = w; y < ImgL.rows - w; y++) {
            for (int x = w + d; x < ImgL.cols - w; x++) {
                if (x - d - w >= 0) {
                    float windowSum = 0.0f;

                    for (int i = -w; i <= w; i++) {
                        const uchar* pL = ImgL.ptr<uchar>(y + i);
                        const uchar* pR = ImgR.ptr<uchar>(y + i);

                        for (int j = -w; j <= w; j++) {
                            windowSum += std::abs((float)pL[x + j] - (float)pR[x + j - d]);
                        }
                    }
                    cost.at<float>(y, x) = windowSum;
                }
            }
        }

        costVolume[idx] = cost;
        cv::Mat integralFloat;
        cv::integral(cost, integralFloat, CV_32F);
        cv::Mat integral;
        integralFloat.convertTo(integral, CV_32S);
        integralCosts[idx] = integral;
    }
}

