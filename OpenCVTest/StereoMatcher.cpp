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

    for (int d = 0; d < d_max; d++) {
		cv::Mat cost = cv::Mat::zeros(ImgR.size(), CV_32F); 

        for(int y = w; y < ImgR.rows - w; y++) {
            for(int x = w; x < ImgR.cols - w; x++) {
                if (x + d + w < ImgL.cols) {
                    float windowSum = 0.0f;
                    
                    for (int i = -w; i <= w; i++) {
                        const uchar* pL = ImgL.ptr<uchar>(y + i);
                        const uchar* pR = ImgR.ptr<uchar>(y + i);
                        
                        for (int j = -w; j <= w; j++) {
                            windowSum += std::abs((float)pL[x + j + d] - (float)pR[x + j]);
                        }
                    }
					cost.at<float>(y, x) = windowSum / 9.0f;
                }
            }
		}

        costVolume[d] = cost;

        cv::Mat integralFloat;
        cv::integral(cost, integralFloat, CV_32F);

        cv::Mat integral;
        integralFloat.convertTo(integral, CV_32S);

        integralCosts[d] = integral;
    }
}

