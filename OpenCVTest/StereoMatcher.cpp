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

void StereoMatcher::populateSpace(const std::vector<cv::Mat>& images, int centerIdx)
{
    const cv::Mat& imgCenter = images[centerIdx];
    int numDisp = Config::MAX_DISPARITY - Config::MIN_DISPARITY;

    costVolume.clear();
    costVolume.resize(numDisp);
    integralCosts.clear();
    integralCosts.resize(numDisp);

    int w = Config::WINDOW_RADIUS;

    std::vector<int> surroundingIdx;
    for (int i = 0; i < (int)images.size(); i++) {
        if (i != centerIdx) surroundingIdx.push_back(i);
    }

    for (int d = Config::MIN_DISPARITY; d < Config::MAX_DISPARITY; d++) {
        int idx = d - Config::MIN_DISPARITY;
        cv::Mat cost = cv::Mat::zeros(imgCenter.size(), CV_32F);

        for (int y = w; y < imgCenter.rows - w; y++) {
            for (int x = w; x < imgCenter.cols - w; x++) {

                std::vector<float> pairCosts;

                for (int camIdx : surroundingIdx) {
                    const cv::Mat& imgOther = images[camIdx];
                    float windowSum = 0.0f;
                    bool valid = true;

                    for (int i = -w; i <= w; i++) {
                        for (int j = -w; j <= w; j++) {
                            int xOther = x + j - d;
                            int yOther = y + i;

                            if (xOther < 0 || xOther >= imgOther.cols ||
                                yOther < 0 || yOther >= imgOther.rows) {
                                valid = false;
                                break;
                            }
                            windowSum += std::abs(
                                (float)imgCenter.at<uchar>(y + i, x + j) -
                                (float)imgOther.at<uchar>(yOther, xOther)
                            );
                        }
                        if (!valid) break;
                    }

                    if (valid) pairCosts.push_back(windowSum);
                }

                if (!pairCosts.empty()) {
                    std::sort(pairCosts.begin(), pairCosts.end());
                    int numToSum = std::max(1, (int)pairCosts.size() / 2);
                    float sortedSum = 0.0f;
                    for (int i = 0; i < numToSum; i++) {
                        sortedSum += pairCosts[i];
                    }
                    cost.at<float>(y, x) = sortedSum;
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

