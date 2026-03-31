#ifndef EVALUATION_H
#define EVALUATION_H
#include <opencv2/opencv.hpp>
#include <vector>

namespace Evaluation {
    double evaluateDisparityMap_G(const cv::Mat& disparityMap, const std::vector<cv::Mat>& costVolume);
}
#endif