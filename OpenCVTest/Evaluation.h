#ifndef EVALUATION_H
#define EVALUATION_H
#include <opencv2/opencv.hpp>
#include <vector>

namespace Evaluation {
    struct EvaluationResult {
        float correctPercentage = 0.0f;
        float correctWithin1Percentage = 0.0f;
        int totalPixels = 0;
    };

    double evaluateDisparityMap_G(const cv::Mat& disparityMap, const std::vector<cv::Mat>& costVolume);
    EvaluationResult evaluateWithGroundTruth(const cv::Mat& disparityMap, const cv::Mat& groundTruth);
}
#endif