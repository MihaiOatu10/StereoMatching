#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include "Evaluation.h"

namespace Evaluation {
    double evaluateDisparityMap_G(const cv::Mat& disparityMap, const std::vector<cv::Mat>& costVolume) {
        int M = disparityMap.rows;
        int N = disparityMap.cols;
        int maxDisp = costVolume.size();

        int totalRegionsR = 0;
        double sumTerm = 0.0;

        cv::Mat dispMapInt;
        disparityMap.convertTo(dispMapInt, CV_32S);

        for (int d = 0; d < maxDisp; d++) {

            cv::Mat maskGE;
            cv::compare(dispMapInt, d, maskGE, cv::CMP_GE);

            cv::Mat labels;
            int numComponents = cv::connectedComponents(maskGE, labels, 4, CV_32S);

            std::vector<int> areaAi(numComponents, 0);
            std::vector<double> errorEi(numComponents, 0.0);

            for (int y = 0; y < M; y++) {
                for (int x = 0; x < N; x++) {

                    if (dispMapInt.at<int>(y, x) == d) {
                        int label = labels.at<int>(y, x);

                        if (label > 0) {
                            areaAi[label]++;
                            float S_cost = costVolume[d].at<float>(y, x);

                            errorEi[label] += (static_cast<double>(S_cost) * static_cast<double>(S_cost));
                        }
                    }
                }
            }

            for (int label = 1; label < numComponents; label++) {
                if (areaAi[label] > 0) {
                    totalRegionsR++;
                    double Ei_squared = errorEi[label] * errorEi[label];
                    sumTerm += Ei_squared / std::sqrt(static_cast<double>(areaAi[label]));
                }
            }
        }
        double G_I = (std::sqrt(static_cast<double>(totalRegionsR)) / (1000000.0 * M * N)) * sumTerm;

        return G_I;
    }
    EvaluationResult evaluateWithGroundTruth(const cv::Mat& disparityMap, const cv::Mat& groundTruth) {
        EvaluationResult result;
        cv::Mat gtGray;
        if (groundTruth.channels() == 3)
            cv::cvtColor(groundTruth, gtGray, cv::COLOR_BGR2GRAY);
        else
            gtGray = groundTruth.clone();

        cv::Mat dispGray;
        if (disparityMap.channels() == 3)
            cv::cvtColor(disparityMap, dispGray, cv::COLOR_BGR2GRAY);
        else
            dispGray = disparityMap.clone();

        CV_Assert(gtGray.size() == dispGray.size());

        int correct = 0;
        int correctWithin1 = 0;
        int total = 0;

        for (int y = 0; y < gtGray.rows; y++) {
            for (int x = 0; x < gtGray.cols; x++) {
                int gtRaw = gtGray.at<uchar>(y, x);
                if (gtRaw == 0) continue;

                float gtDisp = gtRaw / 4.0f;
                int yourDisp = dispGray.at<uchar>(y, x);

                total++;
                float diff = std::abs(gtDisp - yourDisp);
                if (diff < 0.5f) correct++;
                if (diff <= 1.0f) correctWithin1++;
            }
        }

        result.correctPercentage = (float)correct / total * 100.0f;
        result.correctWithin1Percentage = (float)correctWithin1 / total * 100.0f;
        result.totalPixels = total;

        std::cout << "Total pixels evaluated: " << total << std::endl;
        std::cout << "Correct disparities: " << result.correctPercentage << "%" << std::endl;
        std::cout << "Correct within +-1: " << result.correctWithin1Percentage << "%" << std::endl;

        return result;
    }
}