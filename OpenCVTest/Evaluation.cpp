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
}