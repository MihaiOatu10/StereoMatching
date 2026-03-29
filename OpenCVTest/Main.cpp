#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "LinearQuadtree.h"
#include "StereoMatcher.h"
#include "GeneticAlgorithm.h"
#include "Common.h"

int main() {
    srand((unsigned int)time(NULL));

    cv::Mat imgL = cv::imread("scene1.row3.col1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imgR = cv::imread("scene1.row3.col3.png", cv::IMREAD_GRAYSCALE);

    if (imgL.empty() || imgR.empty()) {
        return -1;
    }

    cv::Size quadSize(Config::WIDTH, Config::HEIGHT);
    cv::resize(imgL, imgL, quadSize);
    cv::resize(imgR, imgR, quadSize);

    StereoMatcher matcher(Config::MAX_DISPARITY);
    matcher.populateSpace(imgL, imgR);

    GeneticAlgorithm ga(Config::POP_SIZE, 0, Config::MAX_DISPARITY);
    ga.initialize(matcher, imgL.cols);

    ga.evolve(matcher);

    LinearQuadtree best = ga.getBestIndividual();

    cv::Mat grayResult = cv::Mat::zeros(quadSize, CV_8UC1);
    best.drawDisparity(grayResult, 0, 0, 0);

    cv::Mat normalized;
    cv::normalize(grayResult, normalized, 0, 255, cv::NORM_MINMAX);

    cv::Mat colorResult;
    cv::applyColorMap(normalized, colorResult, cv::COLORMAP_JET);

    cv::imwrite("final_disparity_color.png", colorResult);
    cv::imwrite("final_disparity_gray.png", grayResult);

    cv::imshow("Disparity Map", colorResult);
    cv::imshow("Source Left", imgL);
    cv::waitKey(0);

    return 0;
}