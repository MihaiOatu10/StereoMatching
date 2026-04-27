#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp> 
#include <ctime>
#include <vector>
#include "LinearQuadtree.h"
#include "StereoMatcher.h"
#include "GeneticAlgorithm.h"
#include "Common.h"
#include "Evaluation.h"

int main() {
    srand(time(nullptr));

    cv::Mat imgL = cv::imread("view1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imgR = cv::imread("view5.png", cv::IMREAD_GRAYSCALE);

    if (imgL.empty() || imgR.empty()) {
        return -1;
    }

    int imgOriginalW = imgL.cols;
    int imgOriginalH = imgL.rows;

    int validW = ((imgL.cols + 127) / 128) * 128;
    int validH = ((imgL.rows + 127) / 128) * 128;

    int totalPadW = validW - imgOriginalW;
    int totalPadH = validH - imgOriginalH;

    int padLeft = totalPadW / 2;
    int padRight = totalPadW - padLeft;
    int padTop = totalPadH / 2;
    int padBottom = totalPadH - padTop;

    cv::copyMakeBorder(imgL, imgL, padTop, padBottom, padLeft, padRight, cv::BORDER_REFLECT);
    cv::copyMakeBorder(imgR, imgR, padTop, padBottom, padLeft, padRight, cv::BORDER_REFLECT);

    Config::WIDTH = validW;
    Config::HEIGHT = validH;

    cv::Size quadSize(Config::WIDTH, Config::HEIGHT);

    StereoMatcher matcher(Config::MAX_DISPARITY);
    matcher.populateSpace(imgL, imgR);

    GeneticAlgorithm ga(Config::POP_SIZE, 0, Config::MAX_DISPARITY);
    ga.initialize(matcher, imgL.cols);

    ga.evolve(matcher);

    LinearQuadtree best = ga.getBestIndividual();

    cv::Mat grayResult = cv::Mat::zeros(quadSize, CV_8UC1);
    best.drawDisparity(grayResult, 0, 0, 0);

    grayResult = grayResult(cv::Rect(padLeft, padTop, imgOriginalW, imgOriginalH)).clone();

    imgL = imgL(cv::Rect(padLeft, padTop, imgOriginalW, imgOriginalH)).clone();

    cv::Mat gt = cv::imread("disp1.png", cv::IMREAD_GRAYSCALE);
    gt = gt(cv::Rect(0, 0, imgOriginalW, imgOriginalH)).clone();

    auto results = Evaluation::evaluateWithGroundTruth(grayResult, gt);

    int zeroCount = cv::countNonZero(grayResult == 0);
    std::cout << "Zero pixels: " << zeroCount << "/" << (quadSize.width * quadSize.height) << std::endl;
    std::cout << "Config: " << Config::WIDTH << "x" << Config::HEIGHT << std::endl;
    std::cout << "Image size: " << imgL.cols << "x" << imgL.rows << std::endl;
    std::cout << "Canvas size: " << grayResult.cols << "x" << grayResult.rows << std::endl;

    double gScore = Evaluation::evaluateDisparityMap_G(grayResult, matcher.getCostVolume());
    std::cout << "\nFinal G(I) Quality Measure: " << gScore << std::endl;

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