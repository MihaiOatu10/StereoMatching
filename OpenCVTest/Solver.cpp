#include "Solver.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <vector>
#include "LinearQuadtree.h"
#include "StereoMatcher.h"
#include "GeneticAlgorithm.h"
#include "Common.h"
#include "Evaluation.h"
#include "ImageProcessor.h"

void Solver::run() {
    srand(time(nullptr));
    bool useMulti;
    std::cout << "Use multiple views? (0/1): ";
    std::cin >> useMulti;

    std::vector<cv::Mat> views;
    if (!useMulti) {
        views.push_back(cv::imread("view1.png", cv::IMREAD_GRAYSCALE));
        views.push_back(cv::imread("view5.png", cv::IMREAD_GRAYSCALE));
    }
    else {
        for (int i = 0; i < 4; i++) {
            views.push_back(cv::imread("im" + std::to_string(i) + ".png", cv::IMREAD_GRAYSCALE));
        }
    }

    if (views.empty() || views[0].empty()) return;

    ImagePadding pad = calculatePadding(views[0].cols, views[0].rows);
    for (auto& img : views) img = pad.apply(img);

    StereoMatcher matcher(Config::MAX_DISPARITY);
    if (!useMulti) matcher.populateSpace(views[0], views[1]);
    else           matcher.populateSpace(views, 2);

    GeneticAlgorithm ga(Config::POP_SIZE, 0, Config::MAX_DISPARITY);
    ga.initialize(matcher, views[useMulti ? 2 : 0].cols);
    ga.evolve(matcher);

    LinearQuadtree best = ga.getBestIndividual();
    cv::Mat fullResult = cv::Mat::zeros(cv::Size(Config::WIDTH, Config::HEIGHT), CV_8UC1);
    best.drawDisparity(fullResult, 0, 0, 0);

    cv::Mat finalGray = pad.remove(fullResult);
    cv::Mat displayBase = pad.remove(views[useMulti ? 2 : 0]);

    cv::Mat gt = cv::imread(useMulti ? "gt_multi.pgm" : "disp1.png", cv::IMREAD_GRAYSCALE);
    if (!gt.empty()) {
        gt = gt(cv::Rect(0, 0, pad.originalW, pad.originalH)).clone();
        Evaluation::evaluateWithGroundTruth(finalGray, gt);
    }

    std::cout << "G-Score: " << Evaluation::evaluateDisparityMap_G(finalGray, matcher.getCostVolume()) << std::endl;

    cv::Mat color;
    cv::normalize(finalGray, color, 0, 255, cv::NORM_MINMAX);
    cv::applyColorMap(color, color, cv::COLORMAP_JET);

    cv::imshow("Result", color);
    cv::imshow("Source", displayBase);
    cv::waitKey(0);
}