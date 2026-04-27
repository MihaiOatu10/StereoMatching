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
	bool multipleViews;
	std::cout << "Use multiple views? (0 for 2 views, 1 for 9 views): ";
	std::cin >> multipleViews;
    if(!multipleViews)
    {
        cv::Mat imgL = cv::imread("view1.png", cv::IMREAD_GRAYSCALE);
        cv::Mat imgR = cv::imread("view5.png", cv::IMREAD_GRAYSCALE);

        if (imgL.empty() || imgR.empty()) {
            return -1;
        }

        int imgOriginalW = imgL.cols;
        int imgOriginalH = imgL.rows;

        int alignment = 1 << Config::MAX_DEPTH;
        int validW = ((imgOriginalW + alignment - 1) / alignment) * alignment;
        int validH = ((imgOriginalH + alignment - 1) / alignment) * alignment;

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
    }
    else
    {
        std::vector<cv::Mat> images;
        cv::Mat firstImg = cv::imread("im0.png", cv::IMREAD_GRAYSCALE);
        if (firstImg.empty()) {
            std::cout << "Could not load multiview images!" << std::endl;
            return -1;
        }

        int imgOriginalW = firstImg.cols;
        int imgOriginalH = firstImg.rows;
        int alignment = 1 << Config::MAX_DEPTH;
        int validW = ((imgOriginalW + alignment - 1) / alignment) * alignment;
        int validH = ((imgOriginalH + alignment - 1) / alignment) * alignment;
        int totalPadW = validW - imgOriginalW;
        int totalPadH = validH - imgOriginalH;
        int padLeft = totalPadW / 2;
        int padRight = totalPadW - padLeft;
        int padTop = totalPadH / 2;
        int padBottom = totalPadH - padTop;

        for (int i = 0; i < 4; i++) {
            cv::Mat img = cv::imread("im" + std::to_string(i) + ".png", cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cout << "Could not load im" << i << ".png!" << std::endl;
                return -1;
            }
            cv::copyMakeBorder(img, img, padTop, padBottom, padLeft, padRight, cv::BORDER_REFLECT);
            images.push_back(img);
        }

        Config::WIDTH = validW;
        Config::HEIGHT = validH;
        cv::Size quadSize(Config::WIDTH, Config::HEIGHT);

        StereoMatcher matcher(Config::MAX_DISPARITY);
        matcher.populateSpace(images, 2);
        GeneticAlgorithm ga(Config::POP_SIZE, 0, Config::MAX_DISPARITY);
        ga.initialize(matcher, images[2].cols);
        ga.evolve(matcher);

        LinearQuadtree best = ga.getBestIndividual();
        cv::Mat grayResult = cv::Mat::zeros(quadSize, CV_8UC1);
        best.drawDisparity(grayResult, 0, 0, 0);
        grayResult = grayResult(cv::Rect(padLeft, padTop, imgOriginalW, imgOriginalH)).clone();

        cv::Mat displayImg = images[2](cv::Rect(padLeft, padTop, imgOriginalW, imgOriginalH)).clone();

        int zeroCount = cv::countNonZero(grayResult == 0);
        std::cout << "Zero pixels: " << zeroCount << "/" << (imgOriginalW * imgOriginalH) << std::endl;
        std::cout << "Config: " << Config::WIDTH << "x" << Config::HEIGHT << std::endl;

        double gScore = Evaluation::evaluateDisparityMap_G(grayResult, matcher.getCostVolume());
        std::cout << "\nFinal G(I) Quality Measure: " << gScore << std::endl;

        cv::Mat gt = cv::imread("gt_multi.pgm", cv::IMREAD_GRAYSCALE);
        if (gt.empty()) {
            std::cout << "Could not load ground truth!" << std::endl;
        }
        else {
            double gtMin, gtMax;
            cv::minMaxLoc(gt, &gtMin, &gtMax);
            std::cout << "GT range: " << gtMin << " - " << gtMax << std::endl;

            gt = gt(cv::Rect(0, 0, imgOriginalW, imgOriginalH)).clone();

            auto results = Evaluation::evaluateWithGroundTruth(grayResult, gt);
        }


        cv::Mat normalized;
        cv::normalize(grayResult, normalized, 0, 255, cv::NORM_MINMAX);
        cv::Mat colorResult;
        cv::applyColorMap(normalized, colorResult, cv::COLORMAP_JET);
        cv::imwrite("final_disparity_color_multiview.png", colorResult);
        cv::imwrite("final_disparity_gray_multiview.png", grayResult);
        cv::imshow("Disparity Map", colorResult);
        cv::imshow("Source Center", displayImg);
        cv::waitKey(0);
    }
    return 0;
}