#include "Common.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

struct ImagePadding {
    int left, right, top, bottom;
    int originalW, originalH;

    cv::Mat apply(cv::Mat img) {
        cv::Mat out;
        cv::copyMakeBorder(img, out, top, bottom, left, right, cv::BORDER_REFLECT);
        return out;
    }

    cv::Mat remove(cv::Mat img) {
        return img(cv::Rect(left, top, originalW, originalH)).clone();
    }
};

ImagePadding calculatePadding(int w, int h) {
    int alignment = 1 << Config::MAX_DEPTH;
    int validW = ((w + alignment - 1) / alignment) * alignment;
    int validH = ((h + alignment - 1) / alignment) * alignment;

    Config::WIDTH = validW;
    Config::HEIGHT = validH;

    ImagePadding p;
    p.originalW = w; p.originalH = h;
    int totalW = validW - w;
    int totalH = validH - h;
    p.left = totalW / 2; p.right = totalW - p.left;
    p.top = totalH / 2; p.bottom = totalH - p.top;
    return p;
}