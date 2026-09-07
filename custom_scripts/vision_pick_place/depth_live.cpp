// Live depth viewer for the Astra Pro Plus, colorized to the *observed* min/max
// range each frame instead of a fixed 0-5.12m span - see depth_snapshot.cpp for
// why the stock OrbbecSDK sample viewer looks all-blue on a close-range tabletop
// scene like this one.
#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Frame.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() try {
    ob::Pipeline pipe;
    auto config = std::make_shared<ob::Config>();
    config->enableVideoStream(OB_STREAM_DEPTH);
    pipe.start(config);

    cv::namedWindow("Astra Pro Plus - Depth (auto-ranged)", cv::WINDOW_NORMAL);
    cv::resizeWindow("Astra Pro Plus - Depth (auto-ranged)", 640, 480);

    std::cout << "뎁스 라이브 뷰. 'q' 또는 ESC로 종료.\n";
    while (true) {
        auto frameSet = pipe.waitForFrames(100);
        if (frameSet == nullptr) continue;
        auto depthFrame = frameSet->depthFrame();
        if (depthFrame == nullptr) continue;

        uint32_t width = depthFrame->width();
        uint32_t height = depthFrame->height();
        float scale = depthFrame->getValueScale();
        uint16_t *data = (uint16_t *)depthFrame->data();
        cv::Mat rawMat(height, width, CV_16UC1, data);

        double vmin, vmax;
        cv::Mat validMask = rawMat > 0;
        cv::minMaxLoc(rawMat, &vmin, &vmax, nullptr, nullptr, validMask);
        vmin *= scale;
        vmax *= scale;

        cv::Mat cvtMat;
        if (vmax > vmin) {
            cv::Mat rawMatF;
            rawMat.convertTo(rawMatF, CV_32F, scale);
            cv::Mat clipped;
            cv::min(cv::max(rawMatF, vmin), vmax, clipped);
            clipped.convertTo(cvtMat, CV_8UC1, 255.0 / (vmax - vmin), -255.0 * vmin / (vmax - vmin));
        } else {
            rawMat.convertTo(cvtMat, CV_8UC1, 0);
        }
        cv::Mat colorized;
        cv::applyColorMap(cvtMat, colorized, cv::COLORMAP_JET);
        colorized.setTo(cv::Scalar(0, 0, 0), rawMat == 0);

        cv::putText(colorized, cv::format("range: %.0f-%.0fmm", vmin, vmax), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Astra Pro Plus - Depth (auto-ranged)", colorized);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
    }

    pipe.stop();
    return 0;
} catch (ob::Error &e) {
    std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs()
              << "\nmessage:" << e.getMessage() << "\ntype:" << e.getExceptionType() << std::endl;
    return 1;
}
