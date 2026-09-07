// Live depth viewer with a jet (rainbow) colormap, like the marketing
// thumbnails - closer = red/yellow, farther = blue/purple. This is still
// pure depth data (no real RGB), just colorized for visibility.
#include "libobsensor/ObSensor.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() try {
    ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_OFF);
    ob::Pipeline pipeline;

    auto profiles = pipeline.getStreamProfileList(OB_SENSOR_DEPTH);
    auto profile = profiles->getProfile(0)->as<ob::VideoStreamProfile>();
    auto config = std::make_shared<ob::Config>();
    config->enableStream(profile);
    pipeline.start(config);
    std::cerr << "[depth_colorize] started " << profile->width() << "x" << profile->height() << std::endl;

    const char *win = "Astra S Depth (rainbow colormap) - press q to quit";
    cv::namedWindow(win, cv::WINDOW_NORMAL);
    cv::resizeWindow(win, 800, 600);

    const double max_mm = 3000.0;  // clip range: 0 (near/red) .. 3m (far/blue)

    while (true) {
        auto frameset = pipeline.waitForFrames(1000);
        if (!frameset) continue;
        auto depthFrame = frameset->depthFrame();
        if (!depthFrame) continue;

        cv::Mat raw(depthFrame->height(), depthFrame->width(), CV_16UC1, depthFrame->data());
        cv::Mat scaled;
        raw.convertTo(scaled, CV_8UC1, 255.0 / max_mm);
        // invert so near=bright(red in JET), far=dark(blue in JET); zero(no data)->black
        cv::Mat inverted = 255 - scaled;
        cv::Mat mask = (raw == 0);
        inverted.setTo(0, mask);

        cv::Mat colored;
        cv::applyColorMap(inverted, colored, cv::COLORMAP_JET);
        colored.setTo(cv::Scalar(0, 0, 0), mask);

        cv::imshow(win, colored);
        int k = cv::waitKey(1) & 0xFF;
        if (k == 'q' || k == 27) break;
    }
    return 0;
} catch (ob::Error &e) {
    std::cerr << "ERR: " << e.getMessage() << std::endl;
    return 1;
}
