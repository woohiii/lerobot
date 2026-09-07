// Headless single-frame depth snapshot for the Astra Pro Plus, via OrbbecSDK.
// Saves a JET-colorized PNG (same colorization the GUI DepthViewer sample uses)
// plus prints valid-pixel stats, so the actual depth image can be inspected
// directly instead of guessing from console text.
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

    // Skip the first several frames - depth sensors commonly report bogus/empty
    // frames for the first moment after starting the stream.
    std::shared_ptr<ob::FrameSet> frameSet;
    for (int i = 0; i < 30; i++) {
        frameSet = pipe.waitForFrames(200);
        if (frameSet != nullptr && frameSet->depthFrame() != nullptr) break;
    }
    if (frameSet == nullptr || frameSet->depthFrame() == nullptr) {
        std::cerr << "no depth frame received\n";
        return 1;
    }

    auto depthFrame = frameSet->depthFrame();
    uint32_t width = depthFrame->width();
    uint32_t height = depthFrame->height();
    float scale = depthFrame->getValueScale();
    uint16_t *data = (uint16_t *)depthFrame->data();

    cv::Mat rawMat(height, width, CV_16UC1, data);

    // stats on valid (nonzero) pixels, in mm
    int valid = 0;
    double sum = 0;
    uint16_t vmin = 65535, vmax = 0;
    for (uint32_t i = 0; i < width * height; i++) {
        uint16_t mm = (uint16_t)(data[i] * scale);
        if (mm > 0) {
            valid++;
            sum += mm;
            if (mm < vmin) vmin = mm;
            if (mm > vmax) vmax = mm;
        }
    }
    std::cout << "valid pixels: " << valid << "/" << (width * height)
              << " (" << (100.0 * valid / (width * height)) << "%)\n";
    if (valid > 0) {
        std::cout << "depth range (mm): min=" << vmin << " max=" << vmax
                   << " mean=" << (sum / valid) << "\n";
    }

    // window.hpp's own colorization assumes a fixed 0-5.12m range, which is why the
    // GUI viewer looked all-blue: our actual tabletop scene (490-767mm) only spans
    // ~5-8% of that range, compressing into a sliver of JET that never reaches
    // green/yellow/red. Normalizing to the *observed* min/max instead spreads the
    // real working range across the full colormap - same fix already used for the
    // Astra S's own depth visualization (see orbbec_color_camera.py's DEPTH_MIN_MM/
    // DEPTH_MAX_MM), just measured live here instead of hand-tuned.
    cv::Mat cvtMat;
    if (valid > 0 && vmax > vmin) {
        cv::Mat rawMatF;
        rawMat.convertTo(rawMatF, CV_32F, scale);  // -> mm
        cv::Mat clipped;
        cv::min(cv::max(rawMatF, (double)vmin), (double)vmax, clipped);
        clipped.convertTo(cvtMat, CV_8UC1, 255.0 / (vmax - vmin), -255.0 * vmin / (vmax - vmin));
        // zero out invalid (0mm) pixels post-normalization so they stay black, not
        // wrapping to a valid-looking color
        cv::Mat invalidMask = (rawMatF == 0);
        cvtMat.setTo(0, invalidMask);
    } else {
        rawMat.convertTo(cvtMat, CV_8UC1, 0);
    }
    cv::Mat colorized;
    cv::applyColorMap(cvtMat, colorized, cv::COLORMAP_JET);
    colorized.setTo(cv::Scalar(0, 0, 0), rawMat == 0);
    cv::imwrite("/tmp/depth_snapshot.png", colorized);
    std::cout << "saved /tmp/depth_snapshot.png\n";

    pipe.stop();
    return 0;
} catch (ob::Error &e) {
    std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs()
              << "\nmessage:" << e.getMessage() << "\ntype:" << e.getExceptionType() << std::endl;
    return 1;
}
