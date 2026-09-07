// Minimal Astra S color streamer built on the official OrbbecSDK (v1).
//
// Why this exists: the `openni` PyPI ctypes wrapper + the 2022 beta OpenNI2
// Linux redistributable times out unreliably opening this specific Astra S
// unit (firmware RD109Y-007). The actively-maintained OrbbecSDK v1 (which
// re-implements the same OpenNI host protocol) opens it reliably in
// ~250ms. So: do the USB/protocol work in C++ here, and hand frames to
// Python over a plain pipe.
//
// Protocol on stdout (per frame): 4-byte LE width, 4-byte LE height,
// then width*height*3 raw BGR8 bytes. Python side just reads that.
#include "libobsensor/ObSensor.hpp"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>
#include <unistd.h>

static void write_all(const void *buf, size_t n) {
    const char *p = static_cast<const char *>(buf);
    while (n > 0) {
        ssize_t w = write(STDOUT_FILENO, p, n);
        if (w <= 0) { std::exit(0); }  // reader (Python) closed the pipe
        p += w;
        n -= static_cast<size_t>(w);
    }
}

int main(int argc, char **argv) try {
    int want_w = argc > 1 ? std::atoi(argv[1]) : 640;
    int want_h = argc > 2 ? std::atoi(argv[2]) : 480;

    // The SDK's own logger writes straight to stdout, which corrupts our
    // binary frame protocol - silence console logging (still logs to file).
    ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_OFF);
    ob::Context::setLoggerToFile(OB_LOG_SEVERITY_INFO, "./orbbec_sdk_logs");

    ob::Pipeline pipeline;
    auto colorProfiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);

    // Order candidates: exact requested size first, then everything else
    // (largest first). Some profiles the SDK *lists* still fail to actually
    // start ("Match openni video mode failed") on this firmware, so we try
    // each until pipeline.start() actually succeeds.
    std::vector<std::shared_ptr<ob::VideoStreamProfile>> candidates;
    for (uint32_t i = 0; i < colorProfiles->count(); i++) {
        auto p = colorProfiles->getProfile(i)->as<ob::VideoStreamProfile>();
        if (p->width() == want_w && p->height() == want_h) candidates.insert(candidates.begin(), p);
        else candidates.push_back(p);
    }

    std::shared_ptr<ob::VideoStreamProfile> chosen;
    for (auto &p : candidates) {
        auto config = std::make_shared<ob::Config>();
        config->enableStream(p);
        try {
            pipeline.start(config);
            chosen = p;
            break;
        } catch (ob::Error &e) {
            std::cerr << "[orbbec_stream] profile " << p->width() << "x" << p->height()
                      << " failed: " << e.getMessage() << std::endl;
        }
    }
    if (!chosen) {
        std::cerr << "[orbbec_stream] no color profile could be started" << std::endl;
        return 1;
    }

    ob::FormatConvertFilter fmt;
    std::cerr << "[orbbec_stream] READY " << chosen->width() << "x" << chosen->height() << std::endl;

    while (true) {
        auto frameset = pipeline.waitForFrames(200);
        if (!frameset) continue;
        auto colorFrame = frameset->colorFrame();
        if (!colorFrame) continue;

        if (colorFrame->format() != OB_FORMAT_RGB) {
            if (colorFrame->format() == OB_FORMAT_MJPG) fmt.setFormatConvertType(FORMAT_MJPG_TO_RGB);
            else if (colorFrame->format() == OB_FORMAT_UYVY) fmt.setFormatConvertType(FORMAT_UYVY_TO_RGB);
            else if (colorFrame->format() == OB_FORMAT_YUYV) fmt.setFormatConvertType(FORMAT_YUYV_TO_RGB);
            else { std::cerr << "[orbbec_stream] unsupported format " << colorFrame->format() << std::endl; continue; }
            colorFrame = fmt.process(colorFrame)->as<ob::ColorFrame>();
            if (!colorFrame) continue;
        }
        fmt.setFormatConvertType(FORMAT_RGB_TO_BGR);
        colorFrame = fmt.process(colorFrame)->as<ob::ColorFrame>();
        if (!colorFrame) continue;

        uint32_t w = colorFrame->width(), h = colorFrame->height();
        write_all(&w, 4);
        write_all(&h, 4);
        write_all(colorFrame->data(), static_cast<size_t>(w) * h * 3);
    }
    return 0;
} catch (ob::Error &e) {
    std::cerr << "[orbbec_stream] ERROR: " << e.getMessage() << std::endl;
    return 1;
}
