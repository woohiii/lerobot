#include "libobsensor/ObSensor.hpp"
#include <iostream>
int main() try {
    ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_OFF);
    ob::Pipeline pipeline;
    auto config = std::make_shared<ob::Config>();
    config->enableVideoStream(OB_STREAM_COLOR);  // all ANY
    pipeline.start(config);
    std::cerr << "COLOR STREAM STARTED OK (ANY)" << std::endl;
    for (int i = 0; i < 5; i++) {
        auto fs = pipeline.waitForFrames(3000);
        std::cerr << (fs ? "got frameset" : "no frameset") << std::endl;
        if (fs) {
            auto cf = fs->colorFrame();
            std::cerr << "  colorFrame: " << (cf ? "yes" : "no");
            if (cf) std::cerr << " format=" << cf->format() << " w=" << cf->width() << " h=" << cf->height();
            std::cerr << std::endl;
        }
    }
    return 0;
} catch (ob::Error &e) {
    std::cerr << "ERR: " << e.getMessage() << std::endl;
    return 1;
}
