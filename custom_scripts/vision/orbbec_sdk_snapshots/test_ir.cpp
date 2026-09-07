#include "libobsensor/ObSensor.hpp"
#include <iostream>
int main() try {
    ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_INFO);
    ob::Pipeline pipeline;
    auto profiles = pipeline.getStreamProfileList(OB_SENSOR_IR);
    std::cerr << "ir profile count: " << profiles->count() << std::endl;
    for (uint32_t i = 0; i < profiles->count(); i++) {
        auto p = profiles->getProfile(i)->as<ob::VideoStreamProfile>();
        std::cerr << i << ": " << p->width() << "x" << p->height() << " @" << p->fps() << "fps format=" << p->format() << std::endl;
    }
    auto config = std::make_shared<ob::Config>();
    auto p = profiles->getProfile(0)->as<ob::VideoStreamProfile>();
    config->enableStream(p);
    pipeline.start(config);
    std::cerr << "IR STREAM STARTED OK: " << p->width() << "x" << p->height() << std::endl;
    for (int i = 0; i < 5; i++) {
        auto fs = pipeline.waitForFrames(3000);
        std::cerr << (fs ? "got frameset" : "no frameset") << std::endl;
        if (fs) {
            auto irf = fs->irFrame();
            std::cerr << "  irFrame: " << (irf ? "yes" : "no");
            if (irf) std::cerr << " format=" << irf->format() << " w=" << irf->width() << " h=" << irf->height();
            std::cerr << std::endl;
        }
    }
    return 0;
} catch (ob::Error &e) {
    std::cerr << "ERR: " << e.getMessage() << std::endl;
    return 1;
}
