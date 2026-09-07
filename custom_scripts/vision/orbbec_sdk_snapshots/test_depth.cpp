#include "libobsensor/ObSensor.hpp"
#include <iostream>
int main() try {
    ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_OFF);
    ob::Pipeline pipeline;
    auto profiles = pipeline.getStreamProfileList(OB_SENSOR_DEPTH);
    std::cerr << "depth profile count: " << profiles->count() << std::endl;
    auto config = std::make_shared<ob::Config>();
    auto p = profiles->getProfile(0)->as<ob::VideoStreamProfile>();
    std::cerr << "trying " << p->width() << "x" << p->height() << std::endl;
    config->enableStream(p);
    pipeline.start(config);
    std::cerr << "DEPTH STREAM STARTED OK" << std::endl;
    auto fs = pipeline.waitForFrames(2000);
    std::cerr << (fs ? "got frameset" : "no frameset") << std::endl;
    return 0;
} catch (ob::Error &e) {
    std::cerr << "ERR: " << e.getMessage() << std::endl;
    return 1;
}
