#include "libobsensor/ObSensor.hpp"
#include <iostream>

int main() try {
    ob::Pipeline pipeline;
    auto profiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    std::cout << "color profile count: " << profiles->count() << std::endl;
    for (uint32_t i = 0; i < profiles->count(); i++) {
        auto p = profiles->getProfile(i)->as<ob::VideoStreamProfile>();
        std::cout << i << ": " << p->width() << "x" << p->height()
                  << " @" << p->fps() << "fps format=" << p->format() << std::endl;
    }
    return 0;
} catch (ob::Error &e) {
    std::cerr << "ERR: " << e.getMessage() << std::endl;
    return 1;
}
