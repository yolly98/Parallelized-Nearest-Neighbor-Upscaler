#ifndef _HEADERS_H
#define _HEADERS_H

#include <cstdint>
#include <cstring>

struct Settings {
    uint32_t threadsPerBlockX;
    uint32_t threadsPerBlockY;
    uint32_t threadsPerBlockZ;

    uint32_t blocksPerGridX;
    uint32_t blocksPerGridY;
    uint32_t blocksPerGridZ;

    Settings()
        : threadsPerBlockX(0), threadsPerBlockY(0), threadsPerBlockZ(0), blocksPerGridX(0), blocksPerGridY(0), blocksPerGridZ(0) {}
    Settings(uint32_t tpbX, uint32_t tpbY, uint32_t tpbZ, uint32_t bpgX, uint32_t bpgY, uint32_t bpgZ)
        : threadsPerBlockX(tpbX), threadsPerBlockY(tpbY), threadsPerBlockZ(tpbZ), blocksPerGridX(bpgX), blocksPerGridY(bpgY), blocksPerGridZ(bpgZ) {}

    void print() {
        std::cout << "\n[+] GPU Upscale Settings: " << std::endl;
        std::cout << "--> Threads per Block X: " << threadsPerBlockX << std::endl;
        std::cout << "--> Threads per Block Y: " << threadsPerBlockY << std::endl;
        std::cout << "--> Threads per Block Z: " << threadsPerBlockZ << std::endl;
        std::cout << "--> Blocks per Grid X: " << blocksPerGridX << std::endl;
        std::cout << "--> Blocks per Grid Y: " << blocksPerGridY << std::endl;
        std::cout << "--> Blocks per Grid Z: " << blocksPerGridZ << std::endl;
    }
};

enum Channels {
    ALL = 0,
    GREY = 1,
    GREY_ALPHA = 2,
    RGB = 3,
    RGB_ALPHA = 4
};

void cpuUpscaler(uint8_t upscaleFactor, uint8_t* data, size_t width, size_t height, uint32_t bytePerPixel, std::string imageName = "");
void gpuUpscaler(size_t originalSize, size_t upscaledSize, uint8_t upscaleFactor, Settings settings, uint8_t* data, uint32_t width, uint32_t height, uint32_t bytePerPixel, std::string imageName = "");

#endif  // _HEADERS_H
