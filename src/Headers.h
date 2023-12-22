#ifndef _HEADERS_H
#define _HEADERS_H

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

#include "cuda_runtime.h"

enum UpscalerType {
    UpscaleWithTextureObject,
    UpscaleWithTextureObjectOptimized
};

struct Settings {
    uint32_t threadsPerBlock;
    uint32_t blocksPerGrid;
    uint32_t pixelsHandledByThread;
    uint32_t threadsCount;
    UpscalerType upscalerType;

    Settings()
        : upscalerType(UpscalerType::UpscaleWithTextureObjectOptimized) {}
    Settings(uint32_t tpb, UpscalerType ut, uint32_t width, uint32_t height, uint8_t upscaleFactor, uint32_t phbt)
        : threadsPerBlock(tpb), pixelsHandledByThread(phbt), upscalerType(ut) {
        // compute the number of blocks per grid on x-axis
        blocksPerGrid = (((width * height * upscaleFactor * upscaleFactor) / (float)pixelsHandledByThread) + threadsPerBlock - 1) / (float)threadsPerBlock;
        threadsCount = ceil(width * height * upscaleFactor * upscaleFactor / (float)pixelsHandledByThread);
    }


    void print() {
        std::cout << "\n[+] GPU Upscale Settings: " << std::endl;

        switch (upscalerType)
        {
            case UpscalerType::UpscaleWithTextureObject:
                std::cout << "--> Upscaler Type: UpscaleWithTextureObject" << std::endl;
                break;
            case UpscalerType::UpscaleWithTextureObjectOptimized:
                std::cout << "--> Upscaler Type: UpscaleWithTextureObjectOptimized" << std::endl;
                break;
        }

        printf("--> Threads per Block: (%d)\n", threadsPerBlock);
        printf("--> Blocks per Grid: (%d)\n", blocksPerGrid);
        printf("--> Pixels Handled by Thread: %d\n", pixelsHandledByThread);
        printf("--> Threads Count: %d\n", threadsCount);
    }

    std::string toString() {
        std::string str;
        str = str + std::to_string(threadsPerBlock) + ";" + std::to_string(blocksPerGrid);
        str = str + ";" + std::to_string(pixelsHandledByThread) + ";" + std::to_string(threadsCount); 

        switch (upscalerType)
        {
            case UpscalerType::UpscaleWithTextureObject:
                str = str + ";" + "\"UpscaleWithTextureObject\"";
                break;
            case UpscalerType::UpscaleWithTextureObjectOptimized:
                str = str + ";" + "\"UpscaleWithTextureObjectOptimized\"";
                break;
        }

        return str;
    }
};

float cpuUpscaler(uint32_t numThread, uint8_t upscaleFactor, uint8_t* originalImage, size_t width, size_t height, uint32_t bytePerPixel, std::string imageName = "");
float gpuUpscaler(size_t originalSize, size_t upscaledSize, uint8_t upscaleFactor, Settings settings, uint8_t* data, uint32_t width, uint32_t height, uint32_t bytePerPixel, std::string imageName = "");
float computeCI(const std::vector<float>& values);
cudaTextureObject_t createTextureObject(uint32_t width, uint32_t height, uint32_t bytePerPixel, uint8_t* data);

#endif  // _HEADERS_H
