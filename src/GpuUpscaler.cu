#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>

#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

#include "Headers.h"
#include "GpuTimer.cuh"

using namespace std;

// constant for optimized kernel
__device__ __constant__ uint32_t deviceUpscaledWidth;
__device__ __constant__ uint32_t deviceUpscaledSize;
__device__ __constant__ uint32_t deviceThreadsCount;
__device__ __constant__ uint32_t devicePixelsHandledByThread;
__device__ __constant__ uint8_t deviceUpscaleFactor;

__global__ void upscale(cudaTextureObject_t originalImage, uint8_t* upscaledImage, uint32_t pixelsHandledByThread, uint32_t width, uint32_t height, uint8_t bytePerPixel, uint32_t upscaleFactor)
{
    uint32_t pixelsHandledByBlock = pixelsHandledByThread * blockDim.x;
    uint32_t startNewIndex = blockIdx.x * pixelsHandledByBlock + threadIdx.x * pixelsHandledByThread;
    uint32_t upscaledWidth = width * upscaleFactor;
    uint32_t upscaledSize = width * height * upscaleFactor * upscaleFactor;

    // iterate all pixels handled by this thread
    for (uint32_t i = 0; i < pixelsHandledByThread; i++) {
        // compute the coordinates of the pixel
        uint32_t newIndex = startNewIndex + i;

        if (newIndex < upscaledSize) {
            uint32_t x = newIndex / upscaledWidth;
            uint32_t y = newIndex - (x * upscaledWidth);

            // compute the coordinates of the pixel of the original image
            uint32_t oldX = x / upscaleFactor;
            uint32_t oldY = y / upscaleFactor;

            // copy the pixel
            uchar4  pixelToCopy = tex2D<uchar4>(originalImage, oldY, oldX);
            memcpy(&upscaledImage[newIndex * bytePerPixel], &pixelToCopy, sizeof(uchar4));
        }
    }
}

__global__ void upscaleOptimized(cudaTextureObject_t originalImage, uchar4* upscaledImage)
{
    uint32_t startNewIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // iterate all pixels handled by this thread
    for (uint32_t i = 0; i < devicePixelsHandledByThread; i++) {
        // compute the coordinates of the pixel
        uint32_t newIndex = startNewIndex + (i * deviceThreadsCount);

        if (newIndex < deviceUpscaledSize) {
            uint32_t x = newIndex / deviceUpscaledWidth;
            uint32_t y = newIndex - (x * deviceUpscaledWidth);

            // compute the coordinates of the pixel of the original image
            uint32_t oldX = x / deviceUpscaleFactor;
            uint32_t oldY = y / deviceUpscaleFactor;

            // copy the pixel
            uchar4 pixelToCopy = tex2D<uchar4>(originalImage, oldY, oldX);
            upscaledImage[newIndex] = pixelToCopy;
        }
    }
} 

float gpuUpscaler(size_t originalSize, size_t upscaledSize, uint8_t upscaleFactor, Settings settings, uint8_t* data, uint32_t width, uint32_t height, uint32_t bytePerPixel, string imageName)
{
    uint8_t* upscaledImage = new uint8_t[upscaledSize];

    // event based timer
    GpuTimer timer;

    // allocate GPU memory for input and output array 
    uint8_t* deviceUpscaledImage;
    cudaMalloc((void**)&deviceUpscaledImage, upscaledSize);

    // create the texture object to store the original image
    cudaTextureObject_t originalImage = createTextureObject(width, height, bytePerPixel, data);

    // define resources for the execution
    dim3 grid(settings.blocksPerGrid, 1, 1);                // blocks per grid
    dim3 block(settings.threadsPerBlock, 1, 1);             // threads per block

    // start kernel execution
    switch (settings.upscalerType)
    {
        case UpscalerType::UpscaleWithTextureObject:
            timer.start();
            upscale << <grid, block >> > (originalImage, deviceUpscaledImage, settings.pixelsHandledByThread, width, height, bytePerPixel, upscaleFactor);
            timer.stop();
            break;
        case UpscalerType::UpscaleWithTextureObjectOptimized:
            // precompute device constant values
            uint32_t upscaledWidth = width * upscaleFactor;
            uint32_t upscaledSize = width * height * upscaleFactor * upscaleFactor;

            // store parameters in constant memory
            cudaMemcpyToSymbol(deviceUpscaledWidth, &upscaledWidth, sizeof(uint32_t));
            cudaMemcpyToSymbol(deviceUpscaledSize, &upscaledSize, sizeof(uint32_t));
            cudaMemcpyToSymbol(deviceThreadsCount, &settings.threadsCount, sizeof(uint32_t));
            cudaMemcpyToSymbol(devicePixelsHandledByThread, &settings.pixelsHandledByThread, sizeof(uint32_t));
            cudaMemcpyToSymbol(deviceUpscaleFactor, &upscaleFactor, sizeof(uint8_t));

            timer.start();
            upscaleOptimized << <grid, block >> > (originalImage, (uchar4*)deviceUpscaledImage);
            timer.stop();
            break;
    }

    // wait for the end of the execution and retrieve results from GPU memory
    cudaDeviceSynchronize();
    cudaMemcpy(upscaledImage, deviceUpscaledImage, upscaledSize, cudaMemcpyDeviceToHost);

    // print the upscale duration
    float time = timer.getElapsedMilliseconds();
    cout << "\n---------------------------------------------------------------" << endl;
    settings.print();
    cout << "[+] (GPU) Time needed: " << time << "ms" << endl;

    // save image as PNG
    if (imageName != "") {
        cout << "[+] Saving image..." << endl;
        if (stbi_write_png(imageName.c_str(), width * upscaleFactor, height * upscaleFactor, bytePerPixel, upscaledImage, width * upscaleFactor * bytePerPixel))
            cout << "[+] Image saved successfully" << endl;
        else
            cout << "[-] Failed to save image" << endl;
    }

    // free memory
    delete[] upscaledImage;
    cudaFree(deviceUpscaledImage);

    return time;
}
