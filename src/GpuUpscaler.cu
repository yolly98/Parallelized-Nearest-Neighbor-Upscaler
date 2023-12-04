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

__global__ void upscaleFromOriginalImage(uint8_t* imageToUpscale, uint8_t* upscaledImage, uint32_t width, uint8_t upscaleFactor, uint8_t bytePerPixel, size_t originalSize)
{
    // get the pixel position in the original image vector
    // uint32_t oldIndex = ((((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x) * bytePerPixel;
    uint32_t oldIndex = ((blockIdx.x * blockDim.x) + threadIdx.x) * bytePerPixel;

    if (oldIndex < originalSize) {
        // convert the position in a matrix notation
        uint32_t i = oldIndex / (width * bytePerPixel);
        uint32_t j = oldIndex - (i * width * bytePerPixel);

        // compute the position of the first pixel to duplicate in upscaled image
        uint32_t newi = i * upscaleFactor;
        uint32_t newj = j * upscaleFactor;
        uint32_t upscaledWidth = width * upscaleFactor;

        // read the pixel to copy
        uchar4 pixelToCopy;
        memcpy(&pixelToCopy, &imageToUpscale[oldIndex], 4 * sizeof(uint8_t));

        // iterate the pixel to duplicate in upscaled image
        for (int m = newi; m < newi + upscaleFactor; m++) {
            for (int n = newj; n < newj + upscaleFactor * bytePerPixel; n += bytePerPixel) {
                // compute the pixel position in the upscaled image vector
                uint32_t newIndex = m * upscaledWidth * bytePerPixel + n;
            
                // manage single channel if tridimensional version, else manage all the others
                if (blockDim.z == 1) {
                    memcpy(&upscaledImage[newIndex], &pixelToCopy, 4 * sizeof(uint8_t));
                } else {
                    upscaledImage[newIndex + threadIdx.z] = imageToUpscale[oldIndex + threadIdx.z];
                }
            }
        }
    }
}

__global__ void upscaleFromUpscaledImage(uint8_t* imageToUpscale, uint8_t* upscaledImage, uint32_t width, uint8_t upscaleFactor, uint8_t bytePerPixel, size_t upscaledSize)
{
    // get the pixel position in the upscaled image vector
    // uint32_t newIndex = ((((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x) * bytePerPixel;
    uint32_t newIndex = ((blockIdx.x * blockDim.x) + threadIdx.x) * bytePerPixel;

    if (newIndex < upscaledSize) {
        // convert the position in a matrix notation
        uint32_t newi = newIndex / (width * upscaleFactor * bytePerPixel);
        uint32_t newj = (newIndex - (newi * width * upscaleFactor * bytePerPixel)) / bytePerPixel;

        // compute the position of the pixel to copy from the original image
        uint32_t i = newi / upscaleFactor;
        uint32_t j = newj / upscaleFactor;
        uint32_t oldIndex = (i * width + j) * bytePerPixel;

        // manage single channel if tridimensional version, else manage all the others
        if (blockDim.z == 1) {
            memcpy(&upscaledImage[newIndex], &imageToUpscale[oldIndex], 4 * sizeof(uint8_t));
        } else {
            upscaledImage[newIndex + threadIdx.z] = imageToUpscale[oldIndex + threadIdx.z];
        }      
    }
}

__global__ void upscaleWithSingleThread(uint8_t* imageToUpscale, uint8_t* upscaledImage, uint32_t width, uint32_t height, uint8_t upscaleFactor, uint8_t bytePerPixel)
{
    size_t imageToUpscaleSize = width * bytePerPixel * height;
    size_t upscaledWidth = width * upscaleFactor;
    
    for (size_t oldIndex = 0; oldIndex < imageToUpscaleSize; oldIndex += bytePerPixel) {
        // convert the position in a matrix notation
        uint32_t i = oldIndex / (width * bytePerPixel);
        uint32_t j = oldIndex - (i * width * bytePerPixel);

        // compute the position of the first pixel to duplicate in upscaled image
        uint32_t newi = i * upscaleFactor;
        uint32_t newj = j * upscaleFactor;

        // iterate the pixel to duplicate in upscaled image
        for (int m = newi; m < newi + upscaleFactor; m++) {
            for (int n = newj; n < newj + upscaleFactor * bytePerPixel; n += bytePerPixel) {
                // compute the pixel position in the upscaled image vector
                uint32_t newIndex = m * upscaledWidth * bytePerPixel + n;

                // copy all the channels
                for (int k = 0; k < bytePerPixel; k++)
                    upscaledImage[newIndex + k] = imageToUpscale[oldIndex + k];
            }
        }
    }
}

__global__ void upscaleWithTextureObject(cudaTextureObject_t texObj, uint8_t* __restrict__ upscaledImage, uint32_t pixelsHandledByBlock, uint32_t pixelsHandledByThread, uint32_t width, uint32_t height, uint8_t bytePerPixel, uint32_t upscaleFactor)
{
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
            uchar4  pixelToCopy = tex2D<uchar4>(texObj, oldY, oldX);
            memcpy(&upscaledImage[newIndex * bytePerPixel], &pixelToCopy, sizeof(uchar4));
        }
    }
}

float gpuUpscaler(size_t originalSize, size_t upscaledSize, uint8_t upscaleFactor, Settings settings, uint8_t* data, uint32_t width, uint32_t height, uint32_t bytePerPixel, string imageName)
{
    uint8_t* upscaledImage = new uint8_t[upscaledSize];

    // event based timer
    GpuTimer timer;

    // allocate GPU memory for input and output array 
    uint8_t* d_data, * d_out;
    cudaMalloc((void**)&d_data, originalSize);
    cudaMalloc((void**)&d_out, upscaledSize);
    cudaMemcpy(d_data, data, originalSize, cudaMemcpyHostToDevice);

    // define resources for the execution
    dim3 grid(settings.blocksPerGridX, settings.blocksPerGridY, settings.blocksPerGridZ);               // blocks per grid
    dim3 block(settings.threadsPerBlockX, settings.threadsPerBlockY, settings.threadsPerBlockZ);        // threads per block

    // start kernel execution
    switch (settings.upscalerType)
    {
        case UpscalerType::UpscaleFromOriginalImage:
            timer.start();
            upscaleFromOriginalImage << <grid, block >> > (d_data, d_out, width, upscaleFactor, bytePerPixel, originalSize); 
            timer.stop();
            break;
        case UpscalerType::UpscaleFromUpscaledImage:
            timer.start();
            upscaleFromUpscaledImage << <grid, block >> > (d_data, d_out, width, upscaleFactor, bytePerPixel, upscaledSize);
            timer.stop();
            break;
        case UpscalerType::UpscaleWithSingleThread:
            timer.start();
            upscaleWithSingleThread << <grid, block >> > (d_data, d_out, width, height, upscaleFactor, bytePerPixel);
            timer.stop();
            break;
        case UpscalerType::UpscaleWithTextureObject:
            cudaTextureObject_t texObj = createTextureObject(width, height, bytePerPixel, data);
            timer.start();
            upscaleWithTextureObject << <grid, block >> > (texObj, d_out, settings.pixelsHandledByBlock, settings.pixelsHandledByThread, width, height, bytePerPixel, upscaleFactor);
            timer.stop();
            break;
    }

    // wait for the end of the execution and retrieve results from GPU memory
    cudaDeviceSynchronize();
    cudaMemcpy(upscaledImage, d_out, upscaledSize, cudaMemcpyDeviceToHost);

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
    cudaFree(d_data);
    cudaFree(d_out);

    return time;
}

cudaTextureObject_t createTextureObject(uint32_t width, uint32_t height, uint32_t bytePerPixel, uint8_t* data)
{
    // allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // copy data located at address data in host memory to device memory
    const size_t spitch = width * bytePerPixel * sizeof(uint8_t);
    cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * bytePerPixel * sizeof(uint8_t), height, cudaMemcpyHostToDevice);

    // specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}
