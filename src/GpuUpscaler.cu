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

__global__ void upscaleFromOriginalImage(uint8_t* imageToUpscale, uint8_t* upscaledImage, uint32_t width, uint8_t upscaleFactor, uint8_t bytePerPixel)
{
    // get the pixel position in the original image vector
    uint32_t oldIndex = ((((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x) * bytePerPixel;

    // convert the position in a matrix notation
    uint32_t i = oldIndex / (width * bytePerPixel);
    uint32_t j = oldIndex - (i * width * bytePerPixel);

    // compute the position of the first pixel to duplicate in upscaled image
    uint32_t newi = i * upscaleFactor;
    uint32_t newj = j * upscaleFactor;
    uint32_t upscaledWidth = width * upscaleFactor;

    // iterate the pixel to duplicate in upscaled image
    for (int m = newi; m < newi + upscaleFactor; m++) {
        for (int n = newj; n < newj + upscaleFactor * bytePerPixel; n += bytePerPixel) {
            // compute the pixel position in the upscaled image vector
            uint32_t newIndex = m * upscaledWidth * bytePerPixel + n;
            
            // manage single channel if tridimensional version, else manage all the others
            if (blockDim.z == 1) {
                for (int k = 0; k < bytePerPixel; k++)
                    upscaledImage[newIndex + k] = imageToUpscale[oldIndex + k];
            } else {
                upscaledImage[newIndex + threadIdx.z] = imageToUpscale[oldIndex + threadIdx.z];
            }
        }
    }
}

__global__ void upscaleFromUpscaledImage(uint8_t* imageToUpscale, uint8_t* upscaledImage, uint32_t width, uint8_t upscaleFactor, uint8_t bytePerPixel)
{
    // get the pixel position in the upscaled image vector
    uint32_t newIndex = ((((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x) * bytePerPixel;

    // convert the position in a matrix notation
    uint32_t newi = newIndex / (width * upscaleFactor * bytePerPixel);
    uint32_t newj = (newIndex - (newi * width * upscaleFactor * bytePerPixel)) / bytePerPixel;

    // compute the position of the pixel to copy from the original image
    uint32_t i = newi / upscaleFactor;
    uint32_t j = newj / upscaleFactor;
    uint32_t oldIndex = (i * width + j) * bytePerPixel;

    // manage single channel if tridimensional version, else manage all the others
    if (blockDim.z == 1) {
        for (int k = 0; k < bytePerPixel; k++)
            upscaledImage[newIndex + k] = imageToUpscale[oldIndex + k];
    } else {
        upscaledImage[newIndex + threadIdx.z] = imageToUpscale[oldIndex + threadIdx.z];
    }        
}

void gpuUpscaler(size_t originalSize, size_t upscaledSize, uint8_t upscaleFactor, Settings settings, uint8_t* data, uint32_t width, uint32_t height, uint32_t bytePerPixel, string imageName)
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
    timer.start();
    switch (settings.upscalerType)
    {
        case UpscalerType::UpscaleFromOriginalImage:
            upscaleFromOriginalImage << <grid, block >> > (d_data, d_out, width, upscaleFactor, bytePerPixel);
            break;
        case UpscalerType::UpscaleFromUpscaledImage:
            upscaleFromUpscaledImage << <grid, block >> > (d_data, d_out, width, upscaleFactor, bytePerPixel);
            break;
    }
    timer.stop();

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
}
