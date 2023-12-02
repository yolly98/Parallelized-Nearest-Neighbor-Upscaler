#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "Headers.h"
#include "GpuTimer.cuh"

using namespace std;

void saveImageToPNG(const char* filename, const float* imageData, int width, int height) 
{
    // create the uint8_t array
    uint8_t* byteImageData = new uint8_t[width * height * 4];

    // convert from float to uint8_t ([0, 1] -> [0, 255])
    for (int i = 0; i < width * height * 4; ++i)
        byteImageData[i] = (uint8_t)(imageData[i] * 255);

    // save the image
    stbi_write_png(filename, width, height, 4, byteImageData, width * 4);
    delete[] byteImageData;
}

__global__ void upscaleImage(cudaTextureObject_t texObj, float* upscaledImage, uint32_t width, uint32_t height, uint8_t bytePerPixel, size_t originalSize, uint32_t upscaleFactor)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < height && j < width) {
        // compute the position of the first pixel to duplicate in upscaled image
        uint32_t newi = i * upscaleFactor;
        uint32_t newj = j * upscaleFactor;
        uint32_t upscaledWidth = width * upscaleFactor;

        // convert to normalized coordinates
        float u = j / (float)width;
        float v = i / (float)height;

        // iterate the pixel to duplicate in upscaled image
        for (int m = newi; m < newi + upscaleFactor; m++) {
            for (int n = newj; n < newj + upscaleFactor; n++) {
                // compute the pixel position in the upscaled image vector
                uint32_t newIndex = (m * upscaledWidth + n) * bytePerPixel;

                // copy the pixel
                float4 pixelToCopy = tex2D<float4>(texObj, u, v);
                memcpy(&upscaledImage[newIndex], &pixelToCopy, sizeof(float4));
            }
        }
    }

    /*
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < height && y < width) {
        uint32_t newIndex = (x * width + y) * bytePerPixel;

        // convert to normalized coordinates
        float u = y / (float)width;
        float v = x / (float)height;

        // copy the pixel
        float4 pixelToCopy = tex2D<float4>(texObj, u, v);
        memcpy(&copiedImage[newIndex], &pixelToCopy, sizeof(float4));
    }*/
}

int main(int argc, char* argv[]) 
{
    int channel = Channels::RGB_ALPHA;
    string inputImageName;
    uint8_t upscaleFactor;

    // read image name from input parameters
    if (argc >= 2) {
        inputImageName = argv[1];
        upscaleFactor = atoi(argv[2]);
    }
    else {
        inputImageName = "img/in-small.png";
        upscaleFactor = 2;
    }

    // open the image
    uint32_t width, height, bytePerPixel;
    stbi_ldr_to_hdr_gamma(1.0f);
    float* data = stbi_loadf(inputImageName.c_str(), (int*)&width, (int*)&height, (int*)&bytePerPixel, channel);

    if (!data) {
        cout << "[-] Image not found" << endl;
        return -1;
    }

    // event based timer
    GpuTimer timer;

    // ----------------------------------- Texture Setup -----------------------------------------

    // allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // copy data located at address h_data in host memory to device memory
    const size_t spitch = width * 4 * sizeof(float);
    cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * 4 * sizeof(float), height, cudaMemcpyHostToDevice);

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
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // -------------------------------------------------------------------------------------------

    // create array for the copied image
    size_t originalSize = height * width * bytePerPixel;
    size_t upscaledSize = originalSize * upscaleFactor * upscaleFactor;
    float* copiedImage = new float[upscaledSize];

    // allocate GPU memory for output array 
    float* d_out;
    cudaMalloc((void**)&d_out, upscaledSize * sizeof(float));

    // define resources for the execution
    dim3 block(32, 16);
    dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y);

    // run the kernel
    timer.start();
    upscaleImage << <grid, block >> > (texObj, d_out, width, height, bytePerPixel, originalSize, upscaleFactor);
    timer.stop();
    cout << "[+] (GPU) Time needed: " << timer.getElapsedMilliseconds() << "ms" << endl;

    // retrieve result
    cudaDeviceSynchronize();
    cudaMemcpy(copiedImage, d_out, upscaledSize * sizeof(float), cudaMemcpyDeviceToHost);

    // save image
    saveImageToPNG("img/TEST.png", copiedImage, width * upscaleFactor, height * upscaleFactor);

    // free device memory
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_out);
    delete[] copiedImage;

    // Free host memory
    stbi_image_free(data);
    return 0;
}
