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

__global__ void upscaleImage(cudaTextureObject_t texObj, uint8_t* __restrict__ upscaledImage, uint32_t pixelsHandledByBlock, uint32_t pixelsHandledByThread, uint32_t width, uint32_t height, uint8_t bytePerPixel, uint32_t upscaleFactor)
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
        inputImageName = "img/in-large.png";
        upscaleFactor = 2;
    }

    // open the image
    uint32_t width, height, bytePerPixel;
    uint8_t* data = stbi_load(inputImageName.c_str(), (int*)&width, (int*)&height, (int*)&bytePerPixel, channel);

    if (!data) {
        cout << "[-] Image not found" << endl;
        return -1;
    }

    // event based timer
    GpuTimer timer;

    // ----------------------------------- Texture Setup -----------------------------------------

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

    // -------------------------------------------------------------------------------------------

    // create array for the copied image
    size_t originalSize = height * width * bytePerPixel;
    size_t upscaledSize = originalSize * upscaleFactor * upscaleFactor;
    uint8_t* copiedImage = new uint8_t[upscaledSize];

    // allocate GPU memory for output array 
    uint8_t* d_out;
    cudaMalloc((void**)&d_out, upscaledSize);

    // define resources for the execution
    dim3 block(128, 1, 1);
    dim3 grid(((width * height * upscaleFactor * upscaleFactor) + block.x - 1) / block.x, 1, 1);
    uint32_t pixelsHandledByThread = 1;
    uint32_t pixelsHandledByBlock = block.x * pixelsHandledByThread;

    // run the kernel
    timer.start();
    upscaleImage << <grid, block >> > (texObj, d_out, pixelsHandledByBlock, pixelsHandledByThread, width, height, bytePerPixel, upscaleFactor);
    timer.stop();
    cout << "[+] (GPU) Time needed: " << timer.getElapsedMilliseconds() << "ms" << endl;

    // retrieve result
    cudaDeviceSynchronize();
    cudaMemcpy(copiedImage, d_out, upscaledSize, cudaMemcpyDeviceToHost);

    // save image
    if (stbi_write_png("img/TEST.png", width * upscaleFactor, height * upscaleFactor, bytePerPixel, copiedImage, width * upscaleFactor * bytePerPixel))
        cout << "[+] Image saved successfully" << endl;

    // free device memory
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_out);
    delete[] copiedImage;

    // Free host memory
    stbi_image_free(data);
    return 0;
}
