#include <stdio.h>
#include <iostream>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "Headers.h"

using namespace std;

int main(int argc, char* argv[])
{
    string inputImageName;
    uint8_t upscaleFactor = 2;
    UpscalerType upscaler;
    uint32_t width, height, threadsPerBlock, pixelsHandledByThread;

    // read parameters
    if (argc >= 2) {
        inputImageName = "img/in-large.png";
        upscaler = static_cast<UpscalerType>(atoi(argv[1]));
        threadsPerBlock = atoi(argv[2]);
        pixelsHandledByThread = atoi(argv[3]);
    } else {
        inputImageName = "img/in-small.png";
        upscaler = UpscalerType::UpscaleWithTextureObject;
        threadsPerBlock = 128;
        pixelsHandledByThread = 128;
    }

    // open the image
    uint32_t bytePerPixel, channel;
    uint8_t* data = stbi_load(inputImageName.c_str(), (int*)&width, (int*)&height, (int*)&bytePerPixel, channel);

    // read the custom size if given
    if (argc >= 6) {
        width = atoi(argv[4]);
        height = atoi(argv[5]);
    }

    // check if the image is loaded
    if (!data) {
        cout << "[-] Image not found" << endl;
        return -1;
    }

    // check if it has 4 channels
    if (bytePerPixel != 4) {
        cout << "[-] The image doesn't have 4 channels" << endl;
        return -1;
    }

    // image info
    cout << "[+] Input image name: " << inputImageName << endl;
    cout << "[+] Upscale factor: " << (uint32_t)upscaleFactor << endl;
    cout << "[+] Loaded image info: " << endl;
    cout << "--> Width: " << width << "px" << endl;
    cout << "--> Height: " << height << "px" << endl;
    cout << "--> Byte per Pixel: " << bytePerPixel << endl << endl;

    // compute the new upscaled image size
    size_t originalSize = height * width * bytePerPixel * sizeof(uint8_t);
    size_t upscaledSize = height * upscaleFactor * width * upscaleFactor * bytePerPixel * sizeof(uint8_t);

    // upscale with GPU
    if (upscaler == UpscalerType::UpscaleWithTextureObject){
        Settings settings = Settings(threadsPerBlock, UpscalerType::UpscaleWithTextureObject, width, height, upscaleFactor, pixelsHandledByThread);
        gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel);
    } else {
        Settings settings = Settings(threadsPerBlock, UpscalerType::UpscaleWithTextureObjectOptimized, width, height, upscaleFactor, pixelsHandledByThread);
        gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel);
    }

    // free image
    stbi_image_free(data);
    return 0;
}
