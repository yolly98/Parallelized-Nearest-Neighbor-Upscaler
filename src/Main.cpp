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
    uint8_t upscaleFactor;

    // read image name from input parameters
    if (argc >= 2) {
        inputImageName = argv[1];
        upscaleFactor = atoi(argv[2]);
    } else {
        inputImageName = "img/in-small.png";
        upscaleFactor = 2;
    }

    // open the image
    uint32_t width, height, bytePerPixel, channel;
    uint8_t* data = stbi_load(inputImageName.c_str(), (int*)&width, (int*)&height, (int*)&bytePerPixel, channel);

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
    
    // single core CPU upscaler
    cpuUpscaler(1, upscaleFactor, data, width, height, bytePerPixel, "img/CPU1.png");

    // multi core CPU upscaler
    cpuUpscaler(16, upscaleFactor, data, width, height, bytePerPixel, "img/CPU2.png");

    // GPU Upscaler with Texture Object
    Settings settings(128, UpscalerType::UpscaleWithTextureObject, width, height, upscaleFactor, 2);
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel, "img/GPU1.png");

    // GPU Upscaler with Texture Object Optimized
    settings = Settings(128, UpscalerType::UpscaleWithTextureObjectOptimized, width, height, upscaleFactor, 128);
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel, "img/GPU2.png");

    // free image
    stbi_image_free(data);
    return 0;
}
