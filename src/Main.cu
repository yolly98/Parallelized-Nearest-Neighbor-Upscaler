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
    int channel = Channels::RGB_ALPHA;
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
    uint32_t width, height, bytePerPixel;
    uint8_t* data = stbi_load(inputImageName.c_str(), (int*)&width, (int*)&height, (int*)&bytePerPixel, channel);

    if (!data) {
        cout << "[-] Image not found" << endl;
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
    cpuUpscaler(upscaleFactor, data, width, height, bytePerPixel, "img/singleCPU.png");
    
    // GPU upscaler with one thread per block
    Settings settings;
    settings.threadsPerBlockX = 1;
    settings.threadsPerBlockY = 1;
    settings.threadsPerBlockZ = 1;
    settings.blocksPerGridX = width / (settings.threadsPerBlockX * settings.threadsPerBlockY);
    settings.blocksPerGridY = height;
    settings.blocksPerGridZ = 1;
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel);

    // GPU upscaler with array of threads and arry of blocks
    settings.threadsPerBlockX = 32;
    settings.threadsPerBlockY = 1;
    settings.threadsPerBlockZ = 1;
    settings.blocksPerGridX = width * height / (settings.threadsPerBlockX * settings.threadsPerBlockY);
    settings.blocksPerGridY = 1;
    settings.blocksPerGridZ = 1;
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel);

    // GPU upscaler with bidimensional matrix of threads and bidimensional matrix of blocks
    settings.threadsPerBlockX = 8;
    settings.threadsPerBlockY = 4;
    settings.threadsPerBlockZ = 1;
    settings.blocksPerGridX = width / (settings.threadsPerBlockX * settings.threadsPerBlockY);
    settings.blocksPerGridY = height;
    settings.blocksPerGridZ = 1;
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel, "img/GPU.png");

    // free image
    stbi_image_free(data);
    return 0;
}