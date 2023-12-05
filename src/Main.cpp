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
    cpuUpscaler(upscaleFactor, data, width, height, bytePerPixel, "img/CPU1.png");

    // multi core CPU upscaler
    cout << "\n---------------------------------------------------------------" << endl << endl;
    cpuMultithreadUpscaler(16, upscaleFactor, data, width, height, bytePerPixel, "img/CPU2.png");
    
    // GPU upscaler with one thread per block using UpscaleFromOrginalImage kernel
    Settings settings;
    settings.threadsPerBlockX = 128;
    settings.threadsPerBlockY = 1;
    settings.threadsPerBlockZ = 1;
    settings.blocksPerGridX = ((width * height) + settings.threadsPerBlockX - 1) / settings.threadsPerBlockX;
    settings.blocksPerGridY = 1;
    settings.blocksPerGridZ = 1;
    settings.upscalerType = UpscalerType::UpscaleFromOriginalImage;
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel, "img/GPU1.png");
    
    // GPU upscaler in which a threads manages a pixel channes and using UpscaleFromOrginalImage kernel
    settings.threadsPerBlockX = 128;
    settings.threadsPerBlockY = 1;
    settings.threadsPerBlockZ = bytePerPixel;
    settings.blocksPerGridX = ((width * height) + settings.threadsPerBlockX - 1) / settings.threadsPerBlockX;
    settings.blocksPerGridY = 1;
    settings.blocksPerGridZ = 1;
    settings.upscalerType = UpscalerType::UpscaleFromOriginalImage;
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel, "img/GPU2.png");

    // GPU upscaler with array of threads and array of blocks using UpscaleFromUpscaledImage kernel
    settings.threadsPerBlockX = 128;
    settings.threadsPerBlockY = 1;
    settings.threadsPerBlockZ = 1;
    settings.blocksPerGridX = ((width * height * settings.threadsPerBlockZ * upscaleFactor * upscaleFactor) + (settings.threadsPerBlockX * settings.threadsPerBlockZ) - 1) / (settings.threadsPerBlockX * settings.threadsPerBlockZ);
    settings.blocksPerGridY =  1;
    settings.blocksPerGridZ = 1;
    settings.upscalerType = UpscalerType::UpscaleFromUpscaledImage;
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel, "img/GPU3.png");

    // GPU upscaler in which a threads manages a pixel channes and using UpscaleFromUpscaledImage kernel
    settings.threadsPerBlockX = 128;
    settings.threadsPerBlockY = 1;
    settings.threadsPerBlockZ = bytePerPixel;
    settings.blocksPerGridX = ((width * height * settings.threadsPerBlockZ * upscaleFactor * upscaleFactor) + (settings.threadsPerBlockX * settings.threadsPerBlockZ) - 1) / (settings.threadsPerBlockX * settings.threadsPerBlockZ);
    settings.blocksPerGridY = 1;
    settings.blocksPerGridZ = 1;
    settings.upscalerType = UpscalerType::UpscaleFromUpscaledImage;
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel, "img/GPU4.png");

    // GPU upscaler with a single thread
    settings.threadsPerBlockX = 1;
    settings.threadsPerBlockY = 1;
    settings.threadsPerBlockZ = 1;
    settings.blocksPerGridX = 1;
    settings.blocksPerGridY = 1;
    settings.blocksPerGridZ = 1;
    settings.upscalerType = UpscalerType::UpscaleWithSingleThread;
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel, "img/GPU5.png");

    // GPU upscaler with Texture Object
    settings.threadsPerBlockX = 128;
    settings.threadsPerBlockY = 1;
    settings.threadsPerBlockZ = 1;
    settings.pixelsHandledByThread = 1;
    settings.pixelsHandledByBlock = settings.pixelsHandledByThread * settings.threadsPerBlockX;
    settings.blocksPerGridX = ((width * height * upscaleFactor * upscaleFactor) + settings.threadsPerBlockX - 1) / settings.threadsPerBlockX;;
    settings.blocksPerGridY = 1;
    settings.blocksPerGridZ = 1;
    settings.upscalerType = UpscalerType::UpscaleWithTextureObject;
    gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel, "img/GPU6.png");

    // free image
    stbi_image_free(data);
    return 0;
}
