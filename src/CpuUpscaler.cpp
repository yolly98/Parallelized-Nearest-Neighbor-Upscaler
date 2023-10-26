#include <iostream>
#include <cstring>

#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

#include "Timer.h"
#include "Headers.h"

using namespace std;

void cpuUpscaler(uint8_t upscaleFactor, uint8_t* originalImage, size_t width, size_t height, uint32_t bytePerPixel, string imageName)
{
    uint8_t* upscaledImage = new uint8_t[(width * upscaleFactor * bytePerPixel) * (height * upscaleFactor)];

    size_t sizeOriginalImage = width * bytePerPixel * height; 
    uint32_t upscaledWidth = width * upscaleFactor;

    Timer timer;

    // iterate the pixels of the original image
    for (size_t oldIndex = 0; oldIndex < sizeOriginalImage; oldIndex += bytePerPixel) {
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
                    upscaledImage[newIndex + k] = originalImage[oldIndex + k];
            }
        }
    }

    // print the upscale duration
    float time = timer.getElapsedMilliseconds();
    cout << "[+] (CPU) Time needed: " << time << "ms" << endl;

    // save image as PNG
    if (imageName != "") {
        cout << "[+] Saving image..." << endl;
        if (stbi_write_png(imageName.c_str(), width * upscaleFactor, height * upscaleFactor, bytePerPixel, upscaledImage, width * upscaleFactor * bytePerPixel))
            cout << "[+] Image saved successfully" << endl;
        else
            cout << "[-] Failed to save image" << endl;
    }

    delete[] upscaledImage;
}
