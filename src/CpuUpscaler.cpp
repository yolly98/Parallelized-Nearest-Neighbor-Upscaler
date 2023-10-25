#include <iostream>
#include <cstring>

#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

#include "Timer.h"

using namespace std;

struct Pixel {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;

    Pixel() :
        red(255), green(255), blue(255), alpha(255) {};
    Pixel(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
        red(r), green(g), blue(b), alpha(a) {};
};

void cpuUpscaler(uint8_t upscaleFactor, uint8_t* data, size_t width, size_t height, uint32_t bytePerPixel, string imageName)
{
    Pixel* originalImage = (Pixel*)data;
    Pixel* upscaledImage = new Pixel[(width * upscaleFactor) * (height * upscaleFactor)];

    size_t sizeOriginalImage = width * height;
    size_t sizeRowUpscaledImage = width * upscaleFactor;

    Timer timer;

    // indices to iterate the upscaled image
    uint32_t i = 0;
    uint32_t j = 0;

    // iterate the original image
    for (uint32_t k = 0; k < sizeOriginalImage; k++) {
        // duplicate original pixel in the upscaled image
        for (uint32_t n = 0; n < upscaleFactor; n++)
            for (uint32_t m = 0; m < upscaleFactor; m++)
                upscaledImage[(i + n) * sizeRowUpscaledImage + (j + m)] = originalImage[k];

        // compute new indices for the next pixels in the upscaled image
        if (j >= sizeRowUpscaledImage - upscaleFactor) {
            i += upscaleFactor;
            j = 0;
        } else {
            j += upscaleFactor;
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