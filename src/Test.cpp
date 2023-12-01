#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "Headers.h"

using namespace std;

int main(int argc, char* argv[])
{
    // read test parameters
    if (argc >= 4) {
        // get image info
        string inputImageName = argv[1];
        uint8_t upscaleFactor = atoi(argv[2]);
        ofstream file;
        file.open(argv[3], std::ios_base::app);

        // open the image
        int channel = Channels::RGB_ALPHA;
        uint32_t width, height, bytePerPixel;
        uint8_t* data = stbi_load(inputImageName.c_str(), (int*)&width, (int*)&height, (int*)&bytePerPixel, channel);
        
        // iterate the required upscaler
        if (!strcmp(argv[4], "cpu")) {

            uint32_t numThreads = atoi(argv[5]);
            uint32_t numRepetitions = atoi(argv[6]);
            string result = to_string(upscaleFactor) + ";" + to_string(numThreads);

            for (uint32_t i = 0; i < numRepetitions; i++) {
                if (numThreads == 1) {
                    float elapsedTime = cpuUpscaler(upscaleFactor, data, width, height, bytePerPixel);
                    result = result + ";" + to_string(elapsedTime);
                } else {
                    float elapsedTime = cpuMultithreadUpscaler(numThreads, upscaleFactor, data, width, height, bytePerPixel);
                    result = result + ";" + to_string(elapsedTime);
                }
            }

            result += "\n";
            std::replace(result.begin(), result.end(), '.', ',');
            file.write(result.c_str(), result.size());
        } else if (!strcmp(argv[4], "gpu")) {

            uint32_t numRepetitions = atoi(argv[8]);

            // compute the new upscaled image size
            size_t originalSize = height * width * bytePerPixel * sizeof(uint8_t);
            size_t upscaledSize = height * upscaleFactor * width * upscaleFactor * bytePerPixel * sizeof(uint8_t);

            // get upscale settings from parameters
            Settings settings(atoi(argv[5]), atoi(argv[6]), static_cast<UpscalerType>(atoi(argv[7])), width, height, upscaleFactor);
            string result = to_string(upscaleFactor) + ";" + settings.toString();

            // repeate the tests
            vector<float> elapsedTimes;
            for (uint32_t i = 0; i < numRepetitions; i++) {
                float elapsedTime = gpuUpscaler(originalSize, upscaledSize, upscaleFactor, settings, data, width, height, bytePerPixel);
                elapsedTimes.push_back(elapsedTime);
            }

            // compute the arithmetic mean of the elapsed times, the 95% confidence interval and the worst time
            float meanElapsedTime = std::accumulate(elapsedTimes.begin(), elapsedTimes.end(), 0.0) / elapsedTimes.size();
            std::pair<float, float> confidenceInterval = computeCI(elapsedTimes);
            auto it = std::max_element(elapsedTimes.begin(), elapsedTimes.end());
            float worstElapsedTime = *it;

            // convert and save the values in the CSV
            result = result + ";" + to_string(meanElapsedTime) + ";" + to_string(confidenceInterval.first) + ";" + to_string(confidenceInterval.second) + ";" + to_string(worstElapsedTime) + "\n";
            std::replace(result.begin(), result.end(), '.', ',');
            file.write(result.c_str(), result.size());
        } else {
            cerr << "[-] Wrong parameters" << endl;
            exit(-1);
        }

    } else {
        cerr << "[-] Not enough parameters" << endl;
        exit(-1);
    }
}