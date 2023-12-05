#include <vector>
#include <numeric>
#include <cmath>

#include "cuda_runtime.h"

std::pair<float, float> computeCI(const std::vector<float>& values) {
    // compute mean
    float mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    // compute standard deviation
    float variance = 0.0;
    for (float value : values)
        variance += (value - mean) * (value - mean);
    variance /= (values.size() - 1);
    float standardDeviation = std::sqrt(variance);

    // compute the margin of error
    float standardError = standardDeviation / std::sqrt(static_cast<float>(values.size()));
    float marginOfError = 1.96 * standardError;

    return std::make_pair(mean - marginOfError, mean + marginOfError);
}

cudaTextureObject_t createTextureObject(uint32_t width, uint32_t height, uint32_t bytePerPixel, uint8_t* data)
{
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

    return texObj;
}
