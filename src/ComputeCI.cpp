#include <vector>
#include <numeric>
#include <cmath>

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