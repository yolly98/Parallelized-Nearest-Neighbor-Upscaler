#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class GpuTimer
{
public:
	GpuTimer() { cudaEventCreate(&m_Start); cudaEventCreate(&m_Stop); }

	void start() { cudaEventRecord(m_Start); }
	void stop() { cudaEventRecord(m_Stop); }

	float getElapsedMilliseconds()
	{
		cudaEventSynchronize(m_Stop);
		float time = 0.0f;
		cudaEventElapsedTime(&time, m_Start, m_Stop);
		return time;
	}

private:
	cudaEvent_t m_Start;
	cudaEvent_t m_Stop;
};