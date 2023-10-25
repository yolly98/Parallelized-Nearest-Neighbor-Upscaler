#include <chrono>
using namespace std;

class Timer
{
public:
	Timer() { reset(); }
	void reset() { m_StartingTime = chrono::high_resolution_clock::now(); }
	float getElapsedMilliseconds() { return getElapsed() * 1000.0f; }

private:
	float getElapsed() { return chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - m_StartingTime).count() * 0.001f * 0.001f * 0.001f; }

private:
	chrono::time_point<chrono::high_resolution_clock> m_StartingTime;
};