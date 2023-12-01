import subprocess

REPETITIONS = 30
UPSCALE_RANGE = [ 2, 5 ]
GPU_THREAD_RANGE = [ 32, 128, 256, 512, 1024, 1536 ]
SMALL_IMAGE = { 'name': '"../img/in-small.png"', 'width': 640, 'height': 479, 'byte_per_pixel': 4 }
LARGE_IMAGE = { 'name': '"../img/in-large.png"', 'width': 5472, 'height': 3648, 'byte_per_pixel': 4 }
EXECUTABLE = '..\\x64\Release\Parallelized-Nearest-Neighbor-Upscaler.exe'

def execute(parameters: list):
    parameters = [str(param) for param in parameters]
    cmd = f"{EXECUTABLE} {' '.join(parameters)}"
    subprocess.run(cmd, capture_output=True)

if __name__ == '__main__':
    # GPU (one thread per pixel) with UpscaleFromOriginalImage
    for upscale_factor in range(*UPSCALE_RANGE):
        for threads in GPU_THREAD_RANGE:
            parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_fromOriginalImage_small.csv"', '"gpu"', threads, 1, 0, REPETITIONS ]
            execute(parameters)
            parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_fromOriginalImage_large.csv"', '"gpu"', threads, 1, 0, REPETITIONS ]
            execute(parameters)

    # GPU (one thread per channel) with UpscaleFromOriginalImage
    for upscale_factor in range(*UPSCALE_RANGE):
        for threads in GPU_THREAD_RANGE:
            parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_channel_fromOriginalImage_small.csv"', '"gpu"', threads, SMALL_IMAGE['byte_per_pixel'], 0, REPETITIONS ]
            execute(parameters)
            parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_channel_fromOriginalImage_large.csv"', '"gpu"', threads, LARGE_IMAGE['byte_per_pixel'], 0, REPETITIONS ]
            execute(parameters)

    # GPU (one thread per pixel) with UpscaleFromUpscaledImage
    for upscale_factor in range(*UPSCALE_RANGE):
        for threads in GPU_THREAD_RANGE:
            parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_fromUpscaledImage_small.csv"', '"gpu"', threads, 1, 1, REPETITIONS ]
            execute(parameters)
            parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_fromUpscaledImage_large.csv"', '"gpu"', threads, 1, 1, REPETITIONS ]
            execute(parameters)

    # GPU (one thread per channel) with UpscaleFromUpscaledImage
    for upscale_factor in range(*UPSCALE_RANGE):
        for threads in GPU_THREAD_RANGE:
            parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_channel_fromUpscaledImage_small.csv"', '"gpu"', threads, SMALL_IMAGE['byte_per_pixel'], 1, REPETITIONS ]
            execute(parameters)
            parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_channel_fromUpscaledImage_large.csv"', '"gpu"', threads, LARGE_IMAGE['byte_per_pixel'], 1, REPETITIONS ]
            execute(parameters)
