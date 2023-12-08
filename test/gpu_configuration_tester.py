import subprocess

REPETITIONS = 30
UPSCALE_RANGE = [ 2, 4, 6 ]
GPU_THREAD_RANGE = [ 32, 64, 128, 256, 512, 1024 ]
PIXELS_HANDLED_BY_THREAD = [ 1, 2, 4, 8, 16, 32, 64, 128 ]
MAX_THREADS_PER_BLOCK = 1024
SMALL_IMAGE = { 'name': '"../img/in-small.png"', 'width': 640, 'height': 479, 'byte_per_pixel': 4 }
LARGE_IMAGE = { 'name': '"../img/in-large.png"', 'width': 5472, 'height': 3648, 'byte_per_pixel': 4 }
EXECUTABLE = '..\\x64\Release\Parallelized-Nearest-Neighbor-Upscaler.exe'

def execute(parameters: list):
    parameters = [str(param) for param in parameters]
    cmd = f"{EXECUTABLE} {' '.join(parameters)}"
    subprocess.run(cmd, capture_output=True)

if __name__ == '__main__':
    # GPU (one thread per pixel) with UpscaleFromOriginalImage
    for upscale_factor in UPSCALE_RANGE:
        for threads in GPU_THREAD_RANGE:
            for pixels_handled in PIXELS_HANDLED_BY_THREAD:
                parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_fromOriginalImage_small.csv"', '"gpu"', 0, threads, 1, pixels_handled, REPETITIONS ]
                execute(parameters)
                parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_fromOriginalImage_large.csv"', '"gpu"', 0, threads, 1, pixels_handled, REPETITIONS ]
                execute(parameters)

        print(f'--> Completed Upscale Factor {upscale_factor}')
    
    print('[+] Completed GPU (one thread per pixel) with UpscaleFromOriginalImage')

    # GPU (one thread per channel) with UpscaleFromOriginalImage
    for upscale_factor in UPSCALE_RANGE:
        for threads in GPU_THREAD_RANGE:
            for pixels_handled in PIXELS_HANDLED_BY_THREAD:
                if threads * SMALL_IMAGE['byte_per_pixel'] <= MAX_THREADS_PER_BLOCK:
                    parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_channel_fromOriginalImage_small.csv"', '"gpu"', 0, threads, SMALL_IMAGE['byte_per_pixel'], pixels_handled, REPETITIONS ]
                    execute(parameters)
                if threads * LARGE_IMAGE['byte_per_pixel'] <= MAX_THREADS_PER_BLOCK:
                    parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_channel_fromOriginalImage_large.csv"', '"gpu"', 0, threads, LARGE_IMAGE['byte_per_pixel'], pixels_handled, REPETITIONS ]
                    execute(parameters)

        print(f'--> Completed Upscale Factor {upscale_factor}')
    
    print('[+] Completed GPU (one thread per channel) with UpscaleFromOriginalImage')

    # GPU (one thread per pixel) with UpscaleFromUpscaledImage
    for upscale_factor in UPSCALE_RANGE:
        for threads in GPU_THREAD_RANGE:
            for pixels_handled in PIXELS_HANDLED_BY_THREAD:
                parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_fromUpscaledImage_small.csv"', '"gpu"', 1, threads, 1, pixels_handled, REPETITIONS ]
                execute(parameters)
                parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_fromUpscaledImage_large.csv"', '"gpu"', 1, threads, 1, pixels_handled, REPETITIONS ]
                execute(parameters)

        print(f'--> Completed Upscale Factor {upscale_factor}')
    
    print('[+] Completed GPU (one thread per pixel) with UpscaleFromUpscaledImage')

    # GPU (one thread per channel) with UpscaleFromUpscaledImage
    for upscale_factor in UPSCALE_RANGE:
        for threads in GPU_THREAD_RANGE:
            for pixels_handled in PIXELS_HANDLED_BY_THREAD:
                if threads * SMALL_IMAGE['byte_per_pixel'] <= MAX_THREADS_PER_BLOCK:
                    parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_channel_fromUpscaledImage_small.csv"', '"gpu"', 1, threads, SMALL_IMAGE['byte_per_pixel'], pixels_handled, REPETITIONS ]
                    execute(parameters)
                if threads * LARGE_IMAGE['byte_per_pixel'] <= MAX_THREADS_PER_BLOCK:
                    parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_channel_fromUpscaledImage_large.csv"', '"gpu"', 1, threads, LARGE_IMAGE['byte_per_pixel'], pixels_handled, REPETITIONS ]
                    execute(parameters)

        print(f'--> Completed Upscale Factor {upscale_factor}')
    
    print('[+] Completed GPU (one thread per channel) with UpscaleFromUpscaledImage')

    # GPU with UpscaleWithTextureObject
    for upscale_factor in UPSCALE_RANGE:
        for threads in GPU_THREAD_RANGE:
            for pixels_handled in PIXELS_HANDLED_BY_THREAD:
                if threads <= MAX_THREADS_PER_BLOCK:
                    parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_withTextureObject_small.csv"', '"gpu"', 3, threads, pixels_handled, REPETITIONS ]
                    execute(parameters)
                if threads <= MAX_THREADS_PER_BLOCK:
                    parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_thread_per_pixel_withTextureObject_large.csv"', '"gpu"', 3, threads, pixels_handled, REPETITIONS ]
                    execute(parameters)

        print(f'--> Completed Upscale Factor {upscale_factor}')
    
    print('[+] Completed GPU with UpscaleWithTextureObject')
