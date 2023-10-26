import subprocess

REPETITIONS = 5
UPSCALE_RANGE = [ 2, 5 ]
CPU_THREAD_RANGE = [ 2, 4, 6, 8, 16 ]
SMALL_IMAGE = { 'name': '"../img/in-small.png"', 'width': 640, 'height': 479, 'byte_per_pixel': 4 }
LARGE_IMAGE = { 'name': '"../img/in-large.png"', 'width': 5472, 'height': 3648, 'byte_per_pixel': 4 }
EXECUTABLE = '..\\x64\Debug\Parallelized-Nearest-Neighbor-Upscaler.exe'

def execute(parameters: list):
    parameters = [str(param) for param in parameters]
    cmd = f"{EXECUTABLE} {' '.join(parameters)}"
    subprocess.run(cmd, capture_output=True)


if __name__ == '__main__':
    # CPU Single Thread Test
    for upscale_factor in range(*UPSCALE_RANGE):
        parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/CPU_single_thread_small.csv"', '"cpu"', 1, REPETITIONS ]
        execute(parameters)
        parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/CPU_single_thread_large.csv"', '"cpu"', 1, REPETITIONS ]
        execute(parameters)
    
    print('[+] Completed CPU Single Thread Test')

    # CPU Multi Thread Test
    for upscale_factor in range(*UPSCALE_RANGE):
        for n_threads in CPU_THREAD_RANGE:
            parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/CPU_multi_thread_small.csv"', '"cpu"', n_threads, REPETITIONS ]
            execute(parameters)
            parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/CPU_multi_thread_large.csv"', '"cpu"', n_threads, REPETITIONS ]
            execute(parameters)
    
    print('[+] Completed CPU Multi Thread Test')

    # GPU Single Thread Test
    for upscale_factor in range(*UPSCALE_RANGE):
        parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_single_thread_small.csv"', '"gpu"', 1, 1, 1, 1, 1, 1, 2, REPETITIONS ]
        execute(parameters)
        parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_single_thread_large.csv"', '"gpu"', 1, 1, 1, 1, 1, 1, 2, REPETITIONS ]
        execute(parameters)
    
    print('[+] Completed GPU Single Thread Test')

    # GPU Bidimensional with UpscaleFromOriginalImage
    for upscale_factor in range(*UPSCALE_RANGE):
        for threads in range(*CPU_THREAD_RANGE):
            blocks_per_grid_x = SMALL_IMAGE['width'] / (threads * threads)
            blocks_per_grid_y = SMALL_IMAGE['height']
            parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_multi_thread_bidimensional_fromOriginalImage_small.csv"', '"gpu"', threads, threads, 1, blocks_per_grid_x, blocks_per_grid_y, 1, 0, REPETITIONS ]
            execute(parameters)
            blocks_per_grid_x = LARGE_IMAGE['width'] / (threads * threads)
            blocks_per_grid_y = LARGE_IMAGE['height']
            parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_multi_thread_bidimensional_fromOriginalImage_large.csv"', '"gpu"', threads, threads, 1, blocks_per_grid_x, blocks_per_grid_y, 1, 0, REPETITIONS ]
            execute(parameters)
    
    print('[+] Completed GPU Bidimensional with UpscaleFromOriginalImage')

    # GPU Tridimensional with UpscaleFromOriginalImage
    for upscale_factor in range(*UPSCALE_RANGE):
        for threads in range(*CPU_THREAD_RANGE):
            blocks_per_grid_x = SMALL_IMAGE['width'] / (threads * threads * SMALL_IMAGE['byte_per_pixel'])
            blocks_per_grid_y = SMALL_IMAGE['height']
            parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_multi_thread_tridimensional_fromOriginalImage_small.csv"', '"gpu"', threads, threads, SMALL_IMAGE['byte_per_pixel'], blocks_per_grid_x, blocks_per_grid_y, 1, 0, REPETITIONS ]
            execute(parameters)
            blocks_per_grid_x = LARGE_IMAGE['width'] / (threads * threads * LARGE_IMAGE['byte_per_pixel'])
            blocks_per_grid_y = LARGE_IMAGE['height']
            parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_multi_thread_tridimensional_fromOriginalImage_large.csv"', '"gpu"', threads, threads, LARGE_IMAGE['byte_per_pixel'], blocks_per_grid_x, blocks_per_grid_y, 1, 0, REPETITIONS ]
            execute(parameters)
    
    print('[+] Completed GPU Tridimensional with UpscaleFromOriginalImage')

    # GPU Bidimensional with Upscaler UpscaleFromUpscaledImage
    for upscale_factor in range(*UPSCALE_RANGE):
        for threads in range(*CPU_THREAD_RANGE):
            blocks_per_grid_x = SMALL_IMAGE['width'] / (threads * threads)
            blocks_per_grid_y = SMALL_IMAGE['height']
            parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_multi_thread_bidimensional_fromUpscaledImage_small.csv"', '"gpu"', threads, threads, 1, blocks_per_grid_x, blocks_per_grid_y, 1, 1, REPETITIONS ]
            execute(parameters)
            blocks_per_grid_x = LARGE_IMAGE['width'] / (threads * threads)
            blocks_per_grid_y = LARGE_IMAGE['height']
            parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_multi_thread_bidimensional_fromUpscaledImage_large.csv"', '"gpu"', threads, threads, 1, blocks_per_grid_x, blocks_per_grid_y, 1, 1, REPETITIONS ]
            execute(parameters)
    
    print('[+] Completed GPU Bidimensional with Upscaler UpscaleFromUpscaledImage')

    # GPU Tridimensional with Upscaler UpscaleFromUpscaledImage
    for upscale_factor in range(*UPSCALE_RANGE):
        for threads in range(*CPU_THREAD_RANGE):
            blocks_per_grid_x = SMALL_IMAGE['width'] / (threads * threads * SMALL_IMAGE['byte_per_pixel'])
            blocks_per_grid_y = SMALL_IMAGE['height']
            parameters = [ SMALL_IMAGE['name'], upscale_factor, '"results/GPU_multi_thread_tridimensional_fromUpscaledImage_small.csv"', '"gpu"', threads, threads, SMALL_IMAGE['byte_per_pixel'], blocks_per_grid_x, blocks_per_grid_y, 1, 1, REPETITIONS ]
            execute(parameters)
            blocks_per_grid_x = LARGE_IMAGE['width'] / (threads * threads * LARGE_IMAGE['byte_per_pixel'])
            blocks_per_grid_y = LARGE_IMAGE['height']
            parameters = [ LARGE_IMAGE['name'], upscale_factor, '"results/GPU_multi_thread_tridimensional_fromUpscaledImage_large.csv"', '"gpu"', threads, threads, LARGE_IMAGE['byte_per_pixel'], blocks_per_grid_x, blocks_per_grid_y, 1, 1, REPETITIONS ]
            execute(parameters)
    
    print('[+] Completed GPU Tridimensional with Upscaler UpscaleFromUpscaledImage')
