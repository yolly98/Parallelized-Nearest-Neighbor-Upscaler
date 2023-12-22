import subprocess

REPETITIONS = 30
UPSCALE_FACTOR = 2
CPU_THREAD_RANGE = [ 1, 2, 4, 8, 16, 32, 64, 128 ]
IMAGE_RESOLUTION_RANGE = [ (854, 480), (1280, 720), (1920, 1080) ]
LARGE_IMAGE = { 'name': '../img/in-large.png', 'width': 5472, 'height': 3648, 'byte_per_pixel': 4 }
EXECUTABLE = '../bin/upscaler'

def execute(parameters: list):
    parameters = [str(param) for param in parameters]
    cmd = [EXECUTABLE] + parameters
    subprocess.run(cmd, capture_output=True)

if __name__ == '__main__':
    # iterate all threads
    for threads in CPU_THREAD_RANGE:
        for image_resolution in IMAGE_RESOLUTION_RANGE:
            width = image_resolution[0]
            height = image_resolution[1]
            parameters = [ LARGE_IMAGE['name'], UPSCALE_FACTOR, f'results/CPU_multi_thread_{width}_{height}.csv', 'cpu', threads, REPETITIONS, width, height]
            execute(parameters)
    
        print(f'[+] Completed {threads} Threads')
