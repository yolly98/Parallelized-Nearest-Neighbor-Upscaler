import subprocess
import math

REPETITIONS = 30
UPSCALE_FACTOR = 2
THREADS_PER_BLOCK_RANGE = [ 32, 64, 128 ]
PIXELS_HANDLED = 32
IMAGE_RESOLUTION_RANGE = [ (640, 360), (854, 480), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160) ]
LARGE_IMAGE = { 'name': '../img/in-large.png', 'width': 5472, 'height': 3648, 'byte_per_pixel': 4 }
EXECUTABLE = '../bin/upscaler'

def execute(parameters: list):
    parameters = [str(param) for param in parameters]
    cmd = [ EXECUTABLE ] + parameters
    subprocess.run(cmd, capture_output=True)

if __name__ == '__main__':
    # Test Upscalers Varying Image Size
    for threads_per_block in THREADS_PER_BLOCK_RANGE:
        for image_resolution in IMAGE_RESOLUTION_RANGE:
            parameters = [ LARGE_IMAGE['name'], UPSCALE_FACTOR, 'results/GPU_upscale_with_different_images.csv', 'gpu', 0, threads_per_block, PIXELS_HANDLED, REPETITIONS, image_resolution[0], image_resolution[1] ]
            execute(parameters)
            parameters = [ LARGE_IMAGE['name'], UPSCALE_FACTOR, 'results/GPU_upscale_optimized_with_different_images.csv', 'gpu', 1, threads_per_block, PIXELS_HANDLED, REPETITIONS, image_resolution[0], image_resolution[1] ]
            execute(parameters)

            print(f'- Completed {image_resolution[0]}x{image_resolution[1]} image')
        print(f'[+] Completed {threads_per_block} threads per block')

    # Test Upscalers Varying Pixel Handled by Thread
    for threads_per_block in THREADS_PER_BLOCK_RANGE:
        for image_resolution in IMAGE_RESOLUTION_RANGE[1:4]:
            # compute pixels handled range
            upscaled_image_size = image_resolution[0] * image_resolution[1] * UPSCALE_FACTOR * UPSCALE_FACTOR
            max_exponent = math.floor(math.log2(upscaled_image_size))
            pixels_handled_range = [ 2**i for i in range(0, max_exponent + 1) ] + [ upscaled_image_size ]

            # iterate pixels handled
            for pixels_handled in pixels_handled_range:
                parameters = [ LARGE_IMAGE['name'], UPSCALE_FACTOR, 'results/GPU_upscale_with_different_pixels_handled.csv', 'gpu', 0, threads_per_block, pixels_handled, REPETITIONS, image_resolution[0], image_resolution[1] ]
                execute(parameters)
                parameters = [ LARGE_IMAGE['name'], UPSCALE_FACTOR, 'results/GPU_upscale_optimized_with_different_pixels_handled.csv', 'gpu', 1, threads_per_block, pixels_handled, REPETITIONS, image_resolution[0], image_resolution[1] ]
                execute(parameters)

            print(f'- Completed {image_resolution[0]}x{image_resolution[1]} image')
        print(f'[+] Completed {threads_per_block} threads per block')
