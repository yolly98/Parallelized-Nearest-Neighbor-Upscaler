# Parallelized Nearest Neighbor Upscaler

Computer Architecture project.

## Test

To test the `cpu upscaler`:
```[bash]
.\x64\Release\Parallelized-Nearest-Neighbor-Upscaler.exe IMAGE_NAME UPSCALE_FACTOR RESULTS_FILE "cpu" NUMBER_OF_THREADS REPETITIONS
```

To test the `gpu upscaler`:
```[bash]
.\x64\Release\Parallelized-Nearest-Neighbor-Upscaler.exe IMAGE_NAME UPSCALE_FACTOR RESULTS_FILE "gpu" UPSCALER_TYPE THREADS_PER_BLOCK_X THREADS_PER_BLOCK_Z PIXELS_HANDLED_BY_THREAD REPETITIONS
```

If the gpu upscaler is `UpscaleWithTextureObject`:
```[bash]
.\x64\Release\Parallelized-Nearest-Neighbor-Upscaler.exe IMAGE_NAME UPSCALE_FACTOR RESULTS_FILE "gpu" UPSCALER_TYPE THREADS_PER_BLOCK_X PIXELS_HANDLED_BY_THREAD REPETITIONS
```

The upscaler type can be:
* *0*: Upscale from Original Image
* *1*: Upscale from Upscaled Image
* *2*: Upscale with Single Thread