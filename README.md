# Parallelized Nearest Neighbor Upscaler

Computer Architecture project.

## Test

To test the `cpu upscaler`:
```[bash]
.\x64\Release\Parallelized-Nearest-Neighbor-Upscaler.exe IMAGE_NAME UPSCALE_FACTOR RESULTS_FILE "cpu" NUMBER_OF_THREADS REPETITIONS
```

To test the `gpu upscaler`:
```[bash]
.\x64\Release\Parallelized-Nearest-Neighbor-Upscaler.exe IMAGE_NAME UPSCALE_FACTOR RESULTS_FILE "gpu" THREADS_PER_BLOCK_X THREADS_PER_BLOCK_Z UPSCALER_TYPE REPETITIONS
```

The upscaler type can be:
* *0*: Upscale from Original Image
* *1*: Upscale from Upscaled Image
* *2*: Upscale with Single Thread