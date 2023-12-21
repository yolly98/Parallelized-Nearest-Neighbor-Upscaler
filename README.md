# Parallelized Nearest Neighbor Upscaler

Computer Architecture project.

## Test

To compile the test main:
```[bash]
make mode=release main=test
```

To test the `cpu upscaler`:
```[bash]
bin/upscaler IMAGE_NAME UPSCALE_FACTOR RESULTS_FILE "cpu" NUMBER_OF_THREADS REPETITIONS [WIDTH] [HEIGHT]
```

To test the `gpu upscaler`:
```[bash]
bin/upscaler IMAGE_NAME UPSCALE_FACTOR RESULTS_FILE "gpu" UPSCALER_TYPE THREADS_PER_BLOCK PIXELS_HANDLED_BY_THREAD REPETITIONS [WIDTH] [HEIGHT]
```

The upscaler type can be:
* *0*: Upscale with Texture Object
* *1*: Upscale with Texture Object Optimized
