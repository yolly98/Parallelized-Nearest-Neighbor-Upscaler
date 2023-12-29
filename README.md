# Parallelized Nearest Neighbor Upscaler
<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/882/882731.png" alt="canva-logo" height="64px"/>
</p>

The aim of this project is to develop an **image upscaler** taking advantage of the parallelization capabilities of the modern **CPU** and **GPU**. Image upscaling is the process that allows to increase the resolution of an image, trying to minimize the loss in image quality.
There are several possible algorithms and the one chosen for the project is the nearest neighbor interpolation. The nearest neighbor is the simplest upscaling method in which each pixel in the upscaled image is assigned the value of its nearest neighbor in the original image. 

## How to Run

The project consists of three main files:

* `Main`: execute multiple upscales on CPU and GPU using different configurations.
* `ProfileMain`: setup a profile scenario for *Nvidia Nsight Compute* through command line arguments.
* `Test`: execute a specific configuration through command line arguments.

To compile on linux:
```[bash]
make mode=[debug/release] main=[main/profile/test]
```

To test the `CPU upscaler`:
```[bash]
make mode=release main=test
bin/upscaler IMAGE_NAME UPSCALE_FACTOR RESULTS_FILE "cpu" NUMBER_OF_THREADS REPETITIONS [WIDTH] [HEIGHT]
```

To test the `GPU upscaler`:
```[bash]
make mode=release main=test
bin/upscaler IMAGE_NAME UPSCALE_FACTOR RESULTS_FILE "gpu" UPSCALER_TYPE THREADS_PER_BLOCK PIXELS_HANDLED_BY_THREAD REPETITIONS [WIDTH] [HEIGHT]
```

The upscaler type can be:
* *0*: Upscale with Texture Object
* *1*: Upscale with Texture Object Optimized

## Authors

* [Biagio Cornacchia](https://github.com/biagiocornacchia)
* [Gianluca Gemini](https://github.com/yolly98)
* [Matteo Abaterusso](https://github.com/MatteoAba)