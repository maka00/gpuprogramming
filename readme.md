# GPU Programming

NVidias [Thrust](https://developer.nvidia.com/thrust) library is used to interact with the GPU
CLion integration works fine (code highlighting, code completion, debugging, ...)

The used system:
```
> nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.63.01    Driver Version: 470.63.01    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:0B:00.0  On |                  N/A |
|  0%   57C    P0    39W / 151W |    694MiB /  8116MiB |      7%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
This is a GTX 1070 (Pascal)

### Remarks regarding building with Thrust:

* don't use -pedantic warnings! This creates a lot of warnings which are of no use in this case.
* at least on arch linux the Thrust and CUB cmake targets are not found. Set them manually 
via **Thrust_DIR** and **CUB_DIR**
* All Thurst code has to go into files with the .cu extension! If not the nvidia compiler will not be used.

**CMAKE  example file:**
```
project(gpuapp LANGUAGES CXX CUDA)
set(SOURCES main.cu )

set( Thrust_DIR /opt/cuda/include/thrust/cmake)
set( CUB_DIR /opt/cuda/include/cub/cmake/)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC Thrust )
```



| directory | descritption
|---|---|
|gpuapp | a minimal gpu example (with Thrust) |
|imageproc | a possible interop between OpenCV and Thrust (shared object on the GPU).<br>**This will only compile if you have a OpenCV version compiled with CUDA!** |
