# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)
set(CUDAToolkit_ROOT "/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6")
set(CMAKE_CUDA_COMPILER "/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/bin/nvcc")
# 设置项目名称
project(CUDAMPICompress CUDA CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CUDA_COMPILER_FORCED TRUE)

set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_COMPILER_FORCED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -ccbin g++")
set(CMAKE_CUDA_LINK_FLAGS "${CMAKE_CUDA_LINK_FLAGS} -rdc=true")
# 仅让 nvcc 识别 -maxconstdata 选项
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-maxconstdata=0x18000")

set(NVCC_GENCODE "arch=compute_90,code=sm_90")
set(NVSHMEM_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/nvshmem")
set(HYDRA_HOME "$NVSHMEM_HOME")
set(CUDA_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6")
set(MPI_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi")
set(NCCL_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl")

# 设置目标可执行文件
add_executable(CUDAMPICompress CUDAMPICompress.cu 
                ../include/parallelFunctions.cu 
                ../include/GPUWrapper.cu 
                ../include/kernel.cu )

install(TARGETS CUDAMPICompress
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/../bin)

# Enable device code linking
set_target_properties(CUDAMPICompress PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "CUDAMPI_compress"
)

# Set include directories
target_include_directories(CUDAMPICompress PRIVATE
    ${NVSHMEM_HOME}/include
    ${MPI_HOME}/include
)

# Set link directories
target_link_directories(CUDAMPICompress PRIVATE
    ${NVSHMEM_HOME}/lib
    ${MPI_HOME}/lib
)

# Link libraries
target_link_libraries(CUDAMPICompress PRIVATE
    mpi
)    