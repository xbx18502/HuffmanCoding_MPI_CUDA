# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(Huffmancoding)

# 添加子目录
# 检查是否需要编译CUDAMPI部分
option(BUILD_CUDAMPI "Build CUDA MPI components" ON)

# 有条件地添加CUDA子目录
add_subdirectory(CUDAMPI)

add_subdirectory(MPI)