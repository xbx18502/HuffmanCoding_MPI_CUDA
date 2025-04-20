# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(MPICompressAndDecompress C CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MPI_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi")

# 设置目标可执行文件
add_executable(MPICompress MPICompress.c 
                ../include/serialFunctions.c
)

set_source_files_properties(MPICompress.c ../include/serialFunctions.c PROPERTIES LANGUAGE C)

install(TARGETS MPICompress
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/../bin)


# Enable device code linking
set_target_properties(MPICompress PROPERTIES
    OUTPUT_NAME "MPI_compress"
)

# Set include directories
target_include_directories(MPICompress PRIVATE
    ${NVSHMEM_HOME}/include
    ${MPI_HOME}/include
)

# Set link directories
target_link_directories(MPICompress PRIVATE
    ${NVSHMEM_HOME}/lib
    ${MPI_HOME}/lib
)

# Link libraries
target_link_libraries(MPICompress PRIVATE
    mpi
)    


add_executable(MPIDecompress MPIDecompress.c
                ../include/serialFunctions.c

)

set_source_files_properties(MPIDecompress.c ../include/serialFunctions.c PROPERTIES LANGUAGE C)

install(TARGETS MPIDecompress
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/../bin)


# Enable device code linking
set_target_properties(MPIDecompress PROPERTIES
    OUTPUT_NAME "MPI_decompress"
)

# Set include directories
target_include_directories(MPIDecompress PRIVATE
    ${NVSHMEM_HOME}/include
    ${MPI_HOME}/include
)

# Set link directories
target_link_directories(MPIDecompress PRIVATE
    ${NVSHMEM_HOME}/lib
    ${MPI_HOME}/lib
)

# Link libraries
target_link_libraries(MPIDecompress PRIVATE
    mpi
)    