cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

# project name
set(PROJECT_NAME "cublas_examples")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -G -lineinfo --extended-lambda" )

# set(CUDA_NVCC_FLAGS "-g -G")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --extended-lambda" )

# add header files in ./utils
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE CPP_FILES "src/*.cu" "src/*.cpp" )

# add the executable
add_executable(${PROJECT_NAME}  main.cu ${CPP_FILES})
set_property(TARGET ${PROJECT_NAME}  PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(${PROJECT_NAME}  PUBLIC ${CUDA_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}  PUBLIC cublas CUDA::cudart CUDA::curand CUDA::cusolver)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)










