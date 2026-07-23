cmake_minimum_required(VERSION 3.20)

# Hand-written CUDA Driver API RPC server + pipeline client.
#
# Build scenarios:
#   GPU machine:  cmake -DZLINK_CUDA_EXAMPLES=ON ..  (builds server + client)
#   CPU machine:  cmake -DZLINK_CUDA_EXAMPLES=ON ..  (builds client only)

find_package(CUDAToolkit)

add_executable(cuda_client cuda_client.cpp)
target_link_libraries(cuda_client PRIVATE zlink_core)
target_include_directories(cuda_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(CUDAToolkit_FOUND)
    add_executable(cuda_server cuda_server.cpp)
    target_link_libraries(cuda_server PRIVATE
        zlink_core
        CUDA::cuda_driver
    )
    target_include_directories(cuda_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
else()
    message(STATUS "CUDA toolkit not found - cuda_server target skipped")
endif()
