cmake_minimum_required(VERSION 3.20)

add_executable(test_rpc test_rpc.cpp)
target_link_libraries(test_rpc PRIVATE zlink_core)

add_test(NAME test_rpc COMMAND test_rpc)
