include(FetchContent)

FetchContent_Declare(
    zpp_bits
    GIT_REPOSITORY https://github.com/eyalz800/zpp_bits.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(zpp_bits)

# zpp_bits is header-only; create an INTERFACE target if not provided
if(NOT TARGET zpp_bits)
    add_library(zpp_bits INTERFACE)
    target_include_directories(zpp_bits INTERFACE
        ${zpp_bits_SOURCE_DIR}
    )
endif()

find_package(Threads REQUIRED)
