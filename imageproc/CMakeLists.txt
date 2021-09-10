cmake_minimum_required(VERSION 3.19)
project(imageproc LANGUAGES CXX CUDA)

set(SOURCES main.cu )

set( Thrust_DIR /opt/cuda/include/thrust/cmake)
set( CUB_DIR /opt/cuda/include/cub/cmake/)

find_package(Thrust REQUIRED CONFIG)
find_package(Threads)
find_package( OpenCV REQUIRED )
find_package(PkgConfig REQUIRED)
find_package(Boost COMPONENTS filesystem iostreams system program_options timer REQUIRED)
thrust_create_target(Thrust)

enable_testing()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-D_DEBUG=1)
endif()
add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        DEBUG_POSTFIX "d"
        CXX_STANDARD 20
        )

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${PROJECT_SOURCE_DIR}

        PRIVATE
        ${BOOST_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}

       )
# CMake 3.14 onwards
target_link_directories(${PROJECT_NAME}
        PRIVATE
        ../lib
)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${CONAN_LIBS}
        Threads::Threads
        Boost::program_options
        Boost::filesystem
        Boost::timer
        Thrust
        opencv_core
        opencv_videoio
        opencv_highgui
        )
target_compile_features(${PROJECT_NAME}
        PUBLIC
        cxx_std_20
        )

target_compile_options(${PROJECT_NAME}
        PUBLIC
        PRIVATE
        -Wall -Wextra
        )
