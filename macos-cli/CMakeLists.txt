cmake_minimum_required(VERSION 3.14)

project(MatchTool_macOS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Shared core library
add_library(match_core STATIC match_core.cpp match_core.h)
target_include_directories(match_core PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(match_core PUBLIC ${OpenCV_LIBS})

# ARM NEON + dot product for Apple Silicon
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(match_core PRIVATE -march=armv8.4-a+dotprod)
endif()

# GUI application
add_executable(MatchTool
    main.cpp
    mainwindow.h
    mainwindow.cpp
)
target_link_libraries(MatchTool PRIVATE match_core Qt6::Widgets)

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(match_core PRIVATE -O3)
    target_compile_options(MatchTool PRIVATE -O3)
endif()
