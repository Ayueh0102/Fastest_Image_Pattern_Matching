cmake_minimum_required(VERSION 3.16)
project(MatchToolDesktop LANGUAGES C CXX)
# Compatibility entry point. GUI and CLI now share the portable C++ core.
set(FPM_BUILD_GUI ON CACHE BOOL "Build Qt6 GUI" FORCE)
set(FPM_WITH_OPENCV ON CACHE BOOL "CLI image codecs" FORCE)
enable_testing()
add_subdirectory(../cpp-core portable)
set_target_properties(fpm_gui PROPERTIES OUTPUT_NAME MatchTool RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
