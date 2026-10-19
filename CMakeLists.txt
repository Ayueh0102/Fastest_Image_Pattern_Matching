cmake_minimum_required(VERSION 3.16)

set(project_name "MatchTool")

project(${project_name})

enable_testing()
add_subdirectory(cpp-core)
option(FPM_BUILD_MFC "Build the original Windows MFC application" ${MSVC})
if(FPM_BUILD_MFC)
if(NOT MSVC)
    message(FATAL_ERROR "FPM_BUILD_MFC requires Visual Studio/MFC; use FPM_BUILD_GUI for the Qt GUI")
endif()

#for MFC
add_definitions(-D_AFXDLL)
set(CMAKE_MFC_FLAG 2)
add_definitions(-DUNICODE -D_UNICODE)

#for opencv4
add_definitions(-DOPENCV_4X)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV CONFIG REQUIRED)

add_executable(${project_name} WIN32
                               #MatchTool/CvvImage.h
                               MatchTool/framework.h 
                               MatchTool/MatchTool.h
                               MatchTool/MatchToolDlg.h
                               MatchTool/pch.h
                               MatchTool/resource.h
                               MatchTool/targetver.h

                               #MatchTool/arm_version.cpp
                               #CvvImage.cpp
                               MatchTool/MatchTool.cpp
                               MatchTool/MatchToolDlg.cpp
                               MatchTool/pch.cpp
                               
                               MatchTool/MatchTool.rc
                               MatchTool/MatchTool.Lang
                               MatchTool/res/MatchTool.ico
                               MatchTool/res/MatchTool.rc2
)

# Copy MatchTool.Lang to the output directory
add_custom_command(TARGET ${project_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/MatchTool/MatchTool.Lang
        $<TARGET_FILE_DIR:${project_name}>
)

target_link_libraries(${project_name} PRIVATE opencv_core opencv_imgproc opencv_highgui)
endif()
