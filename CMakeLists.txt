cmake_minimum_required(VERSION 3.5)
project(141044090_Emre_Aydin)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -std=c++11 ")

set(SOURCE_FILES FindVehicles.cpp Constants.h)
add_executable(141044090_Emre_Aydin ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(141044090_Emre_Aydin ${OpenCV_LIBS}) # replace binary_name
