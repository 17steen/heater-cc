cmake_minimum_required(VERSION 3.22)
project(heater_cc)

set(CMAKE_CXX_STANDARD 23)


find_package(Boost 1.74 REQUIRED)
find_package(OpenCL REQUIRED)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(${PROJECT_NAME}
        heater.cc)

target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
