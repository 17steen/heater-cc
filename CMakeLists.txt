cmake_minimum_required(VERSION 3.22)
project(heater_cc)

set(CMAKE_CXX_STANDARD 23)


find_package(Boost 1.74 REQUIRED)
find_package(OpenCL REQUIRED)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_program(RESOURCE_COMPILER xxd)
set(KERNEL_FILE kernel.cl)
set(OUTPUT_FILE kernel.cl.h)

add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND cat ${KERNEL_FILE} | ${RESOURCE_COMPILER} -i > ${CMAKE_SOURCE_DIR}/${OUTPUT_FILE}
        COMMENT "Embedding kernel"
)

add_executable(${PROJECT_NAME}
        heater.cc kernel.cl.h)

target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
