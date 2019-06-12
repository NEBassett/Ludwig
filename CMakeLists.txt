cmake_minimum_required (VERSION 2.6)
project (ludwig)

find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Boost 1.68 REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
add_executable (ludwig "${PROJECT_SOURCE_DIR}/src/main.cpp" "${PROJECT_SOURCE_DIR}/src/ludwig.cpp")
set(CMAKE_CXX_STANDARD 17)
set_source_files_properties( "${PROJECT_SOURCE_DIR}/src/main.cpp" "${PROJECT_SOURCE_DIR}/src/ludwig.cpp" PROPERTIES COMPILE_FLAGS "-std=gnu++17 -g -DGLEW_STATIC -O3" )

include_directories(${Boost_INCLUDE_DIR})

target_link_libraries(ludwig glfw)

target_include_directories(ludwig PUBLIC "${GLEW_INCLUDE_DIR}" "${OPENGL_INCLUDE_DIRS}" "${GLFW_INCLUDE_DIRS}")
target_link_libraries(ludwig "${GLEW_LIBRARIES}" "${OPENGL_LIBRARIES}" "${GLFW_LIBRARIES}")

target_include_directories(ludwig PUBLIC "${GLEW_INCLUDE_DIR}" "${OPENGL_INCLUDE_DIRS}")
target_link_libraries(ludwig "${GLEW_LIBRARIES}" "${OPENGL_LIBRARIES}")

set(GLM_DIR "${PROJECT_SOURCE_DIR}/glm")
target_include_directories(ludwig PUBLIC "${GLM_DIR}")
