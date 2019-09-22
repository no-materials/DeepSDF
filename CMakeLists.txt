project("DeepSDF")
cmake_minimum_required(VERSION 3.0)

# Add directory with macros
#set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

#find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(nanoflann REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

set(TINYXML2_SOURCE
        src/external/tinyxml2/src/tinyxml2.cpp
        )

set(TINYXML2_HEADERS
        src/external/tinyxml2/include/tinyxml2.h
        )

include_directories(src/external/tinyxml2/include/)
include_directories(src/external/Simple-OpenGL-Image-Library/include/)

add_library(TINYXML2 STATIC
        ${TINYXML2_SOURCE}
        ${TINYXML2_HEADERS}
        )
target_link_libraries(TINYXML2)
set_target_properties(TINYXML2
        PROPERTIES
        PROJECT_LABEL tinyxml2
        FOLDER "Libraries"
        )

add_subdirectory(third-party/cnpy)
add_subdirectory(third-party/CL11)
add_subdirectory(src/external/Simple-OpenGL-Image-Library)

set(SOURCES
        src/ShaderProgram.cpp
        src/Utils.cpp
        src/common/util.cpp
        src/common/util.h
        src/common/shader.cpp
        src/common/shader.h
        src/common/camera.cpp
        src/common/camera.h
        src/common/model.cpp
        src/common/model.h
        src/common/texture.cpp
        src/common/texture.h
        src/shaders/fragmentShader.glsl
        src/shaders/vertexShader.glsl
        src/shaders/geometryShader.glsl)

add_executable(PreprocessMesh src/PreprocessMesh.cpp ${SOURCES})
target_include_directories(PreprocessMesh PRIVATE third-party/cnpy ${Pangolin_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR} /usr/local/Cellar/glfw/3.3/include/GLFW )
target_link_libraries(PreprocessMesh PRIVATE CLI11::CLI11 ${Pangolin_LIBRARY} cnpy Eigen3::Eigen ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} glfw glm SOIL TINYXML2)
target_compile_features(PreprocessMesh PRIVATE cxx_std_17)
set_target_properties(PreprocessMesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(SampleVisibleMeshSurface src/SampleVisibleMeshSurface.cpp ${SOURCES})
target_include_directories(SampleVisibleMeshSurface PRIVATE third-party/cnpy ${Pangolin_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR})
target_link_libraries(SampleVisibleMeshSurface PRIVATE CLI11::CLI11 ${Pangolin_LIBRARY} cnpy Eigen3::Eigen ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
target_compile_features(SampleVisibleMeshSurface PRIVATE cxx_std_17)
set_target_properties(SampleVisibleMeshSurface PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
