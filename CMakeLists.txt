project("DeepSDF")
cmake_minimum_required(VERSION 3.0)

# Add directory with macros
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

#find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(nanoflann REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

add_subdirectory(third-party/cnpy)
add_subdirectory(third-party/CL11)

set(SOURCES
    src/ShaderProgram.cpp
    src/Utils.cpp)

add_executable(PreprocessMesh src/PreprocessMesh.cpp ${SOURCES})
target_include_directories(PreprocessMesh PRIVATE third-party/cnpy ${Pangolin_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR})
target_link_libraries(PreprocessMesh PRIVATE CLI11::CLI11 ${Pangolin_LIBRARY} cnpy Eigen3::Eigen ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
target_compile_features(PreprocessMesh PRIVATE cxx_std_17)
set_target_properties(PreprocessMesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(SampleVisibleMeshSurface src/SampleVisibleMeshSurface.cpp ${SOURCES})
target_include_directories(SampleVisibleMeshSurface PRIVATE third-party/cnpy ${Pangolin_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR})
target_link_libraries(SampleVisibleMeshSurface PRIVATE CLI11::CLI11 ${Pangolin_LIBRARY} cnpy Eigen3::Eigen ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
target_compile_features(SampleVisibleMeshSurface PRIVATE cxx_std_17)
set_target_properties(SampleVisibleMeshSurface PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
