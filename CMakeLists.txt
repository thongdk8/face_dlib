cmake_minimum_required (VERSION 3.2)
project(dlib_implementation)

find_package(dlib 19.6)
if(${dlib_FOUND})
    message("dlib information:")
    message("  dlib version: ${dlib_VERSION}")

    if (NOT TARGET dlib)
        add_library(dlib INTERFACE IMPORTED GLOBAL)
    endif()
else()
    message(FATAL_ERROR "dlib not found in the system, please install dlib")
endif()


find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


# Declare the executable target built from your sources
add_executable(dnn_mmod_face_detection_ex dnn_mmod_face_detection_ex.cpp)
# Link your application with OpenCV libraries
target_link_libraries(dnn_mmod_face_detection_ex ${dlib_LIBS} ${OpenCV_LIBS})

# Declare the executable target built from your sources
add_executable(face_detection_ex face_detection_ex.cpp)
# Link your application with OpenCV libraries
target_link_libraries(face_detection_ex ${dlib_LIBS} ${OpenCV_LIBS})

# Declare the executable target built from your sources
add_executable(dnn_export_face_chip dnn_export_face_chip.cpp)
# Link your application with OpenCV libraries
target_link_libraries(dnn_export_face_chip ${dlib_LIBS} ${OpenCV_LIBS})

# Declare the executable target built from your sources
add_executable(dnn_metric_learning_on_images_ex dnn_metric_learning_on_images_ex.cpp)
# Link your application with OpenCV libraries
target_link_libraries(dnn_metric_learning_on_images_ex ${dlib_LIBS} ${OpenCV_LIBS})
