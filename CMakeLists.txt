
find_package(OpenCV REQUIRED)

add_subdirectory(./ball_dectection)

# Add OpenCV headers location to your include paths

include_directories(${OpenCV_INCLUDE_DIRS})

# Declare the executable target built from your sources

add_executable(main main.cpp)

# Link your application with OpenCV libraries

target_link_libraries(main ${OpenCV_LIBS})