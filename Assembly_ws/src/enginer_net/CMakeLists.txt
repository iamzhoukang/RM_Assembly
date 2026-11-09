cmake_minimum_required(VERSION 3.16)
project(enginer_net)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(yaml-cpp REQUIRED)

add_library(yolo_detector src/yolo_detector.cpp)
target_include_directories(yolo_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_detector PUBLIC openvino::runtime yaml-cpp ${OpenCV_LIBS})

add_library(litehr_detector src/litehr_detector.cpp)
target_include_directories(litehr_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(litehr_detector PUBLIC openvino::runtime ${OpenCV_LIBS})

add_library(detector src/detector.cpp)
target_link_libraries(detector PUBLIC yolo_detector litehr_detector)

add_executable(detector_example tools/detector_example.cpp)
ament_target_dependencies(detector_example ament_index_cpp)
target_link_libraries(detector_example detector ${OpenCV_LIBS})

install(TARGETS detector_example DESTINATION lib/${PROJECT_NAME})
install(FILES tools/example.jpg DESTINATION share/${PROJECT_NAME}/tools)

install(TARGETS detector yolo_detector litehr_detector
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config weights DESTINATION share/${PROJECT_NAME})
ament_export_include_directories(include)
ament_export_libraries(detector yolo_detector litehr_detector)
ament_export_dependencies(OpenCV OpenVINO yaml-cpp)
ament_package()
