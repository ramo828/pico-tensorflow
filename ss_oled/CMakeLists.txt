cmake_minimum_required(VERSION 3.12)
project(ss_oled C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
add_library(ss_oled INTERFACE)
target_sources(ss_oled INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/BitBang_I2C.c
  ${CMAKE_CURRENT_LIST_DIR}/ss_oled.c
  ${CMAKE_CURRENT_LIST_DIR}/ss_oled.cpp
)

target_include_directories(ss_oled INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

# Pull in pico libraries that we need
target_link_libraries(ss_oled INTERFACE pico_stdlib hardware_i2c)
