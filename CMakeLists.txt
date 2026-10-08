cmake_minimum_required(VERSION 3.16)                  # Mindestversion von CMake
include($ENV{IDF_PATH}/tools/cmake/project.cmake)     # lädt das ESP-IDF-Buildsystem (setzt den ESP32-Compiler)
project(esp32-mnist-benchmark)                        # Name des Projekts