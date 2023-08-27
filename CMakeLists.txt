cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(rnn-descent)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(faiss)
add_subdirectory(rnn-descent)
add_subdirectory(benches)
