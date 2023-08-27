include(FetchContent)

FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

find_package(OpenMP REQUIRED)

add_executable(bench_rnndescent bench_rnndescent.cpp)
target_include_directories(bench_rnndescent PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_link_libraries(bench_rnndescent 
    rnndescent 
    OpenMP::OpenMP_CXX 
    argparse 
    nlohmann_json::nlohmann_json
)

add_executable(bench_hnsw bench_hnsw.cpp)
target_include_directories(bench_hnsw PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_link_libraries(bench_hnsw 
    faiss 
    OpenMP::OpenMP_CXX 
    argparse 
    nlohmann_json::nlohmann_json
)

add_executable(bench_nsg bench_nsg.cpp)
target_include_directories(bench_nsg PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_link_libraries(bench_nsg
    faiss 
    OpenMP::OpenMP_CXX 
    argparse 
    nlohmann_json::nlohmann_json
)

add_executable(bench_nndescent bench_nndescent.cpp)
target_include_directories(bench_nndescent PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_link_libraries(bench_nndescent 
    faiss 
    OpenMP::OpenMP_CXX 
    argparse 
    nlohmann_json::nlohmann_json
)