add_library(rnndescent IndexRNNDescent.cpp RNNDescent.cpp)

target_include_directories(rnndescent PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

target_compile_definitions(rnndescent PRIVATE FINTEGER=int)

find_package(OpenMP REQUIRED)
target_link_libraries(rnndescent PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(rnndescent PUBLIC faiss)