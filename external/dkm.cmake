project(dkm)

set(DKM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dkm/include)

add_library(${PROJECT_NAME} INTERFACE)
add_library(point_sampler::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} INTERFACE ${DKM_DIR})
