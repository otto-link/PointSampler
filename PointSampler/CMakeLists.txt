project(point_sampler)

# --- sources GUI headers need to be added in add_executable, otherwise the moc
# won't parse them
file(GLOB_RECURSE PS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} STATIC ${PS_SOURCES})

set(PS_INCLUDE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE)

target_include_directories(${PROJECT_NAME} PRIVATE ${PS_INCLUDE}
                                                   ${CMAKE_BINARY_DIR}/include)

# Link libraries
target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC nanoflann::nanoflann
  INTERFACE point_sampler::dkm)
