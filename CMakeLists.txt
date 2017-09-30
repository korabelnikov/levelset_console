cmake_minimum_required(VERSION 2.8)
project(Levelset-app)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -openmp")

find_package(AGTK REQUIRED)
include(${AGTK_USE_FILE})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(OpenCL REQUIRED)

find_package(Boost REQUIRED)

set(IncSegSourcePath "" CACHE PATH)

include_directories(
${Boost_INCLUDE_DIR}
${Level-Set-Segmentation_INCLUDE_DIR}
${IncSegSourcePath}
${IncSegSourcePath}/OpenCL
${IncSegSourcePath}/OpenCLUtilities
${IncSegSourcePath}/OpenCLUtilityLibrary
)

add_executable(levelsetSeg 
main.cxx 
${IncSegSourcePath}/OpenCL/LevelsetOpenCL.cpp
${IncSegSourcePath}/OpenCLUtilities/openCLUtilities.cpp
${IncSegSourcePath}/OpenCLUtilityLibrary/HistogramPyramids.cpp
)

target_link_libraries(levelsetSeg ${AGTK_LIBRARIES} ${ITK_LIBRARIES})


add_executable(resizeTest 
resizeTest.cxx
${IncSegSourcePath}/OpenCL/LevelsetOpenCL.cpp
${IncSegSourcePath}/OpenCLUtilities/openCLUtilities.cpp
)
target_link_libraries(resizeTest ${AGTK_LIBRARIES} ${ITK_LIBRARIES})
