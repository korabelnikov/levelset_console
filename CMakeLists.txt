cmake_minimum_required(VERSION 2.8)
project(Levelset-app)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -openmp")

find_package(AGTK REQUIRED)
include(${AGTK_USE_FILE})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(OpenCL REQUIRED)

find_package(Boost REQUIRED)

set(IncSegSourcePath "" CACHE PATH "Path to autoplan incremental_segmentation plugin src")

#
include(${IncSegSourcePath}/../write_gpu_kernels.cmake)

set(OpenCLSrcs )
set(Kernels
  ${IncSegSourcePath}/HistogramPyramides/HistogramPyramids.cl
  ${IncSegSourcePath}/Levelset/levelset.cl
)

write_gpu_kernels( ${OpenCLSrcs} ${Kernels})

include_directories(
${Boost_INCLUDE_DIR}
${CMAKE_CURRENT_BINARY_DIR}
${IncSegSourcePath}
${IncSegSourcePath}/OpenCLUtilities
${IncSegSourcePath}/Levelset/
${IncSegSourcePath}/HistogramPyramides/
)

add_executable(levelsetSeg 
main.cxx 
${IncSegSourcePath}/Levelset/LevelsetOpenCL.cpp
${IncSegSourcePath}/Levelset/LevelsetOpenCL.h
${IncSegSourcePath}/HistogramPyramides/HistogramPyramids.cpp
${IncSegSourcePath}/HistogramPyramides/HistogramPyramids.hpp
${IncSegSourcePath}/OpenCLUtilities/openCLUtilities.cpp
${IncSegSourcePath}/OpenCLUtilities/openCLUtilities.hpp
${OpenCLSrcs}
)

target_link_libraries(levelsetSeg ${AGTK_LIBRARIES} ${ITK_LIBRARIES})


add_executable(resizeTest 
resizeTest.cxx
${IncSegSourcePath}/Levelset/LevelsetOpenCL.cpp
${IncSegSourcePath}/Levelset/LevelsetOpenCL.h
${IncSegSourcePath}/OpenCLUtilities/openCLUtilities.cpp
${IncSegSourcePath}/OpenCLUtilities/openCLUtilities.hpp
)
target_link_libraries(resizeTest ${AGTK_LIBRARIES} ${ITK_LIBRARIES})
