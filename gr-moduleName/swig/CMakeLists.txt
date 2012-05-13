#  timestamp: Sun May 13 18:04:03 2012
#  Author: Xingzhong
#  Description: Template according to the gr-howto-write-a-block-cmake in GNU-Radio

#
# Include swig generation macros
#
find_package(SWIG)
find_package(PythonLibs)
if(NOT SWIG_FOUND OR NOT PYTHONLIBS_FOUND)
    return()
endif()
include(GrSwig)
include(GrPython)

#
# Setup swig generation
#
foreach(incdir ${GNURADIO_CORE_INCLUDE_DIRS})
    list(APPEND GR_SWIG_INCLUDE_DIR ${incdir}/swig)
endforeach(incdir)

set(GR_SWIG_LIBRARIES gnuradio-ssp)
set(GR_SWIG_DOC_FILE ${CMAKE_CURRENT_BINARY_DIR}/ssp_swig_doc.i)
set(GR_SWIG_DOC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include)

GR_SWIG_MAKE(ssp_swig ssp_swig.i)

#
# Install the build swig module
#
GR_SWIG_INSTALL(TARGETS ssp_swig DESTINATION ${GR_PYTHON_DIR}/ssp)

#
# Install swig .i files for development
#
install(
    FILES
    ssp_swig.i
    ${CMAKE_CURRENT_BINARY_DIR}/ssp_swig_doc.i
    DESTINATION ${GR_INCLUDE_DIR}/ssp/swig
)
