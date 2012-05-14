#  timestamp: Sun May 13 20:00:52 2012
#  Author: Xingzhong
#  Description: Template according to the gr-howto-write-a-block-cmake in GNU-Radio

#
# Include python install macros
#
include(GrPython)
if(NOT PYTHONINTERP_FOUND)
    return()
endif()

#
# Install python sources
#
GR_PYTHON_INSTALL(
    FILES
    __init__.py
    DESTINATION ${GR_PYTHON_DIR}/ssp
)

#
# Handle the unit tests
#
include(GrTest)

set(GR_TEST_TARGET_DEPS gnuradio-ssp)
set(GR_TEST_PYTHON_DIRS ${CMAKE_BINARY_DIR}/swig)
GR_ADD_TEST(qa_ssp ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_ssp.py)
