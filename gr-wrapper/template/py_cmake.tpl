#import datetime
\#  timestamp: $datetime.datetime.now().strftime('%c')
\#  Author: $author
\#  Description: $description

\########################################################################
\# Include python install macros
\########################################################################
include(GrPython)
if(NOT PYTHONINTERP_FOUND)
    return()
endif()

\########################################################################
\# Install python sources
\########################################################################
GR_PYTHON_INSTALL(
    FILES
    __init__.py
    DESTINATION \${GR_PYTHON_DIR}/${prefix}
)

\########################################################################
\# Handle the unit tests
\########################################################################
include(GrTest)

set(GR_TEST_TARGET_DEPS gnuradio-${prefix})
set(GR_TEST_PYTHON_DIRS \${CMAKE_BINARY_DIR}/swig)
GR_ADD_TEST(qa_${prefix} \${PYTHON_EXECUTABLE} \${CMAKE_CURRENT_SOURCE_DIR}/qa_${prefix}.py)
