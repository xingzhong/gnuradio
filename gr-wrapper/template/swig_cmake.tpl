#import datetime
\#  timestamp: $datetime.datetime.now().strftime('%c')
\#  Author: $author
\#  Description: $description

\########################################################################
\# Include swig generation macros
\########################################################################
find_package(SWIG)
find_package(PythonLibs)
if(NOT SWIG_FOUND OR NOT PYTHONLIBS_FOUND)
    return()
endif()
include(GrSwig)
include(GrPython)

\########################################################################
\# Setup swig generation
\########################################################################
foreach(incdir \${GNURADIO_CORE_INCLUDE_DIRS})
    list(APPEND GR_SWIG_INCLUDE_DIR \${incdir}/swig)
endforeach(incdir)

set(GR_SWIG_LIBRARIES gnuradio-${prefix})
set(GR_SWIG_DOC_FILE \${CMAKE_CURRENT_BINARY_DIR}/${prefix}_swig_doc.i)
set(GR_SWIG_DOC_DIRS \${CMAKE_CURRENT_SOURCE_DIR}/../include)

GR_SWIG_MAKE(${prefix}_swig ${prefix}_swig.i)

\########################################################################
\# Install the build swig module
\########################################################################
GR_SWIG_INSTALL(TARGETS ${prefix}_swig DESTINATION \${GR_PYTHON_DIR}/${prefix})

\########################################################################
\# Install swig .i files for development
\########################################################################
install(
    FILES
    ${prefix}_swig.i
    \${CMAKE_CURRENT_BINARY_DIR}/${prefix}_swig_doc.i
    DESTINATION \${GR_INCLUDE_DIR}/${prefix}/swig
)
