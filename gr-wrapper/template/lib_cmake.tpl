#import datetime
\#  timestamp: $datetime.datetime.now().strftime('%c')
\#  Author: $author
\#  Description: $description

\########################################################################
\# Setup library
\########################################################################
include(GrPlatform) \#define LIB_SUFFIX

add_library(gnuradio-${prefix} SHARED ${prefix}_${title}_${IOType}.cc )
target_link_libraries(gnuradio-${prefix} \${Boost_LIBRARIES} \${GRUEL_LIBRARIES} \${GNURADIO_CORE_LIBRARIES})
set_target_properties(gnuradio-${prefix} PROPERTIES DEFINE_SYMBOL "gnuradio_${prefix}_EXPORTS")

\########################################################################
\# Install built library files
\########################################################################
install(TARGETS gnuradio-${prefix}
    LIBRARY DESTINATION lib\${LIB_SUFFIX} \# .so/.dylib file
    ARCHIVE DESTINATION lib\${LIB_SUFFIX} \# .lib file
    RUNTIME DESTINATION bin              \# .dll file
)

\########################################################################
\# Build and register unit test
\########################################################################
find_package(Boost COMPONENTS unit_test_framework)

include(GrTest)
set(GR_TEST_TARGET_DEPS gnuradio-${prefix})
\#turn each test cpp file into an executable with an int main() function
add_definitions(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

add_executable(qa_${prefix}_${title}_${IOType} qa_${prefix}_${title}_${IOType}.cc)
target_link_libraries(qa_${prefix}_${title}_${IOType} gnuradio-${prefix} \${Boost_LIBRARIES})
GR_ADD_TEST(qa_${prefix}_${title}_${IOType} qa_${prefix}_${title}_${IOType})
