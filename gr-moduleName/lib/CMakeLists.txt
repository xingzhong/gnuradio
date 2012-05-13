#  timestamp: Sun May 13 18:04:03 2012
#  Author: Xingzhong
#  Description: Template according to the gr-howto-write-a-block-cmake in GNU-Radio

#
# Setup library
#
include(GrPlatform) #define LIB_SUFFIX

add_library(gnuradio-ssp SHARED ssp_moduleName_cc.cc )
target_link_libraries(gnuradio-ssp ${Boost_LIBRARIES} ${GRUEL_LIBRARIES} ${GNURADIO_CORE_LIBRARIES})
set_target_properties(gnuradio-ssp PROPERTIES DEFINE_SYMBOL "gnuradio_ssp_EXPORTS")

#
# Install built library files
#
install(TARGETS gnuradio-ssp
    LIBRARY DESTINATION lib${LIB_SUFFIX} # .so/.dylib file
    ARCHIVE DESTINATION lib${LIB_SUFFIX} # .lib file
    RUNTIME DESTINATION bin              # .dll file
)

#
# Build and register unit test
#
find_package(Boost COMPONENTS unit_test_framework)

include(GrTest)
set(GR_TEST_TARGET_DEPS gnuradio-ssp)
#turn each test cpp file into an executable with an int main() function
add_definitions(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

add_executable(qa_ssp_moduleName_cc qa_ssp_moduleName_cc.cc)
target_link_libraries(qa_hssp_moduleName_cc gnuradio-ssp ${Boost_LIBRARIES})
GR_ADD_TEST(qa_ssp_moduleName_cc qa_ssp_moduleName_cc)
