# Copyright 2011 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 
# ${Header} 
# by ${Author} at ${Today}
# 

########################################################################
# Setup library
########################################################################
include(GrPlatform) #define LIB_SUFFIX

add_library(gnuradio-ssp SHARED ${funcName}.cc)
target_link_libraries(gnuradio-ssp \${Boost_LIBRARIES} \${GRUEL_LIBRARIES} \${GNURADIO_CORE_LIBRARIES} kernel-ssp)
set_target_properties(gnuradio-ssp PROPERTIES DEFINE_SYMBOL "gnuradio_ssp_EXPORTS")

########################################################################
# Install built library files
########################################################################
install(TARGETS gnuradio-ssp
    LIBRARY DESTINATION lib\${LIB_SUFFIX} # .so/.dylib file
    ARCHIVE DESTINATION lib\${LIB_SUFFIX} # .lib file
    RUNTIME DESTINATION bin              # .dll file
)

########################################################################
# Build and register unit test
########################################################################
find_package(Boost COMPONENTS unit_test_framework)

include(GrTest)
set(GR_TEST_TARGET_DEPS gnuradio-ssp)
#turn each test cpp file into an executable with an int main() function
add_definitions(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

#add_executable(qa_${funcName} qa_${funcName}.cc)
#target_link_libraries(qa_${funcName} gnuradio-ssp \${Boost_LIBRARIES})
#GR_ADD_TEST(qa_${funcName} qa_${funcName})

