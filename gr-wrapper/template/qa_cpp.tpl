#include "./template/header.tpl"

\#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(qa_${prefix}_${title}_${IOType}_t1){
    BOOST_CHECK_EQUAL(2 + 2, 4);
    // TODO BOOST_* test macros here
}

