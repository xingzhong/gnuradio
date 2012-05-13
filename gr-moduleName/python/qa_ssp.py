#!/usr/bin/env python


from gnuradio import gr, gr_unittest
import ssp_swig

class qa_ssp (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_moduleName_cc (self):
        src_data = (-3, 4, -5.5, 2, 3)
        expected_result = (9, 16, 30.25, 4, 9)
        src = gr.vector_source_c (src_data)
        sqr = ssp_swig.moduleName_cc ()
        dst = gr.vector_sink_c ()
        self.tb.connect (src, sqr)
        self.tb.connect (sqr, dst)
        self.tb.run ()
        result_data = dst.data ()
        #self.assertFloatTuplesAlmostEqual (expected_result, result_data, 6)

    def test_002_moduleName_cc (self):
        src_data = (-3, 4, -5.5, 2, 3)
        expected_result = (9, 16, 30.25, 4, 9)
        src = gr.vector_source_c (src_data)
        sqr = ssp_swig.moduleName_cc ()
        dst = gr.vector_sink_c ()
        self.tb.connect (src, sqr)
        self.tb.connect (sqr, dst)
        self.tb.run ()
        result_data = dst.data ()
        #self.assertFloatTuplesAlmostEqual (expected_result, result_data, 6)
        
if __name__ == '__main__':
    gr_unittest.main ()