/* -*- c++ -*- */

#define SSP_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "ssp_swig_doc.i"


%{
#include "ssp_square_ff.h"
#include "ssp_square2_ff.h"
%}

GR_SWIG_BLOCK_MAGIC(ssp,square_ff);
%include "ssp_square_ff.h"

GR_SWIG_BLOCK_MAGIC(ssp,square2_ff);
%include "ssp_square2_ff.h"

#if SWIGGUILE
%scheme %{
(load-extension-global "libguile-gnuradio-ssp_swig" "scm_init_gnuradio_ssp_swig_module")
%}

%goops %{
(use-modules (gnuradio gnuradio_core_runtime))
%}
#endif
