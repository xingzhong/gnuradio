/* -*- c++ -*- */

\#define SSP_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "ssp_swig_doc.i"


%{
\#include "${funcName}.h"
%}

GR_SWIG_BLOCK_MAGIC(ssp, ${Name});
%include "${funcName}.h"


\#if SWIGGUILE
%scheme %{
(load-extension-global "libguile-gnuradio-ssp_swig" "scm_init_gnuradio_ssp_swig_module")
%}

%goops %{
(use-modules (gnuradio gnuradio_core_runtime))
%}
\#endif
