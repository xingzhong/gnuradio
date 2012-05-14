#include "./template/header.tpl"


/* -*- c++ -*- */

\#define ${prefix.upper()}_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
//%include "${prefix}_swig_doc.i"


%{
\#include "${prefix}_${title}_${IOType}.h"
%}

GR_SWIG_BLOCK_MAGIC(${prefix}, ${title}_${IOType});
%include "${prefix}_${title}_${IOType}.h"


\#if SWIGGUILE
%scheme %{
(load-extension-global "libguile-gnuradio_${prefix}_swig" "scm_init_gnuradio_${prefix}_swig_module")
%}


%goops %{
(use-modules (gnuradio gnuradio_core_runtime))
%}
#endif
