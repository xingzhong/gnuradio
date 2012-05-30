#include "./template/header.tpl"


/*
 * config.h is generated by configure.  It contains the results
 * of probing for features, options etc.  It should be the first
 * file included in your .cc file.
 */
\#ifdef HAVE_CONFIG_H
\#include "config.h"
\#endif

\#include <${prefix}_${title}_${IOType}.h>
\#include <gr_io_signature.h>

/*
 * Create a new instance of ${prefix}_${title}_${IOType} and return
 * a boost shared_ptr.  This is effectively the public constructor.
 */
${prefix}_${title}_${IOType}_sptr 
${prefix}_make_${title}_${IOType} ()
{
  return gnuradio::get_initial_sptr(new ${prefix}_${title}_${IOType} ());
}

/*
 * Specify constraints on number of input and output streams.
 * This info is used to construct the input and output signatures
 * (2nd & 3rd args to gr_block's constructor).  The input and
 * output signatures are used by the runtime system to
 * check that a valid number and type of inputs and outputs
 * are connected to this block.  In this case, we accept
 * only 1 input and 1 output.
 */
static const int MIN_IN = 1;	// mininum number of input streams
static const int MAX_IN = 1;	// maximum number of input streams
static const int MIN_OUT = 1;	// minimum number of output streams
static const int MAX_OUT = 1;	// maximum number of output streams

/*
 * The private constructor
 */
${prefix}_${title}_${IOType}::${prefix}_${title}_${IOType} ()
  : gr_sync_block ("${title}_${IOType}",
		   gr_make_io_signature (MIN_IN, MAX_IN, sizeof (float)),
		   gr_make_io_signature (MIN_OUT, MAX_OUT, sizeof (float)))
{
  // nothing else required in this example
}

/*
 * Our virtual destructor.
 */
${prefix}_${title}_${IOType}::~${prefix}_${title}_${IOType} ()
{
  // nothing else required in this example
}

int 
${prefix}_${title}_${IOType}::work (int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items)
{
  const float *in = (const float *) input_items[0];
  float *out = (float *) output_items[0];

  ${kernel}

  // Tell runtime system how many output items we produced.
  return noutput_items;
}
