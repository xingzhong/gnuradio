/* -*- c++ -*- */
/*
 * Copyright 2004 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 * 
 * ${Header} 
 * by ${Author} at ${Today}
 */


\#ifndef INCLUDED_${FUNCNAME}_H
\#define INCLUDED_${FUNCNAME}_H

\#include <api.h>
\#include <gr_block.h>

void ssp_kernel(const ${t1}*, ${t2}*, int);

class ${funcName};

typedef boost::shared_ptr<${funcName}> ${funcName}_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of ${funcName}.
 *
 * To avoid accidental use of raw pointers, ${funcName}'s
 * constructor is private.  ssp_make_${Name} is the public
 * interface for creating new instances.
 */
SSP_API ${funcName}_sptr ssp_make_${Name} ();

/*!
 * \brief square a stream of floats.
 * \ingroup block
 *
 * \sa ${funcName} for a version that subclasses .
 */
class SSP_API ${funcName} : public gr_block
{
private:
  // The friend declaration allows ssp_make_${Name} to
  // access the private constructor.

  friend SSP_API ${funcName}_sptr ssp_make_${Name} ();

  /*!
   * \brief square a stream of floats.
   */
  ${funcName} ();  	// private constructor

 public:
  ~${funcName} ();	// public destructor

  // Where all the action really happens

  int general_work (int noutput_items,
		    gr_vector_int &ninput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
};

\#endif /* INCLUDED_SSP_${FUNCNAME}_H*/
