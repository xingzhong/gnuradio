/* -*- c++ -*- */
/*
 * Copyright 2006 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gr_ofdm_correlator.h>
#include <gr_io_signature.h>

#define VERBOSE 0
#define M_TWOPI (2*M_PI)

gr_ofdm_correlator_sptr
gr_make_ofdm_correlator (unsigned int occupied_carriers, unsigned int vlen, 
			 unsigned int cplen,
			 std::vector<gr_complex> known_symbol1, 
			 std::vector<gr_complex> known_symbol2)
{
  return gr_ofdm_correlator_sptr (new gr_ofdm_correlator (occupied_carriers, vlen, cplen,
							  known_symbol1, known_symbol2));
}

gr_ofdm_correlator::gr_ofdm_correlator (unsigned occupied_carriers, unsigned int vlen, 
					unsigned int cplen,
					std::vector<gr_complex> known_symbol1, 
					std::vector<gr_complex> known_symbol2)
  : gr_block ("ofdm_correlator",
	      gr_make_io_signature (1, 1, sizeof(gr_complex)*vlen),
	      gr_make_io_signature (1, 1, sizeof(gr_complex)*occupied_carriers)),
    d_occupied_carriers(occupied_carriers),
    d_vlen(vlen),
    d_cplen(cplen),
    d_freq_shift_len(5),
    d_known_symbol1(known_symbol1),
    d_known_symbol2(known_symbol2),
    d_coarse_freq(0),
    d_phase_count(0)
{
  d_diff_corr_factor.resize(d_occupied_carriers);
  d_hestimate.resize(d_occupied_carriers);

  std::vector<gr_complex>::iterator i1, i2;

  int i = 0;
  gr_complex one(1.0, 0.0);
  for(i1 = d_known_symbol1.begin(), i2 = d_known_symbol2.begin(); i1 != d_known_symbol1.end(); i1++, i2++) {
    d_diff_corr_factor[i] = one / ((*i1) * conj(*i2));
    i++;
  }
}

gr_ofdm_correlator::~gr_ofdm_correlator(void)
{
}

void
gr_ofdm_correlator::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  unsigned ninputs = ninput_items_required.size ();
  for (unsigned i = 0; i < ninputs; i++)
    ninput_items_required[i] = 2;
}

gr_complex
gr_ofdm_correlator::coarse_freq_comp(int freq_delta, int symbol_count)
{
  return gr_complex(cos(-M_TWOPI*freq_delta*d_cplen/d_vlen*symbol_count),
		    sin(-M_TWOPI*freq_delta*d_cplen/d_vlen*symbol_count));
}

bool
gr_ofdm_correlator::correlate(const gr_complex *previous, const gr_complex *current, 
			      int zeros_on_left)
{
  unsigned int i = 0;
  int search_delta = 0;
  bool found = false;

  gr_complex h_sqrd = gr_complex(0.0,0.0);
  float power = 0.0F;

  while(!found && (abs(search_delta) < d_freq_shift_len)) {
    h_sqrd = gr_complex(0.0,0.0);
    power = 0.0F;

    for(i = 0; i < d_occupied_carriers; i++) {
      h_sqrd = h_sqrd + previous[i+zeros_on_left+search_delta] * 
	conj(coarse_freq_comp(search_delta,1)*current[i+zeros_on_left+search_delta]) * 
	d_diff_corr_factor[i];
      power = power + norm(current[i+zeros_on_left+search_delta]); // No need to do coarse freq here
    }
    
#if VERBOSE
      printf("bin %d\th_sqrd = ( %f, %f )\t power = %f\t real(h)/p = %f\t angle = %f\n", 
	     search_delta, h_sqrd.real(), h_sqrd.imag(), power, h_sqrd.real()/power, arg(h_sqrd)); 
#endif

    if(h_sqrd.real() > 0.75*power) {
      found = true;
      d_coarse_freq = search_delta;
      d_phase_count = 1;
      d_snr_est = 10*log10(power/(power-h_sqrd.real()));

      printf("CORR: Found, bin %d\tSNR Est %f dB\tcorr power fraction %f\n", 
             search_delta, d_snr_est, h_sqrd.real()/power);
      // search_delta,10*log10(h_sqrd.real()/fabs(h_sqrd.imag())),h_sqrd.real()/power);
      break;
    }
    else {
      if(search_delta <= 0)
	search_delta = (-search_delta) + 1;
      else
	search_delta = -search_delta;
    }
  }
  return found;
}

void
gr_ofdm_correlator::calculate_equalizer(const gr_complex *previous, const gr_complex *current, 
					int zeros_on_left)
{
  unsigned int i=0;

  for(i = 0; i < d_occupied_carriers; i++) {
    // FIXME possibly add small epsilon in divisor to protect from div 0
    //d_hestimate[i] = 0.5F * (d_known_symbol1[i] / previous[i+zeros_on_left] +
    //			    d_known_symbol2[i] / (coarse_freq_comp(d_coarse_freq,1)*
    //						  current[i+zeros_on_left+d_coarse_freq]));
    d_hestimate[i] = 0.5F * (d_known_symbol1[i] / previous[i+zeros_on_left+d_coarse_freq] +
			     d_known_symbol2[i] / (coarse_freq_comp(d_coarse_freq,1)*
						   current[i+zeros_on_left+d_coarse_freq]));
    
#if VERBOSE
    fprintf(stderr, "%f %f ", d_hestimate[i].real(), d_hestimate[i].imag());
#endif
  }
#if VERBOSE
  fprintf(stderr, "\n");
#endif
}

int
gr_ofdm_correlator::general_work(int noutput_items,
				 gr_vector_int &ninput_items,
				 gr_vector_const_void_star &input_items,
				 gr_vector_void_star &output_items)
{
  const gr_complex *in = (const gr_complex *)input_items[0];
  const gr_complex *previous = &in[0];
  const gr_complex *current = &in[d_vlen];

  gr_complex *out = (gr_complex *) output_items[0];
  
  unsigned int i=0;

  int unoccupied_carriers = d_vlen - d_occupied_carriers;
  int zeros_on_left = (int)ceil(unoccupied_carriers/2.0);

  bool corr = correlate(previous, current, zeros_on_left);
  if(corr) {
    calculate_equalizer(previous, current, zeros_on_left);
  }

  for(i = 0; i < d_occupied_carriers; i++) {
    out[i] = d_hestimate[i]*coarse_freq_comp(d_coarse_freq,d_phase_count)*current[i+zeros_on_left+d_coarse_freq];
  }
  d_phase_count++;
  consume_each(1);
  return 1;
}