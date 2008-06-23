/* -*- c++ -*- */
/*
 * Copyright 2006 Free Software Foundation, Inc.
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
 */

#ifndef INCLUDED_REALTIME_H
#define INCLUDED_REALTIME_H

namespace gruel {

  typedef enum {
    RT_OK = 0,
    RT_NOT_IMPLEMENTED,
    RT_NO_PRIVS,
    RT_OTHER_ERROR
  } rt_status_t;

  /*!
   * \brief If possible, enable high-priority "real time" scheduling.
   * \ingroup misc
   */
  rt_status_t
  enable_realtime_scheduling();

} // namespace gruel

#endif /* INCLUDED_GR_REALTIME_H */