#!/usr/bin/env python
#
# Copyright 2008 Free Software Foundation, Inc.
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

from gnuradio import gr

class selector(gr.hier_block2):
	"""A hier2 block with N inputs and M outputs, where data is only forwarded through input n to output m."""
	def __init__(self, item_size, num_inputs, num_outputs, input_index, output_index):
		"""
		SelectorHelper constructor.
		@param item_size the size of the gr data stream in bytes
		@param num_inputs the number of inputs (integer)
		@param num_outputs the number of outputs (integer)
		@param input_index the index for the source data
		@param output_index the index for the destination data
		"""
		gr.hier_block2.__init__(
			self, 'selector',
			gr.io_signature(num_inputs, num_inputs, item_size),
			gr.io_signature(num_outputs, num_outputs, item_size),
		)
		#terminator blocks for unused inputs and outputs
		self.input_terminators = [gr.null_sink(item_size)] * num_inputs
		self.output_terminators = [gr.head(item_size, 0)] * num_outputs
		self.copy = None
		#connections
		for i in range(num_inputs): self.connect((self, i), self.input_terminators[i])
		for i in range(num_outputs): self.connect(gr.null_source(item_size), self.output_terminators[i], (self, i))
		self.item_size = item_size
		self.input_index = input_index
		self.output_index = output_index
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self._connect_current()

	def _indexes_valid(self):
		"""
		Are the input and output indexes within range of the number of inputs and outputs?
		@return true if input index and output index are in range
		"""
		return self.input_index in range(self.num_inputs) and self.output_index in range(self.num_outputs)

	def _connect_current(self):
		"""If the input and output indexes are valid:
		disconnect the blocks at the input and output index from their terminators,
		and connect them to one another. Then connect the terminators to one another."""
		if self._indexes_valid():
			self.disconnect((self, self.input_index), self.input_terminators[self.input_index])
			self.disconnect(self.output_terminators[self.output_index], (self, self.output_index))
			self.copy = gr.skiphead(self.item_size, 0)
			self.connect((self, self.input_index), self.copy)
			self.connect(self.copy, (self, self.output_index))
			self.connect(self.output_terminators[self.output_index], self.input_terminators[self.input_index])

	def _disconnect_current(self):
		"""If the input and output indexes are valid:
		disconnect the blocks at the input and output index from one another,
		and the terminators at the input and output index from one another.
		Reconnect the blocks to the terminators."""
		if self._indexes_valid():
			self.disconnect((self, self.input_index), self.copy)
			self.disconnect(self.copy, (self, self.output_index))
			self.disconnect(self.output_terminators[self.output_index], self.input_terminators[self.input_index])
			del self.copy
			self.copy = None
			self.connect((self, self.input_index), self.input_terminators[self.input_index])
			self.connect(self.output_terminators[self.output_index], (self, self.output_index))

	def set_input_index(self, input_index):
		"""
		Change the block to the new input index if the index changed.
		@param input_index the new input index
		"""
		if self.input_index != input_index:
			self.lock()
			self._disconnect_current()
			self.input_index = input_index
			self._connect_current()
			self.unlock()

	def set_output_index(self, output_index):
		"""
		Change the block to the new output index if the index changed.
		@param output_index the new output index
		"""
		if self.output_index != output_index:
			self.lock()
			self._disconnect_current()
			self.output_index = output_index
			self._connect_current()
			self.unlock()

class valve(selector):
	"""Wrapper for selector with 1 input and 1 output."""

	def __init__(self, item_size, open):
		"""
		Constructor for valve.
		@param item_size the size of the gr data stream in bytes
		@param open true if initial valve state is open
		"""
		if open: output_index = -1
		else: output_index = 0
		selector.__init__(self, item_size, 1, 1, 0, output_index)

	def set_open(self, open):
		"""
		Callback to set open state.
		@param open true to set valve state to open
		"""
		if open: output_index = -1
		else: output_index = 0
		self.set_output_index(output_index)