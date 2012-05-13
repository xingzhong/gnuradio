#include "./template/header.tpl"

\#ifndef INCLUDED_${prefix.upper()}_${title.upper()}_${IOType.upper()}_H
\#define INCLUDED_${prefix.upper()}_${title.upper()}_${IOType.upper()}_H

\#include <${prefix}_api.h>
\#include <gr_sync_block.h>

class ${prefix}_${title}_${IOType};

/*
 * We use boost::shared_ptr's instead of raw pointers for all access
 * to gr_blocks (and many other data structures).  The shared_ptr gets
 * us transparent reference counting, which greatly simplifies storage
 * management issues.  This is especially helpful in our hybrid
 * C++ / Python system.
 *
 * See http://www.boost.org/libs/smart_ptr/smart_ptr.htm
 *
 * As a convention, the _sptr suffix indicates a boost::shared_ptr
 */
typedef boost::shared_ptr<${prefix}_${title}_${IOType}> ${prefix}_${title}_${IOType}_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of ${prefix}_${title}_${IOType}.
 *
 * To avoid accidental use of raw pointers, ${prefix}_${title}_${IOType}'s
 * constructor is private.  ${prefix}_${title}_${IOType} is the public
 * interface for creating new instances.
 */
${prefix.upper()}_API ${prefix}_${title}_${IOType}_sptr ${prefix}_make_${title}_${IOType} ();

/*!
 * \brief ${title} a stream of floats.
 * \ingroup block
 *
 * This uses the preferred technique: subclassing gr_sync_block.
 */
class ${prefix.upper()}_API ${prefix}_${title}_${IOType} : public gr_sync_block
{
private:
  // The friend declaration allows ${prefix}_${title}_${IOType} to
  // access the private constructor.

  friend ${prefix.upper()}_API ${prefix}_${title}_${IOType}_sptr ${prefix}_make_${title}_${IOType} ();

  ${prefix}_${title}_${IOType} ();  	// private constructor

 public:
  ~${prefix}_${title}_${IOType} ();	// public destructor

  // Where all the action really happens

  int work (int noutput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star &output_items);
};

\#endif /* INCLUDED_${prefix.upper()}_${title.upper()}_${IOType.upper()}_H */
