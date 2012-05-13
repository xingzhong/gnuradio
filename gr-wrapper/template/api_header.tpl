#include "./template/header.tpl"

\#ifndef INCLUDED_${prefix.upper()}_API_H
\#define INCLUDED_${prefix.upper()}_API_H

\#include <gruel/attributes.h>

\#ifdef gnuradio_${prefix}_EXPORTS
\#  define ${prefix.upper()}_API __GR_ATTR_EXPORT
\#else
\#  define ${prefix.upper()}_API __GR_ATTR_IMPORT
\#endif

\#endif /* INCLUDED_${prefix.upper()}_API_H */