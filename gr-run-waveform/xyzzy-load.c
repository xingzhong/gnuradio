/* Copyright (C) 1995,1996,1998,1999,2000,2001,2004,2006,2010 Free Software Foundation, Inc.
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */



#ifdef HAVE_CONFIG_H
#  include <grconfig.h>
#endif

#include <string.h>
#include <stdio.h>

/* libpath.h is generated whenever the Makefile is rebuilt */
#include "libpath.h"

#include <libguile/__scm.h>
#include <libguile/fports.h>
#include <libguile/read.h>
#include <libguile/eval.h>
#include <libguile/throw.h>
#include <libguile/alist.h>
#include <libguile/dynwind.h>
#include <libguile/root.h>
#include <libguile/strings.h>
#include <libguile/modules.h>
#include <libguile/lang.h>
#include <libguile/chars.h>
#include <libguile/tags.h>
#include <libguile/snarf.h>
#include <libguile/srfi-13.h>
#include <libguile/validate.h>
#include <libguile/load.h>
#include <libguile/fluids.h>

// these headers where not in the original version of this file.
#include <libguile/boolean.h>
#include <libguile/pairs.h>
#include <libguile/gc.h>
#include <libguile/variable.h>

#include <sys/types.h>
#include <sys/stat.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* HAVE_UNISTD_H */

#ifndef R_OK
#define R_OK 4
#endif

#include "xyzzy.h"

// This is the magic number used when loading files
static const char *MAGIC = "-XyZzY-";

static SCM scm_listofnullstr;

static SCM *scm_loc_load_hook;
static SCM *scm_loc_load_path;
static SCM *scm_loc_load_extensions;

/* The current reader (a fluid).  */
static SCM *scm_loc_current_reader;


/* Utility functions for assembling C strings in a buffer.
 */

struct stringbuf {
  char *buf, *ptr;
  size_t buf_len;
};

static void
stringbuf_free (void *data)
{
  struct stringbuf *buf = (struct stringbuf *)data;
  free (buf->buf);
}

static void
stringbuf_grow (struct stringbuf *buf)
{
  size_t ptroff = buf->ptr - buf->buf;
  buf->buf_len *= 2; 
  buf->buf = scm_realloc (buf->buf, buf->buf_len);
  buf->ptr = buf->buf + ptroff;
}

static void
stringbuf_cat_locale_string (struct stringbuf *buf, SCM str)
{
  size_t max_len = buf->buf_len - (buf->ptr - buf->buf) - 1;
  size_t len = scm_to_locale_stringbuf (str, buf->ptr, max_len);
  if (len > max_len)
    {
      /* buffer is too small, double its size and try again. 
       */
      stringbuf_grow (buf);
      stringbuf_cat_locale_string (buf, str);
    }
  else
    {
      /* string fits, terminate it and check for embedded '\0'.
       */
      buf->ptr[len] = '\0';
      if (strlen (buf->ptr) != len)
	scm_misc_error (NULL,
			"string contains #\\nul character: ~S",
			scm_list_1 (str));
      buf->ptr += len;
    }
}

static void
stringbuf_cat (struct stringbuf *buf, char *str)
{
  size_t max_len = buf->buf_len - (buf->ptr - buf->buf) - 1;
  size_t len = strlen (str);
  if (len > max_len)
    {
      /* buffer is too small, double its size and try again. 
       */
      stringbuf_grow (buf);
      stringbuf_cat (buf, str);
    }
  else
    {
      /* string fits, copy it into buffer.
       */
      strcpy (buf->ptr, str);
      buf->ptr += len;
    }
}

/* Search PATH for a directory containing a file named FILENAME.
   The file must be readable, and not a directory.
   If we find one, return its full filename; otherwise, return #f.
   If FILENAME is absolute, return it unchanged.
   If given, EXTENSIONS is a list of strings; for each directory 
   in PATH, we search for FILENAME concatenated with each EXTENSION.  */
SCM_DEFINE (scm_xyzzy_search_path, "xyyzy-search-path", 2, 1, 0,
           (SCM path, SCM filename, SCM extensions),
	    "Search @var{path} for a directory containing a file named\n"
	    "@var{filename}. The file must be readable, and not a directory.\n"
	    "If we find one, return its full filename; otherwise, return\n"
	    "@code{#f}.  If @var{filename} is absolute, return it unchanged.\n"
	    "If given, @var{extensions} is a list of strings; for each\n"
	    "directory in @var{path}, we search for @var{filename}\n"
	    "concatenated with each @var{extension}.")
#define FUNC_NAME s_scm_xyzzy_search_path
{
  struct stringbuf buf;
  char *filename_chars;
  size_t filename_len;
  SCM result = SCM_BOOL_F;

  if (SCM_UNBNDP (extensions))
    extensions = SCM_EOL;

  /* fprintf(stderr, "TRACE %s: %d %s\n", __FUNCTION__, __LINE__, scm_to_locale_string(SCM_CAR(path))); */

  scm_dynwind_begin (0);
  filename_chars = scm_to_locale_string (filename);
  filename_len = strlen (filename_chars);
  scm_dynwind_free (filename_chars);

  /* If FILENAME is absolute, return it unchanged.  */
#ifdef __MINGW32__
  if (((filename_len >= 1) && 
       (filename_chars[0] == '/' || filename_chars[0] == '\\')) ||
      ((filename_len >= 3) && filename_chars[1] == ':' &&
       ((filename_chars[0] >= 'a' && filename_chars[0] <= 'z') ||
	(filename_chars[0] >= 'A' && filename_chars[0] <= 'Z')) &&
       (filename_chars[2] == '/' || filename_chars[2] == '\\')))
#else
  if (filename_len >= 1 && filename_chars[0] == '/')
#endif
    {
      scm_dynwind_end ();
      return filename;
    }

  /* If FILENAME has an extension, don't try to add EXTENSIONS to it.  */
  {
    char *endp;

    for (endp = filename_chars + filename_len - 1;
	 endp >= filename_chars;
	 endp--)
      {
	if (*endp == '.')
	  {
	    /* This filename already has an extension, so cancel the
               list of extensions.  */
	    extensions = SCM_EOL;
	    break;
	  }
#ifdef __MINGW32__
	else if (*endp == '/' || *endp == '\\')
#else
	else if (*endp == '/')
#endif
	  /* This filename has no extension, so keep the current list
             of extensions.  */
	  break;
      }
  }

  /* This simplifies the loop below a bit.
   */
  if (scm_is_null (extensions))
    extensions = scm_listofnullstr;

  buf.buf_len = 512;
  buf.buf = scm_malloc (buf.buf_len);
  scm_dynwind_unwind_handler (stringbuf_free, &buf, SCM_F_WIND_EXPLICITLY);
  
  /* Try every path element.
   */
  for (; scm_is_pair (path); path = SCM_CDR (path))
    {
      SCM dir = SCM_CAR (path);
      SCM exts;
      size_t sans_ext_len;

      buf.ptr = buf.buf;
      stringbuf_cat_locale_string (&buf, dir);
	
      /* Concatenate the path name and the filename. */
      
#ifdef __MINGW32__
      if ((buf.ptr > buf.buf) && (buf.ptr[-1] != '/') && (buf.ptr[-1] != '\\'))
#else
      if ((buf.ptr > buf.buf) && (buf.ptr[-1] != '/'))
#endif
	stringbuf_cat (&buf, "/");

      stringbuf_cat (&buf, filename_chars);
      sans_ext_len = buf.ptr - buf.buf;

      /* Try every extension. */
      for (exts = extensions; scm_is_pair (exts); exts = SCM_CDR (exts))
	{
	  SCM ext = SCM_CAR (exts);
	  struct stat mode;
	  
	  buf.ptr = buf.buf + sans_ext_len;
	  stringbuf_cat_locale_string (&buf, ext);
	  
	  /* If the file exists at all, we should return it.  If the
	     file is inaccessible, then that's an error.  */
	  /* fprintf(stderr, "TRACE %s: %d: \"%s\"\n", __FUNCTION__, __LINE__, buf.buf); */

	  if (xyzzy_file_exists(buf.buf)
	      || (stat (buf.buf, &mode) == 0
		  && ! (mode.st_mode & S_IFDIR)))
	    {
	      result = scm_from_locale_string (buf.buf);
	      goto end;
	    }
	}
      
      if (!SCM_NULL_OR_NIL_P (exts))
	scm_wrong_type_arg_msg (NULL, 0, extensions, "proper list");
    }

  if (!SCM_NULL_OR_NIL_P (path))
    scm_wrong_type_arg_msg (NULL, 0, path, "proper list");

 end:
  
  scm_dynwind_end ();
  
  return result;
}
#undef FUNC_NAME

SCM_DEFINE (scm_xyzzy_primitive_load, "xyzzy-primitive-load", 1, 0, 0, 
           (SCM filename),
	    "Load the file named @var{filename} and evaluate its contents in\n"
	    "the top-level environment. The load paths are not searched;\n"
	    "@var{filename} must either be a full pathname or be a pathname\n"
	    "relative to the current directory.  If the  variable\n"
	    "@code{%load-hook} is defined, it should be bound to a procedure\n"
	    "that will be called before any code is loaded.  See the\n"
	    "documentation for @code{%load-hook} later in this section.")
#define FUNC_NAME s_scm_xyzzy_primitive_load
{
  SCM hook = *scm_loc_load_hook;
  SCM_VALIDATE_STRING (1, filename);

  char *ptr = scm_to_locale_string(filename);
  /* fprintf(stderr, "TRACE %s: %d: %s\n", __FUNCTION__, __LINE__, ptr); */
  
  if (scm_is_true (hook) && scm_is_false (scm_procedure_p (hook)))
    SCM_MISC_ERROR ("value of %load-hook is neither a procedure nor #f",
		    SCM_EOL);

  if (!scm_is_false (hook))
    scm_call_1 (hook, filename);

  { /* scope */
    SCM port;

    if (xyzzy_file_exists(ptr)){
      /* fprintf(stderr, "TRACE: file %s is a XYZZY file system file!\n", ptr); */
      port = xyzzy_open_file(filename);
    } else {
      port = scm_open_file (filename, scm_from_locale_string ("r"));
    }
    scm_dynwind_begin (SCM_F_DYNWIND_REWINDABLE);
    scm_i_dynwind_current_load_port (port);

    while (1)
      {
	SCM reader, form;

	/* Lookup and use the current reader to read the next  expression. */
	reader = scm_fluid_ref(*scm_loc_current_reader);
	if (reader == SCM_BOOL_F)
	  form = scm_read (port);
	else
	  form = scm_call_1 (reader, port);

	if (SCM_EOF_OBJECT_P (form))
	  break;

	scm_primitive_eval_x (form);
      }

    scm_dynwind_end ();
    scm_close_port (port);
  }
  return SCM_UNSPECIFIED;
}
#undef FUNC_NAME

/* Search %load-path for a directory containing a file named FILENAME.
   The file must be readable, and not a directory.
   If we find one, return its full filename; otherwise, return #f.
   If FILENAME is absolute, return it unchanged.  */
SCM_DEFINE (scm_xyzzy_sys_search_load_path, "%xyzzy-search-load-path", 1, 0, 0, 
	    (SCM filename),
	    "Search @var{%load-path} for the file named @var{filename},\n"
	    "which must be readable by the current user.  If @var{filename}\n"
	    "is found in the list of paths to search or is an absolute\n"
	    "pathname, return its full pathname.  Otherwise, return\n"
	    "@code{#f}.  Filenames may have any of the optional extensions\n"
	    "in the @code{%load-extensions} list; @code{%search-load-path}\n"
	    "will try each extension automatically.")
#define FUNC_NAME s_scm_xyzzy_sys_search_load_path
{
  SCM loadpath = scm_c_lookup("%load-path");
  SCM path = scm_variable_ref(loadpath);
  SCM exts = *scm_loc_load_extensions;
  SCM_VALIDATE_STRING (1, filename);

  /* fprintf(stderr, "TRACE %s: %d:\n", __FUNCTION__, __LINE__); */
  
  if (scm_ilength (path) < 0)
    SCM_MISC_ERROR ("%load-path is not a proper list", SCM_EOL);
  if (scm_ilength (exts) < 0)
    SCM_MISC_ERROR ("%load-extension list is not a proper list", SCM_EOL);
  
  return scm_xyzzy_search_path (path, filename, exts);
}
#undef FUNC_NAME

SCM_DEFINE (scm_xyzzy_primitive_load_path, "xyzzy-primitive-load-path", 1, 0, 0, 
	    (SCM filename),
	    "Search @var{%load-path} for the file named @var{filename} and\n"
	    "load it into the top-level environment.  If @var{filename} is a\n"
	    "relative pathname and is not found in the list of search paths,\n"
	    "an error is signalled.")
#define FUNC_NAME s_scm_xyzzy_primitive_load_path
{
  SCM full_filename;
  char *filename_chars;
  size_t filename_len;
  
  filename_chars = scm_to_locale_string (filename);
  filename_len = strlen (filename_chars);
  scm_dynwind_free (filename_chars);
  
  full_filename = scm_xyzzy_sys_search_load_path (filename);

  /* fprintf(stderr, "TRACE %s: %d: %s\n", __FUNCTION__, __LINE__, scm_to_locale_string(full_filename)); */

  if (scm_is_false (full_filename))
    SCM_MISC_ERROR ("Unable to find the file ~S in load path",
		    scm_list_1 (filename));

  return scm_xyzzy_primitive_load (full_filename);
}
#undef FUNC_NAME

SCM_DEFINE (scm_xyzzy_open_file, "xyzzy-open-file", 1, 0, 0,
 (SCM filename),
 "Return a new port which reads from @var{filename}")
#define FUNC_NAME s_scm_xyzzy_open_file
{
    return xyzzy_open_file (filename);
}    
#undef FUNC_NAME

void
scm_xyzzy_init (void)
{
  scm_listofnullstr = scm_permanent_object (scm_list_1 (scm_nullstr));
  scm_loc_load_extensions = SCM_VARIABLE_LOC(scm_c_lookup("%load-extensions"));
  scm_loc_load_hook = SCM_VARIABLE_LOC (scm_c_lookup("%load-hook"));
  scm_loc_current_reader = SCM_VARIABLE_LOC (scm_c_lookup("current-reader"));

  /* initialize our functions in the scheme VM */
  scm_c_define_gsubr ("xyzzy-open-file", 1, 0, 0, (SCM (*)()) scm_xyzzy_open_file);
  scm_c_define_gsubr ("xyzzy-primitive-load", 1, 0, 0, (SCM (*)()) scm_xyzzy_primitive_load);
  scm_c_define_gsubr ("xyzzy-primitive-load-path", 1, 0, 0, (SCM (*)()) scm_xyzzy_primitive_load_path);
  scm_c_define_gsubr ("%xyzzy-search-load-path", 1, 0, 0, (SCM (*)()) scm_xyzzy_sys_search_load_path);
  scm_c_define_gsubr ("xyzzy-search-path", 2, 1, 0, (SCM (*)()) scm_xyzzy_search_path);
}
