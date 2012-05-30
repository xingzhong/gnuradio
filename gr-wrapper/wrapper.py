#!/usr/bin/env pythonw2.7

from Cheetah.Template import Template
from shutil import rmtree
import os
import tarfile

nameSpace = {}

# global configurations
nameSpace['title'] = "moduleName"
nameSpace['author'] = "Xingzhong"
nameSpace['prefix'] = "ssp"
nameSpace['description'] = "Template according to the gr-howto-write-a-block-cmake in GNU-Radio"

# local configurations

nameSpace['InputType'] = "float"
nameSpace['OutputType'] = "float"
nameSpace['IOType'] = nameSpace['InputType'][0]+nameSpace['OutputType'][0]
kernelFile = open('kernel.input', 'r')
nameSpace['kernel'] = kernelFile.read()

# create dir if existes delete all
dirName = "../gr-" + nameSpace['title']
print "dirName: %s"%dirName

if os.path.exists(dirName):
    rmtree(dirName)

# depoly them from archive 
os.mkdir(dirName)
tar = tarfile.open("./template/tpl_dir.tar")
tar.extractall(path=dirName)
tar.close()



## for specific header file
inc_header_file = open('./template/inc_header.tpl', 'r')
inc_header = Template(inc_header_file.read(), searchList=nameSpace)
inc_header_name = "%s_%s_%s.h"%(nameSpace['prefix'], nameSpace['title'], nameSpace['IOType'])
print 'write to: %s'%(dirName+'/include/'+inc_header_name)
filebuf = open(dirName+'/include/'+inc_header_name, 'w')
filebuf.write(str(inc_header))
filebuf.close()




## for api header file
inc_api_file = open('./template/api_header.tpl', 'r')
inc_api = Template(inc_api_file.read(), searchList=nameSpace)
inc_api_name = "%s_api.h"%(nameSpace['prefix'])
print 'write to: %s'%(dirName+'/include/'+inc_api_name)
filebuf = open(dirName+'/include/'+inc_api_name, 'w')
filebuf.write(str(inc_api))
filebuf.close()


## for functional cpp file
lib_cpp_file = open('./template/lib_cpp.tpl', 'r')
lib_cpp = Template(lib_cpp_file.read(), searchList=nameSpace)
lib_cpp_name = "%s_%s_%s.cc"%(nameSpace['prefix'], nameSpace['title'], nameSpace['IOType'])
print 'write to: %s'%(dirName+'/lib/'+lib_cpp_name)
filebuf = open(dirName+'/lib/'+lib_cpp_name, 'w')
filebuf.write(str(lib_cpp))
filebuf.close()

## for swig file
swig_i_file = open('./template/swig_i.tpl', 'r')
swig_i_name = "%s_swig.i"%(nameSpace['prefix'])
swig_i = Template(swig_i_file.read(), searchList=nameSpace)
print 'write to: %s'%(dirName+'/swig/'+swig_i_name)
filebuf = open(dirName+'/swig/'+swig_i_name, 'w')
filebuf.write(str(swig_i))
filebuf.close()

## for qa_python
qa_py_file = open('./template/qa_py.tpl', 'r')
qa_py_name = "qa_%s.py"%(nameSpace['prefix'])
qa_py = Template(qa_py_file.read(), searchList=nameSpace)
print 'write to: %s'%(dirName+'/python/'+qa_py_name)
filebuf = open(dirName+'/python/'+qa_py_name, 'w')
filebuf.write(str(qa_py))
filebuf.close()

## for qa_cpp
qa_cpp_file = open('./template/qa_cpp.tpl', 'r')
qa_cpp_name = "qa_%s_%s_%s.cc"%(nameSpace['prefix'], nameSpace['title'], nameSpace['IOType'])
qa_cpp = Template(qa_cpp_file.read(), searchList=nameSpace)
print 'write to: %s'%(dirName+'/lib/'+qa_cpp_name)
filebuf = open(dirName+'/lib/'+qa_cpp_name, 'w')
filebuf.write(str(qa_cpp))
filebuf.close()

## for main_cmake
cmake_file = open('./template/cmake.tpl', 'r')
cmake_name = "CMakeLists.txt"
cmake = Template(cmake_file.read(), searchList=nameSpace)
print 'write to: %s'%(dirName+'/'+cmake_name)
filebuf = open(dirName+'/'+cmake_name, 'w')
filebuf.write(str(cmake))
filebuf.close()


## for inc_cmake
inc_cmake_file = open('./template/inc_cmake.tpl', 'r')
inc_cmake_name = "CMakeLists.txt"
inc_cmake = Template(inc_cmake_file.read(), searchList=nameSpace)
print 'write to: %s'%(dirName+'/include/'+inc_cmake_name)
filebuf = open(dirName+'/include/'+inc_cmake_name, 'w')
filebuf.write(str(inc_cmake))
filebuf.close()


## for lib_cmake
lib_cmake_file = open('./template/lib_cmake.tpl', 'r')
lib_cmake_name = "CMakeLists.txt"
lib_cmake = Template(lib_cmake_file.read(), searchList=nameSpace)
print 'write to: %s'%(dirName+'/lib/'+lib_cmake_name)
filebuf = open(dirName+'/lib/'+lib_cmake_name, 'w')
filebuf.write(str(lib_cmake))
filebuf.close()

## for swig_cmake
swig_cmake_file = open('./template/swig_cmake.tpl', 'r')
siwg_cmake_name = "CMakeLists.txt"
swig_cmake = Template(swig_cmake_file.read(), searchList=nameSpace)
print 'write to: %s'%(dirName+'/swig/'+siwg_cmake_name)
filebuf = open(dirName+'/swig/'+siwg_cmake_name, 'w')
filebuf.write(str(swig_cmake))
filebuf.close()

## for py_cmake
py_cmake_file = open('./template/py_cmake.tpl', 'r')
py_cmake_name = "CMakeLists.txt"
py_cmake = Template(py_cmake_file.read(), searchList=nameSpace)
print 'write to: %s'%(dirName+'/python/'+py_cmake_name)
filebuf = open(dirName+'/python/'+py_cmake_name, 'w')
filebuf.write(str(py_cmake))
filebuf.close()
