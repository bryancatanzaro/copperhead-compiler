/*
 *   Copyright 2012      NVIDIA Corporation
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * 
 */
/*!\mainpage Copperhead Compiler Index Page

  \section intro_sec Introduction

  This project provides the Copperhead Compiler library, which can be
   used to build data parallel compilers.

   The library provides a set of Abstract Syntax Tree \ref nodes
   "nodes" that describe the text of a program, as well as a set
   of \ref rewriters "rewriters", each of which transform the program
   in a particular way.  These rewriters can be combined to form a
   compiler, such as \ref backend::compiler "this one",
   which is used as the compiler for the Python Copperhead runtime.

   \section build_sec Building
   This code depends on:

   - <a href="http://www.boost.org">Boost</a>. It uses \p boost::variant and
     \p boost::mpl, both of which are header-only libraries and do not
     require compilation. It has been tested with Boost version 1.47.

   - C++11 features:
     - R-value references
     - \p auto, \p decltype
     - strongly-typed enums
     - variadic templates
     - \p std::shared_ptr, \p std::make_shared, \p std::tuple

     This codebase has been tested with g++4.5.  With a little work
     (to workaround the lack of strongly-typed enums and variadic
     templates), it should compile with Visual Studio 10.
     
     To build this code, we provide an SConscript, for use with <a
     href="http://www.scons.org">Scons</a>. It has been built with
     version 2.0.1.
     
 */


 
