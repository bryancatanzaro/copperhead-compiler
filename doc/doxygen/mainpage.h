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
     - initializer lists
     - strongly-typed enums
     - variadic templates
     - \p std::shared_ptr, \p std::make_shared, \p std::tuple
     
     Since Visual Studio 10 (11) does (will) not support initializer lists, this
     code in its current state cannot be compiled by Visual Studio. It
     has been tested with g++ 4.5.

     To build this code, we provide an SConscript, for use with <a
     href="http://www.scons.org">Scons</a>. It has been built with
     version 2.0.1.
     
 */


 
