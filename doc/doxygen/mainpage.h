/*!\mainpage Copperhead Compiler Index Page
   \section intro_sec Introduction
   Documentation for the Copperhead Compiler.  As should be obvious
   from the name, this is only the compiler, and does not include the
   runtime necessary to create working programs.
   The compiler accepts input only in a normalized form, built from
   a set of Abstract Syntax Tree nodes provided by this project.
   - Input is a \p suite of \p procedure nodes.
   - One of the procedures in the input suite is designated as the
   entry point.
   - The input suite includes the entire program: the transitive
   closure of all procedures called from the entry point, excepting
   procedures supplied by the compiler itself, which are tracked by
   a \p registry maintained by the compiler.
   - All expressions in the input suite are flattened.  For example,
   \verbatim a = b + c * d \endverbatim must have been flattened to
   \verbatim e0 = c * d
 a = b + e0 \endverbatim
   - All closures are made explicit with \p closure objects.
   - All \p lambda functions are lifted to procedures.
   - All nested procedures have been flattened.
   - The suite has been typechecked, and AST nodes in the suite
   are populated with type information.
 */


 
