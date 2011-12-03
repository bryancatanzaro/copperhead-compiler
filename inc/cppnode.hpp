#pragma once

#include "node.hpp"
#include "expression.hpp"
#include "statement.hpp"
#include <vector>
#include <memory>

namespace backend {

/*! 
  \addtogroup nodes
  @{
 */

//! An AST node representing a C++ struct
/*! This AST node is only used internally by the compiler.
*/
class structure
    : public statement
{
protected:
    std::shared_ptr<name> m_id;
    std::shared_ptr<suite> m_stmts;
    const std::vector<std::shared_ptr<ctype::type_t> > m_typevars;
public:
    //! Constructor
/*! 
  
  \param name Name of the structure
  \param stmts Statements inside the structure
  \param typevars Template arguments, if any (optional)
*/
    structure(const std::shared_ptr<name> &name,
              const std::shared_ptr<suite> &stmts,
              std::vector<std::shared_ptr<ctype::type_t> > &&typevars =
              std::vector<std::shared_ptr<ctype::type_t> >());
    //! Gets the name of the structure 
/*! 
  \return The name of the structure
*/
    const name& id(void) const;
    //! Gets the statements inside the structure
/*! 
  \return The statements inside the structure
*/
    const suite& stmts(void) const;

    //! Iterator type for accessing template arguments
    typedef decltype(boost::make_indirect_iterator(m_typevars.cbegin())) const_iterator;
    //! Gets iterator to beginning of template arguments
    const_iterator begin(void) const;
    //! Gets iterator to end of template arguments
    const_iterator end(void) const;
};

//! A templated name, such as shared_ptr<int>
/*! In situations where we need a name AST node, but we also need
  to indicate the types that instantiate a C++ template, we use this
  node.  This is only used internally by the compiler.
*/
class templated_name
    : public name
{
protected:
    std::shared_ptr<ctype::tuple_t> m_template_types;
public:
    //! Constructor
/*! 
  
  \param id The value of the name
  \param template_types A set of types to instantiate the name with
  \param type The Copperhead type of the name
  \param ctype The C++ implementation type of the name
  
  \return 
*/
    templated_name(const std::string &id,
                   const std::shared_ptr<ctype::tuple_t> &template_types,
                   const std::shared_ptr<type_t>& type = void_mt,
                   const std::shared_ptr<ctype::type_t>& ctype = ctype::void_mt);
    //! Gets the template instantiation types
/*! 
  \return A tuple of types which are used to instantiate this name
*/
    const ctype::tuple_t& template_types() const;
};

//! AST node representing a C include statement
/*! This node is used to represent a C include statement.

  This node is only used internally by the compiler.
 */
class include
    : public statement
{
protected:
    const std::shared_ptr<literal> m_id;
    const char m_open;
    const char m_close;
public:
    //! Constructor
/*! 
  \param id The name of the file to include
  \param open The character to open the include, eg '<'
  \param close The character to close the include, eg '>'
*/
    include(const std::shared_ptr<literal> &id,
            const char open = '\"',
            const char close = '\"');
    //! Gets the name of the file which is being included
    const literal& id() const;
    //! Gets the character which opens the include statement
    const char& open() const;
    //! Gets the character which closes the include statement
    const char& close() const;
};

//! AST node representing a C typedef statement
/*! This node is only used internally by the compiler
*/
class typedefn
    : public statement
{
protected:
    const std::shared_ptr<ctype::type_t> m_origin;
    const std::shared_ptr<ctype::type_t> m_rename;
public:
    //! Constructor
/*! typedef $origin $rename;
  
  \param origin The original type.
  \param rename The new type.
  
*/
    typedefn(const std::shared_ptr<ctype::type_t> origin,
             const std::shared_ptr<ctype::type_t> rename);
    //! Gets the origin type
    const ctype::type_t& origin() const;
    //! Gets the rename type
    const ctype::type_t& rename() const;
};
/*! 
  @}
 */


}
