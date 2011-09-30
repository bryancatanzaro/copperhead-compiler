#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "py_printer.hpp"
#include "library.hpp"

namespace backend {


class bpl_wrap
    : public copier
{
private:
    const std::string& m_entry_point;
    std::vector<std::shared_ptr<include> > m_includes;
    bool m_outer;
public:
    wrap(const std::string& entry_point,
         const registry& reg)
        : m_entry_point(entry_point),
          m_outer(true) {
        for(auto i = reg.includes.cbegin();
            i != reg.includes.cend();
            i++) {
            m_includes.push_back(
                std::shared_ptr<include>(
                    new include(
                        std::shared_ptr<name>(
                            new name(*i)))));
        }

    }
    using copier::operator();
    result_type operator()(const suite& n) {
        if (!m_outer) {
            return this->copier::operator(n);
        }
        m_outer = false;
        std::vector<std::shared_ptr<statement> > stmts;
        
        //Copy includes into statements
        stmts.insert(stmts.begin(),
                     m_includes.begin(),
                     m_includes.end());
        for(auto i = n.begin();
            i != n.end();
            i++) {
            stmts.insert(boost::apply_visitor(*this, *i));
        }

        std::shared_ptr<name> entry_name(
            new name(m_entry_point));
        std::shared_ptr<name> python_entry_name(
            new name("\"" + m_entry_point + "\""));
        std::shared_ptr<tuple> def_args(
            new tuple(
                std::vector<std::shared_ptr<expression> >{python_entry_name, entry_name}));
        std::shared_ptr<name> def_fn(
            new name(
                detail::boost_python_def()));
        std::shared_ptr<call> def_call(
            new call(
                std::shared_ptr<apply>(
                    new apply(def_fn, def_args))));
        std::shared_ptr<suite> def_suite(
            new suite(
                std::vector<std::shared_ptr<suite> >{def_call}));
        std::shared_ptr<name> bpl_module(
            new name(
                detail::boost_python_module()));
        std::shared_ptr<tuple> bpl_module_args(
            new tuple(
                std::vector<std::shared_ptr<expression>{entry_name}));
        std::shared_ptr<procedure> bpl_proc(
            new procedure(
                bpl_module,
                bpl_module_args,
                def_suite));
        stmts.insert(bpl_proc);

        return result_type(
            new suite(std::move(stmts)));
        
        
    }
};
}
