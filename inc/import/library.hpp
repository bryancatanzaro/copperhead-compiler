#pragma once

#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <set>
#include <ostream>
#include "prelude/phase.hpp"
#include "type.hpp"

namespace backend {

typedef std::tuple<const std::string, const iteration_structure> ident;

class fn_info {
    std::shared_ptr<type_t> m_type;
    std::shared_ptr<phase_t> m_phase;
public:
    fn_info(std::shared_ptr<type_t> type,
            std::shared_ptr<phase_t> phase)
        : m_type(type), m_phase(phase) {}

    fn_info(const fn_info& other) {
        m_type = other.m_type;
        m_phase = other.m_phase;
    }
    
    const type_t& type() const {
        return *m_type;
    }

    const phase_t& phase() const {
        return *m_phase;
    }

    const std::shared_ptr<type_t>& p_type() const {
        return m_type;
    }

    const std::shared_ptr<phase_t>& p_phase() const {
        return m_phase;
    }
        
};

struct library {
private:
    std::map<ident, fn_info> m_fns;
    std::map<std::string, std::string> m_fn_includes;
    std::set<std::string> m_includes;
    std::set<std::string> m_include_dirs;
    std::set<std::string> m_links;
    std::set<std::string> m_link_dirs;

public:
    inline library(std::map<ident, fn_info> &&fns,
                   std::map<std::string, std::string> &&fn_includes=std::map<std::string, std::string>(),
                   std::set<std::string> &&includes=std::set<std::string>(),
                   std::set<std::string> &&include_dirs=std::set<std::string>(),
                   std::set<std::string> &&links=std::set<std::string>(),
                   std::set<std::string> &&link_dirs=std::set<std::string>()
        )
        : m_fns(std::move(fns)),
          m_fn_includes(std::move(fn_includes)),
          m_includes(std::move(includes)),
          m_include_dirs(std::move(include_dirs)),
          m_links(std::move(links)),
          m_link_dirs(std::move(link_dirs))
        {}
    const std::map<ident, fn_info>& fns() const {
        return m_fns;
    }
    const std::map<std::string, std::string>& fn_includes() const {
        return m_fn_includes;
    }
    const std::set<std::string>& includes() const {
        return m_includes;
    }
    const std::set<std::string>& include_dirs() const {
        return m_include_dirs;
    }
    const std::set<std::string>& links() const {
        return m_links;
    }
    const std::set<std::string>& link_dirs() const {
        return m_link_dirs;
    }

};

struct registry {
private:
    std::map<ident, fn_info> m_fns;
    std::map<std::string, std::string> m_fn_includes;
    std::set<std::string> m_includes;
    std::set<std::string> m_include_dirs;
    std::set<std::string> m_links;
    std::set<std::string> m_link_dirs;
public:
    void add_library(std::shared_ptr<library>& l) {
        const std::set<std::string>& includes = l->includes();
        m_includes.insert(includes.begin(), includes.end());
        const std::set<std::string>& include_dirs = l->include_dirs();
        m_include_dirs.insert(include_dirs.begin(), include_dirs.end());
        const std::set<std::string>& links = l->links();
        m_links.insert(links.begin(), links.end());
        const std::set<std::string>& link_dirs = l->link_dirs();
        m_link_dirs.insert(link_dirs.begin(), link_dirs.end());
        const std::map<ident, fn_info>& fns = l->fns();
        m_fns.insert(fns.begin(), fns.end());
        const std::map<std::string, std::string>& fn_includes = l->fn_includes();
        m_fn_includes.insert(fn_includes.begin(), fn_includes.end());
    }
    const std::map<ident, fn_info>& fns() const {
        return m_fns;
    }
    const std::map<std::string, std::string>& fn_includes() const {
        return m_fn_includes;
    }
    const std::set<std::string>& includes() const {
        return m_includes;
    }
    const std::set<std::string>& include_dirs() const {
        return m_include_dirs;
    }
    const std::set<std::string>& links() const {
        return m_links;
    }
    const std::set<std::string>& link_dirs() const {
        return m_link_dirs;
    }
    
};


}
