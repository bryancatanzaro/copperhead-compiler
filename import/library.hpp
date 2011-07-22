#pragma once

#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <set>
#include <ostream>

namespace backend {

enum struct iteration_structure {
    scalar,
    sequential,
    parallel,
    independent
};

//To make iteration_structures comparable
//And thus work as part of a key to a map
bool operator<(iteration_structure l, iteration_structure r) {
    return (int)l < int(r);
}

//To make iteration_structures print
std::ostream& operator<<(std::ostream& strm, iteration_structure is) {
    switch(is) {
    case iteration_structure::scalar:
        strm << "scalar";
        break;
    case iteration_structure::sequential:
        strm << "sequential";
        break;
    case iteration_structure::parallel:
        strm << "parallel";
        break;
    case iteration_structure::independent:
        strm << "independent";
        break;
    default:
        strm << "unknown";
    }
    return strm;
}

typedef std::tuple<const std::string, const iteration_structure> ident;

struct fn_info {
private:
    //shared_ptr<type> m_type;
    //shared_ptr<phase> m_phase;
        

};


// struct scalar
//     : public fn_info {
// public:
//     scalar(const std::string &name)
//         : fn_info(name, iteration_structure::scalar) {}
// };



struct library {
private:
    std::map<ident, fn_info> m_fns;
    std::set<std::string> m_include_dirs;
    std::set<std::string> m_includes;
    std::set<std::string> m_link_dirs;
    std::set<std::string> m_links;
public:
    inline library(std::map<ident, fn_info> &&fns,
                   std::set<std::string> &&includes=std::set<std::string>(),
                   std::set<std::string> &&include_dirs=std::set<std::string>(),
                   std::set<std::string> &&links=std::set<std::string>(),
                   std::set<std::string> &&link_dirs=std::set<std::string>()
        )
        : m_fns(std::move(fns)),
          m_includes(std::move(includes)),
          m_include_dirs(std::move(include_dirs)),
          m_links(std::move(links)),
          m_link_dirs(std::move(link_dirs))
        {}
    const std::map<ident, fn_info>& fns() const {
        return m_fns;
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
    }
    const std::map<ident, fn_info>& fns() const {
        return m_fns;
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
