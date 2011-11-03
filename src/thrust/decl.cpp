#include "thrust/decl.hpp"

using std::shared_ptr;
using std::make_shared;
using std::tuple;
using std::move;
using std::vector;
using std::set;
using std::map;
using std::string;
using std::pair;


namespace backend {

namespace thrust {

namespace detail {

typedef std::tuple<const char*, fn_info> named_info;

shared_ptr<monotype_t> t_a =
    make_shared<monotype_t>("a");

shared_ptr<monotype_t> t_a0 =
    make_shared<monotype_t>("a0");

shared_ptr<monotype_t> t_a1 =
    make_shared<monotype_t>("a1");

shared_ptr<monotype_t> t_a2 =
    make_shared<monotype_t>("a2");

shared_ptr<monotype_t> t_a3 =
    make_shared<monotype_t>("a3");

shared_ptr<monotype_t> t_a4 =
    make_shared<monotype_t>("a4");

shared_ptr<monotype_t> t_a5 =
    make_shared<monotype_t>("a5");

shared_ptr<monotype_t> t_a6 =
    make_shared<monotype_t>("a6");

shared_ptr<monotype_t> t_a7 =
    make_shared<monotype_t>("a7");

shared_ptr<monotype_t> t_a8 =
    make_shared<monotype_t>("a8");

shared_ptr<monotype_t> t_a9 =
    make_shared<monotype_t>("a9");

shared_ptr<monotype_t> t_a10 =
    make_shared<monotype_t>("a10");

shared_ptr<monotype_t> t_b =
    make_shared<monotype_t>("b");

shared_ptr<monotype_t> seq_t_a0 =
    make_shared<sequence_t>(t_a0);

shared_ptr<monotype_t> seq_t_a1 =
    make_shared<sequence_t>(t_a1);

shared_ptr<monotype_t> seq_t_a2 =
    make_shared<sequence_t>(t_a2);

shared_ptr<monotype_t> seq_t_a3 =
    make_shared<sequence_t>(t_a3);

shared_ptr<monotype_t> seq_t_a4 =
    make_shared<sequence_t>(t_a4);

shared_ptr<monotype_t> seq_t_a5 =
    make_shared<sequence_t>(t_a5);

shared_ptr<monotype_t> seq_t_a6 =
    make_shared<sequence_t>(t_a6);

shared_ptr<monotype_t> seq_t_a7 =
    make_shared<sequence_t>(t_a7);

shared_ptr<monotype_t> seq_t_a8 =
    make_shared<sequence_t>(t_a8);

shared_ptr<monotype_t> seq_t_a9 =
    make_shared<sequence_t>(t_a9);

shared_ptr<monotype_t> seq_t_a10 =
    make_shared<sequence_t>(t_a10);

shared_ptr<monotype_t> seq_t_b =
    make_shared<sequence_t>(t_b);

shared_ptr<monotype_t> fn_1_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0}),
        t_b);

shared_ptr<monotype_t> fn_2_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0, t_a1}),
        t_b);

shared_ptr<monotype_t> fn_3_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0, t_a1, t_a2}),
        t_b);

shared_ptr<monotype_t> fn_4_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0, t_a1, t_a2, t_a3}),
        t_b);

shared_ptr<monotype_t> fn_5_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0, t_a1, t_a2, t_a3, t_a4}),
        t_b);

shared_ptr<monotype_t> fn_6_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5}),
        t_b);

shared_ptr<monotype_t> fn_7_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5, t_a6}),
        t_b);

shared_ptr<monotype_t> fn_8_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5, t_a6, t_a7}),
        t_b);

shared_ptr<monotype_t> fn_9_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5, t_a6, t_a7, t_a8}),
        t_b);

shared_ptr<monotype_t> fn_10_mt =
    make_shared<fn_t>(
        make_shared<tuple_t>(
            vector<shared_ptr<type_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5, t_a6, t_a7, t_a8, t_a9}),
        t_b);

shared_ptr<polytype_t> map1_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_1_mt, seq_t_a0}),
            seq_t_b));

shared_ptr<polytype_t> map2_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_a1, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_2_mt, seq_t_a0, seq_t_a1}),
            seq_t_b));

shared_ptr<polytype_t> map3_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_a1, t_a2, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_3_mt, seq_t_a0, seq_t_a1, seq_t_a2}),
            seq_t_b));

shared_ptr<polytype_t> map4_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_a1, t_a2, t_a3, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_4_mt, seq_t_a0, seq_t_a1, seq_t_a2, seq_t_a3}),
            seq_t_b));

shared_ptr<polytype_t> map5_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_5_mt, seq_t_a0, seq_t_a1, seq_t_a2, seq_t_a3, seq_t_a4}),
            seq_t_b));

shared_ptr<polytype_t> map6_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_6_mt, seq_t_a0, seq_t_a1, seq_t_a2, seq_t_a3, seq_t_a4, seq_t_a5}),
            seq_t_b));

shared_ptr<polytype_t> map7_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5, t_a6, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_7_mt, seq_t_a0, seq_t_a1, seq_t_a2, seq_t_a3, seq_t_a4, seq_t_a5, seq_t_a6}),
            seq_t_b));

shared_ptr<polytype_t> map8_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5, t_a6, t_a7, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_8_mt, seq_t_a0, seq_t_a1, seq_t_a2, seq_t_a3, seq_t_a4, seq_t_a5, seq_t_a6, seq_t_a7}),
            seq_t_b));

shared_ptr<polytype_t> map9_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5, t_a6, t_a7, t_a8, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_9_mt, seq_t_a0, seq_t_a1, seq_t_a2, seq_t_a3, seq_t_a4, seq_t_a5, seq_t_a6, seq_t_a7, seq_t_a8}),
            seq_t_b));

shared_ptr<polytype_t> map10_t =
    make_shared<polytype_t>(
        vector<shared_ptr<monotype_t> >{t_a0, t_a1, t_a2, t_a3, t_a4, t_a5, t_a6, t_a7, t_a8, t_a9, t_b},
        make_shared<fn_t>(
            make_shared<tuple_t>(
                vector<shared_ptr<type_t> >{fn_10_mt, seq_t_a0, seq_t_a1, seq_t_a2, seq_t_a3, seq_t_a4, seq_t_a5, seq_t_a6, seq_t_a7, seq_t_a8, seq_t_a9}),
            seq_t_b));



std::vector<const char*> thrust_fn_names = {
    "adjacent_difference",
    "map1", "map2", "map3", "map4", "map5",
    "map6", "map7", "map8", "map9", "map10",
    "scan",
    "rscan",
    "indices",
    "permute"/*,
    "exscan",
    "exrscan"*/
};




}



}


shared_ptr<library> get_thrust() {
    map<ident, fn_info> exported_fns;
    for(auto i = thrust::detail::thrust_fn_names.begin();
        i != thrust::detail::thrust_fn_names.end();
        i++) {
        exported_fns.insert(pair<ident, fn_info>(
                       ident(string(*i), iteration_structure::parallel),
                       //XXX Need to put real types in
                       fn_info(void_mt,
                               make_shared<phase_t>(
                                   vector<completion>{},
                                   completion::invariant)

                           )));

    }
    //XXX HACK.  NEED boost::filesystem path manipulation
    string library_path(string(detail::get_path(PRELUDE_PATH)) +
                             "/../thrust");
    string thrust_path(string(detail::get_path(THRUST_PATH)));
    set<string> include_paths;
    if (library_path.length() > 0) {
        include_paths.insert(library_path);
    }
    if (thrust_path.length() > 0) {
        include_paths.insert(thrust_path);
    }
    return shared_ptr<library>(
        new library(move(exported_fns),
                    set<string>{string(THRUST_FILE)},
                    move(include_paths)));
}


}
