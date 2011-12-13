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
using std::make_pair;
using std::stringstream;
using backend::utility::make_vector;
using backend::utility::make_set;

namespace backend {

namespace thrust {

namespace detail {

typedef std::tuple<const char*, fn_info> named_info;

void declare_maps(int max_arity, map<ident, fn_info>& fns) {
    vector<string> map_ids;
    for(int i = 1; i <= max_arity; i++) {
        stringstream strm;
        strm << "map" << i;
        map_ids.push_back(strm.str());
    }
    shared_ptr<monotype_t> t_b =
        make_shared<monotype_t>("b");
    shared_ptr<monotype_t> seq_t_b =
        make_shared<sequence_t>(t_b);
    vector<shared_ptr<monotype_t> > a_types;
    vector<shared_ptr<monotype_t> > a_seq_types;
    for(int i = 0; i < max_arity; i++) {
        stringstream strm;
        strm << "a" << i;
        auto t_an = make_shared<monotype_t>(strm.str());
        auto t_seq_an = make_shared<sequence_t>(t_an);
        a_types.push_back(t_an);
        a_seq_types.push_back(t_seq_an);
    }
    vector<shared_ptr<fn_t> > fn_mts;
    for(int i = 0; i < max_arity; i++) {
        vector<shared_ptr<type_t> > args;
        for(int j = 0; j <= i; j++) {
            args.push_back(a_types[j]);
        }
        auto tuple_args =
            make_shared<tuple_t>(
                std::move(args));
        auto fn_mt = make_shared<fn_t>(tuple_args,
                                       t_b);
        fn_mts.push_back(fn_mt);
    }
    vector<shared_ptr<type_t> > map_types;
    for(int i = 0; i < max_arity; i++) {
        vector<shared_ptr<monotype_t> > quantifiers;
        vector<shared_ptr<type_t> > args;
        args.push_back(fn_mts[i]);
        for(int j = 0; j <= i; j++) {
            quantifiers.push_back(a_types[j]);
            args.push_back(a_seq_types[j]);
        }
        auto tuple_args =
            make_shared<tuple_t>(std::move(args));
        auto fn_mt = make_shared<fn_t>(tuple_args,
                                       seq_t_b);
        auto fn_pt = make_shared<polytype_t>(
            std::move(quantifiers),
            fn_mt);
        map_types.push_back(fn_pt);
    }
    vector<shared_ptr<phase_t> > map_phases;
    for(int i = 0; i < max_arity; i++) {
        vector<completion> inputs;
        inputs.push_back(completion::invariant);
        for(int j = 0; j <= i; j++) {
            inputs.push_back(completion::local);
        }
        auto map_phase = make_shared<phase_t>(
            std::move(inputs),
            completion::local);
        map_phases.push_back(map_phase);
    }
    for(int i = 0; i < max_arity; i++) {
        fns.insert(make_pair(
                       make_pair(map_ids[i], iteration_structure::independent),
                       fn_info(map_types[i], map_phases[i])));
    }
                        
}

void declare_scans(map<ident, fn_info>& fns) {
    shared_ptr<monotype_t> t_a = make_shared<monotype_t>("a");
    shared_ptr<monotype_t> seq_t_a = make_shared<sequence_t>(t_a);
    shared_ptr<monotype_t> bin_fn_t =
        make_shared<fn_t>(
            make_shared<tuple_t>(
                make_vector<shared_ptr<type_t> >(t_a)(t_a)),
            t_a);
    shared_ptr<polytype_t> scan_t =
        make_shared<polytype_t>(
            make_vector<shared_ptr<monotype_t> >(t_a),
            make_shared<fn_t>(
                make_shared<tuple_t>(
                    make_vector<shared_ptr<type_t> >(bin_fn_t)(seq_t_a)),
                seq_t_a));
    shared_ptr<phase_t> scan_phase_t =
        make_shared<phase_t>(
            make_vector<completion>(completion::invariant)(completion::local),
            completion::total);
    
    fns.insert(make_pair(
                   make_pair("scan", iteration_structure::independent),
                   fn_info(scan_t, scan_phase_t)));
    fns.insert(make_pair(
                   make_pair("rscan", iteration_structure::independent),
                   fn_info(scan_t, scan_phase_t)));
}

void declare_permutes(map<ident, fn_info>& fns) {
    shared_ptr<monotype_t> t_a = make_shared<monotype_t>("a");
    shared_ptr<monotype_t> seq_t_a = make_shared<sequence_t>(t_a);
    shared_ptr<monotype_t> seq_int = make_shared<sequence_t>(int32_mt);
    shared_ptr<polytype_t> permute_t =
        make_shared<polytype_t>(
            make_vector<shared_ptr<monotype_t> >(t_a),
            make_shared<fn_t>(
                make_shared<tuple_t>(
                    make_vector<shared_ptr<type_t> >(seq_t_a)(seq_int)),
                seq_t_a));
    shared_ptr<phase_t> permute_phase_t =
        make_shared<phase_t>(
            make_vector<completion>(completion::total)(completion::local),
            completion::total);
    fns.insert(make_pair(
                   make_pair("permute", iteration_structure::independent),
                   fn_info(permute_t, permute_phase_t)));
}

void declare_special_sequences(map<ident, fn_info>& fns) {
    shared_ptr<monotype_t> t_a = make_shared<monotype_t>("a");
    shared_ptr<monotype_t> seq_t_a = make_shared<sequence_t>(t_a);
    shared_ptr<monotype_t> seq_int = make_shared<sequence_t>(int32_mt);
    shared_ptr<polytype_t> indices_t =
        make_shared<polytype_t>(
            make_vector<shared_ptr<monotype_t> >(t_a),
            make_shared<fn_t>(
                make_shared<tuple_t>(
                    make_vector<shared_ptr<type_t> >(seq_t_a)),
                seq_int));
    shared_ptr<phase_t> indices_phase_t =
        make_shared<phase_t>(
            make_vector<completion>(completion::local),
            completion::local);
    fns.insert(make_pair(
                   make_pair("indices", iteration_structure::independent),
                   fn_info(indices_t, indices_phase_t)));
}

void declare_transforms(map<ident, fn_info>& fns) {
    shared_ptr<monotype_t> t_a = make_shared<monotype_t>("a");
    shared_ptr<monotype_t> seq_t_a = make_shared<sequence_t>(t_a);
    shared_ptr<polytype_t> adj_t =
        make_shared<polytype_t>(
            vector<shared_ptr<monotype_t> >{t_a},
            make_shared<fn_t>(
                make_shared<tuple_t>(
                    make_vector<shared_ptr<type_t> >(seq_t_a)),
                seq_t_a));
    shared_ptr<phase_t> adj_phase_t =
        make_shared<phase_t>(
            make_vector<completion>(completion::total),
            completion::total);
    fns.insert(make_pair(
                   make_pair("adjacent_difference", iteration_structure::independent),
                   fn_info(adj_t, adj_phase_t)));
}

void declare_reductions(map<ident, fn_info>& fns) {
    shared_ptr<monotype_t> t_a = make_shared<monotype_t>("a");
    shared_ptr<monotype_t> seq_t_a = make_shared<sequence_t>(t_a);
    shared_ptr<monotype_t> bin_fn_t =
        make_shared<fn_t>(
            make_shared<tuple_t>(
                make_vector<shared_ptr<type_t> >(t_a)(t_a)),
            t_a);
    shared_ptr<polytype_t> reduce_t =
        make_shared<polytype_t>(
            make_vector<shared_ptr<monotype_t> >(t_a),
            make_shared<fn_t>(
                make_shared<tuple_t>(
                    make_vector<shared_ptr<type_t> >
                    (bin_fn_t)(seq_t_a)(t_a)),
                t_a));
    shared_ptr<phase_t> reduce_phase_t =
        make_shared<phase_t>(
            make_vector<completion>
            (completion::invariant)(completion::local)(completion::local),
            completion::total);
    
    fns.insert(make_pair(
                   make_pair("reduce", iteration_structure::independent),
                   fn_info(reduce_t, reduce_phase_t)));


}

}

}


shared_ptr<library> get_thrust() {
    int max_arity = 10;
    map<ident, fn_info> exported_fns;
    thrust::detail::declare_maps(max_arity, exported_fns);
    thrust::detail::declare_scans(exported_fns);
    thrust::detail::declare_permutes(exported_fns);
    thrust::detail::declare_special_sequences(exported_fns);
    thrust::detail::declare_transforms(exported_fns);
    thrust::detail::declare_reductions(exported_fns);
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
                    make_set<string>(string(THRUST_FILE)),
                    move(include_paths)));
}


}
