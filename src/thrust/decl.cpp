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
using backend::utility::make_map;

namespace backend {

namespace thrust {

namespace detail {

typedef std::tuple<const char*, fn_info> named_info;

void declare_maps(int max_arity,
                  map<ident, fn_info>& fns,
                  map<string, string>& fn_includes) {
    vector<string> map_ids;
    for(int i = 1; i <= max_arity; i++) {
        stringstream strm;
        strm << "map" << i;
        map_ids.push_back(strm.str());
    }
    shared_ptr<const monotype_t> t_b =
        make_shared<const monotype_t>("b");
    shared_ptr<const monotype_t> seq_t_b =
        make_shared<const sequence_t>(t_b);
    vector<shared_ptr<const monotype_t> > a_types;
    vector<shared_ptr<const monotype_t> > a_seq_types;
    for(int i = 0; i < max_arity; i++) {
        stringstream strm;
        strm << "a" << i;
        auto t_an = make_shared<const monotype_t>(strm.str());
        auto t_seq_an = make_shared<const sequence_t>(t_an);
        a_types.push_back(t_an);
        a_seq_types.push_back(t_seq_an);
    }
    vector<shared_ptr<const fn_t> > fn_mts;
    for(int i = 0; i < max_arity; i++) {
        vector<shared_ptr<const type_t> > args;
        for(int j = 0; j <= i; j++) {
            args.push_back(a_types[j]);
        }
        auto tuple_args =
            make_shared<const tuple_t>(
                std::move(args));
        auto fn_mt = make_shared<const fn_t>(tuple_args,
                                       t_b);
        fn_mts.push_back(fn_mt);
    }
    vector<shared_ptr<const type_t> > map_types;
    for(int i = 0; i < max_arity; i++) {
        vector<shared_ptr<const monotype_t> > quantifiers;
        vector<shared_ptr<const type_t> > args;
        args.push_back(fn_mts[i]);
        for(int j = 0; j <= i; j++) {
            quantifiers.push_back(a_types[j]);
            args.push_back(a_seq_types[j]);
        }
        auto tuple_args =
            make_shared<const tuple_t>(std::move(args));
        auto fn_mt = make_shared<const fn_t>(tuple_args,
                                       seq_t_b);
        auto fn_pt = make_shared<const polytype_t>(
            std::move(quantifiers),
            fn_mt);
        map_types.push_back(fn_pt);
    }
    vector<shared_ptr<const phase_t> > map_phases;
    for(int i = 0; i < max_arity; i++) {
        vector<completion> inputs;
        inputs.push_back(completion::invariant);
        for(int j = 0; j <= i; j++) {
            inputs.push_back(completion::local);
        }
        auto map_phase = make_shared<const phase_t>(
            std::move(inputs),
            completion::local);
        map_phases.push_back(map_phase);
    }
    for(int i = 0; i < max_arity; i++) {
        fns.insert(make_pair(
                       make_pair(map_ids[i], iteration_structure::independent),
                       fn_info(map_types[i], map_phases[i])));
        fn_includes.insert(make_pair(map_ids[i], "prelude/primitives/map.h"));
    }
                        
}

void declare_zips(int max_arity,
                  map<ident, fn_info>& fns,
                  map<string, string>& fn_includes) {
    vector<string> zip_ids;
    for(int i = 1; i <= max_arity; i++) {
        stringstream strm;
        strm << "zip" << i;
        zip_ids.push_back(strm.str());
    }
    vector<shared_ptr<const monotype_t> > a_types;
    vector<shared_ptr<const monotype_t> > a_seq_types;
    for(int i = 0; i < max_arity; i++) {
        stringstream strm;
        strm << "a" << i;
        auto t_an = make_shared<const monotype_t>(strm.str());
        auto t_seq_an = make_shared<const sequence_t>(t_an);
        a_types.push_back(t_an);
        a_seq_types.push_back(t_seq_an);
    }

    vector<shared_ptr<const fn_t> > fn_mts;
    for(int i = 0; i < max_arity; i++) {
        vector<shared_ptr<const type_t> > args;
        vector<shared_ptr<const type_t> > results;
        for(int j = 0; j <= i; j++) {
            args.push_back(a_seq_types[j]);
            results.push_back(a_types[j]);
        }
        auto tuple_args =
            make_shared<const tuple_t>(
                std::move(args));
        auto tuple_results =
            make_shared<const tuple_t>(
                std::move(results));
        auto result =
            make_shared<const sequence_t>(tuple_results);
        auto fn_mt = make_shared<const fn_t>(tuple_args,
                                             result);
        fn_mts.push_back(fn_mt);
    }
    vector<shared_ptr<const type_t> > zip_types;
    for(int i = 0; i < max_arity; i++) {
        vector<shared_ptr<const monotype_t> > quantifiers;
        for(int j = 0; j <= i; j++) {
            quantifiers.push_back(a_types[j]);
        }
        auto fn_pt = make_shared<const polytype_t>(
            std::move(quantifiers),
            fn_mts[i]);
        zip_types.push_back(fn_pt);
    }
    vector<shared_ptr<const phase_t> > zip_phases;
    for(int i = 0; i < max_arity; i++) {
        vector<completion> inputs;
        for(int j = 0; j <= i; j++) {
            inputs.push_back(completion::local);
        }
        auto zip_phase = make_shared<const phase_t>(
            std::move(inputs),
            completion::local);
        zip_phases.push_back(zip_phase);
    }
    for(int i = 0; i < max_arity; i++) {
        fns.insert(make_pair(
                       make_pair(zip_ids[i], iteration_structure::independent),
                       fn_info(zip_types[i], zip_phases[i])));
        fn_includes.insert(make_pair(zip_ids[i], "prelude/primitives/zip.h"));
    }
                        
}

void declare_unzips(int max_arity,
                  map<ident, fn_info>& fns,
                  map<string, string>& fn_includes) {
    vector<string> unzip_ids;
    for(int i = 1; i <= max_arity; i++) {
        stringstream strm;
        strm << "unzip" << i;
        unzip_ids.push_back(strm.str());
    }
    vector<shared_ptr<const monotype_t> > a_types;
    vector<shared_ptr<const monotype_t> > a_seq_types;
    for(int i = 0; i < max_arity; i++) {
        stringstream strm;
        strm << "a" << i;
        auto t_an = make_shared<const monotype_t>(strm.str());
        auto t_seq_an = make_shared<const sequence_t>(t_an);
        a_types.push_back(t_an);
        a_seq_types.push_back(t_seq_an);
    }

    vector<shared_ptr<const fn_t> > fn_mts;
    for(int i = 0; i < max_arity; i++) {
        vector<shared_ptr<const type_t> > args;
        vector<shared_ptr<const type_t> > results;
        for(int j = 0; j <= i; j++) {
            args.push_back(a_types[j]);
            results.push_back(a_seq_types[j]);
        }
        auto tuple_args =
            make_shared<const tuple_t>(
                make_vector<shared_ptr<const type_t> >(
                    make_shared<const sequence_t>(
                        make_shared<const tuple_t>(
                            move(args)))));
        auto tuple_results =
            make_shared<const tuple_t>(
                std::move(results));
        auto fn_mt = make_shared<const fn_t>(tuple_args,
                                             tuple_results);
        fn_mts.push_back(fn_mt);
    }
    vector<shared_ptr<const type_t> > unzip_types;
    for(int i = 0; i < max_arity; i++) {
        vector<shared_ptr<const monotype_t> > quantifiers;
        for(int j = 0; j <= i; j++) {
            quantifiers.push_back(a_types[j]);
        }
        auto fn_pt = make_shared<const polytype_t>(
            std::move(quantifiers),
            fn_mts[i]);
        unzip_types.push_back(fn_pt);
    }
    vector<shared_ptr<const phase_t> > unzip_phases;
    for(int i = 0; i < max_arity; i++) {
        vector<completion> inputs;
        auto unzip_phase = make_shared<const phase_t>(
            make_vector<completion>(completion::local),
            completion::local);
        unzip_phases.push_back(unzip_phase);
    }
    for(int i = 0; i < max_arity; i++) {
        fns.insert(make_pair(
                       make_pair(unzip_ids[i], iteration_structure::independent),
                       fn_info(unzip_types[i], unzip_phases[i])));
        fn_includes.insert(make_pair(unzip_ids[i], "prelude/primitives/unzip.h"));
    }
                        
}

void declare_scans(map<ident, fn_info>& fns,
                   map<string, string>& fn_includes) {
    shared_ptr<const monotype_t> t_a = make_shared<const monotype_t>("a");
    shared_ptr<const monotype_t> seq_t_a = make_shared<const sequence_t>(t_a);
    shared_ptr<const monotype_t> bin_fn_t =
        make_shared<const fn_t>(
            make_shared<const tuple_t>(
                make_vector<shared_ptr<const type_t> >(t_a)(t_a)),
            t_a);
    shared_ptr<const polytype_t> scan_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(bin_fn_t)(seq_t_a)),
                seq_t_a));
    shared_ptr<const phase_t> scan_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::invariant)(completion::local),
            completion::total);
    
    fns.insert(make_pair(
                   make_pair("scan", iteration_structure::independent),
                   fn_info(scan_t, scan_phase_t)));
    fns.insert(make_pair(
                   make_pair("rscan", iteration_structure::independent),
                   fn_info(scan_t, scan_phase_t)));
    fn_includes.insert(make_pair("scan", "prelude/primitives/scan.h"));
    fn_includes.insert(make_pair("rscan", "prelude/primitives/scan.h"));
        
}

void declare_permutes(map<ident, fn_info>& fns,
                      map<string, string>& fn_includes) {
    shared_ptr<const monotype_t> t_a = make_shared<const monotype_t>("a");
    shared_ptr<const monotype_t> seq_t_a = make_shared<const sequence_t>(t_a);
    shared_ptr<const monotype_t> seq_int = make_shared<const sequence_t>(int64_mt);
    shared_ptr<const polytype_t> permute_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(seq_t_a)(seq_int)),
                seq_t_a));
    shared_ptr<const phase_t> permute_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::total)(completion::local),
            completion::total);
    fns.insert(make_pair(
                   make_pair("permute", iteration_structure::independent),
                   fn_info(permute_t, permute_phase_t)));
    fn_includes.insert(make_pair("permute", "prelude/primitives/scatter.h"));


    shared_ptr<const polytype_t> scatter_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(seq_t_a)(seq_int)(seq_t_a)),
                seq_t_a));
    shared_ptr<const phase_t> scatter_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::total)(completion::local)(completion::local),
            completion::total);
    fns.insert(make_pair(
                   make_pair("scatter", iteration_structure::independent),
                   fn_info(scatter_t, scatter_phase_t)));
    fn_includes.insert(make_pair("scatter", "prelude/primitives/scatter.h"));

    shared_ptr<const polytype_t> gather_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(seq_t_a)(seq_int)),
                seq_t_a));
    shared_ptr<const phase_t> gather_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::total)(completion::local),
            completion::local);
    fns.insert(make_pair(
                   make_pair("gather", iteration_structure::independent),
                   fn_info(gather_t, gather_phase_t)));
    fn_includes.insert(make_pair("gather", "prelude/primitives/gather.h"));

    
    
}

void declare_special_sequences(map<ident, fn_info>& fns,
                               map<string, string>& fn_includes) {
    shared_ptr<const monotype_t> t_a = make_shared<const monotype_t>("a");
    shared_ptr<const monotype_t> seq_t_a = make_shared<const sequence_t>(t_a);
    shared_ptr<const monotype_t> seq_int = make_shared<const sequence_t>(int64_mt);
    shared_ptr<const polytype_t> indices_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(seq_t_a)),
                seq_int));
    shared_ptr<const phase_t> indices_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::local),
            completion::local);
    fns.insert(make_pair(
                   make_pair("indices", iteration_structure::independent),
                   fn_info(indices_t, indices_phase_t)));
    fn_includes.insert(make_pair("indices", "prelude/primitives/indices.h"));

    shared_ptr<const polytype_t> replicate_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(t_a)(int64_mt)),
                seq_t_a));
    shared_ptr<const phase_t> replicate_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::local)(completion::local),
            completion::local);
    fns.insert(make_pair(
                   make_pair("replicate", iteration_structure::independent),
                   fn_info(replicate_t, replicate_phase_t)));
    fn_includes.insert(make_pair("replicate", "prelude/primitives/replicate.h"));

    shared_ptr<const polytype_t> shift_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(seq_t_a)(int64_mt)(t_a)),
                seq_t_a));
    shared_ptr<const phase_t> shift_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::total)(completion::local)(completion::local),
            completion::local);
    fns.insert(make_pair(
                   make_pair("shift", iteration_structure::independent),
                   fn_info(shift_t, shift_phase_t)));
    fn_includes.insert(make_pair("shift", "prelude/primitives/shift.h"));
               
    shared_ptr<const polytype_t> rotate_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(seq_t_a)(int64_mt)),
                seq_t_a));
    shared_ptr<const phase_t> rotate_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::total)(completion::local),
            completion::local);
    fns.insert(make_pair(
                   make_pair("rotate", iteration_structure::independent),
                   fn_info(rotate_t, rotate_phase_t)));
    fn_includes.insert(make_pair("rotate", "prelude/primitives/rotate.h"));                
}

void declare_transforms(map<ident, fn_info>& fns,
                        map<string, string>& fn_includes) {
    shared_ptr<const monotype_t> t_a = make_shared<const monotype_t>("a");
    shared_ptr<const monotype_t> seq_t_a = make_shared<const sequence_t>(t_a);
    shared_ptr<const polytype_t> adj_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(seq_t_a)),
                seq_t_a));
    shared_ptr<const phase_t> adj_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::total),
            completion::total);
    fns.insert(make_pair(
                   make_pair("adjacent_difference", iteration_structure::independent),
                   fn_info(adj_t, adj_phase_t)));
    fn_includes.insert(make_pair("adjacent_difference", "prelude/primitives/adjacent_difference.h"));
}

void declare_reductions(map<ident, fn_info>& fns,
                        map<string, string>& fn_includes) {
    shared_ptr<const monotype_t> t_a = make_shared<const monotype_t>("a");
    shared_ptr<const monotype_t> seq_t_a = make_shared<const sequence_t>(t_a);
    shared_ptr<const monotype_t> bin_fn_t =
        make_shared<const fn_t>(
            make_shared<const tuple_t>(
                make_vector<shared_ptr<const type_t> >(t_a)(t_a)),
            t_a);
    shared_ptr<const polytype_t> reduce_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >
                    (bin_fn_t)(seq_t_a)(t_a)),
                t_a));
    shared_ptr<const phase_t> reduce_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>
            (completion::invariant)(completion::local)(completion::local),
            completion::total);
    
    fns.insert(make_pair(
                   make_pair("reduce", iteration_structure::independent),
                   fn_info(reduce_t, reduce_phase_t)));
    fn_includes.insert(make_pair("reduce", "prelude/primitives/reduce.h"));
    shared_ptr<const polytype_t> sum_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >
                    (seq_t_a)),
                t_a));
    shared_ptr<const phase_t> sum_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>
            (completion::local),
            completion::total);
                    
    
    fns.insert(make_pair(
                   make_pair("sum", iteration_structure::independent),
                   fn_info(sum_t, sum_phase_t)));
    fn_includes.insert(make_pair("sum", "prelude/primitives/reduce.h"));
}

void declare_sorts(map<ident, fn_info>& fns,
                   map<string, string>& fn_includes) {
    shared_ptr<const monotype_t> t_a = make_shared<const monotype_t>("a");
    shared_ptr<const polytype_t> cmp_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(t_a)(t_a)),
                bool_mt));
    shared_ptr<const monotype_t> seq_t_a = make_shared<const sequence_t>(t_a);
    shared_ptr<const polytype_t> sort_t =
        make_shared<const polytype_t>(
            make_vector<shared_ptr<const monotype_t> >(t_a),
            make_shared<const fn_t>(
                make_shared<const tuple_t>(
                    make_vector<shared_ptr<const type_t> >(cmp_t)(seq_t_a)),
                seq_t_a));
    shared_ptr<const phase_t> sort_phase_t =
        make_shared<const phase_t>(
            make_vector<completion>(completion::invariant)(completion::total),
            completion::total);
    fns.insert(make_pair(
                   make_pair("sort", iteration_structure::independent),
                   fn_info(sort_t, sort_phase_t)));
    fn_includes.insert(make_pair("sort", "prelude/primitives/sort.h"));
}

}

}


shared_ptr<library> get_thrust() {
    int max_arity = 10;
    map<ident, fn_info> exported_fns;
    map<string, string> fn_includes;
    thrust::detail::declare_maps(max_arity, exported_fns, fn_includes);
    thrust::detail::declare_scans(exported_fns, fn_includes);
    thrust::detail::declare_permutes(exported_fns, fn_includes);
    thrust::detail::declare_special_sequences(exported_fns, fn_includes);
    thrust::detail::declare_transforms(exported_fns, fn_includes);
    thrust::detail::declare_reductions(exported_fns, fn_includes);
    thrust::detail::declare_sorts(exported_fns, fn_includes);
    thrust::detail::declare_zips(max_arity, exported_fns, fn_includes);
    thrust::detail::declare_unzips(max_arity, exported_fns, fn_includes);

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
                    move(fn_includes),
                    make_set<string>(string(THRUST_FILE)),
                    move(include_paths)));
}


}
