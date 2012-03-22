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

#include <prelude/runtime/make_cuarray.hpp>
#include <prelude/runtime/cuarray.hpp>
#include <prelude/runtime/monotype.hpp>
#include <prelude/runtime/fake_tags.hpp>

namespace copperhead {

namespace detail {

template<typename T>
struct type_deriver {};


template<>
struct type_deriver<float> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::float32_mt;
    }
};

template<>
struct type_deriver<double> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::float64_mt;
    }
};

template<>
struct type_deriver<int> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::int32_mt;
    }
};

template<>
struct type_deriver<long> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::int64_mt;
    }
};

template<>
struct type_deriver<bool> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::bool_mt;
    }
};

}

template<typename T>
sp_cuarray make_cuarray(size_t s) {
    sp_cuarray r(new cuarray());
    r->m_t =
        std::make_shared<backend::sequence_t>(
            detail::type_deriver<T>::fun());
    r->m_l.push_back(s);
    data_map data;
    data[detail::fake_omp_tag] = std::make_pair(vector<shared_ptr<chunk> >(), true);
   
    vector<std::shared_ptr<chunk> >& local_chunks = data[detail::fake_omp_tag].first;
#ifdef CUDA_SUPPORT
    data[detail::fake_cuda_tag] = std::make_pair(vector<shared_ptr<chunk> >(), true);
    vector<std::shared_ptr<chunk> >& remote_chunks = data[detail::fake_cuda_tag].first;
#endif
    
    local_chunks.push_back(
        std::make_shared<chunk>(detail::fake_omp_tag, s * sizeof(T)));
#ifdef CUDA_SUPPORT
    remote_chunks.push_back(
        std::make_shared<chunk>(detail::fake_cuda_tag, s * sizeof(T)));
#endif
    r->m_d = std::move(data);
    return r;
}


}
