#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include "allocators.h"
#include "chunk.h"
#include "type.hpp"
#include "ctype.hpp"

struct cuarray {
    std::vector<std::shared_ptr<chunk<host_alloc> > > m_local;
#ifdef CUDA_SUPPORT
    std::vector<std::shared_ptr<chunk<cuda_alloc> > > m_remote;
    bool m_clean_local;
    bool m_clean_remote;
#endif
    std::vector<size_t> m_l;
    std::shared_ptr<backend::type_t> m_t;
    std::shared_ptr<backend::ctype::type_t> m_ct;
};
