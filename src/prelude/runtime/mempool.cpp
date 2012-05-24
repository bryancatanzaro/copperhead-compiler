#include <cstdlib>
#include <map>
#include <cassert>
#include <prelude/runtime/mempool.hpp>

namespace copperhead {
namespace detail {

// cached_allocator: a simple allocator for caching allocation
// requests.  Adapted from thrust's custom_temporary_allocator example
template<typename Tag>
struct cached_allocator
{
    typedef typename thrust_memory_tag<Tag>::tag thrust_tag;
    bool m_live;
    cached_allocator() : m_live(true) {}

    void *allocate(std::ptrdiff_t num_bytes)
        {
            void *result = 0;
            
            // search the cache for a free block
            free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

            if(free_block != free_blocks.end())
            {
                // get the pointer
                result = free_block->second;

                // erase from the free_blocks map
                free_blocks.erase(free_block);
            }
            else
            {
                // no allocation of the right size exists
                // create a new one with cuda::malloc
                // throw if cuda::malloc can't satisfy the request
                try
                {
                    result = thrust::detail::tag_malloc(thrust_tag(),
                                                        num_bytes);
                }
                catch(std::runtime_error &e)
                {
                    //Allocation failed
                    //Nuke the cache and try again
                    free_free();
                    
                    try {
                        result = thrust::detail::tag_malloc(thrust_tag(),
                                                            num_bytes);
                    } catch(std::runtime_error &e) {
                        throw;
                    }
                }
            }

            // insert the allocated pointer into the allocated_blocks map
            allocated_blocks.insert(std::make_pair(result, num_bytes));

            return result;
        }

    void deallocate(void *ptr)
        {
            //Check to see if allocator has been closed
            if (!m_live) {
                return;
            }
            // erase the allocated block from the allocated blocks map
            allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
            std::ptrdiff_t num_bytes = iter->second;
            allocated_blocks.erase(iter);

            // insert the block into the free blocks map
            free_blocks.insert(std::make_pair(num_bytes, ptr));
        }

    void free_free() {
        for(free_blocks_type::iterator i = free_blocks.begin();
            i != free_blocks.end();
            ++i) {
            // transform the pointer to cuda::pointer before calling cuda::free
            thrust::detail::tag_free(thrust_tag(), i->second);
        }
        free_blocks.clear();
    }
    

    void free_all()
        {
            free_free();
            for(allocated_blocks_type::iterator i = allocated_blocks.begin();
                i != allocated_blocks.end();
                ++i)
            {
                thrust::detail::tag_free(thrust_tag(), i->first);
            }
            allocated_blocks.clear();
        }

    void close() {
        free_all();
        m_live = false;
    }
    typedef std::multimap<std::ptrdiff_t, void*> free_blocks_type;
    typedef std::map<void *, std::ptrdiff_t>     allocated_blocks_type;

    free_blocks_type      free_blocks;
    allocated_blocks_type allocated_blocks;
};


//XXX Need to protect access to these with mutex
cached_allocator<cpp_tag> g_cpp_allocator;

//XXX Need to protect access to these with mutex
#ifdef CUDA_SUPPORT
cached_allocator<cuda_tag> g_cuda_allocator;
#endif

}

void* malloc(cpp_tag, size_t cnt) {
    return detail::g_cpp_allocator.allocate(cnt);
}

void free(cpp_tag, void* ptr) {
    return detail::g_cpp_allocator.deallocate(ptr);
}

#ifdef CUDA_SUPPORT
void* malloc(cuda_tag, size_t cnt) {
    return detail::g_cuda_allocator.allocate(cnt);
}

void free(cuda_tag, void* ptr) {
    return detail::g_cuda_allocator.deallocate(ptr);
}
#endif

void take_down() {
    detail::g_cpp_allocator.close();
    #ifdef CUDA_SUPPORT
    detail::g_cuda_allocator.close();
    #endif
}

}
