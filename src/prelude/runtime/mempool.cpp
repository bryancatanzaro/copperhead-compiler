#include <thrust/system/cuda/vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <cstdlib>
#include <iostream>
#include <map>
#include <cassert>
#include <prelude/runtime/tag_malloc_and_free.h>

// create a tag derived from system::cuda::tag for distinguishing
// our overloads of get_temporary_buffer and return_temporary_buffer
struct cuda_tag : thrust::system::cuda::tag {};
struct cpp_tag : thrust::system::cpp::tag {};

template<typename Tag>
struct underlying_tag {};

template<>
struct underlying_tag<cuda_tag> {
    typedef thrust::system::cuda::tag tag;
};


template<>
struct underlying_tag<cpp_tag> {
    typedef thrust::system::cpp::tag tag;
};

// cached_allocator: a simple allocator for caching allocation
// requests
template<typename Tag>
struct cached_allocator
{
    typedef typename underlying_tag<Tag>::tag thrust_tag;
    cached_allocator() {}

    void *allocate(std::ptrdiff_t num_bytes)
        {
            void *result = 0;

            std::cout << "Searching for " << num_bytes << " bytes; cache has " << free_blocks.size() << " entries" << std::endl;
            
            // search the cache for a free block
            free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

            if(free_block != free_blocks.end())
            {
                std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

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
                    std::cout << "cached_allocator::allocate(): no free block found for size " << num_bytes << "; calling cuda::malloc" << std::endl;

                    result = thrust::detail::tag_malloc(thrust_tag(),
                                                        num_bytes);
                }
                catch(std::runtime_error &e)
                {
                    //Allocation failed
                    //Nuke the cache and try again
                    std::cout << "cached_allocator::allocator(): memory full, purging cache" << std::endl;
                    
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
            std::cout << "cached_allocator::deallocate() " << std::endl;
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

    typedef std::multimap<std::ptrdiff_t, void*> free_blocks_type;
    typedef std::map<void *, std::ptrdiff_t>     allocated_blocks_type;

    free_blocks_type      free_blocks;
    allocated_blocks_type allocated_blocks;
};


// the cache is simply a global variable
// XXX ideally this variable is declared thread_local
cached_allocator<cuda_tag> g_cuda_allocator;
cached_allocator<cpp_tag> g_cpp_allocator;


void* malloc(cuda_tag, size_t cnt) {
    return g_cuda_allocator.allocate(cnt);
}

void* malloc(cpp_tag, size_t cnt) {
    return g_cpp_allocator.allocate(cnt);
}

void free(cuda_tag, void* ptr) {
    return g_cuda_allocator.deallocate(ptr);
}

void free(cpp_tag, void* ptr) {
    return g_cpp_allocator.deallocate(ptr);
}

int main()
{
    size_t n = 1 << 22;

    thrust::host_vector<int> h_input(n);

    // generate random input
    thrust::generate(h_input.begin(), h_input.end(), rand);

    thrust::system::cuda::vector<int> d_input = h_input;
    thrust::system::cuda::vector<int> d_result(n);

    size_t num_trials = 5;

    for(size_t i = 0; i < num_trials; ++i)
    {
        // initialize data to sort
        d_result = d_input;

        // tag iterators with my_tag to cause invocations of our
        // get_temporary_buffer and return_temporary_buffer
        // during sort
        thrust::sort(thrust::retag<cuda_tag>(d_result.begin()),
                     thrust::retag<cuda_tag>(d_result.end()));

        // ensure the result is sorted
        assert(thrust::is_sorted(d_result.begin(), d_result.end()));
    }

    // free all allocations before the underlying
    // device backend (e.g., CUDART) goes out of scope
    g_cuda_allocator.free_all();
    g_cpp_allocator.free_all();
    
    return 0;
}
