#pragma once
#include <string>

namespace backend {
namespace detail {

//XXX This isn't necessary any more, is it?
std::string mark_generated_id(const std::string &in);

std::string fnize_id(const std::string &in);

std::string wrap_array_id(const std::string &in);

std::string wrap_proc_id(const std::string &in);

std::string typify(const std::string &in);

std::string complete(const std::string &in);

}
}
