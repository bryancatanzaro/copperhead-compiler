#pragma once
#include <string>


namespace backend {
namespace detail {

/*!
  \addtogroup utilities
  @{
*/


//! Gets string for get_remote_r
const std::string get_remote_r();
//! Gets string for get_remote_w
const std::string get_remote_w();
//! Gets string for wrap
const std::string wrap();
//! Gets string for make_remote
const std::string make_remote();
//! Gets string for boost_python_module
const std::string boost_python_module();
//! Gets string for boost_python_def
const std::string boost_python_def();
//! Gets string for phase_boundary
const std::string phase_boundary();

/*!
  @}
*/


}
}

