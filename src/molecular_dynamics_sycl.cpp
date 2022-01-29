#ifndef BUILD_SYCL
#    error "Pass -DSYCL=ON to cmake in order to enable the SYCL backend"
#endif


#include <backend/sycl/sycl_backend.hpp>
#include <molecular_dynamics/molecular_dynamics.hpp>


#ifdef BUILD_DOUBLE
template class sim::sycl_backend<double>;
template class sim::molecular_dynamics<double, sim::sycl_backend>;
#endif

#ifdef BUILD_FLOAT
template class sim::sycl_backend<float>;
template class sim::molecular_dynamics<float, sim::sycl_backend>;
#endif

#ifdef BUILD_HALF
template class sim::sycl_backend<sycl::half>;
template class sim::molecular_dynamics<sycl::half, sim::sycl_backend>;
#endif
