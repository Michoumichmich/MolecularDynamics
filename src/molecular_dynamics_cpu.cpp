#include <backend/cpu/cpu_backend.hpp>
#include <molecular_dynamics.hpp>


#ifdef BUILD_DOUBLE
template class sim::cpu_backend<double>;
template class sim::molecular_dynamics<double, sim::cpu_backend>;
#endif

#ifdef BUILD_FLOAT
template class sim::cpu_backend<float>;
template class sim::molecular_dynamics<float, sim::cpu_backend>;
#endif

#ifdef BUILD_HALF
template class sim::cpu_backend<sycl::half>;
template class sim::molecular_dynamics<sycl::half, sim::cpu_backend>;
#endif
