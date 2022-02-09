#include <backend/cpu/cpu_backend.hpp>
#include <molecular_dynamics/molecular_dynamics.hpp>


#ifdef BUILD_DOUBLE
template class sim::cpu_backend<false, double>;
template class sim::molecular_dynamics<double, sim::cpu_backend_regular>;
template class sim::cpu_backend<true, double>;
template class sim::molecular_dynamics<double, sim::cpu_backend_decompose>;
#endif

#ifdef BUILD_FLOAT
template class sim::cpu_backend<false, float>;
template class sim::molecular_dynamics<float, sim::cpu_backend_regular>;
template class sim::cpu_backend<true, float>;
template class sim::molecular_dynamics<float, sim::cpu_backend_decompose>;
#endif

#ifdef BUILD_HALF
template class sim::cpu_backend<false, sycl::half>;
template class sim::molecular_dynamics<sycl::half, sim::cpu_backend_regular>;
template class sim::cpu_backend<true, sycl::half>;
template class sim::molecular_dynamics<sycl::half, sim::cpu_backend_decompose>;
#endif
