#ifndef DYNAMIC_HYDRA_HPP
#define DYNAMIC_HYDRA_HPP

#include "dynamic_instance.hpp"

namespace dynamic {
const svec &hydra(struct state_t &state, size_t n);

void update_steps(struct state_t &state);
}
#endif /* !DYNAMIC_HYDRA_HPP */
