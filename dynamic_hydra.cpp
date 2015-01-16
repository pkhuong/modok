#include <algorithm>
#include <utility>
#include <cstdio>

#include "dynamic_instance.hpp"

namespace dynamic {

/*
 * Quoth "Fast Distributed Coordinate Descent for Non-Strongly Convex Losses"
 *    (http://arxiv.org/abs/1405.5300v2), Algorithm 1.
 *
 * t^i_k = arg min_t f'_i(\theta^2_k u_k + z_k)t + s\Theta_2 D_ii/2\tau t^2 + [box constraint]
 *
 * f'_i(\theta^2_k u_k + z_k) = \sum_{j \in rows} A_ji violation_j
 *                            = \sum_{j\in \rows} A_ji (\theta^2_k ru_j + rz_j)
 *
 * We're approximating the problem in one dimension with a derivative + fixed quadratic term.
 * Let's solve that alternate problem (find the zero and clamp) and let t_i be the step taken.
 *
 * Next step is to update z and u:
 *
 *   z^i += t^i, u^i += (s/\tau\theta_k - 1/\theta^2_k) t^i
 *
 * and update rz = Az - b and ru = Au.
 *
 * I simplified some of the arithmetic with sample_rate = \tau/s
 */
static void
coordinate_descent(struct state_t &state, const column_t &variable)
{
	const char *name(variable->name);
	const struct instance_t &instance(state.instance);
	const double theta_2(state.theta * state.theta);
	double df(instance.linear.coefs.find(name)->second);

	if (variable->vector.empty()) {
		return;
	}

	for (auto it : variable->vector) {
		const char *name(it.first);
		const double weight(it.second.first);
		const double ax_b(theta_2 * state.ru[name] + state.rz[name]);

		/* static version: assume weight = 1. */
		df += instance.row_weight.find(name)->second * weight * ax_b;
	}

	const double quad(state.var_step[name] * state.theta / state.sample_rate);
	/* zero quad + df = 0 --> zero = -df / quad */
	const double zero(-df / quad);
	{
		double z0 = zero * df + zero * zero * quad * .5;
		double za = (zero - 1e-4) * df + std::pow(zero - 1e-4, 2) * quad * .5;
		double zb = (zero + 1e-4) * df + std::pow(zero + 1e-4, 2) * quad * .5;

		assert(z0 <= za);
		assert(z0 <= zb);
	}
	double &z(state.z[name]);
	const double clamped(std::max(variable->min, std::min(variable->max, zero + z)));
	const double t(clamped - z);
	const double delta_u((1 / (state.sample_rate * state.theta) - 1 / theta_2) * t);

	z = clamped;
	state.u[name] += delta_u;
	for (auto it : variable->vector) {
		const char *name(it.first);
		double w(it.second.first);

		state.ru[name] += w * delta_u;
		state.rz[name] += w * t;
	}

	return;
}

static void
one_iteration(struct state_t &state, std::vector<column_t> &vars)
{
	const size_t n_var(vars.size());
	const size_t n(std::ceil(n_var * state.sample_rate));

#if 0
	state.theta = 2 / (state.n_iter + 2);
#endif

	for (size_t i = 0; i < n; i++) {
		size_t j(i + (1.0 * random() / RAND_MAX) * (n_var - i));

		std::swap(vars[i], vars[j]);
		coordinate_descent(state, vars[i]);
	}

	if (state.accelerated) {
		double theta = state.theta;

		state.n_iter++;
		state.theta = 0.5 * (std::sqrt(std::pow(theta, 4) + 4 * theta * theta) - theta * theta);
	}

	return;
}

const svec &
hydra(struct state_t &state, size_t n)
{
	std::vector<column_t> vars(state.instance.all_vars);

	for (size_t i = 0; i < n; i++) {
		one_iteration(state, vars);
	}

	return state.get_x();
}

/*
 * Same paper, Section 4.
 *
 * \omega_j is the number of non-zeros in row j (# vars that interact w/ j).
 * \omega^\prime_j is the number of *processors* that interact with row j.
 *
 * \tau is # random choice per iteration per processor, s is # variables/processor
 *
 * s_1 = max{1, s - 1}
 *
 * For constraint j:
 *
 * alpha_1 = 1 + (\tau - 1)(\omega_j - 1)/s_1
 *
 * alpha_2 = (\tau/s - (\tau - 1)/s_1) (\omega^\prime_j - 1)/\omega^\prime_j \omega_j
 *
 * There's obviously a lot of redundancy here.
 *
 * Finally, alpha_j = alpha_1 + alpha_2.
 *
 * D_ii = \sum_j \alpha_j \A^2_{ji}.
 *
 * the idea is that the more this variable interacts with other
 * variables (via constraints), the more conservative we want each
 * step to be.
 */
void
update_steps(struct state_t &state)
{
	const size_t n(state.instance.all_vars.size());
	const double tau(state.sample_rate * n);
	const double s1(n > 1 ? n - 1 : 1);
	const double scale_1((tau - 1) / s1);
	const double scale_2(state.sample_rate - scale_1);

	for (auto var : state.instance.all_vars) {
		double acc(0);

		for (auto entry : var->vector) {
			auto name(entry.first);
			auto row(entry.second.second);
			const double alpha_1(1 + scale_1 * (row->vector.size() - 1));
			/* XXX: assuming a single partition for now. */
			const double alpha_2(scale_2 * 0 * row->vector.size());
			const double alpha(alpha_1 + alpha_2);
			const double weight(state.instance.row_weight.find(name)->second);

			acc += alpha * weight * std::pow(entry.second.first, 2);
		}

		state.var_step[var->name] = acc;
	}

	return;
}
}
