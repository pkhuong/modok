#include <utility>

#include "dynamic_instance.hpp"

namespace dynamic {

static void
coordinate_descent(struct state_t &state, const column_t &variable)
{
	const char *name(variable->name);
	const struct instance_t &instance(state.instance);
	const double theta_2(state.theta * state.theta);
	double df(instance.linear.coefs.find(name)->second);

	for (auto it : variable->vector) {
		const char *name(it.first);
		const double ax(theta_2 * state.ru[name] + state.rz[name]);
		const double rhs(it.second.second->rhs);

		df += it.second.first * (ax - rhs);
	}

	const double quad(variable->step * state.theta / state.sample_rate);
	/* zero quad + df = 0 --> zero = -df / quad */
	const double zero(-df / quad);
	double &z(state.z[name]);
	const double clamped(std::max(variable->min, std::min(variable->max, zero + z)));
	const double t(clamped - z);
	const double delta_u((1 / (state.sample_rate * state.theta) - 1 / theta_2) * t);

	z = clamped;
	state.u[name] += delta_u;
	for (auto it : variable->vector) {
		const char *name(it.first);
		const double weight(it.second.first);

		state.ru[name] += weight * delta_u;
		state.rz[name] += weight * t;
	}

	return;
}

static void
one_iteration(struct state_t &state)
{
	std::vector<column_t> vars(state.instance.all_vars);
	const size_t n(std::ceil(vars.size() * state.sample_rate));
	const double theta(state.theta);
	
	std::random_shuffle(vars.begin(), vars.end());
	for (size_t i = 0; i < n; i++) {
		coordinate_descent(state, vars[i]);
	}

	state.theta = 0.5 * (std::sqrt(std::pow(theta, 4) + 4 * theta * theta) - theta * theta);
	return;
}

const svec &
hydra(struct state_t &state, size_t n)
{

	for (size_t i = 0; i < n; i++) {
		one_iteration(state);
	}

	return state.get_x();
}

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
			auto row(entry.second.second);
			const double alpha_1(1 + scale_1 * (row->vector.size() - 1));
			/* XXX: assuming a single partition for now. */
			const double alpha_2(scale_2 * 0 * row->vector.size());
			const double alpha(alpha_1 + alpha_2);

			acc += alpha * std::pow(entry.second.first, 2);
		}

		var->step = acc;
	}

	return;
}
}
