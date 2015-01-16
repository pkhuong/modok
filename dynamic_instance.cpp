#include <cassert>

#include "dynamic_instance.hpp"

namespace dynamic {
void
instance_t::new_var(const char *name_, double min, double max, double z_coef)
{
	auto name(intern_string(name_));
	auto &var(vars[name]);

	if (var) {
		assert(var->min == min);
		assert(var->max == max);
		assert(linear.coefs[name] == z_coef);

		return;
	}

	var = column_value::make(name, min, max);
	all_vars.push_back(var);
	if (z_coef != 0) {
		linear.coefs[name] = z_coef;
	}

	return;
}

void
instance_t::new_penalty(const char *name_, double rhs, double weight)
{
	auto name(intern_string(name_));
	auto &row(penalties[name]);

	if (row) {
		assert(row->rhs == rhs);
		assert(row->weight == weight);

		return;
	}

	row = row_value::make(name, rhs, weight);
	all_rows.push_back(row);
	return;
}

void
instance_t::set_coef(const char *var_name_, const char *penalty_name_, double coef)
{
	auto var_name(intern_string(var_name_));
	auto penalty_name(intern_string(penalty_name_));
	auto var(vars[var_name]);
	auto row(penalties[penalty_name]);

	assert(var);
	assert(row);
	var->vector[penalty_name] = std::make_pair(coef, row);
	row->vector[var_name] = std::make_pair(coef, var);
	return;
}

void
instance_t::set_linear(const char *var_name_, double x)
{
	auto var_name(intern_string(var_name_));

	assert(vars[var_name]);
	linear.coefs[var_name] = x;
	return;
}

void
instance_t::set_rhs(const char *penalty_name, double rhs)
{
	auto penalty(penalties[intern_string(penalty_name)]);

	assert(penalty);
	penalty->rhs = rhs;
	return;
}

void
instance_t::set_weight(const char *penalty_name, double weight)
{
	auto penalty(penalties[intern_string(penalty_name)]);

	assert(penalty);
	penalty->weight = weight;
	return;
}

double
instance_t::eval(const svec &x)
{
	double acc(linear.z0);

	for (auto it : x) {
		auto name(it.first);
		const double value(it.second);

		acc += linear.coefs[name] * value;
		assert(value >= vars[name]->min);
		assert(value <= vars[name]->max);
	}

	for (auto row : all_rows) {
		const double w(row->weight);
		const double rhs(row->rhs);
		double lhs(0);

		for (auto entry : row->vector) {
			auto xi(x.find(entry.first));

			if (xi != x.end()) {
				lhs += xi->second * entry.second.first;
			}
		}

		acc += .5 * w * std::pow(lhs - rhs, 2);
	}

	return acc;
}

const svec &
state_t::get_x()
{
	double theta_2 = theta * theta;
	
	x = z;
	for (auto it : u) {
		x[it.first] += theta_2 * it.second;
	}

	for (auto &it : x) {
		auto name(it.first);
		auto var(instance.vars.find(name)->second);

		it.second = std::min(var->max, std::max(var->min, it.second));
	}

	return x;
}

double
state_t::recompute_discrepancy()
{
	double acc = 0;

	for (auto row : instance.all_rows) {
		auto row_name(row->name);
		double new_ru(0);
		double new_rz(-row->rhs);

		for (auto it : row->vector) {
			auto var_name(it.first);
			double w(it.second.first);

			new_ru += u[var_name] * w;
			new_rz += z[var_name] * w;
		}

		acc += std::pow(ru[row_name] - new_ru, 2);
		acc += std::pow(rz[row_name] - new_rz, 2);

		ru[row_name] = new_ru;
		rz[row_name] = new_rz;
	}

	return acc;
}

double
state_t::projected_gradient_norm()
{
	svec delta;
	double acc = 0;
	double l_inf(0);
	double l_one(0);
	size_t n_vars(instance.all_vars.size());

	get_x();
	for (auto row : instance.all_rows) {
		auto name(row->name);
		double rhs(row->rhs);
		double lhs(0);

		for (auto it : row->vector) {
			auto var_name(it.first);

			lhs += x[var_name] * it.second.first;
		}

		delta[name] = lhs - rhs;
	}

	for (auto var : instance.all_vars) {
		auto name(var->name);
		double g(0);
		double xi(x[name]);

		for (auto entry : var->vector) {
			auto row_name(entry.first);

			g += entry.second.first * delta[row_name];
		}

		double xi_prime(std::min(var->max, std::max(var->min, xi - g)));
		double delta(xi - xi_prime);
		acc += std::pow(delta, 2);
		l_inf = std::max(l_inf, delta);
		l_one += std::abs(delta);
	}

	return l_one / n_vars + std::sqrt(acc) / n_vars + l_inf;
}

static double
round_sample_rate(const struct instance_t &instance, double sample_rate)
{
	double n(instance.vars.size());
	double m(std::ceil(n * sample_rate));

	return 1.0 * m / n;
}
	
state_t::state_t(const struct instance_t &instance_, double sample_rate_, bool accelerated_)
	: instance(instance_),
	  sample_rate(round_sample_rate(instance, sample_rate_)),
	  theta(sample_rate),
	  accelerated(accelerated_)
{
	/*
	 * 2 / (n_iter + 2) ~= sample_rate
	 * n_iter = (2 / sample_rate) - 2;
	 */

	n_iter = (2 / sample_rate) - 2;
	for (auto var : instance.all_vars) {
		auto name(var->name);

		x[name] = u[name] = z[name] = 0;
	}

	for (auto row : instance.all_rows) {
		auto name(row->name);

		ru[name] = 0;
		rz[name] = -row->rhs;
	}

	return;
}

}
