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
	for (auto it : z) {
		x[it.first] += theta_2 * it.second;
	}

	return x;
}

state_t::state_t(const struct instance_t &instance_, double sample_rate_, double theta_)
	: instance(instance_),
	  theta(std::isnan(theta_) ? sample_rate_ : theta_),
	  sample_rate(sample_rate_)
{}

}
