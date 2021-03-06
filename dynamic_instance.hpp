#ifndef DYNAMIC_INSTANCE_HPP
#define DYNAMIC_INSTANCE_HPP
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "interned_string.hpp"

namespace dynamic {

typedef std::map<const char *, double> svec;
	
struct row_value;
struct column_value;

typedef std::shared_ptr<row_value> row_t;
typedef std::shared_ptr<column_value> column_t;
	
struct row_value {
	/* 1/2 weight |vector'x - rhs|_2^2 */
	const char *name;
	double rhs;
	std::map<const char *, std::pair<double, column_t> > vector;

private:
	row_value(const char *name_, double rhs_ = 0)
		: name(name_),
		  rhs(rhs_)
	{};

public:
	static row_t
	make(const char *name, double rhs = 0)
	{
		return row_t(new row_value(intern_string(name), rhs));
	}
};

struct column_value {
	const char *name;
	double min, max;
	std::map<const char *, std::pair<double, row_t> > vector;

private:
	column_value(const char *name_,
	    double min_ = 0, double max_ = HUGE_VAL)
		: name(name_),
		  min(min_),
		  max(max_)
	{};

public:
	static column_t
	make(const char *name,
	    double min = 0, double max = HUGE_VAL)
	{
		return column_t(new column_value(name, min, max));
	}
};

struct linear_t {
	/* coefs'x + z0 */
	std::map<const char *, double> coefs;
	double z0;
};

struct instance_t {
	linear_t linear;
	std::map<const char *, column_t> vars;
	std::map<const char *, row_t> penalties;
	std::map<const char *, double> row_weight;

	std::vector<column_t> all_vars;
	std::vector<row_t> all_rows;

	void new_var(const char *name, double min = 0, double max = HUGE_VAL, double z_coef = 0);
	void new_penalty(const char *name, double rhs = 0, double weight = 1);

	void set_coef(const char *var, const char *penalty, double coef);
	void set_min(const char *var, const char *penalty, double min);
	void set_max(const char *var, const char *penalty, double max);

	void set_linear(const char *name, double coef);

	void set_rhs(const char *name, double rhs);
	void set_weight(const char *name, double weight);

	void
	set_z0(double z0)
	{
		linear.z0 = z0;
		return;
	}

	double eval(const svec &);
};

struct state_t {
	const struct instance_t &instance;
	std::map<const char *, double> var_step;
	svec x;
	svec u, z;
	svec ru, rz;
	const double sample_rate;
	double theta;
	double n_iter;
	bool accelerated;

	const svec &get_x();
	double recompute_discrepancy();
	double projected_gradient_norm();
	state_t(const struct instance_t &, double sample_rate, bool accelerated = true);
};

}
#endif /* !DYNAMIC_INSTANCE_HPP */
