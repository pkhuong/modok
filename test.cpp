#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include "dynamic_hydra.hpp"
#include "dynamic_instance.hpp"
#include "interned_string.hpp"

using namespace dynamic;

typedef std::vector<const char *> range_t;

static range_t
make_range(const char *prefix, size_t n)
{
	char buf[1024];
	range_t ret;

	for (size_t i = 0; i < n; i++) {
		sprintf(buf, "%s-%zu", prefix, i);
		ret.push_back(intern_string(buf));
	}

	return ret;
}

static svec
random_vector(const range_t &range, double min, double max)
{
	svec ret;
	double delta(max - min);

	for (auto name : range) {
		ret[name] = min + delta * random() / RAND_MAX;
	}
	
	return ret;
}

static double
dot(const svec &x, std::map<const char *, std::pair<double, column_t> > &vector)
{
	double acc(0);

	for (auto it : x) {
		auto name(it.first);
		auto xi(it.second);
		auto entry(vector.find(name));

		if (entry != vector.end()) {
			acc += xi * entry->second.first;
		}
	}

	return acc;
}

static instance_t
random_instance(const range_t &rows, const range_t &columns, const svec &expected, size_t n_col)
{
	instance_t ret;

	for (auto row : rows) {
		ret.new_penalty(row);
	}

	for (auto column : columns) {
		ret.new_var(column, -HUGE_VAL, HUGE_VAL);
	}

	for (auto row : rows) {
		range_t cols(columns);

		std::random_shuffle(cols.begin(), cols.end());
		for (size_t i = 0; i < n_col; i++) {
			ret.set_coef(cols[i], row, 2.0 * random() / RAND_MAX - 1);
		}
	}

	for (auto row : rows) {
		double rhs(dot(expected, ret.penalties[row]->vector));

		ret.set_rhs(row, rhs);
	}

	return ret;
}

int
main()
{
	const range_t rows(make_range("row", 1000));
	const range_t cols(make_range("col", 1000));
	const svec expected(random_vector(cols, -1, 1));
	const double scale(1.0 / 100);

	instance_t instance(random_instance(rows, cols, expected, 10));
	state_t state(instance, scale, true);

	update_steps(state);
	std::cout << "It " << 0 << "\t" << instance.eval(state.get_x()) << std::endl;
	for (size_t i = 1; i <= 20; i++) {
		auto x(hydra(state, std::ceil(100 / scale)));

		std::cout << "It " << i << "\t"
			  << instance.eval(x) << "\t"
			  << instance.eval(state.z) << std::endl;
	}

	return 0;
}
