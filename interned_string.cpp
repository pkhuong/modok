#include <map>
#include <cstring>
#include <string>

#include "interned_string.hpp"

static std::map<std::string, const char *> table;

const char *
intern_string(const char *string)
{
	std::string key(string);
	const char *&ret(table[key]);

	if (ret != NULL) {
		return ret;
	}

	ret = strdup(string);
	return ret;
}

#ifdef TEST_INTERNED_STRING
#include <cstdio>
#include <iostream>

int
main()
{
	char buf[1024];

	while (!std::cin.eof()) {
		std::cin.getline(buf, sizeof(buf), '\n');

		printf("%s -> %p (%s)\n",
		    buf, intern_string(buf), intern_string(buf));
	}

	return 0;
}
#endif
