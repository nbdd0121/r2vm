#include "util/assert.h"

namespace util::internal {

void assertion_fail(const char *message) {
    throw Assertion_error { message };
}

}