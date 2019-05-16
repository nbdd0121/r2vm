#ifndef UTIL_MACRO_H
#define UTIL_MACRO_H

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)

#define CONCAT_IMPL(x,y) x##y
#define CONCAT(x,y) CONCAT_IMPL(x,y)

#endif
