#ifndef UTIL_FORMAT_H
#define UTIL_FORMAT_H

#include <iostream>

namespace util {

template<typename T, typename = void>
struct Formatter {
    static void format(std::ostream& stream, const T& value, [[maybe_unused]] char format) {
        stream << value;
    }
};

template<typename T>
struct Formatter<T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 1>> {
    static void format(std::ostream& stream, const unsigned char& value, [[maybe_unused]] char format) {
        switch (format) {
            case 'u': case 'd': case 'i': case 'o': case 'X': case 'x':
                stream << static_cast<int>(value);
                break;
            default:
                stream << value;
                break;
        }
    }
};

namespace internal {

struct Bound_formatter {
    using Format_function = void (*)(std::ostream& stream, const void *value, char format);

    Format_function formatter;
    const void *value;

    template<typename T>
    Bound_formatter(const T& reference) {
        formatter = [](std::ostream& stream, const void *value, char format) {
            Formatter<T>::format(stream, *reinterpret_cast<const T*>(value), format);
        };
        value = reinterpret_cast<const void*>(&reference);
    }
    
    void format(std::ostream& stream, char format) {
        formatter(stream, value, format);
    }
};

void format_impl(std::ostream& stream, const char *format, Bound_formatter* view, size_t size);

}

template<typename... Args>
void format(std::ostream& stream, const char *format, const Args&... args) {
    util::internal::Bound_formatter list[] = { args... };
    format_impl(stream, format, list, sizeof...(args));
}

template<typename... Args>
void print(const char *format, const Args&... args) {
    util::format(std::cout, format, args...);
}

template<typename... Args>
void error(const char *format, const Args&... args) {
    util::format(std::cerr, format, args...);
}

template<typename... Args>
void log(const char *format, const Args&... args) {
    util::format(std::clog, format, args...);
}

}

#endif
