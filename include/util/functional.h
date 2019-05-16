#ifndef UTIL_FUNCTIONAL_H
#define UTIL_FUNCTIONAL_H

namespace util {

namespace internal {

template<typename Type, Type Function>
struct as_function_pointer_helper {};

template<typename Class, typename Ret, typename... Args, Ret(Class::* Function)(Args...)>
struct as_function_pointer_helper<Ret(Class::*)(Args...), Function> {
    static Ret of(Class& obj, Args... args) {
        return (obj.*Function)(std::forward<Args>(args)...);
    }
};

}

template<typename Type, Type Function>
constexpr auto as_function_pointer() {
    return &internal::as_function_pointer_helper<Type, Function>::of;
}

#define AS_FUNCTION_POINTER(func) (util::as_function_pointer<decltype(func), func>())

}

#endif
