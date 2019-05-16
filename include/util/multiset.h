#ifndef UTIL_MULTISET_H
#define UTIL_MULTISET_H

#include <vector>
#include <functional>

#include "util/assert.h"

namespace util {

template<typename T>
class Multiset {
public:
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using reverse_iterator = typename std::vector<T>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;
    using size_type = size_t;
private:
    std::vector<T> _vector;

public:
    Multiset() = default;
    Multiset(const Multiset&) = default;
    Multiset(Multiset&&) = default;
    ~Multiset() = default;

    Multiset& operator =(const Multiset&) = default;
    Multiset& operator =(Multiset&&) = default;

    /* Iterators */
    iterator begin() noexcept { return _vector.begin(); }
    const_iterator begin() const noexcept { return _vector.begin(); }
    const_iterator cbegin() const noexcept { return _vector.cbegin(); }
    iterator end() noexcept { return _vector.end(); }
    const_iterator end() const noexcept { return _vector.end(); }
    const_iterator cend() const noexcept { return _vector.cend(); }
    reverse_iterator rbegin() noexcept { return _vector.rbegin(); }
    const_reverse_iterator rbegin() const noexcept { return _vector.rbegin(); }
    const_reverse_iterator crbegin() const noexcept { return _vector.crbegin(); }
    reverse_iterator rend() noexcept { return _vector.rend(); }
    const_reverse_iterator rend() const noexcept { return _vector.rend(); }
    const_reverse_iterator crend() const noexcept { return _vector.crend(); }

    /* Capacity */
    bool empty() const noexcept { return _vector.empty(); }
    size_type size() const noexcept { return _vector.size(); }
    size_type capacity() const noexcept { return _vector.capacity(); }

    /* Modifiers */
    void clear() noexcept { _vector.clear(); }
    void insert(const T& value) { _vector.push_back(value); }
    void erase(const_iterator pos) {
        auto iter = _vector.begin() + (pos - _vector.cbegin());
        ASSERT(iter != _vector.end());
        auto last_element = _vector.end() - 1;
        if (iter != last_element) {
            *iter = std::move(*last_element);
        }
        _vector.pop_back();
    }

    /* Lookups */
    iterator find(const T& value) noexcept { return std::find(_vector.begin(), _vector.end(), value); }
    const_iterator find(const T& value) const noexcept { return std::find(_vector.cbegin(), _vector.cend(), value); }

    /* Helper functions */
    void remove(const T& value) { erase(find(value)); }
    void replace(const T& value, const T& new_value) {
        auto pos = find(value);
        ASSERT(pos != _vector.end());
        *pos = new_value;
    }
};

}

#endif
