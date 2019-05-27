#ifndef UTIL_SMALL_VECTOR_H
#define UTIL_SMALL_VECTOR_H

#include <algorithm>
#include <initializer_list>
#include <iterator>

#include "util/assert.h"

namespace util {

template<typename T, size_t N>
class Small_vector {
private:
    using storage = std::aligned_storage_t<sizeof(T), alignof(T)>;
public:
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reference = T&;
    using const_reference = const T&;
private:
    T* _begin;
    T* _end;
    T* _capacity;
    storage _small[N];

public:
    Small_vector() {
        unsafe_clear();
    }

    explicit Small_vector(size_type count): Small_vector{} {
        reserve(count);
        _end = std::uninitialized_default_construct_n(_begin, count);
    }

    explicit Small_vector(size_type count, const T& value): Small_vector{} {
        reserve(count);
        _end = std::uninitialized_fill_n(_begin, count, value);
    }

    Small_vector(const Small_vector& other): Small_vector{} {
        reserve(other.size());
        _end = std::uninitialized_copy(other.begin(), other.end(), _begin);
    }

    Small_vector(Small_vector&& other) noexcept {

        // This is easy, simply take over the control
        if (UNLIKELY(!other.is_small())) {
            _begin = other._begin;
            _end = other._end;
            _capacity = other._capacity;
        } else {
            unsafe_clear();
            _end = std::uninitialized_move(other.begin(), other.end(), _begin);
        }

        other.unsafe_clear();
    }

    Small_vector(std::initializer_list<T> init): Small_vector{} {
        reserve(init.size());
        _end = std::uninitialized_move(init.begin(), init.end(), _begin);
    }

    ~Small_vector() {
        std::destroy(_begin, _end);
        if (!is_small()) delete[] (storage*)_begin;
    }

    Small_vector& operator =(const Small_vector& other) = delete;

    Small_vector& operator =(Small_vector&& other) {
        if (UNLIKELY(!other.is_small())) {
            this->~Small_vector();
            _begin = other._begin;
            _end = other._end;
            _capacity = other._capacity;
            other.unsafe_clear();
        } else {
            if (size() > other.size()) {
                auto new_end = std::move(other.begin(), other.end(), _begin);
                std::destroy(new_end, _end);
                _end = new_end;
            } else {
                std::move(other.begin(), other.begin() + size(), _begin);
                _end = std::uninitialized_move(other.begin() + size(), other.end(), _end);
            }
        }
        return *this;
    }

private:
    void unsafe_clear() noexcept {
        _end = _begin = reinterpret_cast<T*>(&_small[0]);
        _capacity = reinterpret_cast<T*>(&_small[N]);
    }

    bool is_small() noexcept {
        return _begin == reinterpret_cast<T*>(&_small[0]);
    }

    void exact_grow(size_type new_cap) noexcept {
        auto new_begin = reinterpret_cast<T*>(new storage[new_cap]);
        _end = std::uninitialized_move(_begin, _end, new_begin);
        if (UNLIKELY(!is_small())) delete[] _begin;
        _begin = new_begin;
        _capacity = new_begin + new_cap;
    }

    void grow(size_type new_cap) noexcept {
        exact_grow(std::max(new_cap, capacity() * 2));
    }

public:

    /* Element access */
    reference operator[](size_type pos) { return _begin[pos]; }
    const_reference operator[](size_type pos) const { return _begin[pos]; }
    reference front() { return *_begin; }
    const_reference front() const { return *_begin; }
    reference back() { return *(_end - 1); }
    const_reference back() const { return *(_end - 1); }
    T* data() noexcept { return _begin; }
    const T* data() const noexcept { return _begin; }

    /* Iterators */
    iterator begin() noexcept { return _begin; }
    const_iterator begin() const noexcept { return _begin; }
    const_iterator cbegin() const noexcept { return _begin; }
    iterator end() noexcept { return _end; }
    const_iterator end() const noexcept { return _end; }
    const_iterator cend() const noexcept { return _end; }
    reverse_iterator rbegin() noexcept { return reverse_iterator(_end); }
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(_end); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(_end); }
    reverse_iterator rend() noexcept { return reverse_iterator(_begin); }
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(_end); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(_end); }

    /* Capacity */
    bool empty() const noexcept { return _begin == _end; }
    size_type size() const noexcept { return _end - _begin; }
    size_type max_size() const { return static_cast<size_type>(-1) / sizeof(T); }

    // Add new_cap > N here to hint the compiler.
    void reserve(size_t new_cap) { if (new_cap > N && new_cap > capacity()) exact_grow(new_cap); }
    size_type capacity() const noexcept { return _capacity - _begin; }

    /* Modifiers */
    void clear() noexcept {
        std::destroy(_begin, _end);
        _end = _begin;
    }

    iterator insert(const_iterator pos, const T& value) = delete;
    iterator insert(const_iterator pos, T&& value) = delete;
    iterator insert(const_iterator pos, size_type count, const T& value) = delete;
    template<typename InputIt>
    iterator insert(const_iterator cpos, InputIt first, InputIt last) {
        auto pos = const_cast<iterator>(cpos);
        size_type count = last - first;
        if (pos == _end) {
            reserve(size() + count);
            _end = std::uninitialized_copy(first, last, _end);
            return _end;
        }

        ASSERT(0);
    }

    iterator insert(const_iterator pos, std::initializer_list<T> ilist) = delete;

    iterator erase(const_iterator pos) {
        auto iter = const_cast<iterator>(pos);
        std::move(iter + 1, end(), iter);
        pop_back();
        return iter;
    }

    iterator erase(const_iterator cfirst, const_iterator clast) {
        auto first = const_cast<iterator>(cfirst);
        auto last = const_cast<iterator>(clast);
        iterator new_end = std::move(last, _end, first);
        std::destroy(new_end, _end);
        _end = new_end;
        return first;
    }

    void push_back(const T& value) {
        if (UNLIKELY(_end == _capacity)) grow(size() + 1);
        new (_end) T(value);
        _end++;
    }

    void push_back(T&& value) {
        if (UNLIKELY(_end == _capacity)) grow(size() + 1);
        new (_end) T(std::move(value));
        _end++;
    }

    void pop_back() {
        std::destroy_at(_end - 1);
        _end--;
    }

    void resize(size_type count) {
        if (UNLIKELY(count == size())) return;
        if (count < size()) {
            std::destroy(_begin + count, _end);
        } else {
            reserve(count);
            std::uninitialized_default_construct(_end, _begin + count);
        }
        _end = _begin + count;
    }

    void resize(size_type count, const value_type& value) {
        if (UNLIKELY(count == size())) return;
        if (count < size()) {
            std::destroy(_begin + count, _end);
        } else {
            reserve(count);
            std::uninitialized_fill(_end, _begin + count, value);
        }
        _end = _begin + count;
    }
};

}

#endif
