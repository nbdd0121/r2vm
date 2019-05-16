#ifndef UTIL_BITFIELD_H
#define UTIL_BITFIELD_H

#include <type_traits>

namespace util {

// It a quite common operation to view an integer as bits and extract from or pack fields into the integer. Doing these
// operation by shift and bit operation each time is tedious, inelegant, and prone to errors. Bitfield class provide a
// easy way to access them.
// Arguments are specified in pairs, Hi followed by Lo. If Hi set to -1, then Lo is the number of zeroes to pad.
// E.g. Bitfield<int, 23, 16, 7, 0> will extract 0x3478 from 0x12345678.
// E.g. Bitfield<int, 23, 16, -1, 8, 7, 0> will extract 0x340078 from 0x12345678.
template<typename Type, int... Range>
class Bitfield {

    // This unspecialized class catches the case that sizeof...(Range) is 0 or odd, but odd number does not make sense.
    static_assert(sizeof...(Range) == 0, "Bitfield must have an even number of integer arguments.");

public:
    static constexpr int width = 0;
    static constexpr Type extract(Type) noexcept { return 0; }
    static constexpr Type pack(Type bits, Type) noexcept { return bits; }
};

template<typename Type, int Hi, int Lo, int... Range>
class Bitfield<Type, Hi, Lo, Range...> {
    static_assert(Hi >= 0 && Lo >= 0, "Hi and Lo must be non-negative.");
    static_assert(Hi >= Lo, "Hi must be >= Lo.");

    using Effective_type = std::make_unsigned_t<Type>;

    // This class will handle only one segment, and then it passes the task to a smaller bitfield.
    using Chain = Bitfield<Effective_type, Range...>;

    // Mask containing ones on all bits within range [Lo, Hi]
    static constexpr Effective_type mask = ((static_cast<Effective_type>(1) << (Hi - Lo + 1)) - 1) << Lo;

public:
    static constexpr int width = Chain::width + (Hi - Lo + 1);

    static constexpr Type extract(Type bits) noexcept {

        // We have right shifts here, so make sure we are only operating on unsigned values.
        Effective_type bits_unsigned = bits;
        Effective_type value = Chain::extract(bits_unsigned) | ((bits_unsigned & mask) >> Lo << Chain::width);

        if constexpr(std::is_signed<Type>::value) {
            return static_cast<Type>(value) << (sizeof(Type) * 8 - width) >> (sizeof(Type) * 8 - width);
        } else {
            return value;
        }
    }

    static constexpr Type pack(Type bits, Type value) noexcept {

        // No need to cast to unsigned here since the righted result is masked.
        return Chain::pack((bits & ~mask) | ((value >> Chain::width << Lo) & mask), value);
    }
};

// Special case for padding 0s. This essentially only increase width, and otherwise is a no-op.
template<typename Type, int Lo, int... Range>
class Bitfield<Type, -1, Lo, Range...> {
    static_assert(Lo > 0, "Lo must be positive.");

    // Since Bitfield<Signed> will be converted to Bitfield<Unsigned> after first segment, if we see signed, then the
    // user must be putting padding as the first segment.
    static_assert(std::is_unsigned<Type>::value, "Padding 0 at the start of a signed integer is meaningless");

    using Chain = Bitfield<Type, Range...>;

public:
    static constexpr int width = Chain::width + Lo;
    static constexpr Type extract(Type bits) noexcept { return Chain::extract(bits); }
    static constexpr Type pack(Type bits, Type value) noexcept { return Chain::pack(bits, value); }
};

} // util

#endif
