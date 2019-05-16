#ifndef UTIL_INT_SIZE_H
#define UTIL_INT_SIZE_H

#include <cstdint>

namespace util {

static inline bool is_int8(uint32_t imm) {
    return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(imm))) == imm;
}

static inline bool is_int8(uint64_t imm) {
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int8_t>(imm))) == imm;
}

static inline bool is_int16(uint64_t imm) {
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int16_t>(imm))) == imm;
}

static inline bool is_int32(uint64_t imm) {
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(imm))) == imm;
}

static inline bool is_uint8(uint64_t imm) {
    return (imm & 0xFF) == imm;
}

static inline bool is_uint16(uint64_t imm) {
    return (imm & 0xFFFF) == imm;
}

static inline bool is_uint32(uint64_t imm) {
    return (imm & 0xFFFFFFFF) == imm;
}

}

#endif
