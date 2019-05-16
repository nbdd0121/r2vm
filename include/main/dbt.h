#ifndef MAIN_DBT_H
#define MAIN_DBT_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <unordered_map>

#include "emu/typedef.h"
#include "main/executor.h"
#include "util/code_buffer.h"

namespace riscv {
    struct Context;
}

namespace util {
class Code_buffer;
};

struct Dbt_block;

class Dbt_runtime final: public Executor {
private:
    // The following two fields are for hot direct-mapped instruction cache that contains recently executed code.
    std::unique_ptr<emu::reg_t[]> icache_tag_;
    std::unique_ptr<std::byte*[]> icache_;

    // The "slow" instruction cache that contains all code that are compiled previously.
    std::unordered_map<emu::reg_t, std::unique_ptr<Dbt_block>> inst_cache_;

    void compile(emu::reg_t);

public:
    Dbt_runtime();
    ~Dbt_runtime();

    void step(riscv::Context& context);
    virtual void flush_cache() override;

    friend class Dbt_compiler;
};

#endif
