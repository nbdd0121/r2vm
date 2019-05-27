#ifndef RISCV_FRONTEND_H
#define RISCV_FRONTEND_H

#include "ir/node.h"

namespace riscv {

struct Basic_block;

ir::Graph compile(const Basic_block& block);

} // riscv

#endif
