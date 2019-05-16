#include "emu/state.h"

namespace emu::state {

std::string exec_path;

std::string sysroot = "/opt/riscv/sysroot";

reg_t original_brk;
reg_t brk;
reg_t heap_start;
reg_t heap_end;

bool disassemble = false;

bool no_instret = true;

int inline_limit = 15;

int compile_threshold = 0;

bool strace = false;

bool strict_exception = false;

bool enable_phi = false;

bool monitor_performance = false;

bool no_direct_memory_access = false;

}
