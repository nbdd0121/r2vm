#include <chrono>
#include <cstring>

#include "emu/state.h"
#include "emu/unwind.h"
#include "ir/analysis.h"
#include "ir/pass.h"
#include "main/ir_dbt.h"
#include "main/signal.h"
#include "riscv/basic_block.h"
#include "riscv/context.h"
#include "riscv/decoder.h"
#include "riscv/disassembler.h"
#include "riscv/frontend.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/assert.h"
#include "util/format.h"
#include "util/memory.h"
#include "x86/backend.h"

// Declare the exception handling registration functions.
extern "C" void __register_frame(void*);
extern "C" void __deregister_frame(void*);

// Denotes a translated block.
struct Ir_block {

    // Translated code.
    util::Code_buffer code;

    // Exception handling frame
    std::unique_ptr<uint8_t[]> cie;

    // Number of times the block is hit. If the number reaches compile_threshold, IR DBT will start to work.
    int num_hit = 0;

    ~Ir_block() {
        if (cie) {
            __deregister_frame(cie.get());
        }
    }
};

_Unwind_Reason_Code ir_dbt_personality(
    [[maybe_unused]] int version,
    [[maybe_unused]] _Unwind_Action actions,
    [[maybe_unused]] uint64_t exception_class,
    [[maybe_unused]] struct _Unwind_Exception *exception_object,
    [[maybe_unused]] struct _Unwind_Context *context
) {
    return _URC_CONTINUE_UNWIND;
}

static void generate_eh_frame(Ir_block& block, int stack_size) {
    // TODO: Create an dwarf generation to replace this hard-coded template.
    static const unsigned char cie_template[] = {
        // CIE
        // Length
        0x1C, 0x00, 0x00, 0x00,
        // CIE
        0x00, 0x00, 0x00, 0x00,
        // Version
        0x01,
        // Augmentation string
        'z', 'P', 'L', 0,
        // Instruction alignment factor = 1
        0x01,
        // Data alignment factor = -8
        0x78,
        // Return register number
        0x10,
        // Augmentation data
        0x0A, // Data for z
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // abs format, personality routine
        0x00, // abs format for LSDA
        // Instructions
        // def_cfa(rsp, 8)
        0x0c, 0x07, 0x08,
        // offset(rsp, cfa-8)
        0x90, 0x01,
        // Padding

        // FDE
        // Length
        0x28, 0x00, 0x00, 0x00,
        // CIE Pointer
        0x24, 0x00, 0x00, 0x00,
        // Initial location
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // Augumentation data
        0x8,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // LSDA
        // advance_loc(1)
        0x41,
        // def_cfa_offset(16)
        0x0E, 0x10,
        // offset(rbp, cfa-16)
        0x86, 0x02,
        // def_cfa_offset(stack_size+16)
        0x0E, 0x00, 0x00,
        // Padding
        0x00, 0x00, 0x00,

        0x00, 0x00, 0x00, 0x00
    };

    block.cie = std::make_unique<uint8_t[]>(sizeof(cie_template));
    uint8_t *cie = block.cie.get();

    memcpy(cie, cie_template, sizeof(cie_template));
    util::write_as<uint64_t>(cie + 0x12, reinterpret_cast<uint64_t>(ir_dbt_personality));
    util::write_as<uint64_t>(cie + 0x28, reinterpret_cast<uint64_t>(block.code.data()));
    util::write_as<uint64_t>(cie + 0x30, 4096);
    util::write_as<uint64_t>(cie + 0x39, 0);

    // We only have logic for two bytes in LEB127
    stack_size += 16;
    ASSERT(stack_size >= 0 && stack_size <= 0x3FFF);
    if (stack_size <= 127) {
        cie[0x47] = stack_size;
    } else {
        cie[0x47] = (stack_size & 127) | 0x80;
        cie[0x48] = stack_size >> 7;
    }

    __register_frame(cie);
}

Ir_dbt::Ir_dbt() noexcept {
    icache_tag_ = std::make_unique<emu::reg_t[]>(4096);
    icache_ = std::make_unique<std::byte*[]>(4096);
    for (size_t i = 0; i < 4096; i++) {
        icache_tag_[i] = 0;
    }
}

Ir_dbt::~Ir_dbt() {
    if (emu::state::monitor_performance) {
        int64_t average_in_ns = (total_compilation_time + (total_block_compiled / 2)) / total_block_compiled;
        int64_t average_in_us = (average_in_ns + 500) / 1000;
        int64_t sum_in_us = (total_compilation_time + 500) / 1000;
        util::log(
            "{} blocks are compiled in {} microseconds. Time per block is {} microseconds.\n",
            total_block_compiled, sum_in_us, average_in_us
        );
    }
}

void Ir_dbt::step(riscv::Context& context) {
    const emu::reg_t pc = context.pc;
    const ptrdiff_t tag = (pc >> 1) & 4095;

    // If the cache misses, compile the current block.
    if (UNLIKELY(icache_tag_[tag] != pc)) {
        compile(context, pc);
        return;
    }

    // The return value is the address to patch.
    auto func = reinterpret_cast<Compiled_function>(icache_[tag]);
    ASSERT(func);
    if (UNLIKELY(_code_ptr_to_patch)) patch_trampoline(func);
    _code_ptr_to_patch = func(context);
}

void Ir_dbt::patch_trampoline(Compiled_function func) {
    // Patch the trampoline.
    // mov rax, i64 => 48 B8 i64
    // jmp rax => FF E0
    // 4 here indicates the length of the prologue.
    util::write_as<uint16_t>(_code_ptr_to_patch, 0xB848);
    util::write_as<uint64_t>(_code_ptr_to_patch + 2, reinterpret_cast<uint64_t>(func) + 4);
    util::write_as<uint16_t>(_code_ptr_to_patch + 10, 0xE0FF);
}

ir::Graph Ir_dbt::decode(emu::reg_t pc) {
    riscv::Decoder decoder {pc};
    riscv::Basic_block basic_block = decoder.decode_basic_block();
    ir::Graph graph = riscv::compile(basic_block);

    // Load/store elimination and LVN are required to allow inlining of auipc/jalr fused pair.
    ir::analysis::Block block_analysis{graph};
    ir::analysis::Local_load_store_elimination{graph, block_analysis, 66}.run();
    ir::pass::Local_value_numbering{graph}.run();

    return graph;
}

void Ir_dbt::compile(riscv::Context& context, emu::reg_t pc) {
    const ptrdiff_t tag = (pc >> 1) & 4095;

    // Check the flush flag here, if it is true then we need to flush cache entries.
    if (UNLIKELY(_need_cache_flush)) {
        inst_cache_.clear();
        _need_cache_flush = false;
        _code_ptr_to_patch = nullptr;
    }

    auto& block_ptr = inst_cache_[pc];
    if (UNLIKELY(!block_ptr) || block_ptr->code.empty()) {
        if (!block_ptr) block_ptr = std::make_unique<Ir_block>();

        if (block_ptr->num_hit < emu::state::compile_threshold) {
            _code_ptr_to_patch = nullptr;
            block_ptr->num_hit++;
            riscv::Decoder decoder {pc};
            riscv::Instruction inst;
            do {
                inst = decoder.decode_instruction();
                context.pc += inst.length();
                context.instret++;
                try {
                    riscv::step(&context, inst);
                } catch(...) {
                    // In case an exception happens, we need to move the pc before the instruction.
                    context.pc -= inst.length();
                    context.instret--;
                    throw;
                }
            } while (!decoder.can_change_control_flow(inst));
            return;
        }

        auto start = emu::state::monitor_performance ?
            std::chrono::high_resolution_clock::now().time_since_epoch().count() : 0;

        ir::Graph graph = decode(pc);
        block_ptr->code.reserve(4096);

        // A map between emulated pc and entry point in the graph.
        std::unordered_map<emu::reg_t, ir::Node*> block_map;
        block_map[pc] = *graph.entry()->value(0).references().begin();

        int counter = 0;
        size_t operand_count = graph.exit()->operand_count();

        for (size_t i = 0; i < operand_count; i++) {
            auto operand = graph.exit()->operand(i);
            ir::Value target_pc_value = ir::analysis::Block::get_tail_jmp_pc(operand, 64);

            // We can inline tail jump.
            if (target_pc_value && target_pc_value.is_const()) {
                auto target_pc = target_pc_value.const_value();
                if (!target_pc) continue;

                auto block = block_map[target_pc];

                if (block) {

                    // Add a new edge to the block, and remove the old edge to exit node.
                    graph.exit()->operand_delete(operand);
                    block->operand_add(operand);

                    // Update constraints
                    i--;
                    operand_count--;

                } else if (counter < emu::state::inline_limit) {

                    // To avoid spending too much time inlining all possible branches, we set an upper limit.

                    // Decode and clone the graph of the block to be inlined.
                    ir::Graph graph_to_inline = decode(target_pc);

                    // Store the entry point of the inlined graph.
                    block_map[target_pc] = *graph_to_inline.entry()->value(0).references().begin();

                    if (emu::state::disassemble) {
                        util::log("inline {:x} to {:x}\n", target_pc, pc);
                    }

                    // Inline the graph. Note that the iterator is invalidated so we need to break.
                    graph.inline_graph(operand, std::move(graph_to_inline));

                    // Update constraints
                    i--;
                    operand_count = graph.exit()->operand_count();
                    counter++;
                }
            }
        }

        // Insert keepalive edges and merge blocks without interesting control flow.
        ir::analysis::Block block_analysis{graph};
        block_analysis.update_keepalive();
        block_analysis.simplify_graph();

        if (emu::state::disassemble) {
            util::log("IR for {:x}\n", pc);
            x86::backend::Dot_printer{}.run(graph);
        }

        {
            // We are making this regional, as simplify graph will break the dominance tree, so we need to reconstruct.
            // TODO: Maybe find a way to incrementally update the tree when the control is simplified?
            ir::analysis::Dominance dom(graph, block_analysis);
            ir::analysis::Load_store_elimination elim{graph, block_analysis, dom, 66};
            elim.eliminate_load();
            elim.eliminate_store();
            block_analysis.simplify_graph();
        }

        ir::pass::Local_value_numbering{graph}.run();

        // Dump IR if --disassemble is used.
        if (emu::state::disassemble) {
            util::log("IR for {:x}-opt\n", pc);
            x86::backend::Dot_printer{}.run(graph);
            util::log("Translating {:x} to {:x}\n", pc, reinterpret_cast<uintptr_t>(block_ptr->code.data()));
        }

        // Lowering and target-specific lowering. Currently lowering is only needed if no_direct_memory_access is on.
        if (emu::state::no_direct_memory_access) {
            ir::pass::Lowering{}.run(graph);
            ir::pass::Local_value_numbering{graph}.run();
        }
        x86::backend::Lowering{graph}.run();

        // This garbage collection is required for Value::references to correctly reflect number of users.
        graph.garbage_collect();

        ir::analysis::Dominance dom{graph, block_analysis};

        // Reorder basic blocks before feeding it to the backend.
        block_analysis.reorder(dom);

        ir::analysis::Scheduler scheduler{graph, block_analysis, dom};
        scheduler.schedule();
        x86::backend::Register_allocator regalloc{graph, block_analysis, scheduler};
        regalloc.allocate();
        x86::backend::Code_generator{block_ptr->code, graph, block_analysis, scheduler, regalloc}.run();
        generate_eh_frame(*block_ptr, regalloc.get_stack_size());

        if (emu::state::monitor_performance) {
            auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            total_compilation_time += end - start;
            total_block_compiled++;
        }
    }

    // Update tag to reflect newly compiled code.
    icache_[tag] = block_ptr->code.data();
    icache_tag_[tag] = pc;

    // Run the newly compiled (or loaded from cache) code.
    auto func = reinterpret_cast<Compiled_function>(icache_[tag]);
    ASSERT(func);
    if (_code_ptr_to_patch) patch_trampoline(func);
    _code_ptr_to_patch = func(context);
}

void Ir_dbt::flush_cache() {
    for (int i = 0; i < 4096; i++)
        icache_tag_[i] = 0;

    // As all cache tags are cleared, next time method compile will be called. We can check the flag there.
    _need_cache_flush = true;
}
