#include <vector>
#include <set>
#include <unordered_set>

#include "rabbitizer.hpp"
#include "fmt/format.h"
#include "fmt/ostream.h"

#include "recomp_port.h"

using InstrId = rabbitizer::InstrId::UniqueId;
using Cop0Reg = rabbitizer::Registers::Cpu::Cop0;

std::string_view ctx_gpr_prefix(int reg) {
    if (reg != 0) {
        return "ctx->r";
    }
    return "";
}

class InstructionWriter {
    const RecompPort::Context& context;
    const RecompPort::Config& config;
    const RecompPort::Function& func;
    const RecompPort::FunctionStats& stats;
    const std::unordered_set<uint32_t>& skipped_insns;
    size_t instr_index;
    const std::vector<rabbitizer::InstructionCpu>& instructions;
    std::ofstream& output_file;
    bool indent;
    bool emit_link_branch;
    int link_branch_index;
    size_t reloc_index;
    bool& needs_link_branch;
    bool& is_branch_likely;
    std::span<std::vector<uint32_t>> static_funcs_out;
    uint32_t func_vram_end;

    const RecompPort::Section& section = context.sections[func.section_index];
    const rabbitizer::InstructionCpu& instr = instructions[instr_index];

public:
    InstructionWriter(
        const RecompPort::Context& context,
        const RecompPort::Config& config,
        const RecompPort::Function& func,
        const RecompPort::FunctionStats& stats,
        const std::unordered_set<uint32_t>& skipped_insns,
        size_t instr_index,
        const std::vector<rabbitizer::InstructionCpu>& instructions,
        std::ofstream& output_file,
        bool indent,
        bool emit_link_branch,
        int link_branch_index,
        size_t reloc_index,
        bool& needs_link_branch,
        bool& is_branch_likely,
        std::span<std::vector<uint32_t>> static_funcs_out,
        uint32_t func_vram_end
    ) :
        context(context),
        config(config),
        func(func),
        stats(stats),
        skipped_insns(skipped_insns),
        instr_index(instr_index),
        instructions(instructions),
        output_file(output_file),
        indent(indent),
        emit_link_branch(emit_link_branch),
        link_branch_index(link_branch_index),
        reloc_index(reloc_index),
        needs_link_branch(needs_link_branch),
        is_branch_likely(is_branch_likely),
        static_funcs_out(static_funcs_out),
        func_vram_end(func_vram_end),
        section(context.sections[func.section_index]),
        instr(instructions[instr_index])
         {}

    void print_indent() {
        fmt::print(output_file, "    ");
    };

    template <typename... Ts>
    void print_line(fmt::format_string<Ts...> fmt_str, Ts ...args) {
        writer.print_indent();
        fmt::vprint(output_file, fmt_str, fmt::make_format_args(args...));
        fmt::print(output_file, ";\n");
    };

    template <typename... Ts>
    void print_branch_condition(fmt::format_string<Ts...> fmt_str, Ts ...args) {
        fmt::vprint(output_file, fmt_str, fmt::make_format_args(args...));
        fmt::print(output_file, " ");
    };

    template <typename... Ts>
    void print_unconditional_branch(fmt::format_string<Ts...> fmt_str, Ts ...args) {
        if (instr_index < instructions.size() - 1) {
            bool dummy_needs_link_branch;
            bool dummy_is_branch_likely;
            size_t next_reloc_index = reloc_index;
            uint32_t next_vram = instr_vram + 4;
            if (reloc_index + 1 < section.relocs.size() && next_vram > section.relocs[reloc_index].address) {
                next_reloc_index++;
            }
            process_instruction(context, config, func, stats, skipped_insns, instr_index + 1, instructions, output_file, false, false, link_branch_index, next_reloc_index, dummy_needs_link_branch, dummy_is_branch_likely, static_funcs_out);
        }
        writer.print_indent();
        fmt::vprint(output_file, fmt_str, fmt::make_format_args(args...));
        if (needs_link_branch) {
            fmt::print(output_file, ";\n    goto after_{};\n", link_branch_index);
        } else {
            fmt::print(output_file, ";\n");
        }
    };

    bool print_func_call(uint32_t target_func_vram, bool link_branch = true) {
        const auto matching_funcs_find = context.functions_by_vram.find(target_func_vram);
        std::string jal_target_name;
        uint32_t section_vram_start = section.ram_addr;
        uint32_t section_vram_end = section.ram_addr + section.size;
        // TODO the current section should be prioritized if the target jal is in its vram even if a function isn't known (i.e. static)
        if (matching_funcs_find != context.functions_by_vram.end()) {
            // If we found matches for the target function by vram, 
            const auto& matching_funcs_vec = matching_funcs_find->second;
            size_t real_func_index;
            bool ambiguous;
            // If there is more than one corresponding function, look for any that have a nonzero size.
            if (matching_funcs_vec.size() > 1) {
                size_t nonzero_func_index = (size_t)-1;
                bool found_nonzero_func = false;
                for (size_t cur_func_index : matching_funcs_vec) {
                    const auto& cur_func = context.functions[cur_func_index];
                    if (cur_func.words.size() != 0) {
                        if (found_nonzero_func) {
                            ambiguous = true;
                            break;
                        }
                        // If this section is relocatable and the target vram is in the section, don't call functions
                        // in any section other than this one.
                        if (cur_func.section_index == func.section_index ||
                            !(section.relocatable && target_func_vram >= section_vram_start && target_func_vram < section_vram_end)) {
                            found_nonzero_func = true;
                            nonzero_func_index = cur_func_index;
                        }
                    }
                }
                if (nonzero_func_index == (size_t)-1) {
                    fmt::print(stderr, "[Warn] Potential jal resolution ambiguity\n");
                    for (size_t cur_func_index : matching_funcs_vec) {
                        fmt::print(stderr, "  {}\n", context.functions[cur_func_index].name);
                    }
                    nonzero_func_index = 0;
                }
                real_func_index = nonzero_func_index;
                ambiguous = false;
            }
            else {
                real_func_index = matching_funcs_vec.front();
                ambiguous = false;
            }
            if (ambiguous) {
                fmt::print(stderr, "Ambiguous jal target: 0x{:08X}\n", target_func_vram);
                for (size_t cur_func_index : matching_funcs_vec) {
                    const auto& cur_func = context.functions[cur_func_index];
                    fmt::print(stderr, "  {}\n", cur_func.name);
                }
                return false;
            }
            jal_target_name = context.functions[real_func_index].name;
        }
        else {
            const auto& section = context.sections[func.section_index];
            if (target_func_vram >= section.ram_addr && target_func_vram < section.ram_addr + section.size) {
                jal_target_name = fmt::format("static_{}_{:08X}", func.section_index, target_func_vram);
                static_funcs_out[func.section_index].push_back(target_func_vram);
            }
            else {
                fmt::print(stderr, "No function found for jal target: 0x{:08X}\n", target_func_vram);
                return false;
            }
        }
        needs_link_branch = link_branch;
        print_unconditional_branch("{}(rdram, ctx)", jal_target_name);
        return true;
    };

    void print_branch(uint32_t branch_target) {
        if (branch_target < func.vram || branch_target >= func_vram_end) {
            // FIXME: how to deal with static functions?
            if (context.functions_by_vram.find(branch_target) != context.functions_by_vram.end()) {
                fmt::print(output_file, "{{\n    ");
                fmt::print("Tail call in {} to 0x{:08X}\n", func.name, branch_target);
                print_func_call(branch_target, false);
                print_line("return");
                fmt::print(output_file, ";\n    }}\n");
                return;
            }

            fmt::print(stderr, "[Warn] Function {} is branching outside of the function (to 0x{:08X})\n", func.name, branch_target);
        }

        fmt::print(output_file, "{{\n    ");
        if (instr_index < instructions.size() - 1) {
            bool dummy_needs_link_branch;
            bool dummy_is_branch_likely;
            size_t next_reloc_index = reloc_index;
            uint32_t next_vram = instr.getVram() + 4;
            if (reloc_index + 1 < section.relocs.size() && next_vram > section.relocs[reloc_index].address) {
                next_reloc_index++;
            }
            process_instruction(context, config, func, stats, skipped_insns, instr_index + 1, instructions, output_file, true, false, link_branch_index, next_reloc_index, dummy_needs_link_branch, dummy_is_branch_likely, static_funcs_out);
        }

        fmt::print(output_file, "        ");
        fmt::print(output_file, "goto L_{:08X}", branch_target);
        if (needs_link_branch) {
            fmt::print(output_file, ";\n        goto after_{}", link_branch_index);
        }
        fmt::print(output_file, ";\n    }}\n");
    };

};

// Major TODO, this function grew very organically and needs to be cleaned up. Ideally, it'll get split up into some sort of lookup table grouped by similar instruction types.
bool process_instruction(const RecompPort::Context& context, const RecompPort::Config& config, const RecompPort::Function& func, const RecompPort::FunctionStats& stats, const std::unordered_set<uint32_t>& skipped_insns, size_t instr_index, const std::vector<rabbitizer::InstructionCpu>& instructions, std::ofstream& output_file, bool indent, bool emit_link_branch, int link_branch_index, size_t reloc_index, bool& needs_link_branch, bool& is_branch_likely, std::span<std::vector<uint32_t>> static_funcs_out) {
    const auto& section = context.sections[func.section_index];
    const auto& instr = instructions[instr_index];
    needs_link_branch = false;
    is_branch_likely = false;

    // Output a comment with the original instruction
    if (instr.isBranch() || instr.getUniqueId() == InstrId::cpu_j) {
        fmt::print(output_file, "    // {}\n", instr.disassemble(0, fmt::format("L_{:08X}", (uint32_t)instr.getBranchVramGeneric())));
    } else if (instr.getUniqueId() == InstrId::cpu_jal) {
        fmt::print(output_file, "    // {}\n", instr.disassemble(0, fmt::format("0x{:08X}", (uint32_t)instr.getBranchVramGeneric())));
    } else {
        fmt::print(output_file, "    // {}\n", instr.disassemble(0));
    }

    uint32_t instr_vram = instr.getVram();

    if (skipped_insns.contains(instr_vram)) {
        return true;
    }


    bool at_reloc = false;
    bool reloc_handled = false;
    RecompPort::RelocType reloc_type = RecompPort::RelocType::R_MIPS_NONE;
    uint32_t reloc_section = 0;
    uint32_t reloc_target_section_offset = 0;

    uint32_t func_vram_end = func.vram + func.words.size() * sizeof(func.words[0]);
    InstructionWriter writer {
        context,
        config,
        func,
        stats,
        skipped_insns,
        instr_index,
        instructions,
        output_file,
        indent,
        emit_link_branch,
        link_branch_index,
        reloc_index,
        needs_link_branch,
        is_branch_likely,
        static_funcs_out,
        func_vram_end
    };

    // Check if this instruction has a reloc.
    if (section.relocatable && section.relocs.size() > 0 && section.relocs[reloc_index].address == instr_vram) {
        // Get the reloc data for this instruction
        const auto& reloc = section.relocs[reloc_index];
        reloc_section = reloc.target_section;
        // Some symbols are in a nonexistent section (e.g. absolute symbols), so check that the section is valid before doing anything else.
        // Absolute symbols will never need to be relocated so it's safe to skip this.
        if (reloc_section < context.sections.size()) {
            // Ignore this reloc if it points to a different section.
            // Also check if the reloc points to the bss section since that will also be relocated with the section.
            if (reloc_section == func.section_index || reloc_section == section.bss_section_index) {
                // Record the reloc's data.
                reloc_type = reloc.type;
                reloc_target_section_offset = reloc.target_address - section.ram_addr;
                // Ignore all relocs that aren't HI16 or LO16.
                if (reloc_type == RecompPort::RelocType::R_MIPS_HI16 || reloc_type == RecompPort::RelocType::R_MIPS_LO16) {
                    at_reloc = true;
                }
            }
        }
    }

    if (indent) {
        writer.print_indent();
    }

    int rd = (int)instr.GetO32_rd();
    int rs = (int)instr.GetO32_rs();
    int base = rs;
    int rt = (int)instr.GetO32_rt();
    int sa = (int)instr.Get_sa();

    int fd = (int)instr.GetO32_fd();
    int fs = (int)instr.GetO32_fs();
    int ft = (int)instr.GetO32_ft();

    int cop1_cs = (int)instr.Get_cop1cs();

    uint16_t imm = instr.Get_immediate();

    std::string unsigned_imm_string;
    std::string signed_imm_string;

    if (!at_reloc) {
        unsigned_imm_string = fmt::format("{:#X}", imm);
        signed_imm_string = fmt::format("{:#X}", (int16_t)imm);
    } else {
        switch (reloc_type) {
            case RecompPort::RelocType::R_MIPS_HI16:
                unsigned_imm_string = fmt::format("RELOC_HI16({}, {:#X})", (uint32_t)func.section_index, reloc_target_section_offset);
                signed_imm_string = "(int16_t)" + unsigned_imm_string;
                reloc_handled = true;
                break;
            case RecompPort::RelocType::R_MIPS_LO16:
                unsigned_imm_string = fmt::format("RELOC_LO16({}, {:#X})", (uint32_t)func.section_index, reloc_target_section_offset);
                signed_imm_string = "(int16_t)" + unsigned_imm_string;
                reloc_handled = true;
                break;
            default:
                throw std::runtime_error(fmt::format("Unexpected reloc type {} in {}\n", static_cast<int>(reloc_type), func.name));
        }
    }

    switch (instr.getUniqueId()) {
    case InstrId::cpu_nop:
        fmt::print(output_file, "\n");
        break;
    // Cop0 (Limited functionality)
    case InstrId::cpu_mfc0:
        {
            Cop0Reg reg = instr.Get_cop0d();
            switch (reg) {
            case Cop0Reg::COP0_Status:
                writer.print_line("{}{} = cop0_status_read(ctx)", ctx_gpr_prefix(rt), rt);
                break;
            default:
                fmt::print(stderr, "Unhandled cop0 register in mfc0: {}\n", (int)reg);
                return false;
            }
            break;
        }
    case InstrId::cpu_mtc0:
        {
            Cop0Reg reg = instr.Get_cop0d();
            switch (reg) {
            case Cop0Reg::COP0_Status:
                writer.print_line("cop0_status_write(ctx, {}{})", ctx_gpr_prefix(rt), rt);
                break;
            default:
                fmt::print(stderr, "Unhandled cop0 register in mtc0: {}\n", (int)reg);
                return false;
            }
            break;
        }
    // Arithmetic
    case InstrId::cpu_lui:
        writer.print_line("{}{} = S32({} << 16)", ctx_gpr_prefix(rt), rt, unsigned_imm_string);
        break;
    case InstrId::cpu_add:
    case InstrId::cpu_addu:
        {
            // Check if this addu belongs to a jump table load
            auto find_result = std::find_if(stats.jump_tables.begin(), stats.jump_tables.end(),
                [instr_vram](const RecompPort::JumpTable& jtbl) {
                return jtbl.addu_vram == instr_vram;
            });
            // If so, create a temp to preserve the addend register's value
            if (find_result != stats.jump_tables.end()) {
                const RecompPort::JumpTable& cur_jtbl = *find_result;
                writer.print_line("gpr jr_addend_{:08X} = {}{}", cur_jtbl.jr_vram, ctx_gpr_prefix(cur_jtbl.addend_reg), cur_jtbl.addend_reg);
            }
        }
        writer.print_line("{}{} = ADD32({}{}, {}{})", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_daddu:
        writer.print_line("{}{} = {}{} + {}{}", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_negu: // pseudo instruction for subu x, 0, y
    case InstrId::cpu_sub:
    case InstrId::cpu_subu:
        writer.print_line("{}{} = SUB32({}{}, {}{})", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_addi:
    case InstrId::cpu_addiu:
        writer.print_line("{}{} = ADD32({}{}, {})", ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs, signed_imm_string);
        break;
    case InstrId::cpu_daddi:
    case InstrId::cpu_daddiu:
        writer.print_line("{}{} = {}{} + {}", ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs, signed_imm_string);
        break;
    case InstrId::cpu_and:
        writer.print_line("{}{} = {}{} & {}{}", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_andi:
        writer.print_line("{}{} = {}{} & {}", ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs, unsigned_imm_string);
        break;
    case InstrId::cpu_or:
        writer.print_line("{}{} = {}{} | {}{}", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_ori:
        writer.print_line("{}{} = {}{} | {}", ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs, unsigned_imm_string);
        break;
    case InstrId::cpu_nor:
        writer.print_line("{}{} = ~({}{} | {}{})", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_xor:
        writer.print_line("{}{} = {}{} ^ {}{}", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_xori:
        writer.print_line("{}{} = {}{} ^ {}", ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs, unsigned_imm_string);
        break;
    case InstrId::cpu_sll:
        writer.print_line("{}{} = S32({}{}) << {}", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, sa);
        break;
    case InstrId::cpu_dsll:
        writer.print_line("{}{} = {}{} << {}", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, sa);
        break;
    case InstrId::cpu_dsll32:
        writer.print_line("{}{} = ((gpr)({}{})) << ({} + 32)", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, sa);
        break;
    case InstrId::cpu_sllv:
        writer.print_line("{}{} = S32({}{}) << ({}{} & 31)", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs);
        break;
    case InstrId::cpu_dsllv:
        writer.print_line("{}{} = {}{} << ({}{} & 63)", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs);
        break;
    case InstrId::cpu_sra:
        writer.print_line("{}{} = S32({}{}) >> {}", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, sa);
        break;
    case InstrId::cpu_dsra:
        writer.print_line("{}{} = SIGNED({}{}) >> {}", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, sa);
        break;
    case InstrId::cpu_dsra32:
        writer.print_line("{}{} = SIGNED({}{}) >> ({} + 32)", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, sa);
        break;
    case InstrId::cpu_srav:
        writer.print_line("{}{} = S32({}{}) >> ({}{} & 31)", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs);
        break;
    case InstrId::cpu_dsrav:
        writer.print_line("{}{} = SIGNED({}{}) >> ({}{} & 63)", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs);
        break;
    case InstrId::cpu_srl:
        writer.print_line("{}{} = S32(U32({}{}) >> {})", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, sa);
        break;
    case InstrId::cpu_dsrl:
        writer.print_line("{}{} = {}{} >> {}", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, sa);
        break;
    case InstrId::cpu_dsrl32:
        writer.print_line("{}{} = ((gpr)({}{})) >> ({} + 32)", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, sa);
        break;
    case InstrId::cpu_srlv:
        writer.print_line("{}{} = S32(U32({}{}) >> ({}{} & 31))", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs);
        break;
    case InstrId::cpu_dsrlv:
        writer.print_line("{}{} = {}{} >> ({}{} & 63))", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs);
        break;
    case InstrId::cpu_slt:
        writer.print_line("{}{} = SIGNED({}{}) < SIGNED({}{}) ? 1 : 0", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_slti:
        writer.print_line("{}{} = SIGNED({}{}) < {} ? 1 : 0", ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs, signed_imm_string);
        break;
    case InstrId::cpu_sltu:
        writer.print_line("{}{} = {}{} < {}{} ? 1 : 0", ctx_gpr_prefix(rd), rd, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_sltiu:
        writer.print_line("{}{} = {}{} < {} ? 1 : 0", ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs, signed_imm_string);
        break;
    case InstrId::cpu_mult:
        writer.print_line("result = S64(S32({}{})) * S64(S32({}{})); lo = S32(result >> 0); hi = S32(result >> 32)", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_dmult:
        writer.print_line("DMULT(S64({}{}), S64({}{}), &lo, &hi)", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_multu:
        writer.print_line("result = U64(U32({}{})) * U64(U32({}{})); lo = S32(result >> 0); hi = S32(result >> 32)", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_dmultu:
        writer.print_line("DMULTU(U64({}{}), U64({}{}), &lo, &hi)", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_div:
        // Cast to 64-bits before division to prevent artihmetic exception for s32(0x80000000) / -1
        writer.print_line("lo = S32(S64(S32({}{})) / S64(S32({}{}))); hi = S32(S64(S32({}{})) % S64(S32({}{})))", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_ddiv:
        writer.print_line("DDIV(S64({}{}), S64({}{}), &lo, &hi)", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_divu:
        writer.print_line("lo = S32(U32({}{}) / U32({}{})); hi = S32(U32({}{}) % U32({}{}))", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_ddivu:
        writer.print_line("DDIVU(U64({}{}), U64({}{}), &lo, &hi)", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_mflo:
        writer.print_line("{}{} = lo", ctx_gpr_prefix(rd), rd);
        break;
    case InstrId::cpu_mfhi:
        writer.print_line("{}{} = hi", ctx_gpr_prefix(rd), rd);
        break;
    case InstrId::cpu_mtlo:
        writer.print_line("lo = {}{}", ctx_gpr_prefix(rd), rd);
        break;
    case InstrId::cpu_mthi:
        writer.print_line("hi = {}{}", ctx_gpr_prefix(rd), rd);
        break;
    // Loads
    case InstrId::cpu_ld:
        writer.print_line("{}{} = LD({}, {}{})", ctx_gpr_prefix(rt), rt, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    case InstrId::cpu_lw:
        writer.print_line("{}{} = MEM_W({}, {}{})", ctx_gpr_prefix(rt), rt, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    case InstrId::cpu_lh:
        writer.print_line("{}{} = MEM_H({}, {}{})", ctx_gpr_prefix(rt), rt, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    case InstrId::cpu_lb:
        writer.print_line("{}{} = MEM_B({}, {}{})", ctx_gpr_prefix(rt), rt, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    case InstrId::cpu_lhu:
        writer.print_line("{}{} = MEM_HU({}, {}{})", ctx_gpr_prefix(rt), rt, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    case InstrId::cpu_lbu:
        writer.print_line("{}{} = MEM_BU({}, {}{})", ctx_gpr_prefix(rt), rt, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    // Stores
    case InstrId::cpu_sd:
        writer.print_line("SD({}{}, {}, {}{})", ctx_gpr_prefix(rt), rt, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    case InstrId::cpu_sw:
        writer.print_line("MEM_W({}, {}{}) = {}{}", signed_imm_string, ctx_gpr_prefix(base), base, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_sh:
        writer.print_line("MEM_H({}, {}{}) = {}{}", signed_imm_string, ctx_gpr_prefix(base), base, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_sb:
        writer.print_line("MEM_B({}, {}{}) = {}{}", signed_imm_string, ctx_gpr_prefix(base), base, ctx_gpr_prefix(rt), rt);
        break;
    // Unaligned loads
    // examples:
    // reg =        11111111 01234567
    // mem @ x =             89ABCDEF

    // LWL x + 0 -> FFFFFFFF 89ABCDEF
    // LWL x + 1 -> FFFFFFFF ABCDEF67
    // LWL x + 2 -> FFFFFFFF CDEF4567
    // LWL x + 3 -> FFFFFFFF EF234567

    // LWR x + 0 -> 00000000 01234589
    // LWR x + 1 -> 00000000 012389AB
    // LWR x + 2 -> 00000000 0189ABCD
    // LWR x + 3 -> FFFFFFFF 89ABCDEF
    case InstrId::cpu_lwl:
        writer.print_line("{}{} = do_lwl(rdram, {}{}, {}, {}{})", ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rt), rt, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    case InstrId::cpu_lwr:
        writer.print_line("{}{} = do_lwr(rdram, {}{}, {}, {}{})", ctx_gpr_prefix(rt), rt, ctx_gpr_prefix(rt), rt, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    // Unaligned stores
    // examples:
    // reg =        11111111 01234567
    // mem @ x =             89ABCDEF

    // SWL x + 0 ->          01234567
    // SWL x + 1 ->          89012345
    // SWL x + 2 ->          89AB0123
    // SWL x + 3 ->          89ABCD01

    // SWR x + 0 ->          67ABCDEF
    // SWR x + 1 ->          4567CDEF
    // SWR x + 2 ->          234567EF
    // SWR x + 3 ->          01234567
    case InstrId::cpu_swl:
        writer.print_line("do_swl(rdram, {}, {}{}, {}{})", signed_imm_string, ctx_gpr_prefix(base), base, ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_swr:
        writer.print_line("do_swr(rdram, {}, {}{}, {}{})", signed_imm_string, ctx_gpr_prefix(base), base, ctx_gpr_prefix(rt), rt);
        break;

    // Branches
    case InstrId::cpu_jal:
        writer.print_func_call(instr.getBranchVramGeneric());
        break;
    case InstrId::cpu_jalr:
        // jalr can only be handled with $ra as the return address register
        if (rd != (int)rabbitizer::Registers::Cpu::GprO32::GPR_O32_ra) {
            fmt::print(stderr, "Invalid return address reg for jalr: f{}\n", rd);
            return false;
        }
        needs_link_branch = true;
        writer.print_unconditional_branch("LOOKUP_FUNC({}{})(rdram, ctx)", ctx_gpr_prefix(rs), rs);
        break;
    case InstrId::cpu_j:
    case InstrId::cpu_b:
        {
            uint32_t branch_target = instr.getBranchVramGeneric();
            if (branch_target == instr_vram) {
                writer.print_line("pause_self(rdram)");
            }
            // Check if the branch is within this function
            else if (branch_target >= func.vram && branch_target < func_vram_end) {
                writer.print_unconditional_branch("goto L_{:08X}", branch_target);
            }
            // This may be a tail call in the middle of the control flow due to a previous check
            // For example:
            // ```c
            // void test() {
            //     if (SOME_CONDITION) {
            //         do_a();
            //     } else {
            //         do_b();
            //     }
            // }
            // ```
            // FIXME: how to deal with static functions?
            else if (context.functions_by_vram.find(branch_target) != context.functions_by_vram.end()) {
                fmt::print("Tail call in {} to 0x{:08X}\n", func.name, branch_target);
                writer.print_func_call(branch_target, false);
                writer.print_line("return");
            }
            else {
                fmt::print(stderr, "Unhandled branch in {} at 0x{:08X} to 0x{:08X}\n", func.name, instr_vram, branch_target);
                return false;
            }
        }
        break;
    case InstrId::cpu_jr:
        if (rs == (int)rabbitizer::Registers::Cpu::GprO32::GPR_O32_ra) {
            writer.print_unconditional_branch("return");
        } else {
            auto jtbl_find_result = std::find_if(stats.jump_tables.begin(), stats.jump_tables.end(),
                [instr_vram](const RecompPort::JumpTable& jtbl) {
                    return jtbl.jr_vram == instr_vram;
                });

            if (jtbl_find_result != stats.jump_tables.end()) {
                const RecompPort::JumpTable& cur_jtbl = *jtbl_find_result;
                bool dummy_needs_link_branch, dummy_is_branch_likely;
                size_t next_reloc_index = reloc_index;
                uint32_t next_vram = instr_vram + 4;
                if (reloc_index + 1 < section.relocs.size() && next_vram > section.relocs[reloc_index].address) {
                    next_reloc_index++;
                }
                process_instruction(context, config, func, stats, skipped_insns, instr_index + 1, instructions, output_file, false, false, link_branch_index, next_reloc_index, dummy_needs_link_branch, dummy_is_branch_likely, static_funcs_out);
                writer.print_indent();
                fmt::print(output_file, "switch (jr_addend_{:08X} >> 2) {{\n", cur_jtbl.jr_vram);
                for (size_t entry_index = 0; entry_index < cur_jtbl.entries.size(); entry_index++) {
                    writer.print_indent();
                    writer.print_line("case {}: goto L_{:08X}; break", entry_index, cur_jtbl.entries[entry_index]);
                }
                writer.print_indent();
                writer.print_line("default: switch_error(__func__, 0x{:08X}, 0x{:08X})", instr_vram, cur_jtbl.vram);
                writer.print_indent();
                fmt::print(output_file, "}}\n");
                break;
            }

            auto jump_find_result = std::find_if(stats.absolute_jumps.begin(), stats.absolute_jumps.end(),
                [instr_vram](const RecompPort::AbsoluteJump& jump) {
                return jump.instruction_vram == instr_vram;
            });

            if (jump_find_result != stats.absolute_jumps.end()) {
                writer.print_unconditional_branch("LOOKUP_FUNC({})(rdram, ctx)", (uint64_t)(int32_t)jump_find_result->jump_target);
                // jr doesn't link so it acts like a tail call, meaning we should return directly after the jump returns
                writer.print_line("return");
                break;
            }

            bool is_tail_call = instr_vram == func_vram_end - 2 * sizeof(func.words[0]);
            if (is_tail_call) {
                fmt::print("Indirect tail call in {}\n", func.name);
                writer.print_unconditional_branch("LOOKUP_FUNC({}{})(rdram, ctx)", ctx_gpr_prefix(rs), rs);
                writer.print_line("return");
                break;
            }

            fmt::print(stderr, "No jump table found for jr at 0x{:08X} and not tail call\n", instr_vram);
        }
        break;
    case InstrId::cpu_syscall:
        writer.print_line("recomp_syscall_handler(rdram, ctx, 0x{:08X})", instr_vram);
        // syscalls don't link, so treat it like a tail call
        writer.print_line("return");
        break;
    case InstrId::cpu_bnel:
        is_branch_likely = true;
        [[fallthrough]];
    case InstrId::cpu_bne:
        writer.print_indent();
        writer.print_branch_condition("if ({}{} != {}{})", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        writer.print_branch((uint32_t)instr.getBranchVramGeneric());
        break;
    case InstrId::cpu_beql:
        is_branch_likely = true;
        [[fallthrough]];
    case InstrId::cpu_beq:
        writer.print_indent();
        writer.print_branch_condition("if ({}{} == {}{})", ctx_gpr_prefix(rs), rs, ctx_gpr_prefix(rt), rt);
        writer.print_branch((uint32_t)instr.getBranchVramGeneric());
        break;
    case InstrId::cpu_bgezl:
        is_branch_likely = true;
        [[fallthrough]];
    case InstrId::cpu_bgez:
        writer.print_indent();
        writer.print_branch_condition("if (SIGNED({}{}) >= 0)", ctx_gpr_prefix(rs), rs);
        writer.print_branch((uint32_t)instr.getBranchVramGeneric());
        break;
    case InstrId::cpu_bgtzl:
        is_branch_likely = true;
        [[fallthrough]];
    case InstrId::cpu_bgtz:
        writer.print_indent();
        writer.print_branch_condition("if (SIGNED({}{}) > 0)", ctx_gpr_prefix(rs), rs);
        writer.print_branch((uint32_t)instr.getBranchVramGeneric());
        break;
    case InstrId::cpu_blezl:
        is_branch_likely = true;
        [[fallthrough]];
    case InstrId::cpu_blez:
        writer.print_indent();
        writer.print_branch_condition("if (SIGNED({}{}) <= 0)", ctx_gpr_prefix(rs), rs);
        writer.print_branch((uint32_t)instr.getBranchVramGeneric());
        break;
    case InstrId::cpu_bltzl:
        is_branch_likely = true;
        [[fallthrough]];
    case InstrId::cpu_bltz:
        writer.print_indent();
        writer.print_branch_condition("if (SIGNED({}{}) < 0)", ctx_gpr_prefix(rs), rs);
        writer.print_branch((uint32_t)instr.getBranchVramGeneric());
        break;
    case InstrId::cpu_break:
        writer.print_line("do_break({})", instr_vram);
        break;
    case InstrId::cpu_bgezall:
        is_branch_likely = true;
        [[fallthrough]];
    case InstrId::cpu_bgezal:
        writer.print_indent();
        writer.print_branch_condition("if (SIGNED({}{}) >= 0) {{", ctx_gpr_prefix(rs), rs);
        writer.print_func_call(instr.getBranchVramGeneric());
        writer.print_line("}}");
        break;

    // Cop1 loads/stores
    case InstrId::cpu_mtc1:
        if ((fs & 1) == 0) {
            // even fpr
            writer.print_line("ctx->f{}.u32l = {}{}", fs, ctx_gpr_prefix(rt), rt);
        }
        else {
            // odd fpr
            writer.print_line("ctx->f_odd[({} - 1) * 2] = {}{}", fs, ctx_gpr_prefix(rt), rt);
        }
        break;
    case InstrId::cpu_mfc1:
        if ((fs & 1) == 0) {
            // even fpr
            writer.print_line("{}{} = (int32_t)ctx->f{}.u32l", ctx_gpr_prefix(rt), rt, fs);
        } else {
            // odd fpr
            writer.print_line("{}{} = (int32_t)ctx->f_odd[({} - 1) * 2]", ctx_gpr_prefix(rt), rt, fs);
        }
        break;
    //case InstrId::cpu_dmfc1:
    //    if ((fs & 1) == 0) {
    //        // even fpr
    //        writer.print_line("{}{} = ctx->f{}.u64", ctx_gpr_prefix(rt), rt, fs);
    //    } else {
    //        fmt::print(stderr, "Invalid operand for dmfc1: f{}\n", fs);
    //        return false;
    //    }
    //    break;
    case InstrId::cpu_lwc1:
        if ((ft & 1) == 0) {
            // even fpr
            writer.print_line("ctx->f{}.u32l = MEM_W({}, {}{})", ft, signed_imm_string, ctx_gpr_prefix(base), base);
        } else {
            // odd fpr
            writer.print_line("ctx->f_odd[({} - 1) * 2] = MEM_W({}, {}{})", ft, signed_imm_string, ctx_gpr_prefix(base), base);
        }
        break;
    case InstrId::cpu_ldc1:
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("ctx->f{}.u64 = LD({}, {}{})", ft, signed_imm_string, ctx_gpr_prefix(base), base);
        break;
    case InstrId::cpu_swc1:
        if ((ft & 1) == 0) {
            // even fpr
            writer.print_line("MEM_W({}, {}{}) = ctx->f{}.u32l", signed_imm_string, ctx_gpr_prefix(base), base, ft);
        } else {
            // odd fpr
            writer.print_line("MEM_W({}, {}{}) = ctx->f_odd[({} - 1) * 2]", signed_imm_string, ctx_gpr_prefix(base), base, ft);
        }
        break;
    case InstrId::cpu_sdc1:
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("SD(ctx->f{}.u64, {}, {}{})", ft, signed_imm_string, ctx_gpr_prefix(base), base);
        break;

    // Cop1 compares
    // TODO allow NaN in ordered and unordered float comparisons, default to a compare result of 1 for ordered and 0 for unordered if a NaN is present
    case InstrId::cpu_c_lt_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl < ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_olt_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl < ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_ult_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl < ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_lt_d:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d < ctx->f{}.d", fs, ft);
        break;
    case InstrId::cpu_c_olt_d:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d < ctx->f{}.d", fs, ft);
        break;
    case InstrId::cpu_c_ult_d:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d < ctx->f{}.d", fs, ft);
        break;
    case InstrId::cpu_c_le_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl <= ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_ole_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl <= ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_ule_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl <= ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_le_d:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d <= ctx->f{}.d", fs, ft);
        break;
    case InstrId::cpu_c_ole_d:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d <= ctx->f{}.d", fs, ft);
        break;
    case InstrId::cpu_c_ule_d:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d <= ctx->f{}.d", fs, ft);
        break;
    case InstrId::cpu_c_eq_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl == ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_ueq_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl == ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_ngl_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl == ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_seq_s:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.fl == ctx->f{}.fl", fs, ft);
        break;
    case InstrId::cpu_c_eq_d:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d == ctx->f{}.d", fs, ft);
        break;
    case InstrId::cpu_c_ueq_d:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d == ctx->f{}.d", fs, ft);
        break;
    case InstrId::cpu_c_ngl_d:
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d == ctx->f{}.d", fs, ft);
        break;
    case InstrId::cpu_c_deq_d: // TODO rename to c_seq_d when fixed in rabbitizer
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("c1cs = ctx->f{}.d == ctx->f{}.d", fs, ft);
        break;
    
    // Cop1 branches
    case InstrId::cpu_bc1tl:
        is_branch_likely = true;
        [[fallthrough]];
    case InstrId::cpu_bc1t:
        writer.print_indent();
        writer.print_branch_condition("if (c1cs)", ctx_gpr_prefix(rs), rs);
        writer.print_branch((uint32_t)instr.getBranchVramGeneric());
        break;
    case InstrId::cpu_bc1fl:
        is_branch_likely = true;
        [[fallthrough]];
    case InstrId::cpu_bc1f:
        writer.print_indent();
        writer.print_branch_condition("if (!c1cs)", ctx_gpr_prefix(rs), rs);
        writer.print_branch((uint32_t)instr.getBranchVramGeneric());
        break;

    // Cop1 arithmetic
    case InstrId::cpu_mov_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.fl = ctx->f{}.fl", fd, fs);
        break;
    case InstrId::cpu_mov_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.d = ctx->f{}.d", fd, fs);
        break;
    case InstrId::cpu_neg_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.fl)", fs);
        writer.print_line("ctx->f{}.fl = -ctx->f{}.fl", fd, fs);
        break;
    case InstrId::cpu_neg_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.d)", fs);
        writer.print_line("ctx->f{}.d = -ctx->f{}.d", fd, fs);
        break;
    case InstrId::cpu_abs_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.fl)", fs);
        writer.print_line("ctx->f{}.fl = fabsf(ctx->f{}.fl)", fd, fs);
        break;
    case InstrId::cpu_abs_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.d)", fs);
        writer.print_line("ctx->f{}.d = fabs(ctx->f{}.d)", fd, fs);
        break;
    case InstrId::cpu_sqrt_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.fl)", fs);
        writer.print_line("ctx->f{}.fl = sqrtf(ctx->f{}.fl)", fd, fs);
        break;
    case InstrId::cpu_sqrt_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.d)", fs);
        writer.print_line("ctx->f{}.d = sqrt(ctx->f{}.d)", fd, fs);
        break;
    case InstrId::cpu_add_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("NAN_CHECK(ctx->f{}.fl); NAN_CHECK(ctx->f{}.fl)", fs, ft);
        writer.print_line("ctx->f{}.fl = ctx->f{}.fl + ctx->f{}.fl", fd, fs, ft);
        break;
    case InstrId::cpu_add_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("NAN_CHECK(ctx->f{}.d); NAN_CHECK(ctx->f{}.d)", fs, ft);
        writer.print_line("ctx->f{}.d = ctx->f{}.d + ctx->f{}.d", fd, fs, ft);
        break;
    case InstrId::cpu_sub_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("NAN_CHECK(ctx->f{}.fl); NAN_CHECK(ctx->f{}.fl)", fs, ft);
        writer.print_line("ctx->f{}.fl = ctx->f{}.fl - ctx->f{}.fl", fd, fs, ft);
        break;
    case InstrId::cpu_sub_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("NAN_CHECK(ctx->f{}.d); NAN_CHECK(ctx->f{}.d)", fs, ft);
        writer.print_line("ctx->f{}.d = ctx->f{}.d - ctx->f{}.d", fd, fs, ft);
        break;
    case InstrId::cpu_mul_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("NAN_CHECK(ctx->f{}.fl); NAN_CHECK(ctx->f{}.fl)", fs, ft);
        writer.print_line("ctx->f{}.fl = MUL_S(ctx->f{}.fl, ctx->f{}.fl)", fd, fs, ft);
        break;
    case InstrId::cpu_mul_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("NAN_CHECK(ctx->f{}.d); NAN_CHECK(ctx->f{}.d)", fs, ft);
        writer.print_line("ctx->f{}.d = MUL_D(ctx->f{}.d, ctx->f{}.d)", fd, fs, ft);
        break;
    case InstrId::cpu_div_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("NAN_CHECK(ctx->f{}.fl); NAN_CHECK(ctx->f{}.fl)", fs, ft);
        writer.print_line("ctx->f{}.fl = DIV_S(ctx->f{}.fl, ctx->f{}.fl)", fd, fs, ft);
        break;
    case InstrId::cpu_div_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("CHECK_FR(ctx, {})", ft);
        writer.print_line("NAN_CHECK(ctx->f{}.d); NAN_CHECK(ctx->f{}.d)", fs, ft);
        writer.print_line("ctx->f{}.d = DIV_D(ctx->f{}.d, ctx->f{}.d)", fd, fs, ft);
        break;
    case InstrId::cpu_cvt_s_w:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.fl = CVT_S_W(ctx->f{}.u32l)", fd, fs);
        break;
    case InstrId::cpu_cvt_d_w:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.d = CVT_D_W(ctx->f{}.u32l)", fd, fs);
        break;
    case InstrId::cpu_cvt_d_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.fl)", fs);
        writer.print_line("ctx->f{}.d = CVT_D_S(ctx->f{}.fl)", fd, fs);
        break;
    case InstrId::cpu_cvt_s_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.d)", fs);
        writer.print_line("ctx->f{}.fl = CVT_S_D(ctx->f{}.d)", fd, fs);
        break;
    case InstrId::cpu_cvt_d_l:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.d = CVT_D_L(ctx->f{}.u64)", fd, fs);
        break;
    case InstrId::cpu_cvt_l_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.d)", fs);
        writer.print_line("ctx->f{}.u64 = CVT_L_D(ctx->f{}.d)", fd, fs);
        break;
    case InstrId::cpu_cvt_s_l:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.fl = CVT_S_L(ctx->f{}.u64)", fd, fs);
        break;
    case InstrId::cpu_cvt_l_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("NAN_CHECK(ctx->f{}.fl)", fs);
        writer.print_line("ctx->f{}.u64 = CVT_L_S(ctx->f{}.fl)", fd, fs);
        break;
    case InstrId::cpu_trunc_w_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = TRUNC_W_S(ctx->f{}.fl)", fd, fs);
        break;
    case InstrId::cpu_trunc_w_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = TRUNC_W_D(ctx->f{}.d)", fd, fs);
        break;
    //case InstrId::cpu_trunc_l_s:
    //    writer.print_line("CHECK_FR(ctx, {})", fd);
    //    writer.print_line("CHECK_FR(ctx, {})", fs);
    //    writer.print_line("ctx->f{}.u64 = TRUNC_L_S(ctx->f{}.fl)", fd, fs);
    //    break;
    //case InstrId::cpu_trunc_l_d:
    //    writer.print_line("CHECK_FR(ctx, {})", fd);
    //    writer.print_line("CHECK_FR(ctx, {})", fs);
    //    writer.print_line("ctx->f{}.u64 = TRUNC_L_D(ctx->f{}.d)", fd, fs);
    //    break;
    case InstrId::cpu_ctc1:
        if (cop1_cs != 31) {
            fmt::print(stderr, "Invalid FP control register for ctc1: {}\n", cop1_cs);
            return false;
        }
        writer.print_line("rounding_mode = ({}{}) & 0x3", ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_cfc1:
        if (cop1_cs != 31) {
            fmt::print(stderr, "Invalid FP control register for cfc1: {}\n", cop1_cs);
            return false;
        }
        writer.print_line("{}{} = rounding_mode", ctx_gpr_prefix(rt), rt);
        break;
    case InstrId::cpu_cvt_w_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = CVT_W_S(ctx->f{}.fl)", fd, fs);
        break;
    case InstrId::cpu_cvt_w_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = CVT_W_D(ctx->f{}.d)", fd, fs);
        break;
    case InstrId::cpu_round_w_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = lroundf(ctx->f{}.fl)", fd, fs);
        break;
    case InstrId::cpu_round_w_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = lround(ctx->f{}.d)", fd, fs);
        break;
    case InstrId::cpu_ceil_w_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = S32(ceilf(ctx->f{}.fl))", fd, fs);
        break;
    case InstrId::cpu_ceil_w_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = S32(ceil(ctx->f{}.d))", fd, fs);
        break;
    case InstrId::cpu_floor_w_s:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = S32(floorf(ctx->f{}.fl))", fd, fs);
        break;
    case InstrId::cpu_floor_w_d:
        writer.print_line("CHECK_FR(ctx, {})", fd);
        writer.print_line("CHECK_FR(ctx, {})", fs);
        writer.print_line("ctx->f{}.u32l = S32(floor(ctx->f{}.d))", fd, fs);
        break;
    default:
        fmt::print(stderr, "Unhandled instruction: {}\n", instr.getOpcodeName());
        return false;
    }

    // TODO is this used?
    if (emit_link_branch) {
        fmt::print(output_file, "    after_{}:\n", link_branch_index);
    }

    return true;
}

bool RecompPort::recompile_function(const RecompPort::Context& context, const RecompPort::Config& config, const RecompPort::Function& func, std::ofstream& output_file, std::span<std::vector<uint32_t>> static_funcs_out, bool write_header) {
    //fmt::print("Recompiling {}\n", func.name);
    std::vector<rabbitizer::InstructionCpu> instructions;

    if (write_header) {
        // Write the file header
        fmt::print(output_file,
            "#include \"recomp.h\"\n"
            "#include \"disable_warnings.h\"\n"
            "\n");
    }

    fmt::print(output_file,
        "void {}(uint8_t* rdram, recomp_context* ctx) {{\n"
        // these variables shouldn't need to be preserved across function boundaries, so make them local for more efficient output
        "    uint64_t hi = 0, lo = 0, result = 0;\n"
        "    unsigned int rounding_mode = DEFAULT_ROUNDING_MODE;\n"
        "    int c1cs = 0; \n", // cop1 conditional signal
        func.name);

    // Skip analysis and recompilation of this function is stubbed.
    if (!func.stubbed) {
        // Use a set to sort and deduplicate labels
        std::set<uint32_t> branch_labels;
        instructions.reserve(func.words.size());

        // First pass, disassemble each instruction and collect branch labels
        uint32_t vram = func.vram;
        for (uint32_t word : func.words) {
            const auto& instr = instructions.emplace_back(byteswap(word), vram);

            // If this is a branch or a direct jump, add it to the local label list
            if (instr.isBranch() || instr.getUniqueId() == rabbitizer::InstrId::UniqueId::cpu_j) {
                branch_labels.insert((uint32_t)instr.getBranchVramGeneric());
            }

            // Advance the vram address by the size of one instruction
            vram += 4;
        }

        // Analyze function
        RecompPort::FunctionStats stats{};
        if (!RecompPort::analyze_function(context, func, instructions, stats)) {
            fmt::print(stderr, "Failed to analyze {}\n", func.name);
            output_file.clear();
            return false;
        }

        std::unordered_set<uint32_t> skipped_insns{};

        // Add jump table labels into function
        for (const auto& jtbl : stats.jump_tables) {
            skipped_insns.insert(jtbl.lw_vram);
            for (uint32_t jtbl_entry : jtbl.entries) {
                branch_labels.insert(jtbl_entry);
            }
        }

        // Second pass, emit code for each instruction and emit labels
        auto cur_label = branch_labels.cbegin();
        vram = func.vram;
        int num_link_branches = 0;
        int num_likely_branches = 0;
        bool needs_link_branch = false;
        bool in_likely_delay_slot = false;
        const auto& section = context.sections[func.section_index];
        bool needs_reloc = section.relocatable && section.relocs.size() > 0;
        size_t reloc_index = 0;
        for (size_t instr_index = 0; instr_index < instructions.size(); ++instr_index) {
            bool had_link_branch = needs_link_branch;
            bool is_branch_likely = false;
            // If we're in the delay slot of a likely instruction, emit a goto to skip the instruction before any labels
            if (in_likely_delay_slot) {
                fmt::print(output_file, "    goto skip_{};\n", num_likely_branches);
            }
            // If there are any other branch labels to insert and we're at the next one, insert it
            if (cur_label != branch_labels.end() && vram >= *cur_label) {
                fmt::print(output_file, "L_{:08X}:\n", *cur_label);
                ++cur_label;
            }

            // If this is a relocatable section, advance the reloc index until we reach the last one or until we get to/pass the current instruction
            if (needs_reloc) {
                while (reloc_index < (section.relocs.size() - 1) && section.relocs[reloc_index].address < vram) {
                    reloc_index++;
                }
            }

            // Process the current instruction and check for errors
            if (process_instruction(context, config, func, stats, skipped_insns, instr_index, instructions, output_file, false, needs_link_branch, num_link_branches, reloc_index, needs_link_branch, is_branch_likely, static_funcs_out) == false) {
                fmt::print(stderr, "Error in recompiling {}, clearing output file\n", func.name);
                output_file.clear();
                return false;
            }
            // If a link return branch was generated, advance the number of link return branches
            if (had_link_branch) {
                num_link_branches++;
            }
            // Now that the instruction has been processed, emit a skip label for the likely branch if needed
            if (in_likely_delay_slot) {
                fmt::print(output_file, "    skip_{}:\n", num_likely_branches);
                num_likely_branches++;
            }
            // Mark the next instruction as being in a likely delay slot if the 
            in_likely_delay_slot = is_branch_likely;
            // Advance the vram address by the size of one instruction
            vram += 4;
        }
    }

    // Terminate the function
    fmt::print(output_file, ";}}\n");
    
    return true;
}
