/*
 * Nexus Core Unikernel Bootloader (x86_64)
 * 
 * The top 1% of quantitative engineering bypasses Linux entirely.
 * A standard OS context switch takes 1000-2000 CPU cycles.
 * This bootloader transitions the CPU from 16-bit real mode -> 32-bit protected mode 
 * -> 64-bit Long Mode, and hands control directly to the trading strategy 
 * running in Ring 0 (Kernel Mode). The strategy IS the operating system.
 */

.global _start
.section .text.boot

_start:
    /* Disable interrupts immediately */
    cli

    /* Set up the Global Descriptor Table (GDT) for 64-bit Flat Mode */
    lgdt (gdt_descriptor)

    /* Enable Physical Address Extension (PAE) */
    mov %cr4, %eax
    or $0x20, %eax
    mov %eax, %cr4

    /* Set Long Mode Enable (LME) bit in EFER MSR */
    mov $0xC0000080, %ecx
    rdmsr
    or $0x100, %eax
    wrmsr

    /* Enable Paging */
    mov %cr0, %eax
    or $0x80000000, %eax
    mov %eax, %cr0

    /* Far jump into 64-bit Ring 0 Long Mode */
    ljmp $0x08, $long_mode_start

.code64
long_mode_start:
    /* Initialize stack pointer */
    mov $0x200000, %rsp

    /* Pass control to the C++ trading logic, running with hardware privileges */
    call nexus_strategy_main

    /* Halt if strategy ever returns */
    hlt

.align 8
gdt_descriptor:
    .word gdt_end - gdt - 1
    .quad gdt

gdt:
    .quad 0x0000000000000000 /* Null descriptor */
    .quad 0x00209A0000000000 /* 64-bit Code descriptor (Ring 0) */
    .quad 0x0000920000000000 /* 64-bit Data descriptor (Ring 0) */
gdt_end:
