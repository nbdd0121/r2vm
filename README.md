# riscv-dbt

This is a dynamic binary translator for running unmodified RISC-V binaries on AMD64 Linux machine. It is written as the University of Cambridge Part II project, and leads to [my dissertation](https://garyguo.net/uploads/riscv-dbt.pdf).

This project has RV64GC interpretation support and can translate instructions in RV64IMC to native AMD64 code for fast execution. It performs the translation by utilising an graph intermediate representation. Theoretically the architecture of this project means this project can be easily extended to support emulation for multiple ISAs on multiple platforms. You can check my dissertation for more technical details.
