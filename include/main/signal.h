#ifndef MAIN_SIGNAL_H
#define MAIN_SIGNAL_H

#include <stdexcept>

struct Segv_exception: std::runtime_error {
    int sig;
    Segv_exception(int sig): std::runtime_error {"segmentation fault"}, sig {sig} {}
};

void setup_fault_handler();

#endif
