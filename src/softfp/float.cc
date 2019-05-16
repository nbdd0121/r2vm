#include "softfp/float.h"

namespace softfp {

Rounding_mode rounding_mode = Rounding_mode::ties_to_even;
Exception_flag exception_flags = Exception_flag::none;

template class Float<8, 23>;
template class Float<11, 52>;

}