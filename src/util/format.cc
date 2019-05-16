#include "util/format.h"

namespace util::internal {

void format_impl(std::ostream& stream, const char *format, util::internal::Bound_formatter* view, size_t size) {

    // Revert stream states to default.
    stream.width(0);
    stream.precision(6);
    stream.fill(' ');
    stream.unsetf(
        std::ios::adjustfield | std::ios::basefield | std::ios::floatfield | std::ios::showbase | std::ios::boolalpha |
        std::ios::showpoint | std::ios::showpos | std::ios::uppercase
    );

    size_t ptr = 0;

    // format will point to the first unprinted character, and pointer will point to the processing character. It is
    // implemented in this way so that all contiguous non-format characters are printed at the same time.
    const char *pointer = format;
    while (*pointer) {

        // Skip normal characters.
        if (*pointer != '{' && *pointer != '}') {
            pointer++;
            continue;
        }

        // Escaped brace character.
        if (pointer[1] == *pointer) {

            // Print all previous characters, including the first brace.
            stream.write(format, pointer - format + 1);
            pointer += 2;
            format = pointer;
            continue;
        }

        // Print out all unprinted normal characters preceeding the brace. Pointers will be updated after formatted print.
        if (pointer != format) stream.write(format, pointer - format);

        if (*pointer == '{') {

            // Move pointer pass the brace.
            pointer++;

            // Deal with positional argument.
            size_t arg_index;
            if (*pointer == ':' || *pointer == '}') {
                if (ptr >= size) throw std::logic_error { "overflow" };
                arg_index = ptr;
                ptr++;
            } else {
                arg_index = strtoul(pointer, const_cast<char**>(&pointer), 10);
                if (!pointer || (*pointer != ':' && *pointer != '}')) throw std::logic_error { "illegal format string" };
                if (arg_index >= size) throw std::logic_error { "overflow" };
            }

            // If the next character is colon, then set stream states accordingly. The format string mimicks the design
            // of printf.
            bool state_changed = false;
            char format_type = '\0';
            if (*pointer == ':') {
                pointer++;

                state_changed = true;

                // Parse flags
                for(;; pointer++) {
                    switch(*pointer) {
                        case '#':
                            stream.setf(std::ios::showbase);
                            continue;
                        case '0':
                            // flag 0 is ignored when flag '-' is present.
                            if(!(stream.flags() & std::ios::left)) {
                                stream.fill('0');
                                stream.setf(std::ios::internal, std::ios::adjustfield);
                            }
                            continue;
                        case '-':
                            // Explicitly set fill to space, just in case 0 is specified beforehand.
                            stream.fill(' ');
                            stream.setf(std::ios::left, std::ios::adjustfield);
                            continue;
                        // TODO: Space is ignored for now as they need additional work.
                        case '+':
                            stream.setf(std::ios::showpos);
                            continue;
                        default:
                            break;
                    }
                    break;
                }

                if (stream.fill() == '0' && (stream.flags() & std::ios::showpos)) {
                    throw std::logic_error {"combination of showpos and zero pad not supported"};
                }

                // Parse width
                // TODO: * not yet supported since it requires additional effort.
                if (*pointer >= '0' && *pointer <= '9') {

                    // This should never fail.
                    stream.width(strtoul(pointer, const_cast<char**>(&pointer), 10));
                }

                // TODO: Precision ignored for now due to no demand.

                // Note that we don't need to specifiy length modifier since this is type safe.
                // Parse the base field. Move pointer in advance, and if move it back in case it is not recognized.
                format_type = *pointer++;
                switch(format_type) {
                    case 'u': case 'd': case 'i':
                        stream.setf(std::ios::dec, std::ios::basefield);
                        break;
                    case 'o':
                        stream.setf(std::ios::oct, std::ios::basefield);
                        break;
                    case 'X':
                        stream.setf(std::ios::uppercase);
                        [[fallthrough]];
                    case 'x':
                        stream.setf(std::ios::hex, std::ios::basefield);
                        break;
                    default:
                        pointer--;
                        format_type = '\0';
                        break;
                }
            }
            
            // Move past the brace.
            if (*pointer != '}') throw std::logic_error { "illegal format string" };
            pointer++;
            format = pointer;

            view[arg_index].format(stream, format_type);

            // Revert stream states to default if we touched it.
            if (state_changed) {
                stream.width(0);
                stream.precision(6);
                stream.fill(' ');
                stream.unsetf(
                    std::ios::adjustfield | std::ios::basefield | std::ios::floatfield | std::ios::showbase | std::ios::boolalpha |
                    std::ios::showpoint | std::ios::showpos | std::ios::uppercase
                );
            }

        } else {

            // Otherwise it must be closing brace, which must be escaped in order to be displayed.
            throw std::logic_error { "illegal format string" };
        }
    }

    // Print out any normal characters remaining.
    if (format != pointer) {
        stream.write(format, pointer - format);
    }
}

}
