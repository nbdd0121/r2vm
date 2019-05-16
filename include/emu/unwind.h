#ifndef EMU_UNWIND_H
#define EMU_UNWIND_H

// This header is used for exception unwinding support. The definition within this file follows the Itanium C++
// Exception handling ABI. https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html.

#include <cstdint>
#include <typeinfo>

extern "C" {

typedef enum {
    _URC_NO_REASON = 0,
    _URC_FOREIGN_EXCEPTION_CAUGHT = 1,
    _URC_FATAL_PHASE2_ERROR = 2,
    _URC_FATAL_PHASE1_ERROR = 3,
    _URC_NORMAL_STOP = 4,
    _URC_END_OF_STACK = 5,
    _URC_HANDLER_FOUND = 6,
    _URC_INSTALL_CONTEXT = 7,
    _URC_CONTINUE_UNWIND = 8
} _Unwind_Reason_Code;

typedef int _Unwind_Action;
static const _Unwind_Action _UA_SEARCH_PHASE = 1;
static const _Unwind_Action _UA_CLEANUP_PHASE = 2;
static const _Unwind_Action _UA_HANDLER_FRAME = 4;
static const _Unwind_Action _UA_FORCE_UNWIND = 8;

typedef void (*_Unwind_Exception_Cleanup_Fn)(_Unwind_Reason_Code reason, struct _Unwind_Exception *exc);

struct _Unwind_Exception {
    uint64_t exception_class;
    _Unwind_Exception_Cleanup_Fn exception_cleanup;
    uint64_t private_1;
    uint64_t private_2;
};

_Unwind_Reason_Code _Unwind_RaiseException(struct _Unwind_Exception *exception_object);

typedef _Unwind_Reason_Code (*_Unwind_Stop_Fn)(
    int version,
    _Unwind_Action actions,
    uint64_t exceptionClass,
    struct _Unwind_Exception *exceptionObject,
    struct _Unwind_Context *context,
    void *stop_parameter
);

_Unwind_Reason_Code _Unwind_ForcedUnwind(
    struct _Unwind_Exception *exception_object,
    _Unwind_Stop_Fn stop,
    void *stop_parameter
);

[[noreturn]] void _Unwind_Resume(struct _Unwind_Exception *exception_object);

void _Unwind_DeleteException(struct _Unwind_Exception *exception_object);

uint64_t _Unwind_GetGR(struct _Unwind_Context *context, int index);

void _Unwind_SetGR(struct _Unwind_Context *context, int index, uint64_t new_value);

uint64_t _Unwind_GetIP(struct _Unwind_Context *context);

void _Unwind_SetIP(struct _Unwind_Context *context, uint64_t new_value);

uint64_t _Unwind_GetLanguageSpecificData(struct _Unwind_Context *context);

uint64_t _Unwind_GetRegionStart(struct _Unwind_Context *context);

typedef _Unwind_Reason_Code (*__personality_routine)(
    int version,
    _Unwind_Action actions,
    uint64_t exceptionClass,
    struct _Unwind_Exception *exceptionObject,
    struct _Unwind_Context *context);

struct __cxa_exception { 
    std::type_info *    exceptionType;
    void (*exceptionDestructor) (void *); 
    void *              unexpectedHandler;
    void *              terminateHandler;
    __cxa_exception *   nextException;

    int                 handlerCount;
    int	                handlerSwitchValue;
    const char *        actionRecord;
    const char *        languageSpecificData;
    void *              catchTemp;
    void *              adjustedPtr;

    _Unwind_Exception   unwindHeader;
};

} // extern "C"

#endif

