

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(_WIN32) || defined(__linux__)
#include <stdint.h>
#endif
/**
 * @brief interface
 *
 */
#if defined(_MSC_VER)
#if defined(BUILDING_MAGIC_XE_DLL)
#define MAGIC_XE_PUBLIC __declspec(dllexport)
#elif defined(USING_MAGIC_XE_DLL)
#define MAGIC_XE_PUBLIC __declspec(dllimport)
#else
#define MAGIC_XE_PUBLIC
#endif // BUILDING_MAGIC_XE_DLL
#else
#define MAGIC_XE_PUBLIC __attribute__((visibility("default")))
#endif // _MSC_VER

#ifdef __cplusplus
#define MAGIC_XE_API extern "C" MAGIC_XE_PUBLIC
#else
#define MAGIC_XE_API MAGIC_XE_PUBLIC
#endif
