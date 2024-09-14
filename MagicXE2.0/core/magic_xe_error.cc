#include "magic_xe_error.h"

#ifndef MAGIC_XE_ARRAY_ELEMS
#define MAGIC_XE_ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))
#endif

typedef struct MagicXEErrorEntry {
    MagicXEError err;
    const char *str;
} MagicXEErrorEntry;

static const struct MagicXEErrorEntry error_entries[] = {
    {MAGIC_XE_SUCCESS, "Successed"},
    {MAGIC_XE_NO_SUCH_FILE, "No such file, the specified file does not exit."},
    {MAGIC_XE_INVALID_FILE, "File is illegal or corrupted."},
    {MAGIC_XE_OUT_OF_MEMORY, "Not enough free memory to allocate object."},
    {MAGIC_XE_NOT_IMPLEMENTED, "Operation has not been implemented."},
    {MAGIC_XE_CANCEL, "Operation has been canceled."},
    {MAGIC_XE_EOF, "End of file."},
    {MAGIC_XE_INVALID_PARAM, "Parameter or data is incorrect or invalid."},
    {MAGIC_XE_NOT_FOUND, "Something is not found."},
    {MAGIC_XE_PREPROCESS_NOT_COMPLETE,
        "Preprocessing required before this operation has not been completed, or the necessary parameters are "
        "missing."},
    {MAGIC_XE_INVALID_MODEL_CONFIG, "Model config is invalid."},
    {MAGIC_XE_INVALID_MODEL, "Model file is incorrect or not exist."},
    {MAGIC_XE_NOT_SUPPORT_DATA_DEVICE, "Data device is not supported by a particular inference framework."},
    {MAGIC_XE_INVALID_BLOB, "Blob may have an invalid name, dimensions or type."},
    {MAGIC_XE_FORWARD_ERROR, "An error occurred during the forward pass of a neural network."},
    {MAGIC_XE_INVALID_MODEL_VERSION, "Current model version is too low."},
    {MAGIC_XE_NOT_SUPPORT_DEVICE, "The specified device or hardware is not supported by the platform."},
    {MAGIC_XE_INVALID_CONTEXT, "Context or environment is invalid or has not been properly initialized."},
    {MAGIC_XE_INVALID_COMMAND_QUEUE, "Command queue is invalid or has not been properly initialized."},
    {MAGIC_XE_INVALID_RUNTIME, "Runtime is invalid or has not been properly initialized."},
    {MAGIC_XE_DEVICE_NOT_FOUND, "The specified device ID is not found in the current platform."},
    {MAGIC_XE_OPENCL_ERROR, "OpenCL error has occurred."},
    {MAGIC_XE_OPENGL_ERROR, "OpenGL error has occurred."},
    {MAGIC_XE_CUDA_ERROR, "CUDA error has occurred."},
    {MAGIC_XE_DECODE_ERROR, "Decode error has occurred."},
    {MAGIC_XE_ENCODE_ERROR, "Encode error has occurred."},

    {MAGIC_XE_NOT_INIT_ENV, "Core not initialize."},
    {MAGIC_XE_INVALID_LICENSE, "License is illegal or corrupted."},

    {MAGIC_XE_PLUGIN_NOT_LOAD, "Plugin is not load."},
    {MAGIC_XE_PLUGIN_FAIL_LOAD, "Plugin is fail load."},

    {MAGIC_XE_INVALID_JSON, "Json string is invalid."},
    {MAGIC_XE_INVALID_PIPELINE, "Pipeline set is invalid."},
    {MAGIC_XE_INVALID_DEVICE, "Device info set is invalid."},
};

const char *MagicXEErrorStringGet(MagicXEError mgxe_err) {

    for (int i = 0; i < MAGIC_XE_ARRAY_ELEMS(error_entries); i++) {
        if (mgxe_err == error_entries[i].err) {
            return error_entries[i].str;
        }
    }
    return "Error type is not support";
}
