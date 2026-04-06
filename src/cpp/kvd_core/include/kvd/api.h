#pragma once

#include <stddef.h>

#if defined(_WIN32)
#if defined(KVD_BUILD_DLL)
#define KVD_API __declspec(dllexport)
#else
#define KVD_API __declspec(dllimport)
#endif
#else
#define KVD_API
#endif

#if defined(_WIN32)
#define KVD_CALL __cdecl
#else
#define KVD_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kvd_handle kvd_handle;

typedef struct kvd_config {
    const char* model_path;
    const char* model_normal_path;
    const char* model_packed_path;
    const char* family_classifier_json_path;
    const char* allowed_scan_root;
    unsigned int max_file_size;
    float prediction_threshold;
    const char* onnx_model_path;
    const char* onnx_model_normal_path;
    const char* onnx_model_packed_path;
} kvd_config;

typedef enum kvd_model_check_result {
    KVD_MODEL_OK = 0,
    KVD_MODEL_ERR_INVALID_ARGUMENT = -1,
    KVD_MODEL_ERR_MAIN_MISSING = -10,
    KVD_MODEL_ERR_MAIN_INVALID = -11,
    KVD_MODEL_ERR_ROUTE_INCOMPLETE = -12,
    KVD_MODEL_ERR_NORMAL_MISSING = -13,
    KVD_MODEL_ERR_NORMAL_INVALID = -14,
    KVD_MODEL_ERR_PACKED_MISSING = -15,
    KVD_MODEL_ERR_PACKED_INVALID = -16,
    KVD_MODEL_ERR_FAMILY_MISSING = -17,
    KVD_MODEL_ERR_FAMILY_INVALID = -18,
    KVD_MODEL_ERR_HARDCASE_MANIFEST_MISSING = -19,
    KVD_MODEL_ERR_HARDCASE_MANIFEST_INVALID = -20,
    KVD_MODEL_ERR_HARDCASE_MODEL_MISSING = -21,
    KVD_MODEL_ERR_HARDCASE_MODEL_INVALID = -22,
    KVD_MODEL_ERR_ONNX_MAIN_MISSING = -30,
    KVD_MODEL_ERR_ONNX_MAIN_INVALID = -31,
    KVD_MODEL_ERR_ONNX_NORMAL_MISSING = -32,
    KVD_MODEL_ERR_ONNX_NORMAL_INVALID = -33,
    KVD_MODEL_ERR_ONNX_PACKED_MISSING = -34,
    KVD_MODEL_ERR_ONNX_PACKED_INVALID = -35,
    KVD_MODEL_ERR_OOM = -100
} kvd_model_check_result;

KVD_API kvd_handle* KVD_CALL kvd_create(const kvd_config* config);
KVD_API void KVD_CALL kvd_destroy(kvd_handle* handle);

KVD_API int KVD_CALL kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len);
KVD_API int KVD_CALL kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json,
                                    size_t* out_len);
KVD_API int KVD_CALL kvd_scan_paths(kvd_handle* handle, const char** paths, size_t count, char** out_json,
                                    size_t* out_len);
KVD_API int KVD_CALL kvd_train_path(kvd_handle* handle, const char* path, int label, char** out_json, size_t* out_len);
KVD_API int KVD_CALL kvd_train_paths(kvd_handle* handle, const char** paths, size_t count, int label, char** out_json,
                                     size_t* out_len);
KVD_API int KVD_CALL kvd_train_from_path(kvd_handle* handle, const char* path, int label, char** out_json,
                                         size_t* out_len);
KVD_API void KVD_CALL kvd_signature_flush(kvd_handle* handle);
KVD_API void KVD_CALL kvd_free(char* p);
KVD_API int KVD_CALL kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len);
KVD_API int KVD_CALL kvd_extract_pe_features(const char* path, float* out_features, size_t out_len);
KVD_API int KVD_CALL kvd_extract_pe_features_batch(const char** paths, size_t count, float* out_features,
                                                   size_t feature_dim, int* out_status, unsigned int thread_count);

KVD_API size_t KVD_CALL kvd_get_pe_feature_dimension(void);

#ifdef __cplusplus
}
#endif
