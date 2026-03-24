#include "llama-zipserv.h"

#include "llama-impl.h"
#include "llama-model-loader.h"

LLM_BN_ZIPSERV_IMPL::LLM_BN_ZIPSERV_IMPL(const LLM_TN_IMPL & base)
    : base(base) {}

std::string LLM_BN_ZIPSERV_IMPL::tensor_name(const char * field) const {
    return format("%s.%s", base.str().c_str(), field);
}

std::string llama_zipserv_meta_key(const std::string & base_name, const char * field) {
    return format("zipserv.%s.%s", base_name.c_str(), field);
}

llama_zipserv_meta llama_zipserv_load_meta(llama_model_loader & ml, const std::string & base_name, bool required) {
    llama_zipserv_meta meta;

    ml.get_key(llama_zipserv_meta_key(base_name, "rows"), meta.rows, required);
    ml.get_key(llama_zipserv_meta_key(base_name, "cols"), meta.cols, required);

    ml.get_key(llama_zipserv_meta_key(base_name, "sign_mantissa_size"),       meta.sign_mantissa_size,       required);
    ml.get_key(llama_zipserv_meta_key(base_name, "compressed_full_size"),     meta.compressed_full_size,     required);
    ml.get_key(llama_zipserv_meta_key(base_name, "bitmap1_size"),             meta.bitmap1_size,             required);
    ml.get_key(llama_zipserv_meta_key(base_name, "bitmap2_size"),             meta.bitmap2_size,             required);
    ml.get_key(llama_zipserv_meta_key(base_name, "bitmap3_size"),             meta.bitmap3_size,             required);
    ml.get_key(llama_zipserv_meta_key(base_name, "tile_offsets_median_size"), meta.tile_offsets_median_size, required);
    ml.get_key(llama_zipserv_meta_key(base_name, "tile_offsets_global_size"), meta.tile_offsets_global_size, required);

    ml.get_key(llama_zipserv_meta_key(base_name, "max_high_freq_count"), meta.max_high_freq_count, required);
    ml.get_key(llama_zipserv_meta_key(base_name, "max_full_count"),      meta.max_full_count,      required);
    ml.get_key(llama_zipserv_meta_key(base_name, "start_exp"),           meta.start_exp,           required);

    return meta;
}
