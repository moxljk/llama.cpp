#pragma once

#include "llama-arch.h"
#include <cstdint>
#include <string>

struct llama_model_loader;

struct llama_zipserv_meta {
    uint32_t rows = 0;
    uint32_t cols = 0;

    uint32_t sign_mantissa_size       = 0;
    uint32_t compressed_full_size     = 0;
    uint32_t bitmap1_size             = 0;
    uint32_t bitmap2_size             = 0;
    uint32_t bitmap3_size             = 0;
    uint32_t tile_offsets_median_size = 0;
    uint32_t tile_offsets_global_size = 0;

    uint32_t max_high_freq_count = 0;
    uint32_t max_full_count      = 0;
    uint32_t start_exp           = 0;
};

struct LLM_BN_ZIPSERV_IMPL {
    const LLM_TN_IMPL base;

    explicit LLM_BN_ZIPSERV_IMPL(const LLM_TN_IMPL & base);

    std::string tensor_name(const char * field) const;
};

std::string llama_zipserv_meta_key(const std::string & base_name, const char * field);
llama_zipserv_meta llama_zipserv_load_meta(llama_model_loader & ml, const std::string & base_name, bool required = true);
