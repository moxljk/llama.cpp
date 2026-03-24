#include "ggml.h"
#include "models.h"

llm_build_qwen3zip::llm_build_qwen3zip(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto build_zipserv_mm = [&](const llama_block_zipserv & block, ggml_tensor * input) -> ggml_tensor * {
        GGML_ASSERT(block.cols > 0);
        GGML_ASSERT(input->ne[0] == (int64_t) block.cols);
        int32_t op_params[ZIPSERV_OPPARAM_COUNT];
        op_params[ZIPSERV_OPPARAM_MAX_HIGH_FREQ_COUNT_IDX] = (int32_t) block.max_high_freq_count;
        op_params[ZIPSERV_OPPARAM_MAX_FULL_COUNT_IDX]      = (int32_t) block.max_full_count;
        op_params[ZIPSERV_OPPARAM_START_EXP_IDX]           = (int32_t) block.start_exp;

        return ggml_mul_mat_zipserv(
                ctx0,
                block.sign_mantissa,
                block.compressed_full,
                block.bitmap1,
                block.bitmap2,
                block.bitmap3,
                block.tile_offsets_median,
                block.tile_offsets_global,
                input,
                op_params,
                (int) block.rows);
    };

    auto build_zipserv_ffn = [&](const llama_layer & layer, ggml_tensor * input, int il) -> ggml_tensor * {
        ggml_tensor * up = build_zipserv_mm(layer.ffn_up_zipserv, input);
        cb(up, "ffn_up", il);

        ggml_tensor * gate = build_zipserv_mm(layer.ffn_gate_zipserv, input);
        cb(gate, "ffn_gate", il);

        ggml_tensor * act = ggml_swiglu_split(ctx0, gate, up);
        cb(act, "ffn_swiglu", il);

        return build_zipserv_mm(layer.ffn_down_zipserv, act);
    };

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_zipserv_mm(model.layers[il].wq_zipserv, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_zipserv_mm(model.layers[il].wk_zipserv, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_zipserv_mm(model.layers[il].wv_zipserv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            cur = build_zipserv_mm(model.layers[il].wo_zipserv, cur);
            
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_zipserv_ffn(model.layers[il], cur, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
