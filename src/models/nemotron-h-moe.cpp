#include "models.h"
#include "llama-kv-cache.h"

std::unique_ptr<llm_graph_context> llama_model_nemotron_h_moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// MTP draft head for Nemotron-H MoE
llama_model_nemotron_h_moe::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params), model(model), layer(model.layers[hparams.n_layer()]) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "NEMOTRON_H_MOE MTP currently supports a single MTP block");
    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm);
    GGML_ASSERT(layer.ffn_gate_inp);
    GGML_ASSERT(layer.nextn.shared_head_norm && "NEMOTRON_H_MOE MTP: missing final head norm");

    head_norm_w = layer.nextn.shared_head_norm;
    head_w      = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    head_s      = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w != nullptr && "NEMOTRON_H_MOE MTP requires an output projection");
    GGML_ASSERT(embd_weights() != nullptr && "NEMOTRON_H_MOE MTP requires token embeddings");

    const int il = hparams.n_layer();

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        tok_embd = ggml_get_rows(ctx0, embd_weights(), inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_embd = nullptr;
    if (cparams.mtp_chain == 0) {
        inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        ggml_set_input(inp->h);
        ggml_set_name(inp->h, "mtp_h_input");
        h_embd = inp->h;
    }
    ggml_tensor * inp_tokens = ubatch.token ? inp->tokens : nullptr;

    res->add_input(std::move(inp));

    auto * inp_attn = build_attn_inp_kv();

    if (cparams.mtp_chain > 0) {
        inp_attn_cur = inp_attn;
        build_mtp_round(*this, llm_graph_mtp_attn_of(inp_attn), tok_embd, inp_tokens, nullptr, cparams.mtp_chain);
        return;
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // attention fills KV over all tokens, but the MoE is position-wise: gather output rows before
    // it to save FFN compute (unless unmasked embeddings_nextn needs the full-length hidden state)
    const bool emit_h_nextn = cparams.embeddings_nextn;
    out_ids_cur = (inp_out_ids && (!emit_h_nextn || cparams.embeddings_nextn_masked)) ? inp_out_ids : nullptr;

    ggml_tensor * cur = build_step(inp_attn, tok_embd, h_embd, nullptr, -1, il);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!out_ids_cur && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

bool llama_model_nemotron_h_moe::has_mtp_round() const {
    return hparams.n_layer_nextn == 1;
}

ggml_tensor * llama_model_nemotron_h_moe::graph_mtp::mtp_step(
        ggml_tensor * tok_embd,
        ggml_tensor * h,
        ggml_tensor * pos,
                int   row,
               bool   kv_only,
        ggml_tensor * mask) {
    return build_step(inp_attn_cur, tok_embd, h, pos, row, hparams.n_layer(), kv_only, mask);
}

ggml_tensor * llama_model_nemotron_h_moe::graph_mtp::mtp_embd_weights() const {
    return embd_weights();
}

ggml_tensor * llama_model_nemotron_h_moe::graph_mtp::mtp_head(ggml_tensor * h) {
    return build_lora_mm(head_w, h, head_s);
}

ggml_tensor * llama_model_nemotron_h_moe::graph_mtp::embd_weights() const {
    return layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
}

// the MTP block: a dense NoPE attention sub-layer then a MoE FFN sub-layer
ggml_tensor * llama_model_nemotron_h_moe::graph_mtp::build_step(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * tok_embd,
                    ggml_tensor * h_embd,
                    ggml_tensor * inp_pos,
                            int   step,
                            int   il,
                           bool   kv_only,
                    ggml_tensor * mask_in) {
    GGML_UNUSED(inp_pos);
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_rows      = tok_embd->ne[1];

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    {
        auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, hparams.n_head(il), hparams.n_head_kv(il), il, n_rows);
        const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        if (step < 0) {
            cur = build_attn(inp_attn, layer.wo, layer.wo_b, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        } else {
            cur = build_attn_mtp(llm_graph_mtp_attn_of(inp_attn), layer.wo, layer.wo_b, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, kq_scale, il, step, kv_only, mask_in);
            if (!cur) {
                return nullptr;
            }
        }
        cb(cur, "mtp_attn_out", il);
    }

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    // gather the output rows here so the MoE FFN below only runs on the positions we keep
    if (out_ids_cur) {
        cur = ggml_get_rows(ctx0, cur, out_ids_cur);
    }

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    {
        ggml_tensor * router_logits = build_lora_mm(layer.ffn_gate_inp, cur);
        cb(router_logits, "mtp_ffn_moe_logits", il);

        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                nullptr, // no gate
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_RELU_SQR, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                il,
                router_logits, nullptr,
                layer.ffn_up_exps_s,
                nullptr, // no gate
                layer.ffn_down_exps_s);
        cb(moe_out, "mtp_ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   NULL, layer.ffn_up_shexp_s,
                NULL,                 NULL, NULL,
                layer.ffn_down_shexp, NULL, layer.ffn_down_shexp_s,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "mtp_ffn_out", il);
    }

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    // the MTP head has its own LayerNorm
    return build_norm(cur, head_norm_w, nullptr, LLM_NORM, -1);
}
