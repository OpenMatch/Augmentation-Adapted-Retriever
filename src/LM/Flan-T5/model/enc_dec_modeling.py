import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import mpu

from .configuration_enc_dec import EncDecConfig


def init_method_normal(std):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


class EncDecModel(nn.Module):
    
    def __init__(
        self,
        config: EncDecConfig,
        parallel_output=True,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
        prompt_config=None,
        args=None):
        
        super(EncDecModel, self).__init__()
        if config.vocab_size is None:
            raise RuntimeError("Should set vocab size")
        self.enc_config = copy.deepcopy(config)
        self.dec_config = copy.deepcopy(config)

        self.parallel_output = parallel_output

        init_method = init_method_normal(std=config.init_method_std) # NOTE: good?

        self.word_embeds = mpu.VocabParallelEmbedding(config.vocab_size, config.d_model, init_method=init_method)

        self.prompt_config = prompt_config
        
        self.args = args

        self.lm_head = mpu.VocabParallelEmbedding(config.vocab_size, config.d_model, init_method=init_method)

        self.encoder = mpu.ParallelTransformer(self.enc_config, word_embeds=self.word_embeds, is_decoder=False, prompt_config=prompt_config["enc"] if prompt_config is not None else None,
                                               checkpoint_activations=checkpoint_activations, checkpoint_num_layers=checkpoint_num_layers, args=args)
        self.decoder = mpu.ParallelTransformer(self.dec_config, word_embeds=self.word_embeds, is_decoder=True, prompt_config=prompt_config["dec"] if prompt_config is not None else None,
                                               checkpoint_activations=checkpoint_activations, checkpoint_num_layers=checkpoint_num_layers, args=args)

        if config.tie_weights:
            self.tie_weights()

    def init_prompt_embeds(self):
        self.encoder.init_prompt_embeds()
        self.decoder.init_prompt_embeds()

    def load_prompt_embeds(self, prompt_embeds):
        self.encoder.load_prompt_embeds(prompt_embeds)
        self.decoder.load_prompt_embeds(prompt_embeds)

    def get_prompt_embeds(self):
        return {
            "encoder": self.encoder.get_prompt(),
            "decoder": self.decoder.get_prompt()
        }

    def tie_weights(self):
        self.lm_head.weight = self.word_embeds.weight

    def reset_score_storage(self):
        for mod in self.decoder.blocks:
            mod.cross_attn.cross_attn.score_storage = None

    def get_crossattention_scores(self, context_mask):
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.blocks:
            scores.append(mod.cross_attn.cross_attn.score_storage)
        scores = torch.cat(scores, dim=2)
        # FiD n_layers beacuse dec seq = 1, auto regressive
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        # batch_size, 1, 1, n_passages, text_maxlength
        scores = scores.masked_fill(~context_mask[:, None, None], 0.).float()
        # batch_size, n_passages
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores / ntokens
        return scores

    def forward(
        self, 
        enc_input_ids=None,
        enc_position_ids=None,
        enc_attention_mask=None,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attention_mask=None,
        cross_attention_mask=None,
        enc_hidden_states=None,
        past_key_values=None,
        only_encoder=False,):

        provided_hidden = (enc_hidden_states is not None)

        if enc_hidden_states is None:
            enc_outputs = self.encoder(
                input_ids=enc_input_ids,
                attention_mask=enc_attention_mask,
            )

            enc_hidden_states = enc_outputs["last_hidden_state"]

        if only_encoder:
            outputs = {
                "encoder_last_hidden_state": enc_hidden_states,
                "encoder_hidden_states": enc_outputs["hidden_states"],
                "encoder_attentions": enc_outputs["attentions"],
            }

            return outputs

        dec_outputs = self.decoder(
            input_ids=dec_input_ids,
            attention_mask=dec_attention_mask,
            cross_attention_mask=cross_attention_mask,
            enc_hidden_states=enc_hidden_states,
            past_key_values=past_key_values,
        )

        last_hidden_state_parallel = mpu.copy_to_model_parallel_region(dec_outputs["last_hidden_state"])
        logits_parallel = F.linear(last_hidden_state_parallel, self.lm_head.weight)

        if self.parallel_output:
            lm_logits = logits_parallel
        else:
            lm_logits = mpu.gather_from_model_parallel_region(logits_parallel)

        outputs = {
            "lm_logits": lm_logits,
            "last_hidden_state": dec_outputs["last_hidden_state"],
            "past_key_values": dec_outputs["past_key_values"],
            "encoder_last_hidden_state": enc_hidden_states,
            "encoder_attentions": enc_outputs["attentions"] if not provided_hidden else None,
            "decoder_self_attentions": dec_outputs["attentions"],
            "decoder_cross_attentions": dec_outputs["cross_attentions"]
        }

        return outputs


def enc_dec_get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, nn.LayerNorm, mpu.transformer_enc_dec.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


def enc_dec_get_params_for_prompt_optimization(module: nn.Module):
    params = [{'params': []}]
    for t in module.named_modules():
        if "prompt" in t[0]:
            if torch.distributed.get_rank() == 0:
                print("Update params", t[0])
            params[0]['params'].extend([p for p in list(t[1]._parameters.values()) if p is not None])

    for t in module.named_parameters():
        if "prompt" not in t[0]:
            t[1].requires_grad_(False)

    if torch.distributed.get_rank() == 0:
        print("print params", params)
    return params

