import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import sys
from math import sqrt
from sklearn.decomposition import PCA
from peft import LoraConfig, get_peft_model
from transformers import GPT2Tokenizer, GPT2Config
# 获取当前文件所在目录的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 将上一级目录添加到 Python 的模块搜索路径中
sys.path.insert(0, parent_dir)
from Embed import DataEmbedding

os.environ['http_proxy'] = 'http://proxy_address:port'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class AccustumGPT2Model(GPT2Model):
    def accustum_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def forward(self, input_ids=None, labels=None, **kwargs):
        outputs = self.accustum_forward(input_ids, **kwargs)
        return outputs.last_hidden_state, outputs.hidden_states, outputs.hidden_states[-2], outputs.attentions # final feat, intermidiate feat

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    

    

class Res_block(nn.Module):
    def __init__(self, in_planes):
        super(Res_block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs
    
    




    
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.5):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
    



class Prompt(nn.Module):
    def __init__(self,  embed_dim=768,
                pool_size=48, top_k=4, wte = None):
        super().__init__()

        self.embed_dim = embed_dim
        self.top_k = top_k
        self.wte = wte

        # self.text_prototype_linear = nn.Linear(600, pool_size)
        self.text_prototype_linear = nn.Linear(500, pool_size)
        # self.prompt = nn.Parameter(torch.randn(pool_size, embed_dim),requires_grad=False)
        # nn.init.uniform_(self.prompt, -5, 5)
          
    
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        
        prompt_key = self.text_prototype_linear(self.wte.transpose(0, 1)).transpose(0, 1)
        prompt_norm = self.l2_normalize(prompt_key, dim=1)
        x_embed_norm = torch.mean(x_embed,dim=1)
        
        # prompt_norm = prompt_norm.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        
        _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
        
        batched_prompt_raw = prompt_key[idx] # B, top_k, length, C
 
        
    

        reduce_sim = torch.sum(similarity) / x_embed.shape[0] # Scalar

        
        return batched_prompt_raw, reduce_sim
    
    
class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=16, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.word = word_embedding
        # self.linear_word = nn.Linear(600, 484)
        self.linear_word = nn.Linear(500, 384)
        
    
        self.mask_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()       

    def forward(self, x):
        B = x.shape[0]
        if self.word.ndim == 2:
            self.word = self.word.repeat(B, 1, 1)
        elif self.word.shape[0] != B:
            self.word = self.word[0].repeat(B, 1, 1)
        
        self.word_embedding = (self.linear_word(self.word.transpose(1, 2))).transpose(1, 2)
        
        x = self.linear(x)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        x_time = x
        
        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)
        

        mask_input = torch.cat([x_time, x], dim=-1)  
        mask = self.sigmoid(self.mask_layer(mask_input)) 


        x = mask * x_time + (1 - mask) * x
        # x = x_time + x
        return x_time, x

        
class Model(nn.Module):

    def __init__(self, gpt_type='gpt2', d_ff=768, d_model=768, gpt_layers=6,
                 pred_len=4, prev_len=16, use_gpu=1, gpu_id=0, mlp=0, res_layers=4,
                 K=48, UQh=4, UQv=1, BQh=2, BQv=1,
                 patch_size=4, stride=1, res_dim=64,
                 embed='timeF', freq='h', dropout=0.1, lora = False):
        super(Model, self).__init__()
        self.device = torch.device('cuda:{}'.format(gpu_id))
        self.mlp = mlp
        self.res_layers = res_layers
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_ff = d_ff
        self.d_model = d_model

        self.K = K
        self.UQh = UQh
        self.UQv = UQv
        self.BQh = BQh
        self.BQv = BQv
        self.Nt = UQh * UQv
        self.Nr = BQh * BQv
        self.mul = prev_len * K * UQh * UQv * BQh * BQv
        self.enc_in = K * UQh * UQv * BQh * BQv
        self.c_out = K * UQh * UQv * BQh * BQv
    
        
        self.enc_embedding1 = DataEmbedding(2 * self.enc_in, self.d_model, embed, freq, dropout)

        if gpt_type == 'gpt2-medium':
            self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2-medium', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2-medium', output_attentions=True, output_hidden_states=True)
            self.gpt2_text.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1024

        elif gpt_type == 'gpt2-large':
            self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2-large', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1280
        elif gpt_type == 'gpt2-xl':
            self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2-xl', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2-xl', output_attentions=True, output_hidden_states=True)
            self.gpt2_text.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1600
        else:
            self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 768

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  
                param.requires_grad = True
            elif 'mlp' in name and mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

   
    
        self.lora_config = LoraConfig(
            inference_mode=False,        
            r=16,             
            lora_alpha=48,         
            lora_dropout=0.2,            
            target_modules=["c_attn", "c_proj", "mlp.c_fc", "mlp.c_proj"]      
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] 
        )


        if lora:
            self.gpt2 = get_peft_model(self.gpt2, self.lora_config)
        
        if use_gpu:
            device = torch.device('cuda:{}'.format(gpu_id))
            self.gpt2.to(device=device)
        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.prefix_length = 4
        self.predict_linear_pre = nn.Linear(self.prev_len, self.prev_len)
        
        self.out_layer_dim_time = nn.Linear(d_ff, self.c_out * 2)
 
        
        self.output_layer_time = nn.Sequential(
            nn.Linear(self.prefix_length+self.prev_len, self.pred_len)
        )


        self.RB_e = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))

        for i in range(self.res_layers):
            self.RB_e.append(Res_block(res_dim))
 
        self.RB_e.append(nn.Conv2d(res_dim, 2, 3, 1, 1))

      
        
        wte = self.gpt2.wte.state_dict()['weight'].cpu().numpy()
        
        pca = PCA(n_components=500)

        self.pca_weights = torch.tensor((pca.fit_transform(wte.T)).T).to(self.device)
        self.in_layer = Encoder_PCA(self.gpt_dim, self.pca_weights, hidden_dim=self.gpt_dim)
        
        self.prompt_model = Prompt(embed_dim=self.d_model, pool_size=100, top_k=self.prefix_length, wte=self.pca_weights)
        # self.soft_prompt = nn.Parameter(torch.randn(self.prefix_length, self.gpt_dim))  # 可学习参数
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,  mask=None, output_attentions=False):
        
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        B, L, enc_in = x_enc.shape  # [B, L, D]
        
        # soft_prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)
    


        x_enc_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc_fre = self.patch_layer(x_enc_fre.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_fre = x_enc_fre.reshape(B, L, enc_in)
        x_enc_fre = rearrange(x_enc_fre, 'b l (k o) -> b o l k', o=2)
        x_enc = self.RB_e(x_enc_fre)
      

        x_enc = x_enc_fre #+ x_enc_delay
        
        x_enc = rearrange(x_enc, 'b o l k -> b l (k o)', o=2)  # [B, L, D]
      
        enc_out = self.enc_embedding1(x_enc, x_mark_enc)  # [B, L, 768]

        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        
        
        _, outputs_text1 = self.in_layer(enc_out)

        time_prompt, reduce_sim = self.prompt_model(outputs_text1)

       
        outputs_time1 = torch.cat([time_prompt, outputs_text1], dim=1)  # [batch_size, seq_len + len, dim]tdd和fdd场景下拼接顺序不同

        
        outputs_time, intermidiate_feat_time, second_last_hidden_state_time, attentions  = self.gpt2(inputs_embeds=outputs_time1, output_hidden_states=True,
                            output_attentions=True)
       
        # residue connection
        outputs_time += outputs_time1
        

        
        outputs_time = outputs_time[:, :, :self.d_ff]

        outputs_time2 = self.out_layer_dim_time(outputs_time)
        outputs_time = self.output_layer_time(outputs_time2.permute(0, 2, 1)).permute(0, 2, 1)

        outputs_time = outputs_time * std + mean
        
        return  {

            'outputs_time':outputs_time[:, -self.pred_len:, :],
            'reduce_sim' : reduce_sim,
            'last_hidden': outputs_time1,
            'penultimate_hidden':second_last_hidden_state_time, 
            'attentions': attentions
        }

