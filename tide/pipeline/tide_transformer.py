from typing import Any, Dict, Optional
import math
from functools import partial

import matplotlib
matplotlib.use('Agg')

import torch
from torch import nn

from diffusers.models import PixArtTransformer2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .transformer_blocks import BasicTransformerTIDEBlock
from .layers import ResidualMLP

class PixArtSpecialAttnTransformerModel(PixArtTransformer2DModel):
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 72,
            in_channels: int = 4,
            out_channels: Optional[int] = 8,
            num_layers: int = 28,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = 1152,
            attention_bias: bool = True,
            sample_size: int = 128,
            patch_size: int = 2,
            activation_fn: str = "gelu-approximate",
            num_embeds_ada_norm: Optional[int] = 1000,
            upcast_attention: bool = False,
            norm_type: str = "ada_norm_single",
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-6,
            interpolation_scale: Optional[int] = None,
            use_additional_conditions: Optional[bool] = None,
            caption_channels: Optional[int] = None,
            attention_type: Optional[str] = "default",
    ):
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            num_layers,
            dropout,
            norm_num_groups,
            cross_attention_dim,
            attention_bias,
            sample_size,
            patch_size,
            activation_fn,
            num_embeds_ada_norm,
            upcast_attention,
            norm_type,
            norm_elementwise_affine,
            norm_eps,
            interpolation_scale,
            use_additional_conditions,
            caption_channels,
            attention_type,
        )

        # Validate inputs.
        if norm_type != "ada_norm_single":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_single" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerTIDEBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.num_layers = num_layers

    def _replace_in_out_proj_conv(self):
        # replace the in_proj layer to accept 8 in_channels
        _in_weight = self.pos_embed.proj.weight.clone()  # [320, 4, 3, 3]
        _in_bias = self.pos_embed.proj.bias.clone()  # [320]
        _in_weight = _in_weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _in_weight *= 0.5
        # new conv_in channel
        _n_convin_out_channel = self.pos_embed.proj.out_channels
        _new_conv_in = nn.Conv2d(
            8, _n_convin_out_channel, kernel_size=(2, 2), stride=(2, 2)
        )
        _new_conv_in.weight = nn.Parameter(_in_weight)
        _new_conv_in.bias = nn.Parameter(_in_bias)
        self.pos_embed.proj = _new_conv_in

        self.register_to_config(in_channels=8)
        return

    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())

        # 计算可学习参数的总数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 打印可学习参数的占比
        print(f"trainable params: {trainable_params}")
        print(f"all params: {total_params}")
        print(f"trainable%: {trainable_params / total_params * 100:.2f}%")

class MiniTransformerModel(PixArtTransformer2DModel):
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 72,
            in_channels: int = 4,
            out_channels: Optional[int] = 8,
            num_layers: int = 28,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = 1152,
            attention_bias: bool = True,
            sample_size: int = 128,
            patch_size: int = 2,
            activation_fn: str = "gelu-approximate",
            num_embeds_ada_norm: Optional[int] = 1000,
            upcast_attention: bool = False,
            norm_type: str = "ada_norm_single",
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-6,
            interpolation_scale: Optional[int] = None,
            use_additional_conditions: Optional[bool] = None,
            caption_channels: Optional[int] = None,
            attention_type: Optional[str] = "default",
    ):
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            num_layers,
            dropout,
            norm_num_groups,
            cross_attention_dim,
            attention_bias,
            sample_size,
            patch_size,
            activation_fn,
            num_embeds_ada_norm,
            upcast_attention,
            norm_type,
            norm_elementwise_affine,
            norm_eps,
            interpolation_scale,
            use_additional_conditions,
            caption_channels,
            attention_type,
        )

        # Validate inputs.
        if norm_type != "ada_norm_single":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_single" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerTIDEBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )




    def _clip_transformer_layers(self, num_layers):
        self.transformer_blocks = self.transformer_blocks[:num_layers]
        self.target_ids = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        self.register_to_config(num_layers=num_layers)

    def _map_clip_transformer_layers(self, num_layers, map=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]):
        self.transformer_blocks = nn.ModuleList([self.transformer_blocks[i] for i in map])
        self.target_ids = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        self.register_to_config(num_layers=num_layers)


    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())

        # 计算可学习参数的总数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 打印可学习参数的占比
        print(f"trainable params: {trainable_params}")
        print(f"all params: {total_params}")
        print(f"trainable%: {trainable_params / total_params * 100:.2f}%")

class TAN(nn.Module):
    def __init__(self, nhidden=1152, hidden_dim=256, ks=1, time_adaptive=True):
        super().__init__()

        pw = ks // 2
        affine_func = partial(nn.Conv2d, kernel_size=ks, padding=pw)
        self.gamma_mlp = ResidualMLP(
            input_dim=nhidden,
            hidden_dim=hidden_dim,
            output_dim=nhidden,
            num_mlp=1,
            num_layer_per_mlp=3,
            affine_func=affine_func
        )
        self.beta_mlp = ResidualMLP(
            input_dim=nhidden,
            hidden_dim=hidden_dim,
            output_dim=nhidden,
            num_mlp=1,
            num_layer_per_mlp=3,
            affine_func=affine_func
        )
        self.time_adaptive_scale = None

        if time_adaptive:
            self.time_adaptive_scale = nn.Sequential(
                nn.Linear(6*nhidden, 1),
                nn.Sigmoid(),
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)

    def forward(self, x, timestep, modal1_feats, modal2_feats=None):
        b, l, c = x.shape
        h = w = int(math.sqrt(l))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        modal1_feats = modal1_feats.reshape(b, h, w, c).permute(0, 3, 1, 2)

        if modal2_feats is not None:
            modal2_feats = modal2_feats.reshape(b, h, w, c).permute(0, 3, 1, 2)

            gamma1, beta1 = self._forward(modal1_feats)
            gamma2, beta2 = self._forward(modal2_feats)

            gamma = (gamma1 + gamma2) / 2
            beta = (beta1 + beta2) / 2
        else:
            gamma = self.gamma_mlp(modal1_feats)
            beta = self.beta_mlp(modal1_feats)

        if self.time_adaptive_scale is not None:
            assert timestep is not None
            sigma = self.time_adaptive_scale(timestep)
            sigma = sigma.unsqueeze(-1).unsqueeze(-1)
            out = x * (1 + sigma * gamma) + sigma * beta
        else:
            out = x * (1 + gamma) + beta

        out = out.permute(0, 2, 3, 1).reshape(b, -1, c)
        return out

    def _forward(self, modal_feats):
        gamma = self.gamma_mlp(modal_feats)
        beta = self.beta_mlp(modal_feats)
        return gamma, beta

class TIDE_TANs(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, num_layers=10, time_adaptive=True) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.DM2I_joint_tan_blocks = nn.ModuleList(
            [
                TAN(time_adaptive=time_adaptive)
                for i in range(num_layers)
            ]
        )

        self.D2M_tan_blocks = nn.ModuleList(
            [
                TAN(time_adaptive=time_adaptive)
                for i in range(num_layers)
            ]
        )

        self.M2D_tan_blocks = nn.ModuleList(
            [
                TAN(time_adaptive=time_adaptive)
                for i in range(num_layers)
            ]
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def train(self, mode: bool = True):
        for block in self.DM2I_joint_tan_blocks:
            block.train(mode)
        for block in self.D2M_tan_blocks:
            block.train(mode)
        for block in self.M2D_tan_blocks:
            block.train(mode)

class TIDETransformerModel(ModelMixin, ConfigMixin):
    def __init__(
            self,
            transformer_image: PixArtSpecialAttnTransformerModel,
            transformer_depth: MiniTransformerModel,
            transformer_mask: MiniTransformerModel,
            featctrol_modules: TIDE_TANs,
            training=False
    ):
        super().__init__()

        self.gradient_checkpointing = False
        self.register_to_config(**transformer_image.config)
        # self.register_to_config(**transformer_l2i.config)
        self.register_to_config(**featctrol_modules.config)
        self.training = training

        self.transformer_image = transformer_image
        self.transformer_depth = transformer_depth
        self.transformer_mask = transformer_mask
        self.featctrol_modules = featctrol_modules

        self.mini_blocks_num = transformer_depth.num_layers
        # self.target_ids = transformer_depth.target_ids
        self.target_ids = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        self.blocks_num = featctrol_modules.num_layers

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_input(
            self,
            transformer,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        if transformer.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // transformer.config.patch_size,
            hidden_states.shape[-1] // transformer.config.patch_size,
        )
        with torch.cuda.amp.autocast():
            hidden_states = transformer.pos_embed(hidden_states)

        timestep, embedded_timestep = transformer.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if transformer.caption_projection is not None:
            encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        #
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        return (
            (height, width),
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            embedded_timestep
        )

    def forward(
            self,
            image_hidden_states: torch.Tensor,
            depth_hidden_states: torch.Tensor,
            mask_hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):

        (
            (height_image, width_image),
            hidden_states_image,
            attention_mask_image,
            encoder_hidden_states_image,
            encoder_attention_mask_image,
            timestep_image,
            embedded_timestep_image
        ) = self.get_input(
            self.transformer_image,
            image_hidden_states,
            encoder_hidden_states,
            timestep,
            added_cond_kwargs,
            attention_mask,
            encoder_attention_mask
        )
        with torch.cuda.amp.autocast():
            (
                (height_depth, width_depth),
                hidden_states_depth,
                attention_mask_depth,
                encoder_hidden_states_depth,
                encoder_attention_mask_depth,
                timestep_depth,
                embedded_timestep_depth
            ) = self.get_input(
                self.transformer_depth,
                depth_hidden_states,
                encoder_hidden_states,
                timestep,
                added_cond_kwargs,
                attention_mask,
                encoder_attention_mask
            )

            (
                (height_mask, width_mask),
                hidden_states_mask,
                attention_mask_mask,
                encoder_hidden_states_mask,
                encoder_attention_mask_mask,
                timestep_mask,
                embedded_timestep_mask
            ) = self.get_input(
                self.transformer_mask,
                mask_hidden_states,
                encoder_hidden_states,
                timestep,
                added_cond_kwargs,
                attention_mask,
                encoder_attention_mask
            )

        # 2. Blocks
        for block_index, transformer_image_block in enumerate(self.transformer_image.transformer_blocks):
            hidden_states_image, cross_attn_map = transformer_image_block(
                hidden_states_image,
                attention_mask=attention_mask_image,
                encoder_hidden_states=encoder_hidden_states_image,
                encoder_attention_mask=encoder_attention_mask_image,
                timestep=timestep_image,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )
            with torch.cuda.amp.autocast():
                if block_index in self.target_ids:
                    id = self.target_ids.index(block_index)

                    hidden_states_mask = self.featctrol_modules.D2M_tan_blocks[id](
                        hidden_states_mask, timestep_mask, hidden_states_depth
                    )

                    hidden_states_depth, _ = self.transformer_depth.transformer_blocks[id](
                        hidden_states_depth,
                        attention_mask=attention_mask_depth,
                        encoder_hidden_states=encoder_hidden_states_depth,
                        encoder_attention_mask=encoder_attention_mask_depth,
                        timestep=timestep_depth,
                        cross_attn_map=cross_attn_map,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=None,
                    )

                    hidden_states_mask, _ = self.transformer_mask.transformer_blocks[id](
                        hidden_states_mask,
                        attention_mask=attention_mask_mask,
                        encoder_hidden_states=encoder_hidden_states_mask,
                        encoder_attention_mask=encoder_attention_mask_mask,
                        timestep=timestep_mask,
                        cross_attn_map=cross_attn_map,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=None,
                    )

                    hidden_states_depth = self.featctrol_modules.M2D_tan_blocks[id](
                        hidden_states_depth, timestep_depth, hidden_states_mask
                    )

                    # feedback to image
                    hidden_states_image = self.featctrol_modules.DM2I_joint_tan_blocks[id](
                        hidden_states_image, timestep_image, hidden_states_depth, hidden_states_mask
                    )



        # 3. Output
        image_noise_output = self.output(
            self.transformer_image,
            hidden_states_image,
            embedded_timestep_image,
            height_image,
            width_image,
            return_dict
        )
        depth_noise_output = self.output(
            self.transformer_depth,
            hidden_states_depth,
            embedded_timestep_depth,
            height_depth,
            width_depth,
            return_dict
        )
        mask_noise_output = self.output(
            self.transformer_mask,
            hidden_states_mask,
            embedded_timestep_mask,
            height_mask,
            width_mask,
            return_dict
        )
        return image_noise_output[0], depth_noise_output[0], mask_noise_output[0]

    def output(
            self,
            transformer,
            hidden_states,
            embedded_timestep,
            height,
            width,
            return_dict
    ):
        shift, scale = (
                transformer.scale_shift_table[None] + embedded_timestep[:, None].to(
            transformer.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = transformer.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        with torch.cuda.amp.autocast():
            hidden_states = transformer.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, transformer.config.patch_size, transformer.config.patch_size,
                   transformer.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, transformer.out_channels, height * transformer.config.patch_size,
                   width * transformer.config.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
