import argparse
import glob
import importlib
import itertools
import os
import random
import re
import time
from typing import List, Union, Optional

import PIL
import diffusers
import numpy as np
import torch
import torchvision
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, PNDMScheduler, LMSDiscreteScheduler, \
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, \
    HeunDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler
from einops import rearrange
from transformers import CLIPModel

from XTI_hijack import unet_forward_XTI, downblock_forward_XTI, upblock_forward_XTI
from library import model_util, train_util
from sinri.drawer import FlashAttentionFunction
from sinri.drawer.BatchData import BatchData
from sinri.drawer.BatchDataBase import BatchDataBase
from sinri.drawer.BatchDataExt import BatchDataExt
from sinri.drawer.DrawerMeta import HighResFixUpscalerMeta, ClipMeta, ControlNetMeta, VaeMeta, ModelMeta, PromptMeta, \
    Vgg16Meta, InteractiveMeta, Img2ImgMeta, OutputMeta, NetworkMeta, BatchDrawMeta, TextualInversionMeta
from sinri.drawer.NoiseManager import NoiseManager
from sinri.drawer.PipelineLike import PipelineLike
from sinri.drawer.TorchRandReplacer import TorchRandReplacer
from tools import original_control_net
from tools.original_control_net import ControlNetInfo


class KohyaSSImageGenerator:
    CLIP_MODEL_PATH = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

    # scheduler:
    SCHEDULER_LINEAR_START = 0.00085
    SCHEDULER_LINEAR_END = 0.0120
    SCHEDULER_TIMESTEPS = 1000
    SCHEDLER_SCHEDULE = "scaled_linear"

    # その他の設定
    LATENT_CHANNELS = 4
    DOWNSAMPLING_FACTOR = 8

    def __init__(self):
        pass

    def __build_args(self, **kwargs) -> argparse.Namespace:
        """
        tokenizer_cache_dir
        """
        # todo seems ok
        return argparse.Namespace(**kwargs)

    def main(
            self,
            model_meta: ModelMeta,
            W: int,
            H: int,
            prompt_meta: PromptMeta,
            output_meta: OutputMeta,
            sampler: str,
            steps,
            vgg_meta: Vgg16Meta = Vgg16Meta(),
            vae_meta: VaeMeta = VaeMeta(),
            control_net_meta: ControlNetMeta = ControlNetMeta(),
            clip_meta: ClipMeta = ClipMeta(),
            highres_fix_upscaler_meta: HighResFixUpscalerMeta = HighResFixUpscalerMeta(),
            interactive_meta: InteractiveMeta = InteractiveMeta(),
            img2img_meta: Img2ImgMeta = Img2ImgMeta(),
            network_meta: NetworkMeta = NetworkMeta(),
            batch_draw_meta: BatchDrawMeta = BatchDrawMeta(),
            textual_inversion_meta: TextualInversionMeta = TextualInversionMeta(),
            seed: Optional[int] = None,

            diffusers_xformers: bool = False,
            xformers: bool = False,
            opt_channels_last: bool = False,

            fp16: bool = False,
            bf16: bool = False,
            # such as torch.float32,

            tokenizer_cache_dir: Optional[str] = None,
    ):
        """

        Args:
            model_meta:
            W: image width, in pixel space / 生成画像幅
            H: image height, in pixel space / 生成画像高さ
            prompt_meta:
            output_meta:
            sampler: sampler (scheduler) type / サンプラー（スケジューラ）の種類
            seed: seed, or seed of seeds in multiple generation / 1枚生成時のseed、または複数枚生成時の乱数seedを決めるためのseed
            steps: number of ddim sampling steps / サンプリングステップ数
            vgg_meta:
            vae_meta:
            control_net_meta:
            clip_meta:
            highres_fix_upscaler_meta:
            interactive_meta:
            img2img_meta:
            network_meta:
            batch_draw_meta:
            textual_inversion_meta:
            diffusers_xformers: use xformers by diffusers (Hypernetworks doesn't work) / Diffusersでxformersを使用する（Hypernetwork利用不可）
            xformers: use xformers / xformersを使用し高速化する
            opt_channels_last: set channels last option to model / モデルにchannels lastを指定し最適化する
            fp16: use fp16 / fp16を指定し省メモリ化する
            bf16: use bfloat16 / bfloat16を指定し省メモリ化する

        Returns:

        """
        parameters = {
            'v2':model_meta.v2,
            'v_parameterization':model_meta.v_parameterization,
            'prompt':prompt_meta.prompt,
            'from_file':prompt_meta.from_file,
            'interactive':interactive_meta.interactive,
            'no_preview':interactive_meta.no_preview,
            'image_path':img2img_meta.image_path,
            'mask_path':img2img_meta.mask_path,
        }
        built_args = argparse.Namespace(**parameters)

        if fp16:
            dtype = torch.float16
        elif bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        highres_fix = highres_fix_upscaler_meta.highres_fix_scale is not None
        # assert not highres_fix or args.image_path is None, f"highres_fix doesn't work with img2img / highres_fixはimg2imgと同時に使えません"

        if model_meta.v_parameterization and not model_meta.v2:
            print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
        if model_meta.v2 and clip_meta.clip_skip is not None:
            print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

        # モデルを読み込む
        ckpt = model_meta.ckpt
        if not os.path.isfile(ckpt):  # ファイルがないならパターンで探し、一つだけ該当すればそれを使う
            files = glob.glob(ckpt)
            if len(files) == 1:
                ckpt = files[0]

        use_stable_diffusion_format = os.path.isfile(ckpt)
        if use_stable_diffusion_format:
            print("load StableDiffusion checkpoint")
            text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(model_meta.v2, ckpt)
        else:
            print("load Diffusers pretrained models")
            loading_pipe = StableDiffusionPipeline.from_pretrained(ckpt, safety_checker=None, torch_dtype=dtype)
            text_encoder = loading_pipe.text_encoder
            vae = loading_pipe.vae
            unet = loading_pipe.unet
            tokenizer = loading_pipe.tokenizer
            del loading_pipe

        # VAEを読み込む
        if vae is not None:
            vae = model_util.load_vae(vae, dtype)
            print("additional VAE loaded")

        # # 置換するCLIPを読み込む
        # if args.replace_clip_l14_336:
        #   text_encoder = load_clip_l14_336(dtype)
        #   print(f"large clip {CLIP_ID_L14_336} is loaded")

        if clip_meta.clip_guidance_scale > 0.0 or clip_meta.clip_image_guidance_scale:
            print("prepare clip model")
            clip_model = CLIPModel.from_pretrained(self.CLIP_MODEL_PATH, torch_dtype=dtype)
        else:
            clip_model = None

        if vgg_meta.vgg16_guidance_scale > 0.0:
            print("prepare resnet model")
            vgg16_model = torchvision.models.vgg16(torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg16_model = None

        # xformers、Hypernetwork対応
        if not diffusers_xformers:
            KohyaSSImageGenerator.replace_unet_modules(unet, not xformers, xformers)
            KohyaSSImageGenerator.replace_vae_modules(vae, not xformers, xformers)

        # tokenizerを読み込む
        print("loading tokenizer")
        if use_stable_diffusion_format:
            tokenizer = train_util.load_tokenizer(built_args)

        # schedulerを用意する
        sched_init_args = {}
        scheduler_num_noises_per_step = 1
        if sampler == "ddim":
            scheduler_cls = DDIMScheduler
            scheduler_module = diffusers.schedulers.scheduling_ddim
        elif sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
            scheduler_cls = DDPMScheduler
            scheduler_module = diffusers.schedulers.scheduling_ddpm
        elif sampler == "pndm":
            scheduler_cls = PNDMScheduler
            scheduler_module = diffusers.schedulers.scheduling_pndm
        elif sampler == "lms" or sampler == "k_lms":
            scheduler_cls = LMSDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_lms_discrete
        elif sampler == "euler" or sampler == "k_euler":
            scheduler_cls = EulerDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_euler_discrete
        elif sampler == "euler_a" or sampler == "k_euler_a":
            scheduler_cls = EulerAncestralDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
        elif sampler == "dpmsolver" or sampler == "dpmsolver++":
            scheduler_cls = DPMSolverMultistepScheduler
            sched_init_args["algorithm_type"] = sampler
            scheduler_module = diffusers.schedulers.scheduling_dpmsolver_multistep
        elif sampler == "dpmsingle":
            scheduler_cls = DPMSolverSinglestepScheduler
            scheduler_module = diffusers.schedulers.scheduling_dpmsolver_singlestep
        elif sampler == "heun":
            scheduler_cls = HeunDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_heun_discrete
        elif sampler == "dpm_2" or sampler == "k_dpm_2":
            scheduler_cls = KDPM2DiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_discrete
        elif sampler == "dpm_2_a" or sampler == "k_dpm_2_a":
            scheduler_cls = KDPM2AncestralDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete
            scheduler_num_noises_per_step = 2
        else:
            raise Exception('unknown scheduler')

        if model_meta.v_parameterization:
            sched_init_args["prediction_type"] = "v_prediction"

        # samplerの乱数をあらかじめ指定するための処理

        noise_manager = NoiseManager()
        if scheduler_module is not None:
            scheduler_module.torch = TorchRandReplacer(noise_manager)

        scheduler = scheduler_cls(
            num_train_timesteps=self.SCHEDULER_TIMESTEPS,
            beta_start=self.SCHEDULER_LINEAR_START,
            beta_end=self.SCHEDULER_LINEAR_END,
            beta_schedule=self.SCHEDLER_SCHEDULE,
            **sched_init_args,
        )

        # clip_sample=Trueにする
        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
            print("set clip_sample to True")
            scheduler.config.clip_sample = True

        # deviceを決定する
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "mps"を考量してない

        # custom pipelineをコピったやつを生成する
        if vae_meta.vae_slices:
            from library.slicing_vae import SlicingAutoencoderKL

            sli_vae = SlicingAutoencoderKL(
                act_fn="silu",
                block_out_channels=(128, 256, 512, 512),
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",
                                  "DownEncoderBlock2D"],
                in_channels=3,
                latent_channels=4,
                layers_per_block=2,
                norm_num_groups=32,
                out_channels=3,
                sample_size=512,
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                num_slices=vae_meta.vae_slices,
            )
            sli_vae.load_state_dict(vae.state_dict())  # vaeのパラメータをコピーする
            vae = sli_vae
            del sli_vae
        vae.to(dtype).to(device)

        text_encoder.to(dtype).to(device)
        unet.to(dtype).to(device)
        if clip_model is not None:
            clip_model.to(dtype).to(device)
        if vgg16_model is not None:
            vgg16_model.to(dtype).to(device)

        # networkを組み込む
        if network_meta.network_module:
            networks = []
            network_default_muls = []
            network_pre_calc = network_meta.network_pre_calc

            for i, network_module in enumerate(network_meta.network_module):
                print("import network module:", network_module)
                imported_module = importlib.import_module(network_module)

                network_mul = 1.0 if network_meta.network_mul is None or len(network_meta.network_mul) <= i else \
                    network_meta.network_mul[i]
                network_default_muls.append(network_mul)

                net_kwargs = {}
                if network_meta.network_args and i < len(network_meta.network_args):
                    network_args = network_meta.network_args[i]
                    # TODO escape special chars
                    network_args = network_args.split(";")
                    for net_arg in network_args:
                        key, value = net_arg.split("=")
                        net_kwargs[key] = value

                if network_meta.network_weights and i < len(network_meta.network_weights):
                    network_weight = network_meta.network_weights[i]
                    print("load network weights from:", network_weight)

                    if model_util.is_safetensors(network_weight) and network_meta.network_show_meta:
                        from safetensors.torch import safe_open

                        with safe_open(network_weight, framework="pt") as f:
                            metadata = f.metadata()
                        if metadata is not None:
                            print(f"metadata for: {network_weight}: {metadata}")

                    network, weights_sd = imported_module.create_network_from_weights(
                        network_mul, network_weight, vae, text_encoder, unet, for_inference=True, **net_kwargs
                    )
                else:
                    raise ValueError("No weight. Weight is required.")
                if network is None:
                    return

                mergeable = network.is_mergeable()
                if network_meta.network_merge and not mergeable:
                    print("network is not mergiable. ignore merge option.")

                if not network_meta.network_merge or not mergeable:
                    network.apply_to(text_encoder, unet)
                    info = network.load_state_dict(weights_sd, False)  # network.load_weightsを使うようにするとよい
                    print(f"weights are loaded: {info}")

                    if opt_channels_last:
                        network.to(memory_format=torch.channels_last)
                    network.to(dtype).to(device)

                    if network_pre_calc:
                        print("backup original weights")
                        network.backup_weights()

                    networks.append(network)
                else:
                    network.merge_to(text_encoder, unet, weights_sd, dtype, device)

        else:
            networks = []

        # upscalerの指定があれば取得する
        upscaler = None
        if highres_fix_upscaler_meta.highres_fix_upscaler:
            print("import upscaler module:", highres_fix_upscaler_meta.highres_fix_upscaler)
            imported_module = importlib.import_module(highres_fix_upscaler_meta.highres_fix_upscaler)

            us_kwargs = {}
            if highres_fix_upscaler_meta.highres_fix_upscaler_args:
                for net_arg in highres_fix_upscaler_meta.highres_fix_upscaler_args.split(";"):
                    key, value = net_arg.split("=")
                    us_kwargs[key] = value

            print("create upscaler")
            upscaler = imported_module.create_upscaler(**us_kwargs)
            upscaler.to(dtype).to(device)

        # ControlNetの処理
        control_nets: List[ControlNetInfo] = []
        if control_net_meta.control_net_models:
            for i, model in enumerate(control_net_meta.control_net_models):
                prep_type = None if not control_net_meta.control_net_preps or len(
                    control_net_meta.control_net_preps) <= i else \
                    control_net_meta.control_net_preps[i]
                weight = 1.0 if not control_net_meta.control_net_weights or len(
                    control_net_meta.control_net_weights) <= i else \
                    control_net_meta.control_net_weights[i]
                ratio = 1.0 if not control_net_meta.control_net_ratios or len(
                    control_net_meta.control_net_ratios) <= i else \
                    control_net_meta.control_net_ratios[i]

                ctrl_unet, ctrl_net = original_control_net.load_control_net(model_meta.v2, unet, model)
                prep = original_control_net.load_preprocess(prep_type)
                control_nets.append(ControlNetInfo(ctrl_unet, ctrl_net, prep, weight, ratio))

        if opt_channels_last:
            print(f"set optimizing: channels last")
            text_encoder.to(memory_format=torch.channels_last)
            vae.to(memory_format=torch.channels_last)
            unet.to(memory_format=torch.channels_last)
            if clip_model is not None:
                clip_model.to(memory_format=torch.channels_last)
            if networks:
                for network in networks:
                    network.to(memory_format=torch.channels_last)
            if vgg16_model is not None:
                vgg16_model.to(memory_format=torch.channels_last)

            for cn in control_nets:
                cn.unet.to(memory_format=torch.channels_last)
                cn.net.to(memory_format=torch.channels_last)

        pipe = PipelineLike(
            device,
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            clip_meta.clip_skip,
            clip_model,
            clip_meta.clip_guidance_scale,
            clip_meta.clip_image_guidance_scale,
            vgg16_model,
            vgg_meta.vgg16_guidance_scale,
            vgg_meta.vgg16_guidance_layer,
        )
        pipe.set_control_nets(control_nets)
        print("pipeline is ready.")

        if diffusers_xformers:
            pipe.enable_xformers_memory_efficient_attention()

        # Extended Textual Inversion および Textual Inversionを処理する
        if textual_inversion_meta.XTI_embeddings:
            diffusers.models.UNet2DConditionModel.forward = unet_forward_XTI
            diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D.forward = downblock_forward_XTI
            diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D.forward = upblock_forward_XTI

        if textual_inversion_meta.textual_inversion_embeddings:
            token_ids_embeds = []
            for embeds_file in textual_inversion_meta.textual_inversion_embeddings:
                if model_util.is_safetensors(embeds_file):
                    from safetensors.torch import load_file

                    data = load_file(embeds_file)
                else:
                    data = torch.load(embeds_file, map_location="cpu")

                if "string_to_param" in data:
                    data = data["string_to_param"]
                embeds = next(iter(data.values()))

                if type(embeds) != torch.Tensor:
                    raise ValueError(
                        f"weight file does not contains Tensor / 重みファイルのデータがTensorではありません: {embeds_file}")

                num_vectors_per_token = embeds.size()[0]
                token_string = os.path.splitext(os.path.basename(embeds_file))[0]
                token_strings = [token_string] + [f"{token_string}{i + 1}" for i in range(num_vectors_per_token - 1)]

                # add new word to tokenizer, count is num_vectors_per_token
                num_added_tokens = tokenizer.add_tokens(token_strings)
                assert (
                        num_added_tokens == num_vectors_per_token
                ), f"tokenizer has same word to token string (filename). please rename the file / 指定した名前（ファイル名）のトークンが既に存在します。ファイルをリネームしてください: {embeds_file}"

                token_ids = tokenizer.convert_tokens_to_ids(token_strings)
                print(f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids}")
                assert (
                        min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(token_ids) - 1
                ), f"token ids is not ordered"
                assert len(tokenizer) - 1 == token_ids[-1], f"token ids is not end of tokenize: {len(tokenizer)}"

                if num_vectors_per_token > 1:
                    pipe.add_token_replacement(token_ids[0], token_ids)

                token_ids_embeds.append((token_ids, embeds))

            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            for token_ids, embeds in token_ids_embeds:
                for token_id, embed in zip(token_ids, embeds):
                    token_embeds[token_id] = embed

        if textual_inversion_meta.XTI_embeddings:
            XTI_layers = [
                "IN01",
                "IN02",
                "IN04",
                "IN05",
                "IN07",
                "IN08",
                "MID",
                "OUT03",
                "OUT04",
                "OUT05",
                "OUT06",
                "OUT07",
                "OUT08",
                "OUT09",
                "OUT10",
                "OUT11",
            ]
            token_ids_embeds_XTI = []
            for embeds_file in textual_inversion_meta.XTI_embeddings:
                if model_util.is_safetensors(embeds_file):
                    from safetensors.torch import load_file

                    data = load_file(embeds_file)
                else:
                    data = torch.load(embeds_file, map_location="cpu")
                if set(data.keys()) != set(XTI_layers):
                    raise ValueError("NOT XTI")
                embeds = torch.concat(list(data.values()))
                num_vectors_per_token = data["MID"].size()[0]

                token_string = os.path.splitext(os.path.basename(embeds_file))[0]
                token_strings = [token_string] + [f"{token_string}{i + 1}" for i in range(num_vectors_per_token - 1)]

                # add new word to tokenizer, count is num_vectors_per_token
                num_added_tokens = tokenizer.add_tokens(token_strings)
                assert (
                        num_added_tokens == num_vectors_per_token
                ), f"tokenizer has same word to token string (filename). please rename the file / 指定した名前（ファイル名）のトークンが既に存在します。ファイルをリネームしてください: {embeds_file}"

                token_ids = tokenizer.convert_tokens_to_ids(token_strings)
                print(f"XTI embeddings `{token_string}` loaded. Tokens are added: {token_ids}")

                # if num_vectors_per_token > 1:
                pipe.add_token_replacement(token_ids[0], token_ids)

                token_strings_XTI = []
                for layer_name in XTI_layers:
                    token_strings_XTI += [f"{t}_{layer_name}" for t in token_strings]
                tokenizer.add_tokens(token_strings_XTI)
                token_ids_XTI = tokenizer.convert_tokens_to_ids(token_strings_XTI)
                token_ids_embeds_XTI.append((token_ids_XTI, embeds))
                for t in token_ids:
                    t_XTI_dic = {}
                    for i, layer_name in enumerate(XTI_layers):
                        t_XTI_dic[layer_name] = t + (i + 1) * num_added_tokens
                    pipe.add_token_replacement_XTI(t, t_XTI_dic)

                text_encoder.resize_token_embeddings(len(tokenizer))
                token_embeds = text_encoder.get_input_embeddings().weight.data
                for token_ids, embeds in token_ids_embeds_XTI:
                    for token_id, embed in zip(token_ids, embeds):
                        token_embeds[token_id] = embed

        # promptを取得する
        if prompt_meta.from_file is not None:
            print(f"reading prompts from {prompt_meta.from_file}")
            with open(prompt_meta.from_file, "r", encoding="utf-8") as f:
                prompt_list = f.read().splitlines()
                prompt_list = [d for d in prompt_list if len(d.strip()) > 0]
        elif prompt_meta.prompt is not None:
            prompt_list = [prompt_meta.prompt]
        else:
            prompt_list = []

        if interactive_meta.interactive:
            n_iter = 1

        # img2imgの前処理、画像の読み込みなど
        def load_images(path):
            if os.path.isfile(path):
                paths = [path]
            else:
                paths = (
                        glob.glob(os.path.join(path, "*.png"))
                        + glob.glob(os.path.join(path, "*.jpg"))
                        + glob.glob(os.path.join(path, "*.jpeg"))
                        + glob.glob(os.path.join(path, "*.webp"))
                )
                paths.sort()

            images = []
            for p in paths:
                image = Image.open(p)
                if image.mode != "RGB":
                    print(f"convert image to RGB from {image.mode}: {p}")
                    image = image.convert("RGB")
                images.append(image)

            return images

        def resize_images(imgs, size):
            resized = []
            for img in imgs:
                r_img = img.resize(size, Image.Resampling.LANCZOS)
                if hasattr(img, "filename"):  # filename属性がない場合があるらしい
                    r_img.filename = img.filename
                resized.append(r_img)
            return resized

        if img2img_meta.image_path is not None:
            print(f"load image for img2img: {img2img_meta.image_path}")
            init_images = load_images(img2img_meta.image_path)
            assert len(init_images) > 0, f"No image / 画像がありません: {img2img_meta.image_path}"
            print(f"loaded {len(init_images)} images for img2img")
        else:
            init_images = None

        if img2img_meta.mask_path is not None:
            print(f"load mask for inpainting: {img2img_meta.mask_path}")
            mask_images = load_images(img2img_meta.mask_path)
            assert len(mask_images) > 0, f"No mask image / マスク画像がありません: {img2img_meta.image_path}"
            print(f"loaded {len(mask_images)} mask images for inpainting")
        else:
            mask_images = None

        # promptがないとき、画像のPngInfoから取得する
        if init_images is not None and len(prompt_list) == 0 and not interactive_meta.interactive:
            print("get prompts from images' meta data")
            for img in init_images:
                if "prompt" in img.text:
                    prompt = img.text["prompt"]
                    if "negative-prompt" in img.text:
                        prompt += " --n " + img.text["negative-prompt"]
                    prompt_list.append(prompt)

            # プロンプトと画像を一致させるため指定回数だけ繰り返す（画像を増幅する）
            l = []
            for im in init_images:
                l.extend([im] * batch_draw_meta.images_per_prompt)
            init_images = l

            if mask_images is not None:
                l = []
                for im in mask_images:
                    l.extend([im] * batch_draw_meta.images_per_prompt)
                mask_images = l

        # 画像サイズにオプション指定があるときはリサイズする
        if W is not None and H is not None:
            # highres fix を考慮に入れる
            w, h = W, H
            if highres_fix:
                w = int(w * highres_fix_upscaler_meta.highres_fix_scale + 0.5)
                h = int(h * highres_fix_upscaler_meta.highres_fix_scale + 0.5)

            if init_images is not None:
                print(f"resize img2img source images to {w}*{h}")
                init_images = resize_images(init_images, (w, h))
            if mask_images is not None:
                print(f"resize img2img mask images to {w}*{h}")
                mask_images = resize_images(mask_images, (w, h))

        regional_network = False
        if networks and mask_images:
            # mask を領域情報として流用する、現在は一回のコマンド呼び出しで1枚だけ対応
            regional_network = True
            print("use mask as region")

            size = None
            for i, network in enumerate(networks):
                if i < 3:
                    np_mask = np.array(mask_images[0])
                    np_mask = np_mask[:, :, i]
                    size = np_mask.shape
                else:
                    np_mask = np.full(size, 255, dtype=np.uint8)
                mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)
                network.set_region(i, i == len(networks) - 1, mask)
            mask_images = None

        prev_image = None  # for VGG16 guided
        if clip_meta.guide_image_path is not None:
            print(f"load image for CLIP/VGG16/ControlNet guidance: {clip_meta.guide_image_path}")
            guide_images = []
            for p in clip_meta.guide_image_path:
                guide_images.extend(load_images(p))

            print(f"loaded {len(guide_images)} guide images for guidance")
            if len(guide_images) == 0:
                print(
                    f"No guide image, use previous generated image. / ガイド画像がありません。直前に生成した画像を使います: {img2img_meta.image_path}")
                guide_images = None
        else:
            guide_images = None

        # seed指定時はseedを決めておく
        if seed is not None:
            # dynamic promptを使うと足りなくなる→images_per_promptを適当に大きくしておいてもらう
            random.seed(seed)
            predefined_seeds = [random.randint(0, 0x7FFFFFFF) for _ in
                                range(batch_draw_meta.n_iter * len(prompt_list) * batch_draw_meta.images_per_prompt)]
            if len(predefined_seeds) == 1:
                predefined_seeds[0] = seed
        else:
            predefined_seeds = None

        # デフォルト画像サイズを設定する：img2imgではこれらの値は無視される（またはW*Hにリサイズ済み）
        if W is None:
            W = 512
        if H is None:
            H = 512

        # 画像生成のループ
        os.makedirs(output_meta.outdir, exist_ok=True)
        max_embeddings_multiples = 1 if prompt_meta.max_embeddings_multiples is None else prompt_meta.max_embeddings_multiples

        for gen_iter in range(n_iter):
            print(f"iteration {gen_iter + 1}/{n_iter}")
            iter_seed = random.randint(0, 0x7FFFFFFF)

            # バッチ処理の関数
            def process_batch(batch: List[BatchData], highres_fix, highres_1st=False):
                batch_size = len(batch)

                # highres_fixの処理
                if highres_fix and not highres_1st:
                    # 1st stageのバッチを作成して呼び出す：サイズを小さくして呼び出す
                    is_1st_latent = upscaler.support_latents() if upscaler else highres_fix_upscaler_meta.highres_fix_latents_upscaling

                    print("process 1st stage")
                    batch_1st = []
                    for _, base, ext in batch:
                        width_1st = int(ext.width * highres_fix_upscaler_meta.highres_fix_scale + 0.5)
                        height_1st = int(ext.height * highres_fix_upscaler_meta.highres_fix_scale + 0.5)
                        width_1st = width_1st - width_1st % 32
                        height_1st = height_1st - height_1st % 32

                        strength_1st = ext.strength if highres_fix_upscaler_meta.highres_fix_strength is None else highres_fix_upscaler_meta.highres_fix_strength

                        ext_1st = BatchDataExt(
                            width_1st,
                            height_1st,
                            highres_fix_upscaler_meta.highres_fix_steps,
                            ext.scale,
                            ext.negative_scale,
                            strength_1st,
                            ext.network_muls,
                            ext.num_sub_prompts,
                        )
                        batch_1st.append(BatchData(is_1st_latent, base, ext_1st))

                    pipe.set_enable_control_net(True)  # 1st stageではControlNetを有効にする
                    images_1st = process_batch(batch_1st, True, True)

                    # 2nd stageのバッチを作成して以下処理する
                    print("process 2nd stage")
                    width_2nd, height_2nd = batch[0].ext.width, batch[0].ext.height

                    if upscaler:
                        # upscalerを使って画像を拡大する
                        lowreso_imgs = None if is_1st_latent else images_1st
                        lowreso_latents = None if not is_1st_latent else images_1st

                        # 戻り値はPIL.Image.Imageかtorch.Tensorのlatents
                        batch_size = len(images_1st)
                        vae_batch_size = (
                            batch_size
                            if vae_meta.vae_batch_size is None
                            else (max(1,
                                      int(batch_size * vae_meta.vae_batch_size)) if vae_meta.vae_batch_size < 1 else vae_meta.vae_batch_size)
                        )
                        vae_batch_size = int(vae_batch_size)
                        images_1st = upscaler.upscale(
                            vae, lowreso_imgs, lowreso_latents, dtype, width_2nd, height_2nd, batch_size, vae_batch_size
                        )

                    elif highres_fix_upscaler_meta.highres_fix_latents_upscaling:
                        # latentを拡大する
                        org_dtype = images_1st.dtype
                        if images_1st.dtype == torch.bfloat16:
                            images_1st = images_1st.to(torch.float)  # interpolateがbf16をサポートしていない
                        images_1st = torch.nn.functional.interpolate(
                            images_1st, (batch[0].ext.height // 8, batch[0].ext.width // 8), mode="bilinear"
                        )  # , antialias=True)
                        images_1st = images_1st.to(org_dtype)

                    else:
                        # 画像をLANCZOSで拡大する
                        images_1st = [image.resize((width_2nd, height_2nd), resample=PIL.Image.LANCZOS) for image in
                                      images_1st]

                    batch_2nd = []
                    for i, (bd, image) in enumerate(zip(batch, images_1st)):
                        bd_2nd = BatchData(False,
                                           BatchDataBase(*bd.base[0:3], bd.base.seed + 1, image, None, *bd.base[6:]),
                                           bd.ext)
                        batch_2nd.append(bd_2nd)
                    batch = batch_2nd

                    if highres_fix_upscaler_meta.highres_fix_disable_control_net:
                        pipe.set_enable_control_net(False)  # オプション指定時、2nd stageではControlNetを無効にする

                # このバッチの情報を取り出す
                (
                    return_latents,
                    (step_first, _, _, _, init_image, mask_image, _, guide_image),
                    (width, height, steps, scale, negative_scale, strength, network_muls, num_sub_prompts),
                ) = batch[0]
                noise_shape = (
                    KohyaSSImageGenerator.LATENT_CHANNELS, height // KohyaSSImageGenerator.DOWNSAMPLING_FACTOR,
                    width // KohyaSSImageGenerator.DOWNSAMPLING_FACTOR)

                prompts = []
                negative_prompts = []
                start_code = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
                noises = [
                    torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
                    for _ in range(steps * scheduler_num_noises_per_step)
                ]
                seeds = []
                clip_prompts = []

                if init_image is not None:  # img2img?
                    i2i_noises = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
                    init_images = []

                    if mask_image is not None:
                        mask_images = []
                    else:
                        mask_images = None
                else:
                    i2i_noises = None
                    init_images = None
                    mask_images = None

                if guide_image is not None:  # CLIP image guided?
                    guide_images = []
                else:
                    guide_images = None

                # バッチ内の位置に関わらず同じ乱数を使うためにここで乱数を生成しておく。あわせてimage/maskがbatch内で同一かチェックする
                all_images_are_same = True
                all_masks_are_same = True
                all_guide_images_are_same = True
                for i, (_, (_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image),
                        _) in enumerate(batch):
                    prompts.append(prompt)
                    negative_prompts.append(negative_prompt)
                    seeds.append(seed)
                    clip_prompts.append(clip_prompt)

                    if init_image is not None:
                        init_images.append(init_image)
                        if i > 0 and all_images_are_same:
                            all_images_are_same = init_images[-2] is init_image

                    if mask_image is not None:
                        mask_images.append(mask_image)
                        if i > 0 and all_masks_are_same:
                            all_masks_are_same = mask_images[-2] is mask_image

                    if guide_image is not None:
                        if type(guide_image) is list:
                            guide_images.extend(guide_image)
                            all_guide_images_are_same = False
                        else:
                            guide_images.append(guide_image)
                            if i > 0 and all_guide_images_are_same:
                                all_guide_images_are_same = guide_images[-2] is guide_image

                    # make start code
                    torch.manual_seed(seed)
                    start_code[i] = torch.randn(noise_shape, device=device, dtype=dtype)

                    # make each noises
                    for j in range(steps * scheduler_num_noises_per_step):
                        noises[j][i] = torch.randn(noise_shape, device=device, dtype=dtype)

                    if i2i_noises is not None:  # img2img noise
                        i2i_noises[i] = torch.randn(noise_shape, device=device, dtype=dtype)

                noise_manager.reset_sampler_noises(noises)

                # すべての画像が同じなら1枚だけpipeに渡すことでpipe側で処理を高速化する
                if init_images is not None and all_images_are_same:
                    init_images = init_images[0]
                if mask_images is not None and all_masks_are_same:
                    mask_images = mask_images[0]
                if guide_images is not None and all_guide_images_are_same:
                    guide_images = guide_images[0]

                # ControlNet使用時はguide imageをリサイズする
                if control_nets:
                    # TODO resampleのメソッド
                    guide_images = guide_images if type(guide_images) == list else [guide_images]
                    guide_images = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in guide_images]
                    if len(guide_images) == 1:
                        guide_images = guide_images[0]

                # generate
                if networks:
                    # 追加ネットワークの処理
                    shared = {}
                    for n, m in zip(networks, network_muls if network_muls else network_default_muls):
                        n.set_multiplier(m)
                        if regional_network:
                            n.set_current_generation(batch_size, num_sub_prompts, width, height, shared)

                    if not regional_network and network_pre_calc:
                        for n in networks:
                            n.restore_weights()
                        for n in networks:
                            n.pre_calculation()
                        print("pre-calculation... done")

                images = pipe(
                    prompts,
                    negative_prompts,
                    init_images,
                    mask_images,
                    height,
                    width,
                    steps,
                    scale,
                    negative_scale,
                    strength,
                    latents=start_code,
                    output_type="pil",
                    max_embeddings_multiples=max_embeddings_multiples,
                    img2img_noise=i2i_noises,
                    vae_batch_size=vae_meta.vae_batch_size,
                    return_latents=return_latents,
                    clip_prompts=clip_prompts,
                    clip_guide_images=guide_images,
                )[0]
                if highres_1st and not highres_fix_upscaler_meta.highres_fix_save_1st:  # return images or latents
                    return images

                # save image
                highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
                ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                for i, (image, prompt, negative_prompt, seed, clip_prompt) in enumerate(
                        zip(images, prompts, negative_prompts, seeds, clip_prompts)
                ):
                    metadata = PngInfo()
                    metadata.add_text("prompt", prompt)
                    metadata.add_text("seed", str(seed))
                    metadata.add_text("sampler", sampler)
                    metadata.add_text("steps", str(steps))
                    metadata.add_text("scale", str(scale))
                    if negative_prompt is not None:
                        metadata.add_text("negative-prompt", negative_prompt)
                    if negative_scale is not None:
                        metadata.add_text("negative-scale", str(negative_scale))
                    if clip_prompt is not None:
                        metadata.add_text("clip-prompt", clip_prompt)

                    if output_meta.use_original_file_name and init_images is not None:
                        if type(init_images) is list:
                            fln = os.path.splitext(os.path.basename(init_images[i % len(init_images)].filename))[
                                      0] + ".png"
                        else:
                            fln = os.path.splitext(os.path.basename(init_images.filename))[0] + ".png"
                    elif output_meta.sequential_file_name:
                        fln = f"im_{highres_prefix}{step_first + i + 1:06d}.png"
                    else:
                        fln = f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"

                    image.save(os.path.join(output_meta.outdir, fln), pnginfo=metadata)

                if not interactive_meta.no_preview and not highres_1st and interactive_meta.interactive:
                    try:
                        import cv2

                        for prompt, image in zip(prompts, images):
                            cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])  # プロンプトが長いと死ぬ
                            cv2.waitKey()
                            cv2.destroyAllWindows()
                    except ImportError:
                        print(
                            "opencv-python is not installed, cannot preview / opencv-pythonがインストールされていないためプレビューできません")

                return images

            # 画像生成のプロンプトが一周するまでのループ
            prompt_index = 0
            global_step = 0
            batch_data = []
            while interactive_meta.interactive or prompt_index < len(prompt_list):
                if len(prompt_list) == 0:
                    # interactive
                    valid = False
                    while not valid:
                        print("\nType prompt:")
                        try:
                            raw_prompt = input()
                        except EOFError:
                            break

                        valid = len(raw_prompt.strip().split(" --")[0].strip()) > 0
                    if not valid:  # EOF, end app
                        break
                else:
                    raw_prompt = prompt_list[prompt_index]

                # sd-dynamic-prompts like variants:
                # count is 1 (not dynamic) or images_per_prompt (no enumeration) or arbitrary (enumeration)
                raw_prompts = KohyaSSImageGenerator.handle_dynamic_prompt_variants(raw_prompt,
                                                                                   batch_draw_meta.images_per_prompt)

                # repeat prompt
                for pi in range(batch_draw_meta.images_per_prompt if len(raw_prompts) == 1 else len(raw_prompts)):
                    raw_prompt = raw_prompts[pi] if len(raw_prompts) > 1 else raw_prompts[0]

                    if pi == 0 or len(raw_prompts) > 1:
                        # parse prompt: if prompt is not changed, skip parsing
                        width = W
                        height = H
                        scale = prompt_meta.scale
                        negative_scale = prompt_meta.negative_scale
                        steps = steps
                        seed = None
                        seeds = None
                        strength = 0.8 if img2img_meta.strength is None else img2img_meta.strength
                        negative_prompt = ""
                        clip_prompt = None
                        network_muls = None

                        prompt_args = raw_prompt.strip().split(" --")
                        prompt = prompt_args[0]
                        print(f"prompt {prompt_index + 1}/{len(prompt_list)}: {prompt}")

                        for parg in prompt_args[1:]:
                            try:
                                m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                                if m:
                                    width = int(m.group(1))
                                    print(f"width: {width}")
                                    continue

                                m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                                if m:
                                    height = int(m.group(1))
                                    print(f"height: {height}")
                                    continue

                                m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                                if m:  # steps
                                    steps = max(1, min(1000, int(m.group(1))))
                                    print(f"steps: {steps}")
                                    continue

                                m = re.match(r"d ([\d,]+)", parg, re.IGNORECASE)
                                if m:  # seed
                                    seeds = [int(d) for d in m.group(1).split(",")]
                                    print(f"seeds: {seeds}")
                                    continue

                                m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                                if m:  # scale
                                    scale = float(m.group(1))
                                    print(f"scale: {scale}")
                                    continue

                                m = re.match(r"nl ([\d\.]+|none|None)", parg, re.IGNORECASE)
                                if m:  # negative scale
                                    if m.group(1).lower() == "none":
                                        negative_scale = None
                                    else:
                                        negative_scale = float(m.group(1))
                                    print(f"negative scale: {negative_scale}")
                                    continue

                                m = re.match(r"t ([\d\.]+)", parg, re.IGNORECASE)
                                if m:  # strength
                                    strength = float(m.group(1))
                                    print(f"strength: {strength}")
                                    continue

                                m = re.match(r"n (.+)", parg, re.IGNORECASE)
                                if m:  # negative prompt
                                    negative_prompt = m.group(1)
                                    print(f"negative prompt: {negative_prompt}")
                                    continue

                                m = re.match(r"c (.+)", parg, re.IGNORECASE)
                                if m:  # clip prompt
                                    clip_prompt = m.group(1)
                                    print(f"clip prompt: {clip_prompt}")
                                    continue

                                m = re.match(r"am ([\d\.\-,]+)", parg, re.IGNORECASE)
                                if m:  # network multiplies
                                    network_muls = [float(v) for v in m.group(1).split(",")]
                                    while len(network_muls) < len(networks):
                                        network_muls.append(network_muls[-1])
                                    print(f"network mul: {network_muls}")
                                    continue

                            except ValueError as ex:
                                print(f"Exception in parsing / 解析エラー: {parg}")
                                print(ex)

                    # prepare seed
                    if seeds is not None:  # given in prompt
                        # 数が足りないなら前のをそのまま使う
                        if len(seeds) > 0:
                            seed = seeds.pop(0)
                    else:
                        if predefined_seeds is not None:
                            if len(predefined_seeds) > 0:
                                seed = predefined_seeds.pop(0)
                            else:
                                print("predefined seeds are exhausted")
                                seed = None
                        elif batch_draw_meta.iter_same_seed:
                            seeds = iter_seed
                        else:
                            seed = None  # 前のを消す

                    if seed is None:
                        seed = random.randint(0, 0x7FFFFFFF)
                    if interactive_meta.interactive:
                        print(f"seed: {seed}")

                    # prepare init image, guide image and mask
                    init_image = mask_image = guide_image = None

                    # 同一イメージを使うとき、本当はlatentに変換しておくと無駄がないが面倒なのでとりあえず毎回処理する
                    if init_images is not None:
                        init_image = init_images[global_step % len(init_images)]

                        # img2imgの場合は、基本的に元画像のサイズで生成する。highres fixの場合はargs.W, args.Hとscaleに従いリサイズ済みなので無視する
                        # 32単位に丸めたやつにresizeされるので踏襲する
                        if not highres_fix:
                            width, height = init_image.size
                            width = width - width % 32
                            height = height - height % 32
                            if width != init_image.size[0] or height != init_image.size[1]:
                                print(
                                    f"img2img image size is not divisible by 32 so aspect ratio is changed / img2imgの画像サイズが32で割り切れないためリサイズされます。画像が歪みます"
                                )

                    if mask_images is not None:
                        mask_image = mask_images[global_step % len(mask_images)]

                    if guide_images is not None:
                        if control_nets:  # 複数件の場合あり
                            c = len(control_nets)
                            p = global_step % (len(guide_images) // c)
                            guide_image = guide_images[p * c: p * c + c]
                        else:
                            guide_image = guide_images[global_step % len(guide_images)]
                    elif clip_meta.clip_image_guidance_scale > 0 or vgg_meta.vgg16_guidance_scale > 0:
                        if prev_image is None:
                            print("Generate 1st image without guide image.")
                        else:
                            print("Use previous image as guide image.")
                            guide_image = prev_image

                    if regional_network:
                        num_sub_prompts = len(prompt_meta.prompt.split(" AND "))
                        assert (
                                len(networks) <= num_sub_prompts
                        ), "Number of networks must be less than or equal to number of sub prompts."
                    else:
                        num_sub_prompts = None

                    b1 = BatchData(
                        False,
                        BatchDataBase(global_step, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt,
                                      guide_image),
                        BatchDataExt(
                            width,
                            height,
                            steps,
                            scale,
                            negative_scale,
                            strength,
                            tuple(network_muls) if network_muls else None,
                            num_sub_prompts,
                        ),
                    )
                    if len(batch_data) > 0 and batch_data[-1].ext != b1.ext:  # バッチ分割必要？
                        process_batch(batch_data, highres_fix)
                        batch_data.clear()

                    batch_data.append(b1)
                    if len(batch_data) == batch_draw_meta.batch_size:
                        prev_image = process_batch(batch_data, highres_fix)[0]
                        batch_data.clear()

                    global_step += 1

                prompt_index += 1

            if len(batch_data) > 0:
                process_batch(batch_data, highres_fix)
                batch_data.clear()

        print("done!")

    @staticmethod
    def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
        if mem_eff_attn:
            KohyaSSImageGenerator.replace_unet_cross_attn_to_memory_efficient()
        elif xformers:
            KohyaSSImageGenerator.replace_unet_cross_attn_to_xformers()

    @staticmethod
    def replace_vae_modules(vae: diffusers.models.AutoencoderKL, mem_eff_attn, xformers):
        if mem_eff_attn:
            KohyaSSImageGenerator.replace_vae_attn_to_memory_efficient()
        elif xformers:
            # とりあえずDiffusersのxformersを使う。AttentionがあるのはMidBlockのみ
            print("Use Diffusers xformers for VAE")
            vae.set_use_memory_efficient_attention_xformers(True)

        """
        # VAEがbfloat16でメモリ消費が大きい問題を解決する
        upsamplers = []
        for block in vae.decoder.up_blocks:
            if block.upsamplers is not None:
                upsamplers.extend(block.upsamplers)

        def forward_upsample(_self, hidden_states, output_size=None):
            assert hidden_states.shape[1] == _self.channels
            if _self.use_conv_transpose:
                return _self.conv(hidden_states)

            dtype = hidden_states.dtype
            if dtype == torch.bfloat16:
                assert output_size is None
                # repeat_interleaveはすごく遅いが、回数はあまり呼ばれないので許容する
                hidden_states = hidden_states.repeat_interleave(2, dim=-1)
                hidden_states = hidden_states.repeat_interleave(2, dim=-2)
            else:
                if hidden_states.shape[0] >= 64:
                    hidden_states = hidden_states.contiguous()

                # if `output_size` is passed we force the interpolation output
                # size and do not make use of `scale_factor=2`
                if output_size is None:
                    hidden_states = torch.nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
                else:
                    hidden_states = torch.nn.functional.interpolate(hidden_states, size=output_size, mode="nearest")

            if _self.use_conv:
                if _self.name == "conv":
                    hidden_states = _self.conv(hidden_states)
                else:
                    hidden_states = _self.Conv2d_0(hidden_states)
            return hidden_states

        # replace upsamplers
        for upsampler in upsamplers:
            # make new scope
            def make_replacer(upsampler):
                def forward(hidden_states, output_size=None):
                    return forward_upsample(upsampler, hidden_states, output_size)

                return forward

            upsampler.forward = make_replacer(upsampler)
    """

    @staticmethod
    def get_weighted_text_embeddings(
            pipe: PipelineLike,
            prompt: Union[str, List[str]],
            uncond_prompt: Optional[Union[str, List[str]]] = None,
            max_embeddings_multiples: Optional[int] = 1,
            no_boseos_middle: Optional[bool] = False,
            skip_parsing: Optional[bool] = False,
            skip_weighting: Optional[bool] = False,
            clip_skip=None,
            layer=None,
            **kwargs,
    ):
        r"""
        Prompts can be assigned with local weights using brackets. For example,
        prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
        and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
        Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
        Args:
            pipe (`DiffusionPipeline`):
                Pipe to provide access to the tokenizer and the text encoder.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            uncond_prompt (`str` or `List[str]`):
                The unconditional prompt or prompts for guide the image generation. If unconditional prompt
                is provided, the embeddings of prompt and uncond_prompt are concatenated.
            max_embeddings_multiples (`int`, *optional*, defaults to `1`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            no_boseos_middle (`bool`, *optional*, defaults to `False`):
                If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
                ending token in each of the chunk in the middle.
            skip_parsing (`bool`, *optional*, defaults to `False`):
                Skip the parsing of brackets.
            skip_weighting (`bool`, *optional*, defaults to `False`):
                Skip the weighting. When the parsing is skipped, it is forced True.
        """
        max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
        if isinstance(prompt, str):
            prompt = [prompt]

        # split the prompts with "AND". each prompt must have the same number of splits
        new_prompts = []
        for p in prompt:
            new_prompts.extend(p.split(" AND "))
        prompt = new_prompts

        if not skip_parsing:
            prompt_tokens, prompt_weights = KohyaSSImageGenerator.get_prompts_with_weights(pipe, prompt, max_length - 2,
                                                                                           layer=layer)
            if uncond_prompt is not None:
                if isinstance(uncond_prompt, str):
                    uncond_prompt = [uncond_prompt]
                uncond_tokens, uncond_weights = KohyaSSImageGenerator.get_prompts_with_weights(pipe, uncond_prompt,
                                                                                               max_length - 2,
                                                                                               layer=layer)
        else:
            prompt_tokens = [token[1:-1] for token in
                             pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids]
            prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
            if uncond_prompt is not None:
                if isinstance(uncond_prompt, str):
                    uncond_prompt = [uncond_prompt]
                uncond_tokens = [
                    token[1:-1] for token in
                    pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
                ]
                uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

        # round up the longest length of tokens to a multiple of (model_max_length - 2)
        max_length = max([len(token) for token in prompt_tokens])
        if uncond_prompt is not None:
            max_length = max(max_length, max([len(token) for token in uncond_tokens]))

        max_embeddings_multiples = min(
            max_embeddings_multiples,
            (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
        )
        max_embeddings_multiples = max(1, max_embeddings_multiples)
        max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

        # pad the length of tokens and weights
        bos = pipe.tokenizer.bos_token_id
        eos = pipe.tokenizer.eos_token_id
        pad = pipe.tokenizer.pad_token_id
        prompt_tokens, prompt_weights = KohyaSSImageGenerator.pad_tokens_and_weights(
            prompt_tokens,
            prompt_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
        if uncond_prompt is not None:
            uncond_tokens, uncond_weights = KohyaSSImageGenerator.pad_tokens_and_weights(
                uncond_tokens,
                uncond_weights,
                max_length,
                bos,
                eos,
                pad,
                no_boseos_middle=no_boseos_middle,
                chunk_length=pipe.tokenizer.model_max_length,
            )
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

        # get the embeddings
        text_embeddings = KohyaSSImageGenerator.get_unweighted_text_embeddings(
            pipe,
            prompt_tokens,
            pipe.tokenizer.model_max_length,
            clip_skip,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
        )
        prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
        if uncond_prompt is not None:
            uncond_embeddings = KohyaSSImageGenerator.get_unweighted_text_embeddings(
                pipe,
                uncond_tokens,
                pipe.tokenizer.model_max_length,
                clip_skip,
                eos,
                pad,
                no_boseos_middle=no_boseos_middle,
            )
            uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)

        # assign weights to the prompts and normalize in the sense of mean
        # TODO: should we normalize by chunk or in a whole (current implementation)?
        # →全体でいいんじゃないかな
        if (not skip_parsing) and (not skip_weighting):
            previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= prompt_weights.unsqueeze(-1)
            current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
            if uncond_prompt is not None:
                previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= uncond_weights.unsqueeze(-1)
                current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

        if uncond_prompt is not None:
            return text_embeddings, uncond_embeddings, prompt_tokens
        return text_embeddings, None, prompt_tokens

    @staticmethod
    def get_prompts_with_weights(pipe: PipelineLike, prompt: List[str], max_length: int, layer=None):
        r"""
        Tokenize a list of prompts and return its tokens with weights of each token.
        No padding, starting or ending token is included.
        """
        tokens = []
        weights = []
        truncated = False

        for text in prompt:
            texts_and_weights = KohyaSSImageGenerator.parse_prompt_attention(text)
            text_token = []
            text_weight = []
            for word, weight in texts_and_weights:
                if word.strip() == "BREAK":
                    # pad until next multiple of tokenizer's max token length
                    pad_len = pipe.tokenizer.model_max_length - (len(text_token) % pipe.tokenizer.model_max_length)
                    print(f"BREAK pad_len: {pad_len}")
                    for i in range(pad_len):
                        # v2のときEOSをつけるべきかどうかわからないぜ
                        # if i == 0:
                        #     text_token.append(pipe.tokenizer.eos_token_id)
                        # else:
                        text_token.append(pipe.tokenizer.pad_token_id)
                        text_weight.append(1.0)
                    continue

                # tokenize and discard the starting and the ending token
                token = pipe.tokenizer(word).input_ids[1:-1]

                token = pipe.replace_token(token, layer=layer)

                text_token += token
                # copy the weight by length of token
                text_weight += [weight] * len(token)
                # stop if the text is too long (longer than truncation limit)
                if len(text_token) > max_length:
                    truncated = True
                    break
            # truncate
            if len(text_token) > max_length:
                truncated = True
                text_token = text_token[:max_length]
                text_weight = text_weight[:max_length]
            tokens.append(text_token)
            weights.append(text_weight)
        if truncated:
            print("warning: Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
        return tokens, weights

    re_attention = re.compile(
        r"""
    \\\(|
    \\\)|
    \\\[|
    \\]|
    \\\\|
    \\|
    \(|
    \[|
    :([+-]?[.\d]+)\)|
    \)|
    ]|
    [^\\()\[\]:]+|
    :
    """,
        re.X,
    )

    @staticmethod
    def parse_prompt_attention(text):
        """
        Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
        Accepted tokens are:
          (abc) - increases attention to abc by a multiplier of 1.1
          (abc:3.12) - increases attention to abc by a multiplier of 3.12
          [abc] - decreases attention to abc by a multiplier of 1.1
          \( - literal character '('
          \[ - literal character '['
          \) - literal character ')'
          \] - literal character ']'
          \\ - literal character '\'
          anything else - just text
        >>> parse_prompt_attention('normal text')
        [['normal text', 1.0]]
        >>> parse_prompt_attention('an (important) word')
        [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
        >>> parse_prompt_attention('(unbalanced')
        [['unbalanced', 1.1]]
        >>> parse_prompt_attention('\(literal\]')
        [['(literal]', 1.0]]
        >>> parse_prompt_attention('(unnecessary)(parens)')
        [['unnecessaryparens', 1.1]]
        >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
        [['a ', 1.0],
         ['house', 1.5730000000000004],
         [' ', 1.1],
         ['on', 1.0],
         [' a ', 1.1],
         ['hill', 0.55],
         [', sun, ', 1.1],
         ['sky', 1.4641000000000006],
         ['.', 1.1]]
        """

        res = []
        round_brackets = []
        square_brackets = []

        round_bracket_multiplier = 1.1
        square_bracket_multiplier = 1 / 1.1

        def multiply_range(start_position, multiplier):
            for p in range(start_position, len(res)):
                res[p][1] *= multiplier

        # keep break as separate token
        text = text.replace("BREAK", "\\BREAK\\")

        for m in KohyaSSImageGenerator.re_attention.finditer(text):
            text = m.group(0)
            weight = m.group(1)

            if text.startswith("\\"):
                res.append([text[1:], 1.0])
            elif text == "(":
                round_brackets.append(len(res))
            elif text == "[":
                square_brackets.append(len(res))
            elif weight is not None and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), float(weight))
            elif text == ")" and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), round_bracket_multiplier)
            elif text == "]" and len(square_brackets) > 0:
                multiply_range(square_brackets.pop(), square_bracket_multiplier)
            else:
                res.append([text, 1.0])

        for pos in round_brackets:
            multiply_range(pos, round_bracket_multiplier)

        for pos in square_brackets:
            multiply_range(pos, square_bracket_multiplier)

        if len(res) == 0:
            res = [["", 1.0]]

        # merge runs of identical weights
        i = 0
        while i + 1 < len(res):
            if res[i][1] == res[i + 1][1] and res[i][0].strip() != "BREAK" and res[i + 1][0].strip() != "BREAK":
                res[i][0] += res[i + 1][0]
                res.pop(i + 1)
            else:
                i += 1

        return res

    @staticmethod
    def get_unweighted_text_embeddings(
            pipe: PipelineLike,
            text_input: torch.Tensor,
            chunk_length: int,
            clip_skip: int,
            eos: int,
            pad: int,
            no_boseos_middle: Optional[bool] = True,
    ):
        """
        When the length of tokens is a multiple of the capacity of the text encoder,
        it should be split into chunks and sent to the text encoder individually.
        """
        max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
        if max_embeddings_multiples > 1:
            text_embeddings = []
            for i in range(max_embeddings_multiples):
                # extract the i-th chunk
                text_input_chunk = text_input[:, i * (chunk_length - 2): (i + 1) * (chunk_length - 2) + 2].clone()

                # cover the head and the tail by the starting and the ending tokens
                text_input_chunk[:, 0] = text_input[0, 0]
                if pad == eos:  # v1
                    text_input_chunk[:, -1] = text_input[0, -1]
                else:  # v2
                    for j in range(len(text_input_chunk)):
                        if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                            text_input_chunk[j, -1] = eos
                        if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                            text_input_chunk[j, 1] = eos

                if clip_skip is None or clip_skip == 1:
                    text_embedding = pipe.text_encoder(text_input_chunk)[0]
                else:
                    enc_out = pipe.text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                    text_embedding = enc_out["hidden_states"][-clip_skip]
                    text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)

                if no_boseos_middle:
                    if i == 0:
                        # discard the ending token
                        text_embedding = text_embedding[:, :-1]
                    elif i == max_embeddings_multiples - 1:
                        # discard the starting token
                        text_embedding = text_embedding[:, 1:]
                    else:
                        # discard both starting and ending tokens
                        text_embedding = text_embedding[:, 1:-1]

                text_embeddings.append(text_embedding)
            text_embeddings = torch.concat(text_embeddings, axis=1)
        else:
            if clip_skip is None or clip_skip == 1:
                text_embeddings = pipe.text_encoder(text_input)[0]
            else:
                enc_out = pipe.text_encoder(text_input, output_hidden_states=True, return_dict=True)
                text_embeddings = enc_out["hidden_states"][-clip_skip]
                text_embeddings = pipe.text_encoder.text_model.final_layer_norm(text_embeddings)
        return text_embeddings

    @staticmethod
    def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
        r"""
        Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
        """
        max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
        weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
        for i in range(len(tokens)):
            tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
            if no_boseos_middle:
                weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
            else:
                w = []
                if len(weights[i]) == 0:
                    w = [1.0] * weights_length
                else:
                    for j in range(max_embeddings_multiples):
                        w.append(1.0)  # weight for starting token in this chunk
                        w += weights[i][j * (chunk_length - 2): min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                        w.append(1.0)  # weight for ending token in this chunk
                    w += [1.0] * (weights_length - len(w))
                weights[i] = w[:]

        return tokens, weights

    @staticmethod
    def replace_vae_attn_to_memory_efficient():
        print("AttentionBlock.forward has been replaced to FlashAttention (not xformers)")
        flash_func = FlashAttentionFunction

        def forward_flash_attn(self, hidden_states):
            print("forward_flash_attn")
            q_bucket_size = 512
            k_bucket_size = 1024

            residual = hidden_states
            batch, channel, height, width = hidden_states.shape

            # norm
            hidden_states = self.group_norm(hidden_states)

            hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

            # proj to q, k, v
            query_proj = self.query(hidden_states)
            key_proj = self.key(hidden_states)
            value_proj = self.value(hidden_states)

            query_proj, key_proj, value_proj = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (query_proj, key_proj, value_proj)
            )

            out = flash_func.apply(query_proj, key_proj, value_proj, None, False, q_bucket_size, k_bucket_size)

            out = rearrange(out, "b h n d -> b n (h d)")

            # compute next hidden_states
            hidden_states = self.proj_attn(hidden_states)
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

            # res connect and rescale
            hidden_states = (hidden_states + residual) / self.rescale_output_factor
            return hidden_states

        diffusers.models.attention.AttentionBlock.forward = forward_flash_attn

    @staticmethod
    def replace_unet_cross_attn_to_memory_efficient():
        print("CrossAttention.forward has been replaced to FlashAttention (not xformers) and NAI style Hypernetwork")
        flash_func = FlashAttentionFunction

        def forward_flash_attn(self, x, context=None, mask=None):
            q_bucket_size = 512
            k_bucket_size = 1024

            h = self.heads
            q = self.to_q(x)

            context = context if context is not None else x
            context = context.to(x.dtype)

            if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
                context_k, context_v = self.hypernetwork.forward(x, context)
                context_k = context_k.to(x.dtype)
                context_v = context_v.to(x.dtype)
            else:
                context_k = context
                context_v = context

            k = self.to_k(context_k)
            v = self.to_v(context_v)
            del context, x

            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

            out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

            out = rearrange(out, "b h n d -> b n (h d)")

            # diffusers 0.7.0~
            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out

        diffusers.models.attention.CrossAttention.forward = forward_flash_attn

    @staticmethod
    def replace_unet_cross_attn_to_xformers():
        print("CrossAttention.forward has been replaced to enable xformers and NAI style Hypernetwork")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        def forward_xformers(self, x, context=None, mask=None):
            h = self.heads
            q_in = self.to_q(x)

            context = KohyaSSImageGenerator.default(context, x)
            context = context.to(x.dtype)

            if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
                context_k, context_v = self.hypernetwork.forward(x, context)
                context_k = context_k.to(x.dtype)
                context_v = context_v.to(x.dtype)
            else:
                context_k = context
                context_v = context

            k_in = self.to_k(context_k)
            v_in = self.to_v(context_v)

            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
            del q_in, k_in, v_in

            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

            out = rearrange(out, "b n h d -> b n (h d)", h=h)

            # diffusers 0.7.0~
            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out

        diffusers.models.attention.CrossAttention.forward = forward_xformers

    # @staticmethod
    # def exists(val):
    #     return val is not None

    @staticmethod
    def default(val, d):
        # return val if exists(val) else d
        return val if (val is not None) else d

    @staticmethod
    def handle_dynamic_prompt_variants(prompt, repeat_count):
        founds = list(KohyaSSImageGenerator.RE_DYNAMIC_PROMPT.finditer(prompt))
        if not founds:
            return [prompt]

        # make each replacement for each variant
        enumerating = False
        replacers = []
        for found in founds:
            # if "e$$" is found, enumerate all variants
            found_enumerating = found.group(2) is not None
            enumerating = enumerating or found_enumerating

            separator = ", " if found.group(6) is None else found.group(6)
            variants = found.group(7).split("|")

            # parse count range
            count_range = found.group(4)
            if count_range is None:
                count_range = [1, 1]
            else:
                count_range = count_range.split("-")
                if len(count_range) == 1:
                    count_range = [int(count_range[0]), int(count_range[0])]
                elif len(count_range) == 2:
                    count_range = [int(count_range[0]), int(count_range[1])]
                else:
                    print(f"invalid count range: {count_range}")
                    count_range = [1, 1]
                if count_range[0] > count_range[1]:
                    count_range = [count_range[1], count_range[0]]
                if count_range[0] < 0:
                    count_range[0] = 0
                if count_range[1] > len(variants):
                    count_range[1] = len(variants)

            if found_enumerating:
                # make function to enumerate all combinations
                def make_replacer_enum(vari, cr, sep):
                    def replacer():
                        values = []
                        for count in range(cr[0], cr[1] + 1):
                            for comb in itertools.combinations(vari, count):
                                values.append(sep.join(comb))
                        return values

                    return replacer

                replacers.append(make_replacer_enum(variants, count_range, separator))
            else:
                # make function to choose random combinations
                def make_replacer_single(vari, cr, sep):
                    def replacer():
                        count = random.randint(cr[0], cr[1])
                        comb = random.sample(vari, count)
                        return [sep.join(comb)]

                    return replacer

                replacers.append(make_replacer_single(variants, count_range, separator))

        # make each prompt
        if not enumerating:
            # if not enumerating, repeat the prompt, replace each variant randomly
            prompts = []
            for _ in range(repeat_count):
                current = prompt
                for found, replacer in zip(founds, replacers):
                    current = current.replace(found.group(0), replacer()[0], 1)
                prompts.append(current)
        else:
            # if enumerating, iterate all combinations for previous prompts
            prompts = [prompt]

            for found, replacer in zip(founds, replacers):
                if found.group(2) is not None:
                    # make all combinations for existing prompts
                    new_prompts = []
                    for current in prompts:
                        replecements = replacer()
                        for replecement in replecements:
                            new_prompts.append(current.replace(found.group(0), replecement, 1))
                    prompts = new_prompts

            for found, replacer in zip(founds, replacers):
                # make random selection for existing prompts
                if found.group(2) is None:
                    for i in range(len(prompts)):
                        prompts[i] = prompts[i].replace(found.group(0), replacer()[0], 1)

        return prompts

    # regular expression for dynamic prompt:
    # starts and ends with "{" and "}"
    # contains at least one variant divided by "|"
    # optional framgments divided by "$$" at start
    # if the first fragment is "E" or "e", enumerate all variants
    # if the second fragment is a number or two numbers, repeat the variants in the range
    # if the third fragment is a string, use it as a separator

    RE_DYNAMIC_PROMPT = re.compile(r"\{((e|E)\$\$)?(([\d\-]+)\$\$)?(([^\|\}]+?)\$\$)?(.+?((\|).+?)*?)\}")
