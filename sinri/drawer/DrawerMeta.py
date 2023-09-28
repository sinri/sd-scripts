from typing import Optional, List, Union

import torch


class HighResFixUpscalerMeta:
    def __init__(self,
                 highres_fix_scale: Optional[float] = None,
                 highres_fix_upscaler: Optional[str] = None,
                 highres_fix_upscaler_args: Optional[str] = None,
                 highres_fix_latents_upscaling: bool = False,
                 highres_fix_strength: Optional[float] = None,
                 highres_fix_steps: int = 28,
                 highres_fix_disable_control_net: bool = False,
                 highres_fix_save_1st: bool = False,
                 ):
        """

        Args:
            highres_fix_scale: enable highres fix, reso scale for 1st stage / highres fixを有効にして最初の解像度をこのscaleにする
            highres_fix_upscaler: upscaler module for highres fix / highres fixで使うupscalerのモジュール名
            highres_fix_upscaler_args: additional argmuments for upscaler (key=value) / upscalerへの追加の引数
            highres_fix_latents_upscaling: use latents upscaling for highres fix / highres fixでlatentで拡大する
            highres_fix_strength: 1st stage img2img strength for highres fix / highres fixの最初のステージのimg2img時のstrength、省略時はstrengthと同じ
            highres_fix_steps: 1st stage steps for highres fix / highres fixの最初のステージのステップ数
            highres_fix_disable_control_net: disable ControlNet for highres fix / highres fixでControlNetを使わない
            highres_fix_save_1st: save 1st stage images for highres fix / highres fixの最初のステージの画像を保存する
        """

        self.highres_fix_scale = highres_fix_scale
        self.highres_fix_upscaler = highres_fix_upscaler
        self.highres_fix_upscaler_args = highres_fix_upscaler_args
        self.highres_fix_latents_upscaling = highres_fix_latents_upscaling
        self.highres_fix_strength = highres_fix_strength
        self.highres_fix_steps = highres_fix_steps
        self.highres_fix_disable_control_net = highres_fix_disable_control_net
        self.highres_fix_save_1st = highres_fix_save_1st


class ClipMeta:
    def __init__(self,
                 clip_skip: Optional[int] = None,
                 clip_guidance_scale: float = 0.0,
                 guide_image_path: Optional[List[str]] = None,
                 clip_image_guidance_scale: float = 0.0,
                 ):
        """

        Args:
            clip_skip: layer number from bottom to use in CLIP / CLIPの後ろからn層目の出力を使う
            clip_guidance_scale: enable CLIP guided SD, scale for guidance (DDIM, PNDM, LMS samplers only) / CLIP guided SDを有効にしてこのscaleを適用する（サンプラーはDDIM、PNDM、LMSのみ）
            guide_image_path: image to CLIP guidance / CLIP guided SDでガイドに使う画像
            clip_image_guidance_scale: enable CLIP guided SD by image, scale for guidance / 画像によるCLIP guided SDを有効にしてこのscaleを適用する
        """
        self.clip_skip = clip_skip
        self.clip_guidance_scale = clip_guidance_scale
        self.guide_image_path = guide_image_path
        self.clip_image_guidance_scale = clip_image_guidance_scale


class ControlNetMeta:
    """
    現在はControlNet 1.0のみ動作確認しています。プリプロセスはCannyのみサポートしています。

    以下のオプションがあります。

    --control_net_models：ControlNetのモデルファイルを指定します。 複数指定すると、それらをstepごとに切り替えて利用します（Web UIのControlNet拡張の実装と異なります）。diffと通常の両方をサポートします。
    --guide_image_path：ControlNetに使うヒント画像を指定します。--img_pathと同様にフォルダを指定すると、そのフォルダの画像を順次利用します。Canny以外のモデルの場合には、あらかじめプリプロセスを行っておいてください。
    --control_net_preps：ControlNetのプリプロセスを指定します。--control_net_modelsと同様に複数指定可能です。現在はcannyのみ対応しています。対象モデルでプリプロセスを使用しない場合は none を指定します。 cannyの場合 --control_net_preps canny_63_191のように、閾値1と2を'_'で区切って指定できます。
    --control_net_weights：ControlNetの適用時の重みを指定します（1.0で通常、0.5なら半分の影響力で適用）。--control_net_modelsと同様に複数指定可能です。
    --control_net_ratios：ControlNetを適用するstepの範囲を指定します。0.5の場合は、step数の半分までControlNetを適用します。--control_net_modelsと同様に複数指定可能です。
    """

    def __init__(self,
                 control_net_models: Optional[List[str]] = None,
                 control_net_preps: Optional[List[str]] = None,
                 control_net_weights: Optional[List[float]] = None,
                 control_net_ratios: Optional[List[float]] = None,
                 guide_image_path: Optional[List[str]] = None,
                 ):
        """

        Args:
            control_net_models: ControlNet models to use / 使用するControlNetのモデル名
            control_net_preps: ControlNet preprocess to use / 使用するControlNetのプリプロセス名
            control_net_weights: ControlNet weights / ControlNetの重み
            control_net_ratios: ControlNet guidance ratio for steps / ControlNetでガイドするステップ比率
        """
        self.control_net_models = control_net_models
        self.control_net_preps = control_net_preps
        self.control_net_weights = control_net_weights
        self.control_net_ratios = control_net_ratios
        self.guide_image_path = guide_image_path


class VaeMeta:
    def __init__(self,
                 vae: Optional[str] = None,
                 vae_slices: Optional[int] = None,
                 vae_batch_size: Optional[float] = None,
                 ):
        """

        Args:
            vae: path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ
            vae_slices: number of slices to split image into for VAE to reduce VRAM usage, None for no splitting (default), slower if specified. 16 or 32 recommended / VAE処理時にVRAM使用量削減のため画像を分割するスライス数、Noneの場合は分割しない（デフォルト）、指定すると遅くなる。16か32程度を推奨
            vae_batch_size: batch size for VAE, < 1.0 for ratio / VAE処理時のバッチサイズ、1未満の値の場合は通常バッチサイズの比率
        """
        self.vae = vae
        self.vae_slices = vae_slices
        self.vae_batch_size = vae_batch_size


class ModelMeta:
    def __init__(self,
                 ckpt: str,
                 v2: bool = False,
                 v_parameterization: bool = False,
                 ):
        """

        Args:
            ckpt:
            v2: load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む
            v_parameterization: enable v-parameterization training / v-parameterization学習を有効にする
        """
        self.ckpt = ckpt
        self.v2 = v2
        self.v_parameterization = v_parameterization


class PromptMeta:
    def __init__(self,
                 prompt: Optional[str] = None,
                 negative_prompt: Optional[str] = None,
                 from_file: Optional[str] = None,
                 max_embeddings_multiples: Optional[int] = None,
                 scale: float = 7.5,
                 negative_scale: Optional[float] = None,
                 ):
        """

        Args:
            prompt: prompt / プロンプト
            from_file: if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む
            max_embeddings_multiples: max embeding multiples, max token length is 75 * multiples / トークン長をデフォルトの何倍とするか 75*この値 がトークン長となる
            scale: unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) / guidance scale
            negative_scale: set another guidance scale for negative prompt / ネガティブプロンプトのscaleを指定する
        """
        self.prompt = prompt
        # todo --n negative_prompt
        if negative_prompt is not None and negative_scale != '' and self.prompt is not None:
            self.prompt += ' --n ' + negative_prompt
        self.from_file = from_file
        self.max_embeddings_multiples = max_embeddings_multiples
        self.scale = scale
        self.negative_scale = negative_scale


class Vgg16Meta:
    """
    指定した画像に近づくように画像生成する機能です。
    通常のプロンプトによる生成指定に加えて、追加でVGG16の特徴量を取得し、生成中の画像が指定したガイド画像に近づくよう、生成される画像をコントロールします。
    img2imgでの使用をお勧めします（通常の生成では画像がぼやけた感じになります）。
    CLIP Guided Stable Diffusionの仕組みを流用した独自の機能です。
    またアイデアはVGGを利用したスタイル変換から拝借しています。
    なお選択できるサンプラーはDDIM、PNDM、LMSのみとなります。

    --vgg16_guidance_scaleオプションにどの程度、VGG16特徴量を反映するかを数値で指定します。試した感じでは100くらいから始めて増減すると良いようです。
    --guide_image_pathオプションでguideに使用する画像（ファイルまたはフォルダ）を指定してください。

    複数枚の画像を一括でimg2img変換し、元画像をガイド画像とする場合、--guide_image_pathと--image_pathに同じ値を指定すればOKです。
    """

    def __init__(self,
                 guide_image_path: str,
                 vgg16_guidance_scale: float = 0.0,
                 vgg16_guidance_layer: int = 20,

                 ):
        """

        Args:
            vgg16_guidance_scale: enable VGG16 guided SD by image, scale for guidance / 画像によるVGG16 guided SDを有効にしてこのscaleを適用する
            vgg16_guidance_layer: layer of VGG16 to calculate contents guide (1~30, 20 for conv4_2) / VGG16のcontents guideに使うレイヤー番号 (1~30、20はconv4_2)
        """
        self.guide_image_path = guide_image_path
        self.vgg16_guidance_scale = vgg16_guidance_scale
        self.vgg16_guidance_layer = vgg16_guidance_layer


class InteractiveMeta:
    def __init__(self,
                 interactive: bool = False,
                 no_preview: bool = False,
                 ):
        """

        Args:
            interactive: interactive mode (generates one image) / 対話モード（生成される画像は1枚になります）
            no_preview: do not show generated image in interactive mode / 対話モードで画像を表示しない
        """
        self.interactive = interactive
        self.no_preview = no_preview


class OutputMeta:
    def __init__(self,
                 outdir: Optional[str] = None,
                 use_original_file_name: bool = False,
                 sequential_file_name: bool = False,
                 ):
        """

        Args:
            outdir: dir to write results to / 生成画像の出力先
            use_original_file_name: prepend original file name in img2img / img2imgで元画像のファイル名を生成画像のファイル名の先頭に付ける
            sequential_file_name: sequential output file name / 生成画像のファイル名を連番にする
        """
        self.outdir = outdir
        self.use_original_file_name = use_original_file_name
        self.sequential_file_name = sequential_file_name


class Img2ImgMeta:
    def __init__(self,
                 image_path: Optional[str] = None,
                 mask_path: Optional[str] = None,
                 strength: Optional[float] = None,
                 ):
        """

        Args:
            image_path: image to inpaint or to generate from / img2imgまたはinpaintを行う元画像
            mask_path: mask in inpainting / inpaint時のマスク
            strength: img2img strength / img2img時のstrength
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.strength = strength


class NetworkMeta:
    def __init__(self,
                 network_module: Optional[List[str]] = None,
                 network_pre_calc: bool = False,
                 network_mul: Optional[List[float]] = None,
                 network_args: Optional[List[str]] = None,
                 network_weights: Optional[List[str]] = None,
                 network_merge: bool = False,
                 network_show_meta: bool = False,
                 ):
        """

        Args:
            network_module: additional network module to use / 追加ネットワークを使う時そのモジュール名 such as `networks.lora`
            network_pre_calc: pre-calculate network for generation / ネットワークのあらかじめ計算して生成する
            network_mul: additional network multiplier / 追加ネットワークの効果の倍率
            network_args: additional argmuments for network (key=value) / ネットワークへの追加の引数
            network_weights: additional network weights to load / 追加ネットワークの重み
            network_merge: merge network weights to original model / ネットワークの重みをマージする
            network_show_meta:show metadata of network model / ネットワークモデルのメタデータを表示する
        """

        self.network_module = network_module
        self.network_pre_calc = network_pre_calc
        self.network_mul = network_mul
        self.network_args = network_args
        self.network_weights = network_weights
        self.network_merge = network_merge
        self.network_show_meta = network_show_meta


class BatchDrawMeta:
    def __init__(self,
                 n_iter: int = 1,
                 images_per_prompt: int = 1,
                 iter_same_seed: bool = False,
                 batch_size: int = 1,
                 ):
        """

        Args:
            n_iter: sample this often / 繰り返し回数
            images_per_prompt: number of images per prompt / プロンプトあたりの出力枚数
            iter_same_seed: use same seed for all prompts in iteration if no seed specified / 乱数seedの指定がないとき繰り返し内はすべて同じseedを使う（プロンプト間の差異の比較用）
            batch_size: batch size / バッチサイズ
        """
        self.n_iter = n_iter
        self.images_per_prompt = images_per_prompt
        self.iter_same_seed = iter_same_seed
        self.batch_size = batch_size


class TextualInversionMeta:
    def __init__(self,
                 textual_inversion_embeddings: Optional[List[str]] = None,
                 XTI_embeddings: Optional[List[str]] = None,
                 ):
        """

        Args:
            textual_inversion_embeddings: Embeddings files of Textual Inversion / Textual Inversionのembeddings
            XTI_embeddings: Embeddings files of Extended Textual Inversion / Extended Textual Inversionのembeddings
        """
        self.textual_inversion_embeddings = textual_inversion_embeddings
        self.XTI_embeddings = XTI_embeddings
