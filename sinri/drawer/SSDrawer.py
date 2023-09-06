from typing import Optional, List

from sinri.drawer.DrawerMeta import ModelMeta, OutputMeta, PromptMeta, VaeMeta, TextualInversionMeta, NetworkMeta, \
    ClipMeta
from sinri.drawer.KohyaSSImageGenerator import KohyaSSImageGenerator


class SSDrawer(KohyaSSImageGenerator):
    def __init__(self, env: dict):
        self.__env = env
        self.__parameters: dict = {}
        self.__textual_inversion_embeddings: List[str] = []
        self.__networks: List[dict] = []

        self.reset_all()

    def reset_all(self):
        self.__parameters = {
            'xformers': True,
            'W': 512,
            'H': 512,
            'sampler': 'euler_a',
            'steps': 20,
            'fp16': True,
        }
        self.__textual_inversion_embeddings = []
        self.__networks = []

    def set_xformers_switch(self, use_xformers: bool):
        self.__parameters['xformers'] = use_xformers
        return self

    def set_fp16_switch(self, use_fp16: bool):
        self.__parameters['fp16'] = use_fp16
        return self

    def set_bp16_switch(self, use_bp16: bool):
        self.__parameters['bp16'] = use_bp16
        return self

    def set_model(self, key: str):
        model_dict = self.__env.get('model')
        model_meta = model_dict.get(key, {})
        self.__parameters['model_meta'] = ModelMeta(**model_meta)
        return self

    def load_size(self, width: int, height: int):
        self.__parameters['W'] = width
        self.__parameters['H'] = height
        return self

    def set_sampler(self, sampler: str):
        self.__parameters['sampler'] = sampler
        return self

    def set_steps(self, steps: int):
        self.__parameters['steps'] = steps
        return self

    def set_prompt(self, positive_content: str, positive_scale: float, negative_content: Optional[str] = None,
                   negative_scale: Optional[str] = None):
        k = {}
        if positive_content:
            k['prompt'] = positive_content
        if positive_scale:
            k['scale'] = positive_scale
        if negative_content:
            k['negative_prompt'] = negative_content
        if negative_scale:
            k['negative_scale'] = negative_scale
        self.__parameters['prompt_meta'] = PromptMeta(**k)
        return self

    def set_output_dir(self, store_path: str):
        self.__parameters['output_meta'] = OutputMeta(
            outdir=store_path
        )
        return self

    def set_vae(self, key: str):
        vae_dict = self.__env.get('vae')
        vae_meta = vae_dict.get(key, {})
        self.__parameters['vae_meta'] = VaeMeta(**vae_meta)
        return self

    def set_clip_skip(self, clip_skip: int):
        self.__parameters['clip_meta'] = ClipMeta(
            clip_skip=clip_skip
        )
        return self

    def set_seed(self, seed: int):
        self.__parameters['seed'] = seed
        return self

    def add_textual_inversion(self, key: str):
        textual_inversion_dict = self.__env.get('textual_inversion')
        textual_inversion_meta = textual_inversion_dict.get(key, {})
        self.__textual_inversion_embeddings.append(textual_inversion_meta.get('path'))
        return self

    def add_network(self, key: str, network_mul: Optional[float] = None, ):
        # NetworkMeta
        network_dict = self.__env.get('network')
        network_meta = network_dict.get(key, {})
        self.__networks.append({
            'network_module': network_meta.get('network_module', 'networks.lora'),
            'network_mul': network_mul,
            'network_weight': network_meta.get('path'),
        })
        return self

    def draw(self):
        self.__parameters['textual_inversion_meta'] = TextualInversionMeta(
            textual_inversion_embeddings=self.__textual_inversion_embeddings if len(
                self.__textual_inversion_embeddings) > 0 else None,
        )

        network_module_list = []
        network_weight_list = []
        network_mul_list = []

        for network in self.__networks:
            network_module_list.append(network.get('network_module'))
            network_weight_list.append(network.get('network_weight'))
            network_mul_list.append(network.get('network_mul'))

        self.__parameters['network_meta'] = NetworkMeta(
            network_module=network_module_list,
            network_weights=network_weight_list,
            network_mul=network_mul_list,
        )

        files = self.execute(**self.__parameters)
        return files[0]
