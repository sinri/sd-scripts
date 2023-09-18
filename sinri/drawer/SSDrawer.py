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
        self.__network_pre_calc = False
        self.__network_merge = False
        self.__network_show_meta = False

        self.reset_all()

    def reset_all(self):
        self.__parameters = {
            'xformers': True,
            'the_width': 512,
            'the_height': 512,
            'sampler': 'euler_a',
            'the_steps': 20,
            # 'fp16': True,
        }
        self.__textual_inversion_embeddings = []
        self.__networks = []
        self.__network_pre_calc = False
        self.__network_merge = False
        self.__network_show_meta = False

    def set_xformers_switch(self, use_xformers: bool):
        self.__parameters['xformers'] = bool(use_xformers)
        return self

    def set_fp16_switch(self, use_fp16: bool):
        self.__parameters['fp16'] = bool(use_fp16)
        return self

    def set_bp16_switch(self, use_bp16: bool):
        self.__parameters['bp16'] = bool(use_bp16)
        return self

    def set_model(self, key: str):
        model_dict = self.__env.get('model')
        model_meta = model_dict.get(key, {})
        self.__parameters['model_meta'] = ModelMeta(**model_meta)
        return self

    def set_size(self, width: int, height: int):
        self.__parameters['the_width'] = int(width)
        self.__parameters['the_height'] = int(height)
        return self

    def set_sampler(self, sampler: str):
        self.__parameters['sampler'] = sampler
        return self

    def set_steps(self, steps: int):
        self.__parameters['the_steps'] = int(steps)
        return self

    def set_prompt(self, positive_content: str, positive_scale: float, negative_content: Optional[str] = None,
                   negative_scale: Optional[str] = None):
        k = {
            'max_embeddings_multiples': 1,
        }
        if positive_content:
            k['prompt'] = positive_content
            if 70 * k['max_embeddings_multiples'] < len(positive_content):
                k['max_embeddings_multiples'] = int(len(positive_content) / 70) + 1
        if positive_scale:
            k['scale'] = float(positive_scale)
        if negative_content:
            k['negative_prompt'] = negative_content
            if 70 * k['max_embeddings_multiples'] < len(negative_content):
                k['max_embeddings_multiples'] = int(len(negative_content) / 70) + 1
        if negative_scale:
            k['negative_scale'] = float(negative_scale)

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
            clip_skip=int(clip_skip)
        )
        return self

    def set_seed(self, seed: int):
        self.__parameters['the_seed'] = int(seed)
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
            'network_weight': network_meta.get('path'),
            'network_mul': float(network_mul),
            'network_pre_calc': bool(network_meta.get('network_pre_calc', False)),
            'network_merge': bool(network_meta.get('network_merge', False)),
            'network_show_meta': True,
        })
        return self

    def set_network_flags(self,
                          network_pre_calc: bool,
                          network_merge: bool,
                          network_show_meta: bool,
                          ):
        self.__network_merge = network_merge
        self.__network_show_meta = network_show_meta
        self.__network_pre_calc = network_pre_calc
        return self

    def draw(self) -> str:
        if len(self.__textual_inversion_embeddings) > 0:
            self.__parameters['textual_inversion_meta'] = TextualInversionMeta(
                textual_inversion_embeddings=self.__textual_inversion_embeddings
            )

        if len(self.__networks)>0:
            network_parameters = {
                'network_module': [],  # Optional[List[str]] = None,
                'network_pre_calc': self.__network_pre_calc,
                'network_mul': [],  # Optional[List[float]] = None,
                'network_args': [],  # Optional[List[str]] = None,
                'network_weights': [],  # Optional[List[str]] = None,
                'network_merge': self.__network_merge,
                'network_show_meta': self.__network_show_meta,
            }

            for network in self.__networks:
                network_parameters['network_module'].append(network.get('network_module'))
                network_parameters['network_weights'].append(network.get('network_weight'))
                network_parameters['network_mul'].append(network.get('network_mul'))

            self.__parameters['network_meta'] = NetworkMeta(**network_parameters)

        # debug
        # print('SHOW THY PARAMETERS!')
        # print(self.__parameters)

        files = self.execute(**self.__parameters)
        return files[0]
