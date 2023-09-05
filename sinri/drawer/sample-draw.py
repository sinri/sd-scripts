from sinri.drawer.DrawerMeta import ModelMeta, PromptMeta, OutputMeta
from sinri.drawer.KohyaSSImageGenerator import KohyaSSImageGenerator

output_dir = 'E:\\sinri\\sd-scripts\\workspace\\gen\\out'
ckpt_dreamshaper_8 = 'E:\\OneDrive\\Leqee\\ai\\civitai\\ckpt_DreamShaper\\dreamshaper_8.safetensors'

drawer = KohyaSSImageGenerator()


base_kwargs={

}

def draw_1():
    drawer.execute(
        model_meta=ModelMeta(
            ckpt=ckpt_dreamshaper_8,
        ),
        W=512,
        H=768,
        prompt_meta=PromptMeta(
            prompt='masterpiece, high resolution, a girl is playing with a cat, ',
            negative_prompt='bad quality, low quality, nsfw, ',
            scale=7,
        ),
        output_meta=OutputMeta(
            outdir=output_dir,
        ),
        sampler='euler_a',
        steps=20,
        xformers=True,
    )


if __name__ == '__main__':
    draw_1()
