E:\sinri\sd-scripts\venv\Scripts\python.exe ^
    E:\sinri\sd-scripts\gen_img_diffusers.py ^
    --ckpt E:\sinri\HuggingFace\MoonFilm ^
    --outdir E:\sinri\sd-scripts\workspace\gen\out ^
    --xformers ^
    --fp16 ^
    --W 512 ^
    --H 768 ^
    --scale 12.5 ^
    --sampler k_euler_a ^
    --steps 32 ^
    --batch_size 1 ^
    --images_per_prompt 1 ^
    --clip_skip 2 ^
    --max_embeddings_multiples 2 ^
    --prompt "masterpiece, 8k, a girl in flowers, sunny morning, bodybuilder --n bad quality, nsfw" ^
    --network_weights E:\\OneDrive\\Leqee\\ai\\civitai\\LoCon_Yog_Sothoth\\fcBodybuildingXL-FBB-000010.safetensors ^
    --network_mul 0.7
