FROM woodrex/rocm612-torch24-gfx803

ENV HSA_OVERRIDE_GFX_VERSION=8.0.3
ENV PYTORCH_ROCM_ARCH=gfx803
ENV PYTHONUNBUFFERED=1
ENV STABLE_DIFFUSION_REPO=https://github.com/CompVis/stable-diffusion.git
ENV STABLE_DIFFUSION_COMMIT_HASH=21f890f9da3cfbeaba8e2ac3c425ee9e998d5229

WORKDIR /stable-diffusion-webui

RUN git clone https://github.com/woodrex83/stable-diffusion-webui-rx580.git .

RUN pip install -r requirements_versions.txt
RUN git clone --depth 1 https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion-stability-ai \
 && pip install taming-transformers-rom1504 --no-cache-dir
RUN sed -i '/padding=0))$/a\        # Match sgm.modules.attention.SpatialTransformer API expected by WebUI hijacks.\n        self.use_linear = False' \
    repositories/stable-diffusion-stability-ai/ldm/modules/attention.py
RUN mkdir -p repositories/stable-diffusion-stability-ai/ldm/modules/midas \
 && printf "ISL_PATHS = {\n    'dpt_large': 'midas/dpt_large-midas-2f21e586.pt',\n    'dpt_hybrid': 'midas/dpt_hybrid-midas-501f0c75.pt',\n    'midas_v21': 'midas/midas_v21-f6b98070.pt',\n    'midas_v21_small': 'midas/midas_v21_small-70d6b9c8.pt',\n}\n\n\nclass _UnavailableMidas(Exception):\n    pass\n\n\ndef load_model(model_type: str):\n    raise _UnavailableMidas(\"MiDaS models are not bundled. Mount models/midas with weights or provide a repository containing ldm.modules.midas.\")\n\n\n__all__ = ['ISL_PATHS', 'load_model']\n" > repositories/stable-diffusion-stability-ai/ldm/modules/midas/api.py \
 && printf "from .api import ISL_PATHS, load_model\n" > repositories/stable-diffusion-stability-ai/ldm/modules/midas/__init__.py \
 && mkdir -p repositories/stable-diffusion-stability-ai/ldm/data \
 && printf "# Stub for ldm.data.util\nclass AddMiDaS:\n    def __init__(self, *args, **kwargs):\n        pass\n    def __call__(self, x):\n        return x\n" > repositories/stable-diffusion-stability-ai/ldm/data/util.py \
 && touch repositories/stable-diffusion-stability-ai/ldm/data/__init__.py \
 && printf "\n\n# Stub for LatentDepth2ImageDiffusion\nfrom ldm.models.diffusion.ddpm import LatentDiffusion\nclass LatentDepth2ImageDiffusion(LatentDiffusion):\n    pass\n" >> repositories/stable-diffusion-stability-ai/ldm/models/diffusion/ddpm.py
RUN sed -i '/ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES\["softmax-xformers"\]/i\
if not hasattr(ldm.modules.attention.BasicTransformerBlock, "ATTENTION_MODES"):\\\n    ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES = {}' \
    modules/sd_hijack.py

RUN sed -i 's/fetch --refetch/fetch --all/g' modules/launch_utils.py

ENTRYPOINT ["python", "launch.py",  "--listen", "--medvram"]
# ENTRYPOINT ["python", "launch.py", "--api", "--listen", "--no-half", "--precision", "full", "--opt-sdp-attention", "--medvram"]
