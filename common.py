from modal import Stub, Image, Volume, Secret

N_GPUS = 2
GPU_MEM = 80
BASE_MODELS = {
    # Training 70B requires experimental flag fsdp_peft_cpu_offload_for_save.
    "base70": "meta-llama/Llama-2-70b-hf",

    "code7": "codellama/CodeLlama-7b-hf",
    "code13": "codellama/CodeLlama-13b-hf",
    "code34": "codellama/CodeLlama-34b-hf",
}

image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.8",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "llama-recipes @ git+https://github.com/swiftmetrics/llama-recipes.git@ee5f6dc0b17c4b48b4d298dd1fad8ae5108274a5",
        extra_index_url="https://download.pytorch.org/whl/nightly/cu118",
        pre=True,
    )
    .pip_install("huggingface_hub==0.17.1", "hf-transfer==0.1.3", "scipy", "wandb")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)

stub = Stub("llama-finetuning", image=image, secrets=[Secret.from_name("huggingface"), Secret.from_name("wandb")])

# Download pre-trained models into this volume.
stub.pretrained_volume = Volume.persisted("pretrained-vol")

# Save trained models into this volume.
stub.results_volume = Volume.persisted("results-vol")

VOLUME_CONFIG = {
    "/pretrained": stub.pretrained_volume,
    "/results": stub.results_volume,
}
