"""Published SeedVR2 checkpoints supported by this node."""
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import Callable
from urllib.request import urlopen

from invokeai.backend.util.logging import InvokeAILogger

Progress = Callable[[str], None]


@dataclass(frozen=True)
class ModelFile:
    repo: str
    filename: str
    checksum: str
    limit_bytes: int
    remote_prefix: str = ""

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.remote_prefix}{self.filename}"


REPO = "numz/SeedVR2_comfyUI"
COMFY_REPO = "Comfy-Org/SeedVR2"
GIB = 1024**3
MIB = 1024**2
CONSOLE_INTERVAL = 10
CHECKPOINTS = {
    "3B FP8": ModelFile(
        REPO, "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff", 3_391_544_696,
    ),
    "3B FP16": ModelFile(
        REPO, "seedvr2_ema_3b_fp16.safetensors",
        "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304", 6_783_018_808,
    ),
    "7B FP16": ModelFile(
        REPO, "seedvr2_ema_7b_fp16.safetensors",
        "7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a", 16_479_334_424,
    ),
    "7B FP8": ModelFile(
        REPO, "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
        "1fdbf3877b7d1eb266038d3a165a977f17dbb4daa4a0f0d334d5461476963037", 8_239_729_704,
    ),
    "7B Sharp FP16": ModelFile(
        REPO, "seedvr2_ema_7b_sharp_fp16.safetensors",
        "20a93e01ff24beaeebc5de4e4e5be924359606c356c9c51509fba245bd2d77dd", 16_479_334_424,
    ),
    "7B Sharp FP8": ModelFile(
        REPO, "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors",
        "4261d18fd9c331f4c8f14b475c9148bd8c3f1512240ace55fe31a179e0a960b0", 8_239_729_704,
    ),
    "7B Mixed FP8/FP16": ModelFile(
        "AInVFX/SeedVR2_comfyUI", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "3d68b5ec0b295ae28092e355c8cad870edd00b817b26587d0cb8f9dd2df19bb2", 8_466_296_338,
    ),
    "7B Sharp Mixed FP8/FP16": ModelFile(
        "AInVFX/SeedVR2_comfyUI", "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "0d2c5b8be0fda94351149c5115da26aef4f4932a7a2a928c6f184dda9186e0be", 8_466_296_338,
    ),
    "3B GGUF Q3_K_M": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q3_K_M.gguf",
        "44b67c65b272cb778cef68c67370a5703a4e1ab09e2cd5b9a8e2a4eb4e3bdb8c", 1_553_006_944,
    ),
    "3B GGUF Q4_K_M": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q4_K_M.gguf",
        "e665e3909de1a8c88a69c609bca9d43ff5a134647face2ce4497640cc3597f0e", 1_995_344_224,
    ),
    "3B GGUF Q5_K_M": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q5_K_M.gguf",
        "1d09f8894d1cc6c8545b8b5d29e337f2acf42cb14a0660539aea52acb0d22b28", 2_411_661_664,
    ),
    "3B GGUF Q6_K": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q6_K.gguf",
        "67e49503a8c7fb1112adf1e15668efbdfee0a3950e8b4921726fd34d8356b0e0", 2_853_998_944,
    ),
    "3B GGUF Q8_0": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q8_0.gguf",
        "be0d60083a2051a265eb4b77f28edf494e6db67ffc250216f32b72292e5cbd96", 3_660_613_984,
    ),
    "7B GGUF Q3_K_M": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b-Q3_K_M.gguf",
        "1791ab8fa1973db490166b6800ab1cb903e099676f55bf7db610fc96b858023d", 3_675_389_728,
    ),
    "7B GGUF Q4_K_M": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b-Q4_K_M.gguf",
        "db9cb2ad90ebd40d2e8c29da2b3fc6fd03ba87cd58cbadceccca13ad27162789", 4_758_306_592,
    ),
    "7B GGUF Q5_K_M": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b-Q5_K_M.gguf",
        "30cf3359ff06293e021d6f21705a9b82c2a38b3dd03fa5c585b8c9234db6cb1d", 5_777_522_464,
    ),
    "7B GGUF Q6_K": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b-Q6_K.gguf",
        "86c43b686b28d3db7d461610ed68cc2969bbdd7ec5ec96999087004d2bdcb77d", 6_860_439_328,
    ),
    "7B GGUF Q8_0": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b-Q8_0.gguf",
        "669788655e8f15f306284f267a444e9766c8a421869577b16a961e43029c737b", 8_835_170_080,
    ),
    "7B Sharp GGUF Q3_K_M": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b_sharp-Q3_K_M.gguf",
        "7cfbf57474369ab0bf06102279f61f09bc07dd1312d4a9b08db07c71428f7436", 3_675_389_728,
    ),
    "7B Sharp GGUF Q4_K_M": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
        "7aed800ac4eb8e0d18569a954c0ff35f5a1caa3ed5d920e66cc31405f75b6e69", 4_758_306_592,
    ),
    "7B Sharp GGUF Q5_K_M": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b_sharp-Q5_K_M.gguf",
        "709db1c6f19c2e5a49cdf46141aab5b849731b36536464acaaabceaf61773e3f", 5_777_522_464,
    ),
    "7B Sharp GGUF Q6_K": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b_sharp-Q6_K.gguf",
        "89ab72e527a96cf16644797fa73c1e0580bff4970e587f77b5d561e5331e48b1", 6_860_439_328,
    ),
    "7B Sharp GGUF Q8_0": ModelFile(
        "cmeka/SeedVR2-GGUF", "seedvr2_ema_7b_sharp-Q8_0.gguf",
        "b1f81cb5700b0b1f432f2c785528356c952c41c74d03d205c6f14b0bd6da303d", 8_835_170_080,
    ),
    "3B FP16 (Comfy-Org)": ModelFile(
        COMFY_REPO, "seedvr2_3b_fp16.safetensors",
        "98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660", 6_784_268_336, "diffusion_models/",
    ),
    "3B FP8 (Comfy-Org)": ModelFile(
        COMFY_REPO, "seedvr2_3b_fp8_e4m3fn.safetensors",
        "a0226eaa2c3e6f47ae5ce83225120f16479da890ced1a3bc32b1a14619787914", 3_392_794_232, "diffusion_models/",
    ),
    "3B INT8 ConvRot": ModelFile(
        COMFY_REPO, "seedvr2_3b_int8_convrot.safetensors",
        "c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157", 3_458_259_704, "diffusion_models/",
    ),
    "3B MXFP8": ModelFile(
        COMFY_REPO, "seedvr2_3b_mxfp8.safetensors",
        "768623e3bfb1752b4d0668782751b5fead58b1bcb153f0b5e03a423095630297", 3_557_857_550, "diffusion_models/",
    ),
    "3B NVFP4": ModelFile(
        COMFY_REPO, "seedvr2_3b_nvfp4.safetensors",
        "c8dea38b04d43295621726e2cd371c0d2d001006169c113aea17950f2cb2e295", 1_996_687_726, "diffusion_models/",
    ),
    "7B FP16 (Comfy-Org)": ModelFile(
        COMFY_REPO, "seedvr2_7b_fp16.safetensors",
        "2742ca6fee63bc5cc1773f426dd4b07b78cad27f51c9ea5cd42b035e6b592252", 16_480_583_960, "diffusion_models/",
    ),
    "7B FP8 (Comfy-Org)": ModelFile(
        COMFY_REPO, "seedvr2_7b_fp8_e4m3fn.safetensors",
        "5065e77d647dd553d9090a81e20d6de590d931a61df79d785e008433926ee418", 8_240_979_248, "diffusion_models/",
    ),
    "7B INT8 ConvRot": ModelFile(
        COMFY_REPO, "seedvr2_7b_int8_convrot.safetensors",
        "5aa0d25fc9d35e449b659d0c9a5dcb22e2a4fa04032101b95a39da42b32c1be6", 8_334_897_976, "diffusion_models/",
    ),
    "7B MXFP8": ModelFile(
        COMFY_REPO, "seedvr2_7b_mxfp8.safetensors",
        "b40804f47910d96c5089c728cc7ec8b57b956750eabb6397dc4e6e697477263d", 8_581_725_960, "diffusion_models/",
    ),
    "7B NVFP4": ModelFile(
        COMFY_REPO, "seedvr2_7b_nvfp4.safetensors",
        "cc4af1a7bd5377066496f393555478323e806fa21163bdbe3409451aface9b93", 4_759_694_792, "diffusion_models/",
    ),
    "7B Sharp FP16 (Comfy-Org)": ModelFile(
        COMFY_REPO, "seedvr2_7b_sharp_fp16.safetensors",
        "70823bca54b9c24eeb56e1c452697c7c2a430867e58db0e376c6e260f3a4489d", 16_480_583_960, "diffusion_models/",
    ),
    "7B Sharp FP8 (Comfy-Org)": ModelFile(
        COMFY_REPO, "seedvr2_7b_sharp_fp8_e4m3fn.safetensors",
        "7602c5f70868d28e7730035e4e9d745b05d661c8f0a7eb758e63f9c8603596ef", 8_240_979_248, "diffusion_models/",
    ),
    "7B Sharp INT8 ConvRot": ModelFile(
        COMFY_REPO, "seedvr2_7b_sharp_int8_convrot.safetensors",
        "db48be2f1cc7e36b01a2aa529810f5d9c6a971edd29be225cf1b0eb18d51c366", 8_334_897_976, "diffusion_models/",
    ),
    "7B Sharp MXFP8": ModelFile(
        COMFY_REPO, "seedvr2_7b_sharp_mxfp8.safetensors",
        "0d621ec1561a11ca9b5f432ec6d4e09b263b61f4b83b0280552c8b4add030ec3", 8_581_725_960, "diffusion_models/",
    ),
    "7B Sharp NVFP4": ModelFile(
        COMFY_REPO, "seedvr2_7b_sharp_nvfp4.safetensors",
        "80d57af7722f5a5bd4c01d2ab2688f2bf05e552e59d3d3287257de709db10397", 4_759_694_792, "diffusion_models/",
    ),
    "7B GGUF Q4_K (SeedVR2X)": ModelFile(
        "hekmon/seedvr2x", "seedvr2x_ema_7b_Q4_K.gguf",
        "7f4642d0701c343a33fbd0a72518ccdb08c5b02d6fbb5f58624b9957b42ecde5", 4_758_307_552,
    ),
    "7B GGUF Q4_K iMatrix (SeedVR2X)": ModelFile(
        "hekmon/seedvr2x", "seedvr2x_ema_7b_Q4_K_imatrix.gguf",
        "2d9f7e607d9cdb1dd86debf9b89c2e3dbd34d19d99016454f048d827ca3f1385", 4_758_308_480,
    ),
    "7B GGUF Q8_0 (SeedVR2X)": ModelFile(
        "hekmon/seedvr2x", "seedvr2x_ema_7b_Q8_0.gguf",
        "67ea572a82bea623da75d7b275ace0d64af4f1af67454fae4fd9478cb03d8b52", 8_835_171_040,
    ),
    "7B GGUF Dynamic (SeedVR2X)": ModelFile(
        "hekmon/seedvr2x", "seedvr2x_ema_7b_dyn.gguf",
        "f8c0c50d233dad8a26fb82d5eda4072c2ad935100e842b3fee953994849a487d", 4_757_055_264,
    ),
    "7B Sharp GGUF Q4_K (SeedVR2X)": ModelFile(
        "hekmon/seedvr2x", "seedvr2x_ema_7b_sharp_Q4_K.gguf",
        "a5e423a50bdb0b0b491a0c981a5357fa1dba82819791ed0d7cf241153eeee023", 4_758_307_584,
    ),
    "7B Sharp GGUF Q4_K iMatrix (SeedVR2X)": ModelFile(
        "hekmon/seedvr2x", "seedvr2x_ema_7b_sharp_Q4_K_imatrix.gguf",
        "486a7da980d83a5fa475d623cf9d29675355975f0e57bacb390f30008163fab1", 4_758_308_512,
    ),
    "7B Sharp GGUF Q8_0 (SeedVR2X)": ModelFile(
        "hekmon/seedvr2x", "seedvr2x_ema_7b_sharp_Q8_0.gguf",
        "03fad523334f2721ae4396fb41f9d273042980dea2a0f16071da65c9aad8ca1a", 8_835_171_072,
    ),
    "7B Sharp GGUF Dynamic (SeedVR2X)": ModelFile(
        "hekmon/seedvr2x", "seedvr2x_ema_7b_sharp_dyn.gguf",
        "90ba80c85e25ee08162882be58c0e05f1b4fc19c8dd9837b6f87c0efc4fad779", 4_757_129_024,
    ),
}
VAE = ModelFile(
    REPO, "ema_vae_fp16.safetensors",
    "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1", GIB,
)
CONDITIONING = ModelFile(
    "ByteDance-Seed/SeedVR2-3B", "pos_emb.pt",
    "fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799", 2 * 1024**2,
)


def _check_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"Expected a model file at {path}.")
    return path.resolve()


def find_or_download(folder: Path, spec: ModelFile, progress: Progress) -> Path:
    destination = folder / spec.filename
    if destination.exists() or destination.is_symlink():
        return _check_file(destination)

    progress(f"Downloading {spec.filename}")
    logger = InvokeAILogger.get_logger("SeedVR2")
    logger.info("Downloading %s to %s", spec.filename, folder)
    next_console_report = monotonic() + CONSOLE_INTERVAL
    digest = sha256()
    received = 0
    next_report = 0
    temporary_path = None
    try:
        with urlopen(spec.url, timeout=30) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError("The SeedVR2 download was redirected away from HTTPS.")
            content_length = response.headers.get("Content-Length", "")
            total = int(content_length) if content_length.isdecimal() else 0
            with NamedTemporaryFile(dir=folder, prefix=".seedvr2-", suffix=".part", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := response.read(1024**2):
                    received += len(chunk)
                    if received > spec.limit_bytes:
                        raise ValueError(f"The download for {spec.filename} exceeds its expected size.")
                    if received >= next_report:
                        progress(f"Downloading {spec.filename} ({received // 1024**2} MiB)")
                        next_report = received + 16 * 1024**2
                    digest.update(chunk)
                    temporary.write(chunk)
                    now = monotonic()
                    if now >= next_console_report:
                        if total > 0:
                            logger.info(
                                "Downloading %s: %.1f / %.1f MiB (%.0f%%)",
                                spec.filename, received / MIB, total / MIB, received * 100 / total,
                            )
                        else:
                            logger.info("Downloading %s: %.1f MiB received", spec.filename, received / MIB)
                        next_console_report = now + CONSOLE_INTERVAL
        if digest.hexdigest() != spec.checksum:
            raise ValueError(f"The download for {spec.filename} did not match its published SHA-256.")
        progress(f"Finishing {spec.filename}")
        try:
            # Publish a complete file without replacing a file another client just installed.
            if os.name == "nt":
                temporary_path.rename(destination)
            else:
                os.link(temporary_path, destination)
        except FileExistsError:
            pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    downloaded = _check_file(destination)
    logger.info("Download verified: %s (%.1f MiB)", downloaded, received / MIB)
    return downloaded
