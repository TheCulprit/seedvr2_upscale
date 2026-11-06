"""Invoke cache integration and one-step still-image restoration."""
from hashlib import sha256
from io import BytesIO

from diffusers import AutoencoderKL
from PIL import Image
from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_dequantizing_linear import (
    CustomDequantizingLinear,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.quantization.fp8_scaled import attach_fp8_scales, extract_fp8_scaled_layers, parse_quantization_metadata
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.quantization.int8_convrot import (
    extract_int8_convrot_markers, parse_comfy_quant_marker, swap_in_int8_linears,
)
from invokeai.backend.quantization.nvfp4 import install_nvfp4_layers, pop_nvfp4_layers

from .catalog import CONDITIONING, GIB
from .transformer import SeedVR2Transformer

LATENT_SCALE = 0.9152


class FrameUpsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upscale_conv = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, sample):
        return self.conv(F.pixel_shuffle(self.upscale_conv(sample), 2))


def _one_frame_weight(name, value, shape):
    if ".upsamplers.0.upscale_conv." in name:
        # The causal decoder keeps only time index zero for a one-frame output.
        channels = shape[0] // 4
        remaining = value.shape[1:]
        value = value.reshape(4, -1, channels, *remaining)[:, 0].transpose(0, 1)
        value = value.reshape(4 * channels, *remaining)
    if value.ndim == 5 and len(shape) == 4:
        # The first frame is repeated across causal padding, so fold the time taps into one 2D kernel.
        value = value.sum(dim=2, dtype=torch.float32)
    return value


def _load_weights(model, path, vae=False):
    targets = model.state_dict()
    state = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for original in checkpoint.keys():
            name = original.replace(".rope.rope.freqs", ".rope.freqs")
            if name not in targets or name in state:
                raise ValueError(f"{path.name} contains an unexpected tensor: {original}.")
            value = checkpoint.get_tensor(original)
            if vae:
                value = _one_frame_weight(name, value, targets[name].shape)
            dtype = torch.float32 if name.endswith(".rope.freqs") else torch.bfloat16
            if value.shape != targets[name].shape:
                raise ValueError(f"{path.name} has incompatible dimensions for {original}.")
            state[name] = value.to(dtype=dtype)
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as error:
        raise ValueError(f"{path.name} is incomplete or incompatible with this SeedVR2 variant.") from error
    return model.requires_grad_(False).eval()


class SeedVR2GGUFLinear(CustomDequantizingLinear):
    def __init__(self, source):
        super().__init__()
        CustomModuleMixin.__init__(self)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.quantization = source.weight._ggml_quantization_type
        self.register_buffer("weight", source.weight.quantized_data)
        self.register_buffer("bias", source.bias.detach() if source.bias is not None else None)

    def _dequantized_weight(self, device, dtype):
        packed = GGMLTensor(
            self.weight.to(device), self.quantization, torch.Size((self.out_features, self.in_features)), dtype
        )
        return packed.get_dequantized_tensor()

    def dequant_transient_bytes(self, compute_dtype):
        return self.weight.nbytes + self.in_features * self.out_features * (compute_dtype.itemsize + 8)


def _wrap_gguf_linears(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and isinstance(module.weight, GGMLTensor):
            parent, _, attribute = name.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            setattr(owner, attribute, SeedVR2GGUFLinear(module))


def _transformer_state(path):
    if path.suffix == ".gguf":
        raw = gguf_sd_loader(path, torch.bfloat16, convrot="decode")
        metadata = {}
    else:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            raw = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
            metadata = checkpoint.metadata() or {}

    state = {}
    for original, value in raw.items():
        name = original.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.")
        name = name.replace(".rope.rope.freqs", ".rope.freqs")
        if name in state:
            raise ValueError(f"{path.name} has duplicate tensors for {name}.")
        state[name] = value
    return state, metadata


def _load_transformer(path):
    state, metadata = _transformer_state(path)
    projection = state.get("vid_in.proj.weight")
    if projection is None or tuple(projection.shape) not in ((2560, 132), (3072, 132)):
        raise ValueError(f"{path.name} does not match the SeedVR2 3B or 7B transformer layout.")
    large = projection.shape[0] == 3072
    with torch.device("meta"):
        model = (
            SeedVR2Transformer(width=3072, heads=24, layers=36, separate_layers=36, large=True)
            if large else SeedVR2Transformer()
        )
    model.requires_grad_(False)

    conditioning = state.pop("positive_conditioning", None)
    state.pop("negative_conditioning", None)
    hints = parse_quantization_metadata(metadata)
    hints.update({
        name.removesuffix(".comfy_quant"): parse_comfy_quant_marker(value)
        for name, value in state.items() if name.endswith(".comfy_quant")
    })
    int_layers = extract_int8_convrot_markers(state, accept_int4=True, header_layers=hints)
    nv_layers = pop_nvfp4_layers(state, header_layers=hints)
    # FP8 scale extraction consumes all weight_scale keys, including non-FP8 scales.
    int_scales = {}
    for layer in int_layers:
        scale_key = f"{layer}.weight_scale"
        if scale_key in state:
            int_scales[scale_key] = state.pop(scale_key)
    fp8_layers = extract_fp8_scaled_layers(state, layer_hints=hints)
    state.update(int_scales)
    quantized_keys = {
        f"{name}.{suffix}" for name in int_layers for suffix in ("weight", "weight_scale")
    }
    for name, value in state.items():
        if name in quantized_keys:
            continue
        is_linear_weight = name.endswith(".weight") and isinstance(
            model.get_submodule(name.removesuffix(".weight")), nn.Linear
        )
        if isinstance(value, GGMLTensor):
            if is_linear_weight:
                continue
            value = value.get_dequantized_tensor()
        if is_linear_weight and value.dtype == torch.float8_e4m3fn and (
            large or conditioning is not None or name.removesuffix(".weight") in fp8_layers
        ):
            continue
        if not value.is_floating_point():
            raise ValueError(f"{path.name} has an unsupported quantized tensor: {name}.")
        state[name] = value.to(torch.float32 if name.endswith(".rope.freqs") else torch.bfloat16)

    swap_in_int8_linears(model, state, int_layers)
    install_nvfp4_layers(model, state, nv_layers, torch.bfloat16)
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as error:
        raise ValueError(f"{path.name} is incomplete or incompatible with this SeedVR2 variant.") from error

    if attach_fp8_scales(model, fp8_layers) != len(fp8_layers):
        raise ValueError(f"{path.name} contains FP8 scales that could not be applied.")
    if conditioning is not None:
        _validate_conditioning(conditioning)
    model.register_buffer("conditioning", conditioning, persistent=False)
    _wrap_gguf_linears(model)
    apply_custom_layers_to_model(model)
    return model.eval()


def _load_vae(path):
    # A still frame uses the SD3-style spatial path of SeedVR2's causal VAE.
    blocks = ("DownEncoderBlock2D",) * 4
    up_blocks = ("UpDecoderBlock2D",) * 4
    with torch.device("meta"):
        model = AutoencoderKL(
            in_channels=3, out_channels=3, block_out_channels=(128, 256, 512, 512),
            layers_per_block=2, latent_channels=16, norm_num_groups=32,
            down_block_types=blocks, up_block_types=up_blocks,
            use_quant_conv=False, use_post_quant_conv=False, force_upcast=False, sample_size=512,
        )
        for block, channels in zip(model.decoder.up_blocks[:3], (512, 512, 256), strict=True):
            block.upsamplers[0] = FrameUpsample(channels)

    model.enable_tiling()
    return _load_weights(model, path, vae=True)


def _read_conditioning(path):
    with path.open("rb") as handle:
        data = handle.read(CONDITIONING.limit_bytes + 1)
    if sha256(data).hexdigest() != CONDITIONING.checksum:
        raise ValueError(f"{path.name} does not match the published SeedVR2 conditioning file.")
    tensor = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    _validate_conditioning(tensor)
    return tensor


def _validate_conditioning(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[1] != 5120 or tensor.shape[0] == 0:
        raise ValueError("The SeedVR2 checkpoint requires a nonempty conditioning tensor with 5120 channels.")


@torch.inference_mode()
def upscale(image, output_size, seed, checkpoint, vae_path, conditioning, load_model, progress):
    width, height = output_size
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    source = pil_to_tensor(resized).float().div_(127.5).sub_(1).unsqueeze(0)
    source = F.pad(source, (0, -width % 16, 0, -height % 16), mode="replicate")
    working_bytes = max(3 * GIB, 2048 * source.shape[-1] * source.shape[-2])
    text = _read_conditioning(conditioning) if conditioning is not None else None

    progress("Encoding the SeedVR2 input")
    vae_info = load_model(vae_path, _load_vae)
    with vae_info.model_on_device(working_mem_bytes=working_bytes) as (_, model):
        pixels = source.to(device=vae_info.compute_device, dtype=torch.bfloat16)
        latent = model.encode(pixels).latent_dist.mode().squeeze(0).cpu() * LATENT_SCALE
        del pixels, model

    progress("Loading the SeedVR2 transformer")
    transformer = load_model(checkpoint, _load_transformer)
    with transformer.model_on_device(working_mem_bytes=working_bytes) as (_, model):
        device = transformer.compute_device
        latent = latent.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.empty_like(latent).normal_(generator=generator)
        inputs = torch.cat((noise, latent, torch.ones_like(latent[:1])), dim=0)
        if text is None:
            text = model.conditioning
        _validate_conditioning(text)
        velocity = model(inputs, text.to(device=device, dtype=latent.dtype), progress)
        del text
        # For the one-step linear schedule, the clean latent is x at T minus the predicted velocity.
        restored = (noise.float() - velocity.float()).cpu()
        del inputs, velocity, noise, latent, model

    progress("Decoding the SeedVR2 result")
    vae_info = load_model(vae_path, _load_vae)
    with vae_info.model_on_device(working_mem_bytes=working_bytes) as (_, model):
        vae_input = (restored / LATENT_SCALE).unsqueeze(0).to(device=vae_info.compute_device, dtype=torch.bfloat16)
        pixels = model.decode(vae_input).sample[0, :, :height, :width].float().cpu()
        del vae_input, model

    if not torch.isfinite(pixels).all():
        raise RuntimeError("SeedVR2 returned invalid pixels.")
    # Match global per-channel colour statistics to the input while keeping restored detail.
    original = source[0, :, :height, :width]
    ref_std, ref_mean = torch.std_mean(original, dim=(1, 2), keepdim=True, correction=0)
    out_std, out_mean = torch.std_mean(pixels, dim=(1, 2), keepdim=True, correction=0)
    pixels = (pixels - out_mean) * (ref_std / out_std.clamp_min(1e-6)).clamp(0.5, 2.0) + ref_mean
    result = to_pil_image(pixels.clamp(-1, 1).add_(1).div_(2))
    if "A" in image.getbands() or "transparency" in image.info:
        alpha = image.convert("RGBA").getchannel("A")
        result.putalpha(alpha.resize((width, height), Image.Resampling.BICUBIC))
    return result
