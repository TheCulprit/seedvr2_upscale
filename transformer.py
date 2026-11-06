"""Single-frame SeedVR2 transformer using PyTorch attention.

Checkpoint layout and restoration math were checked against the public
ByteDance SeedVR2, NumZ, and vLLM-Omni implementations.
"""
from math import ceil, sqrt

from diffusers.models.embeddings import get_timestep_embedding
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F


def _segments(length, window, shifted):
    # The shifted grid clips its first and last windows at the image boundaries.
    start = -((window + 1) // 2) if shifted and length > window else 0
    return [slice(max(0, i), min(i + window, length)) for i in range(start, length, window)]


def _image_windows(height, width, shifted):
    reference_scale = sqrt(3600 / (height * width))
    tile_h = max(1, ceil(round(height * reference_scale) / 3))
    tile_w = max(1, ceil(round(width * reference_scale) / 3))
    return [(rows, cols) for cols in _segments(width, tile_w, shifted)
            for rows in _segments(height, tile_h, shifted)]


class Streams(nn.Module):
    """Keep the released checkpoint's independent or shared stream weights."""

    def __init__(self, factory, shared, image_only=False):
        super().__init__()
        self.shared = shared
        self.image_only = image_only
        if shared:
            self.all = factory()
        else:
            self.vid = factory()
            if not image_only:
                self.txt = factory()

    def forward(self, image, text, *args):
        if self.shared:
            return self.all(image, *args), text if self.image_only else self.all(text, *args)
        return self.vid(image, *args), text if self.image_only else self.txt(text, *args)


class SpatialRotary(nn.Module):
    def __init__(self, axis_width, text_rotary):
        super().__init__()
        self.register_buffer("freqs", torch.empty(axis_width // 2))
        self.text_rotary = text_rotary

    def _angles(self, positions, device):
        return (positions.unsqueeze(-1) * self.freqs.to(device=device, dtype=torch.float32)).repeat_interleave(2, -1)

    def text_angles(self, count, device):
        return self._angles(torch.arange(count, device=device), device).repeat(1, 3)

    def image_angles(self, height, width, text_count, device):
        if self.text_rotary:
            row_positions = torch.arange(height, device=device)
            col_positions = torch.arange(width, device=device)
            time_position = text_count
        else:
            row_positions = torch.linspace(-1, 1, height, device=device)
            col_positions = torch.linspace(-1, 1, width, device=device)
            time_position = -1
        row, col = torch.meshgrid(row_positions, col_positions, indexing="ij")
        coordinates = (torch.full_like(row, time_position), row, col)
        return torch.cat([self._angles(axis.flatten(), device) for axis in coordinates], dim=-1)

    @staticmethod
    def apply_angles(values, angles):
        # Rotate only the paired prefix of each head. The last two dimensions stay unchanged.
        size = angles.shape[-1]
        active = values[..., :size].float()
        even, odd = active.unflatten(-1, (-1, 2)).unbind(-1)
        perpendicular = torch.stack((-odd, even), dim=-1).flatten(-2)
        rotated = active * angles.cos() + perpendicular * angles.sin()
        return torch.cat((rotated.to(values.dtype), values[..., size:]), dim=-1)


class WindowAttention(nn.Module):
    def __init__(self, width, heads, shared, text_rotary):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.proj_qkv = Streams(lambda: nn.Linear(width, 3 * width, bias=False), shared)
        self.proj_out = Streams(lambda: nn.Linear(width, width), shared)
        self.norm_q = Streams(lambda: nn.RMSNorm(self.head_width, eps=1e-5), shared)
        self.norm_k = Streams(lambda: nn.RMSNorm(self.head_width, eps=1e-5), shared)
        self.rope = SpatialRotary(self.head_width // (3 if text_rotary else 6), text_rotary)

    def forward(self, image, text, grid, windows):
        image_qkv, text_qkv = self.proj_qkv(image, text)
        def split(projected):
            return rearrange(projected, "n (q h d) -> q h n d", q=3, h=self.heads).unbind(0)

        iq, ik, iv = split(image_qkv)
        tq, tk, tv = split(text_qkv)
        iq, tq = self.norm_q(iq, tq)
        ik, tk = self.norm_k(ik, tk)
        count = text.shape[0]
        if self.rope.text_rotary:
            text_angles = self.rope.text_angles(count, image.device)
            tq = self.rope.apply_angles(tq, text_angles)
            tk = self.rope.apply_angles(tk, text_angles)
        iq, ik, iv = [x.unflatten(1, grid) for x in (iq, ik, iv)]
        image_output = torch.empty_like(iv)
        text_sum = torch.zeros(tv.shape, device=tv.device, dtype=torch.float32)

        for rows, cols in windows:
            height = rows.stop - rows.start
            width = cols.stop - cols.start
            angles = self.rope.image_angles(height, width, count, image.device)
            if not self.rope.text_rotary:
                angles = angles.to(iq.dtype)
            q = self.rope.apply_angles(iq[:, rows, cols].flatten(1, 2), angles)
            k = self.rope.apply_angles(ik[:, rows, cols].flatten(1, 2), angles)
            v = iv[:, rows, cols].flatten(1, 2)
            attended = F.scaled_dot_product_attention(
                torch.cat((q, tq), dim=1).unsqueeze(0),
                torch.cat((k, tk), dim=1).unsqueeze(0),
                torch.cat((v, tv), dim=1).unsqueeze(0),
            ).squeeze(0)
            image_output[:, rows, cols] = attended[:, :-count].unflatten(1, (height, width))
            text_sum.add_(attended[:, -count:])

        text_output = text_sum.div_(len(windows)).to(text.dtype)
        return self.proj_out(
            rearrange(image_output, "h y x d -> (y x) (h d)"),
            rearrange(text_output, "h n d -> n (h d)"),
        )


class Modulation(nn.Module):
    def __init__(self, width, names=("attn", "mlp")):
        super().__init__()
        self.names = names
        for name in names:
            self.register_parameter(f"{name}_shift", nn.Parameter(torch.zeros(width)))
            self.register_parameter(f"{name}_scale", nn.Parameter(torch.ones(width)))
            if name != "out":
                self.register_parameter(f"{name}_gate", nn.Parameter(torch.zeros(width)))

    def forward(self, value, embedding, name, residual=False):
        shift, scale, gate = embedding[:, self.names.index(name)].unbind(-1)
        if residual:
            return value * (gate + getattr(self, f"{name}_gate"))
        return value * (scale + getattr(self, f"{name}_scale")) + shift + getattr(self, f"{name}_shift")


class FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        expansion = 256 * ceil((width * 8 // 3) / 256)
        self.proj_in_gate = nn.Linear(width, expansion, bias=False)
        self.proj_in = nn.Linear(width, expansion, bias=False)
        self.proj_out = nn.Linear(expansion, width, bias=False)

    def forward(self, value):
        return self.proj_out(F.silu(self.proj_in_gate(value)) * self.proj_in(value))


class GeluFeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj_in = nn.Linear(width, 4 * width)
        self.proj_out = nn.Linear(4 * width, width)

    def forward(self, value):
        return self.proj_out(F.gelu(self.proj_in(value), approximate="tanh"))


class RestorationBlock(nn.Module):
    def __init__(self, width, heads, shared, last, large):
        super().__init__()
        self.attn = WindowAttention(width, heads, shared, text_rotary=not large)
        self.ada = Streams(lambda: Modulation(width), shared, image_only=last)
        feed_forward = GeluFeedForward if large else FeedForward
        self.mlp = Streams(lambda: feed_forward(width), shared, image_only=last)
        self.width = width
        self.last = last

    def forward(self, image, text, embedding, grid, windows):
        normalized = (F.rms_norm(image, (self.width,), eps=1e-5), F.rms_norm(text, (self.width,), eps=1e-5))
        attn_image, attn_text = self.attn(*self.ada(*normalized, embedding, "attn"), grid, windows)
        attn_image, attn_text = self.ada(attn_image, attn_text, embedding, "attn", True)
        image = image + attn_image
        if not self.last:
            text = text + attn_text

        normalized = (F.rms_norm(image, (self.width,), eps=1e-5), F.rms_norm(text, (self.width,), eps=1e-5))
        out_image, out_text = self.mlp(*self.ada(*normalized, embedding, "mlp"))
        out_image, out_text = self.ada(out_image, out_text, embedding, "mlp", True)
        return image + out_image, text if self.last else text + out_text


class TimestepProjection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj_in = nn.Linear(256, width)
        self.proj_hid = nn.Linear(width, width)
        self.proj_out = nn.Linear(width, 6 * width)

    def forward(self, sample):
        timestep = torch.full((1,), 1000, device=sample.device, dtype=torch.float32)
        sinusoidal = get_timestep_embedding(timestep, 256, downscale_freq_shift=0).to(sample.dtype)
        hidden = F.silu(self.proj_in(sinusoidal))
        return self.proj_out(F.silu(self.proj_hid(hidden)))


class SeedVR2Transformer(nn.Module):
    def __init__(self, width=2560, heads=20, layers=32, separate_layers=10, *, large=False):
        super().__init__()
        # The checkpoint groups patch projections below vid_in.proj and vid_out.proj.
        self.vid_in = nn.ModuleDict({"proj": nn.Linear(4 * 33, width)})
        self.vid_out = nn.ModuleDict({"proj": nn.Linear(width, 4 * 16)})
        self.txt_in = nn.Linear(5120, width)
        self.emb_in = TimestepProjection(width)
        self.blocks = nn.ModuleList(
            RestorationBlock(width, heads, index >= separate_layers, index == layers - 1 and not large, large)
            for index in range(layers)
        )
        self.vid_out_norm = None if large else nn.RMSNorm(width, eps=1e-5)
        if self.vid_out_norm is not None:
            self.vid_out_ada = Modulation(width, names=("out",))

    def forward(self, pixels, text, progress):
        _, height, width = pixels.shape
        grid = (height // 2, width // 2)
        image = rearrange(pixels, "c (h a) (w b) -> (h w) (a b c)", a=2, b=2)
        image = self.vid_in["proj"](image)
        text = self.txt_in(text)
        embedding = self.emb_in(image).reshape(image.shape[-1], 2, 3)
        layouts = tuple(_image_windows(*grid, shifted) for shifted in (False, True))
        for index, block in enumerate(self.blocks):
            if index % 4 == 0:
                progress(f"Restoring SeedVR2 detail ({index + 1}/{len(self.blocks)})")
            image, text = block(image, text, embedding, grid, layouts[index % 2])
        if self.vid_out_norm is not None:
            image = self.vid_out_ada(self.vid_out_norm(image), embedding[:, :1], "out")
        return rearrange(self.vid_out["proj"](image), "(h w) (a b c) -> c (h a) (w b)", h=grid[0], w=grid[1], a=2, b=2)
