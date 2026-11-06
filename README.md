# SeedVR2 Upscale for InvokeAI

Upscale still images in InvokeAI workflows. Built for InvokeAI 7.0.0-alpha.4.

## Install

Copy the `seedvr2_upscale` folder into your InvokeAI install's `nodes` folder. The resulting path should be `nodes/seedvr2_upscale/__init__.py`. Restart InvokeAI.

## Use

1. In **Workflows**, add **Upscale (SeedVR2)** and connect an image to its **Image** input.
2. In **Model folder**, enter the full path to any folder you want to use for SeedVR2 models. You can use the same folder as ComfyUI.
3. Pick a **Variant** and **Scale**, then run the workflow. The defaults are **3B FP8** and **2×**. Leave **Seed** at `-1` for a new random seed each run.

The node reuses matching files in **Model folder** and downloads anything missing there. Download progress appears in the Invoke console. The size shown next to each variant is its checkpoint size on disk.

## License

The node pack is licensed under [Apache 2.0](LICENSE). The SeedVR2 model downloads are also published under Apache 2.0.

The implementation draws on [ByteDance SeedVR2](https://github.com/ByteDance-Seed/SeedVR), the [NumZ and AInVFX ComfyUI implementation](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), and [vLLM-Omni's SeedVR2 implementation](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/diffusion/models/seedvr2).
