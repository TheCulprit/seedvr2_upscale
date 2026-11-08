from pathlib import Path
from secrets import randbits
from typing import Literal

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.invocation_api import BaseInvocation, ImageField, ImageOutput, Input, InputField, InvocationContext, invocation

from .catalog import CHECKPOINTS, COMFY_REPO, CONDITIONING, VAE, find_or_download


Variant = Literal[tuple(CHECKPOINTS)]


@invocation(
    "seedvr2_upscale",
    title="Upscale (SeedVR2)",
    tags=["upscale", "restoration", "seedvr2"],
    category="upscale",
    version="1.0.0",
)
class SeedVR2UpscaleInvocation(BaseInvocation):
    """Restore and upscale one still image with a SeedVR2 checkpoint."""

    image: ImageField = InputField(description="Image to upscale.")
    model_folder: str = InputField(
        input=Input.Direct, description="Folder to use for existing SeedVR2 weights and missing model downloads."
    )
    variant: Variant = InputField(
        default="3B FP8",
        input=Input.Direct,
        description="SeedVR2 checkpoint to use. Labels show checkpoint file size.",
        ui_choice_labels={
            name: f"{name} ({model.limit_bytes / 1_000_000_000:.2f} GB)"
            for name, model in CHECKPOINTS.items()
        },
    )
    scale: Literal["1.5x", "2x", "2.5x", "3x", "3.5x", "4x"] = InputField(
        default="2x", input=Input.Direct, description="Multiply the image width and height by this factor."
    )
    seed: int = InputField(
        default=-1, ge=-1, le=4294967295, description="Use -1 for a new random seed each run."
    )

    def invoke_internal(self, context, services):
        if self.seed == -1:
            # Random runs must not reuse a previous output from Invoke's invocation cache.
            self._validate_invoke_fields()
            return self.invoke(context)
        return super().invoke_internal(context, services)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        from .inference import upscale

        def report(message: str) -> None:
            if context.util.is_canceled():
                raise CanceledException
            context.util.signal_progress(message)

        folder = Path(self.model_folder.strip()).expanduser()
        if not folder.is_absolute():
            raise ValueError("Enter the full path to the folder you want to use for SeedVR2 models.")

        image = context.images.get_pil(self.image.image_name)
        output_size = self._output_size(image.size)

        report("Finding SeedVR2 models")
        folder.mkdir(parents=True, exist_ok=True)
        folder = folder.resolve()
        variant = CHECKPOINTS[self.variant]
        checkpoint = find_or_download(folder, variant, report)
        vae = find_or_download(folder, VAE, report)
        conditioning = None if variant.repo == COMFY_REPO else find_or_download(folder, CONDITIONING, report)
        seed = randbits(32) if self.seed == -1 else self.seed
        result = upscale(
            image, output_size, seed, checkpoint, vae, conditioning,
            context.models.load_local_model, report,
        )
        report("Saving the SeedVR2 image")
        return ImageOutput.build(context.images.save(image=result))

    def _output_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        factor = float(self.scale.removesuffix("x"))
        width, height = (int(dimension * factor + 0.5) for dimension in image_size)
        return width, height
