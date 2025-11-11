import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_c0_perturb_example() -> dict:
    """Creates a random input example for the C0Perturb policy."""
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),
        "actions": np.random.rand(7),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class C0PerturbInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    state_angle_to_radians: bool = False

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        state = data["observation/state"]
        if self.state_angle_to_radians and "actions" in data:
            # Ad-hoc changes. Convert angles in state from degrees to radians. 
            # Only do this in training as the training data states are in angles.
            state[3:6] = np.deg2rad(state[3:6])
        state = np.append(state, 0.0)
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]
            
        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            inputs["prompt"] = generate_c0_perturb_prompt()

        return inputs

def generate_c0_perturb_prompt() -> str:
    """Generates a random prompt for the C0Perturb policy."""
    prompt_library = [
        "Flip one lying phone box upright.",
        "Rotate one box 90 degrees so it stands vertically.",
        "Make one of the phone boxes stand upright for grasping.",
        "In the cardboard tray at the corner of the shelf, flip one lying box upright.",
        "From the stacked phone boxes in the tray, rotate one box 90 degrees to stand it up.",
        "The phone boxes are lying flat in the tray. Flip one box upright so it is ready for grasping.",
        "A few boxes are stacked in the corner tray. Choose one lying box and make it stand vertically.",
        "From the stacked boxes at the shelf corner, flip one lying phone box upright to enable grasping."
        "Approach the cardboard tray at the corner of the shelf and flip a box upright.",
    ]
    random_index = np.random.randint(len(prompt_library))
    return prompt_library[random_index]

@dataclasses.dataclass(frozen=True)
class C0PerturbOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}





