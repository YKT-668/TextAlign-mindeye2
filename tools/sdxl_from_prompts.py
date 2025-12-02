import os
import json
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# --- Configuration ---
# Paths and filenames from environment variables
out_dir = os.environ.get("OUT_DIR", "enhanced_images")
prompts_json_path = os.environ.get("PROMPTS_JSON", "prompts.json")

# Generation parameters from environment variables
limit = int(os.environ.get("LIMIT", 0))
steps = int(os.environ.get("STEPS", 30))
cfg_scale = float(os.environ.get("CFG", 7.5))
width = int(os.environ.get("W", 1024))
height = int(os.environ.get("H", 1024))
seed = int(os.environ.get("SEED", 12345))

# --- Main Script ---
def main():
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load prompts from JSON file
    try:
        with open(prompts_json_path, 'r') as f:
            prompts_data = json.load(f)
        print(f"Loaded {len(prompts_data)} prompts from {prompts_json_path}")
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompts_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {prompts_json_path}")
        return

    # Apply limit if specified
    if limit > 0:
        prompts_data = prompts_data[:limit]
        print(f"Processing a limited set of {len(prompts_data)} prompts.")

    # Set up the generator for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed)

    # --- Model Loading ---
    # Use a community-vetted base model and a high-quality refiner
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "stabilityai/stable-diffusion-xl-refiner-1.0"
    # For better results, use a high-quality UNet, e.g., from ByteDance
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    try:
        # Try to load the faster SDXL-Lightning UNet
        unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors"), device="cuda"))
        print("Successfully loaded SDXL-Lightning UNet.")
    except Exception as e:
        print(f"Could not load custom UNet, falling back to base. Error: {e}")
        # If it fails, the pipeline will use the default UNet from the base model
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
    else:
        # If custom UNet loads, initialize pipeline with it
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")

    # A faster scheduler can be beneficial
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    print("SDXL pipeline loaded and moved to CUDA.")

    # --- Image Generation Loop ---
    for i, item in enumerate(prompts_data):
        # The prompt is the second element in each item list
        prompt_text = item['positive']

        
        print(f"\n[{i+1}/{len(prompts_data)}] Generating image...")
        print(f"  Prompt: {prompt_text}")

        # Generate the image
        with torch.inference_mode():
            image = pipe(
                prompt=prompt_text,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
                generator=generator
            ).images[0]

        # Save the image
        output_filename = f"enhanced_{i:04d}.png"
        output_path = os.path.join(out_dir, output_filename)
        image.save(output_path)
        print(f"  Saved to: {output_path}")

    print("\nBatch generation complete.")

if __name__ == "__main__":
    main()
