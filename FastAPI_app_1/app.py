from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request
import torch
from diffusers import (AutoencoderKL, DDPMScheduler,
                       StableDiffusionInstructPix2PixPipeline,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import json
import io
import os
import asyncio

app = FastAPI()

# Load model and checkpoints
unet = UNet2DConditionModel.from_pretrained("/home/tasnim/FastAPI_app/checkpoints")
vae = AutoencoderKL.from_pretrained("timbrooks/instruct-pix2pix", subfolder="vae")
text_encoder = CLIPTextModel.from_pretrained("timbrooks/instruct-pix2pix", subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder="tokenizer")
noise_scheduler = DDPMScheduler.from_pretrained("timbrooks/instruct-pix2pix", subfolder="scheduler")

# Initialize the pipeline
model_path = "timbrooks/instruct-pix2pix"
model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_path,
    unet=unet,
    text_encoder=text_encoder,
    vae=vae,
)

def run_inference(image: Image.Image, instruction: str) -> Image.Image:
    print(f"Processing image with prompt: {instruction}")
    images = model(instruction, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
    output_image = images[0]
    return output_image

@app.get("/")
async def read_root():
    return JSONResponse(content={"message": "Welcome to the Image Editor. Use the /edit-image/ endpoint to upload an image and an instruction."})

@app.post("/edit-image/")
async def edit_image(instruction: str = Form(...), file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    edited_image = run_inference(image, instruction)
    output_path = "output.png"
    edited_image.save(output_path)
    return FileResponse(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)