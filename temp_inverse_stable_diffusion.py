from functools import partial
from typing import Callable, List, Optional, Union, Tuple
import copy
import gc
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
# from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler,PNDMScheduler, LMSDiscreteScheduler

from modified_stable_diffusion import ModifiedStableDiffusionPipeline

### credit to: https://github.com/cccntu/efficient-prompt-to-prompt

# Backward_ddim 
def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """ from noise to image"""
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )

def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)
 
def exact_forward_ddim(
        x, 
        alpha_t, 
        alpha_tp1, 
        sig_t, 
        sig_tp1, 
        index, 
        unet, 
        scheduler, 
        time, 
        prev_time, 
        n_iter, 
        total_inference_step, 
        text_embeddings):
    # n_iter : number of differential correction
    lamb_t = torch.log(alpha_t/sig_t)
    lamb_tp1 = torch.log(alpha_tp1/sig_tp1)
    h = lamb_tp1 - lamb_t # change?

    phi_1 = torch.expm1(h)
    noise_pred = unet(x, time, encoder_hidden_states=text_embeddings).sample
    model_s = scheduler.step(noise_pred, time, x).prev_sample
    x_t = x

    # naive DDIM inversion
    x = (
        sig_tp1 / sig_t * x
        + sig_tp1 / sig_t * alpha_t * phi_1 * model_s
    )
    
    # Correction
    torch.set_grad_enabled(True)
    x = differential_correction(x, time, prev_time, x_t, scheduler, unet, n_iter, text_embeddings)
    torch.set_grad_enabled(False)
    return x

def differential_correction(x, time, prev_time, x_t, scheduler, unet, n_iter, text_embeddings):
    scheduler = copy.deepcopy(scheduler)
    unet = copy.deepcopy(unet)
    input = x.clone().float().requires_grad_(True)
    x_t = x_t.clone()
    th = 1e-6

    loss_function = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD([input], lr=0.05)
    for i in range(n_iter):
        model_output = unet(input, time, text_embeddings)
        model_output = scheduler.step(model_output, time, x)

        loss = loss_function(model_output, x_t)
        print(f"t: {time}, Iteration {i}, Loss: {loss.item():.6f}")
        if loss.item() < th:
            break             
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return input

class InversableStableDiffusionPipeline(ModifiedStableDiffusionPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super(InversableStableDiffusionPipeline, self).__init__(vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                requires_safety_checker)

        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)
    
    def get_random_latents(self, latents=None, height=512, width=512, generator=None):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1
        device = self._execution_device

        num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )

        return latents

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings
    
    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents


    @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: True = False,
        **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False


        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
            )
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            
            # ddim 
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )

        return latents

    @torch.inference_mode()
    def exact_forward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: True = False,
        **kwargs,
    ):
        """ Generate latent 'exactly" from image and prompt
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False


        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
            )
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            
            # ddim 
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            sig_t = self.scheduler.betas[t]
            sig_t_prev = (
                self.scheduler.betas[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.betas[-1] # May be a problem, end or the beginning
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
                sig_t, sig_t_prev = sig_t_prev, sig_t
            latents = exact_forward_ddim(
                x=latents,
                alpha_t=alpha_prod_t,
                alpha_tp1=alpha_prod_t_prev,
                sig_t = sig_t,
                sig_tp1 = sig_t_prev,
                index = i,
                unet = self.unet,
                scheduler = self.scheduler,
                time = t,
                prev_time = prev_timestep,
                n_iter = 100,
                total_inference_step = num_inference_steps,
                text_embeddings = text_embeddings,
            )
            # TODO : put n_iter at kwargs or something

        return latents

    def edcorrector(self, x):
        """
        edcorrector calculates latents z of the image x by solving optimization problem ||D(z)-x||,
        not by directly encoding with VAE encoder. "Decoder inversion"

        INPUT
        x : image data (1, 3, 512, 512) -> given data
        OUTPUT
        z : modified latent data (1, 4, 64, 64)

        Goal : minimize norm(e(x)-z)
        """
        input = x.clone().float()

        z = self.get_image_latents(x).clone().float() # initial z
        z.requires_grad_(True)

        # Loss function
        loss_function = torch.nn.MSELoss(reduction='sum')

        # Optimizer
        optimizer = torch.optim.Adam([z], lr=1e-2)

        print("Starting Decoder inversion... ", end=" ")
        # Optimization
        for i in range(1000):
            x_pred = self.decode_image_for_gradient_float(z)

            #if, without regularizer
            loss = loss_function(x_pred, input)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Decoder Inversion ended")
        z.requires_grad_(False)

        # For GPU memory
        del x_pred
        torch.cuda.empty_cache()

        return z.half()

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    def decode_image_for_gradient_float(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        vae = copy.deepcopy(self.vae).float()
        image = [
            vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image