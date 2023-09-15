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

        #self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)
    
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
            #print(f"mean : {latents.mean().item()}, std : {latents.std().item()}")
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

    # @torch.inference_mode()
    def forward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process=True,
        inverse_opt=True,
        inv_order=None,
        **kwargs,
    ):  
        with torch.no_grad():
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

            if inv_order is None:
                inv_order = self.scheduler.solver_order
    
            if (reverse_process):
                timesteps_tensor = reversed(timesteps_tensor)

            for i, t in enumerate(self.progress_bar(timesteps_tensor)):
                print(f"mean : {latents.mean().item()}, std : {latents.std().item()}")
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
                
                # Our Algorithm, latents is x0 at algorithm 2

                # Algorithm 1
                if inv_order == 1:
                    with torch.no_grad():
                        if (i + 2 < len(timesteps_tensor)):
                            s = timesteps_tensor[i + 1]
                            r = timesteps_tensor[i + 2]
                            
                            lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                            sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                            h = lambda_t - lambda_s
                            alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                            phi_1 = torch.expm1(-h)
                            model_s = self.unet(latent_model_input, s, encoder_hidden_states=text_embeddings).sample

                            x_t = latents

                            latents = (sigma_s / sigma_t * latents + sigma_s / sigma_t * alpha_t * phi_1 * model_s)

                            if (inverse_opt):
                                torch.set_grad_enabled(True)
                                latents = self.differential_correction(latents, s, t, x_t, order=inv_order, r=r, text_embeddings=text_embeddings)
                                torch.set_grad_enabled(False)

                        elif (i + 2 == len(timesteps_tensor)):
                            s = timesteps_tensor[i + 1]

                            lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                            sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                            h = lambda_t - lambda_s
                            alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                            phi_1 = torch.expm1(-h)
                            model_s = self.unet(latent_model_input, s, encoder_hidden_states=text_embeddings).sample

                            x_t = latents

                            latents = (sigma_s / sigma_t * latents + sigma_s / sigma_t * alpha_t * phi_1 * model_s)

                            if (inverse_opt):
                                torch.set_grad_enabled(True)
                                latents = self.differential_correction(latents, s, t, x_t, order=1, text_embeddings=text_embeddings)
                                torch.set_grad_enabled(False)

                # Algorithm 2
                elif inv_order == 2:
                    with torch.no_grad():
                        if (i + 2 < len(timesteps_tensor)):
                            x_tM = latents.clone() # line 3
                            for j in range(i, i+2, 1): # line 4
                                if j+2 == len(timesteps_tensor):
                                    break
                                    #temporary
                                y_tj = x_tM # do at line 4, further used as y_tj
                                t_j = timesteps_tensor[j+1]
                                t_jm1 = timesteps_tensor[j+2]
                                
                                r = timesteps_tensor[j]

                                lambda_j, lambda_jm1 = self.scheduler.lambda_t[t_j], self.scheduler.lambda_t[t_jm1]
                                sigma_j, sigma_jm1 = self.scheduler.sigma_t[t_j], self.scheduler.sigma_t[t_jm1]
                                alpha_j, alpha_jm1 = self.scheduler.alpha_t[t_j], self.scheduler.alpha_t[t_jm1]

                                h = lambda_j - lambda_jm1
                                phi_1 = torch.expm1(-h)
                                model_s = self.unet(y_tj, t_jm1, encoder_hidden_states=text_embeddings).sample
                                x_theta = self.scheduler.step(model_s, t_jm1, y_tj).prev_sample

                                #y = (sigma_r / sigma_s * y + sigma_r / sigma_s * alpha_s * phi_1 * model_s)
                                y_tjm1 = (sigma_jm1/sigma_j*y_tj + sigma_jm1/sigma_j*alpha_j*phi_1*x_theta) # line 5

                                if inverse_opt:
                                    torch.set_grad_enabled(True)
                                    y_tjm1 = self.differential_correction(y_tjm1, y_tj, t_j, t_jm1, r=r, text_embeddings=text_embeddings)
                                    torch.set_grad_enabled(False)
                                
                                x_tM = y_tjm1
                            # 이 for가 끝나고 나오는 y_tm1, y_tj가 바로 적용되는게 맞나?

                            # outer step, from line 12
                            t_j = timesteps_tensor[i+1]
                            t_jm1 = timesteps_tensor[i+2]

                            lambda_j, lambda_jm1 = self.scheduler.lambda_t[t_j], self.scheduler.lambda_t[t_jm1]
                            sigma_j, sigma_jm1 = self.scheduler.sigma_t[t_j], self.scheduler.sigma_t[t_jm1]
                            alpha_j, alpha_jm1 = self.scheduler.alpha_t[t_j], self.scheduler.alpha_t[t_jm1]

                            h = lambda_j - lambda_jm1

                            phi_1 = torch.expm1(-h)
                            model_s = self.unet(x_tM, r, encoder_hidden_states=text_embeddings).sample
                            x_theta = self.scheduler.step(model_s, r, latent_model_input).prev_sample

                            if not inverse_opt:
                                # naive DDIM inversion
                                x_tim1 = (
                                    sigma_jm1/sigma_j*x_tM
                                    + sigma_jm1 / sigma_j * alpha_j * phi_1 * x_theta
                                )
                                latents = x_tim1

                            if inverse_opt:
                                # check
                                y_t_model_input = (torch.cat([y_tj] * 2) if do_classifier_free_guidance else y_tj)
                                y_t_model_input = self.scheduler.scale_model_input(y_tj, t_j)
                                model_t_output = self.unet(y_t_model_input, t_j, encoder_hidden_states=text_embeddings).sample
                                model_t_output = self.scheduler.step(model_t_output, t_j, latent_model_input).prev_sample

                                # check
                                y_tm1_model_input = (torch.cat([y_tjm1] * 2) if do_classifier_free_guidance else y_tjm1)
                                y_tm1_model_input = self.scheduler.scale_model_input(y_tjm1, t_jm1)
                                model_tm1_output = self.unet(y_tm1_model_input, t_jm1, encoder_hidden_states=text_embeddings).sample
                                model_tm1_output = self.scheduler.step(model_tm1_output, t_jm1, latent_model_input).prev_sample
                            
                                # OK
                                x_tim1 = (
                                    sigma_jm1/sigma_j*x_tM
                                    + sigma_jm1 / sigma_j * alpha_j * phi_1 * x_theta
                                )

                                # check
                                torch.set_grad_enabled(True)
                                x_tim1 = self.differential_correction(x_tim1, x_tM, t_j, t_jm1, r=r, text_embeddings=text_embeddings)
                                torch.set_grad_enabled(False)
                            
                                latents = x_tim1

                        elif (i + 2 == len(timesteps_tensor)):
                            t_1 = timesteps_tensor[i]
                            t_0 = timesteps_tensor[i+1]

                            lambda_1, lambda_0 = self.scheduler.lambda_t[t_1], self.scheduler.lambda_t[t_0]
                            sigma_1, sigma_0 = self.scheduler.sigma_t[t_1], self.scheduler.sigma_t[t_0]
                            alpha_1 = self.scheduler.alpha_t[t_1]

                            h = lambda_1 - lambda_0
                            
                            phi_1 = torch.expm1(-h)
                            model_s = self.unet(latents, t_0, encoder_hidden_states=text_embeddings).sample
                            model_s = self.scheduler.step(model_s, t_0, latents).prev_sample
                            x_theta = latents

                            # naive DDIM inversion
                            x_0 = (
                                sigma_1 / sigma_0 * latents
                                + sigma_1 / sigma_0 * alpha_1 * phi_1 * x_theta
                            )
                            if inverse_opt:
                                torch.set_grad_enabled(True)
                                x_0 = self.differential_correction(x_0, latents, t_1, t_0, order=1, text_embeddings=text_embeddings)
                                torch.set_grad_enabled(False)    
                                    
                            latents = x_0
                else:
                    pass

        return latents    

    # @torch.inference_mode
    # def inversion(
    #     self,
    #     use_old_emb_i=25,
    #     text_embeddings=None,
    #     old_text_embeddings=None,
    #     new_text_embeddings=None,
    #     latents: Optional[torch.FloatTensor] = None,
    #     num_inference_steps: int = 50,
    #     guidance_scale: float = 7.5,
    #     callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    #     callback_steps: Optional[int] = 1,
    #     reverse_process=True,
    #     inverse_opt=True,
    #     **kwargs,
    # ):
    #     """ Generate image from text prompt and latents
    #     """
    #     # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    #     # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    #     # corresponds to doing no classifier free guidance.
    #     do_classifier_free_guidance = guidance_scale > 1.0
    #     # set timesteps
    #     self.scheduler.set_timesteps(num_inference_steps)
    #     # Some schedulers like PNDM have timesteps as arrays
    #     # It's more optimized to move all timesteps to correct device beforehand
    #     timesteps_tensor = self.scheduler.timesteps.to(self.device)
    #     # scale the initial noise by the standard deviation required by the scheduler
    #     latents = latents * self.scheduler.init_noise_sigma

    #     if old_text_embeddings is not None and new_text_embeddings is not None:
    #         prompt_to_prompt = True
    #     else:
    #         prompt_to_prompt = False


    #     for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
    #         if prompt_to_prompt:
    #             if i < use_old_emb_i:
    #                 text_embeddings = old_text_embeddings
    #             else:
    #                 text_embeddings = new_text_embeddings

    #         # expand the latents if we are doing classifier free guidance
    #         latent_model_input = (
    #             torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    #         )
    #         latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    #         # predict the noise residual
    #         noise_pred = self.unet(
    #             latent_model_input, t, encoder_hidden_states=text_embeddings
    #         ).sample

    #         # perform guidance
    #         if do_classifier_free_guidance:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + guidance_scale * (
    #                 noise_pred_text - noise_pred_uncond
    #             )

    #         prev_timestep = (
    #             t
    #             - self.scheduler.config.num_train_timesteps
    #             // self.scheduler.num_inference_steps
    #         )
    #         # call the callback, if provided
    #         if callback is not None and i % callback_steps == 0:
    #             callback(i, t, latents)
            
    #         # Our Algorithm

    #         # Algorithm 1
    #         if self.scheduler.solver_order == 1:
    #             with torch.no_grad():
    #                 if (i + 2 < len(timesteps_tensor)):
    #                     s = timesteps_tensor[i + 1]
    #                     r = timesteps_tensor[i + 2]
                        
    #                     lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
    #                     sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
    #                     h = lambda_t - lambda_s
    #                     alphas, alphat = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
    #                     phi_1 = torch.expm1(-h)
    #                     model_s = noise_pred

    #                     x_t = latents

    #                     latents = (sigma_s / sigma_t * latents + sigma_s / sigma_t * alpha_t * phi_1 * model_s)

    #                     if (inverse_opt):
    #                         torch.set_grad_enabled(True)
    #                         latents = self.differential_correction(latents, s, t, x_t, order=self.scheduler.solver_order, r=r, text_embeddings=text_embeddings)
    #                         torch.set_grad_enabled(False)

    #                 elif (i + 2 == len(timesteps_tensor)):
    #                     s = timesteps_tensor[i + 1]

    #                     lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
    #                     sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
    #                     h = lambda_t - lambda_s
    #                     alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
    #                     phi_1 = torch.expm1(-h)
    #                     model_s = noise_pred

    #                     x_t = latents

    #                     latents = (sigma_s / sigma_t * latents + sigma_s / sigma_t * alpha_t * phi_1 * model_s)

    #                     if (inverse_opt):
    #                         torch.set_grad_enabled(True)
    #                         latents = self.differential_correction(latents, s, t, x_t, order=1, text_embeddings=text_embeddings)
    #                         torch.set_grad_enabled(False)

    #         # Algorithm 2
    #         elif self.scheduler.solver_order == 2:
    #             with torch.no_grad():
    #                 if (i + 2 < len(timesteps_tensor)):
    #                     y = latents.clone()
    #                     for step in range(i, i+2, 1):
    #                         s = timesteps_tensor[i+1]
    #                         r = timesteps_tensor[i+2]

    #                         lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
    #                         sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
    #                         h = lambda_t - lambda_s
    #                         alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
    #                         phi_1 = torch.expm1(-h)
    #                         model_s = noise_pred

    #                         y_t = y

    #                         y = (sigma_s / sigma_t * y + sigma_s / sigma_t * alpha_t * phi_1 * model_s)
                            
    #                         if inverse_opt:
    #                             torch.set_grad_enabled(True)
    #                             y = self.differential_correction(y, s, t, y_t, order=self.scheduler.solver_order, r=r, text_embeddings=text_embeddings)
    #                             torch.set_grad_enabled(False)
                        
    #                     # outer step
    #                     s = timesteps_tensor[i+1]
    #                     r = timesteps_tensor[i+2]

    #                     lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
    #                     sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
    #                     h = lambda_t - lambda_s
    #                     alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
    #                     phi_1 = torch.expm1(-h)
    #                     model_s = noise_pred

    #                     x_t = latents
    #                     if not inverse_opt:
    #                         # naive DDIM inversion
    #                         latents = (
    #                             sigma_s / sigma_t * latents
    #                             + sigma_s / sigma_t * alpha_t * phi_1 * model_s
    #                         )

    #                     if inverse_opt:
    #                         y_t_model_input = (torch.cat([y_t] * 2) if do_classifier_free_guidance else y_t)
    #                         y_t_model_input = self.scheduler.scale_model_input(y_t, s)
    #                         model_s_output = self.unet(y_t_model_input, s, encoder_hidden_states=text_embeddings).sample

    #                         y_model_input = (torch.cat([y] * 2) if do_classifier_free_guidance else y)
    #                         y_model_input = self.scheduler.scale_model_input(y, r)
    #                         model_r_output = self.unet(y_model_input, r, encoder_hidden_states=text_embeddings).sample
                        
    #                         # not naive DDIM inversion
    #                         latents = (
    #                             sigma_s / sigma_t * latents
    #                             + sigma_s / sigma_t * alpha_t * phi_1 * model_s_output
    #                         )
    #                         torch.set_grad_enabled(True)
    #                         latents = self.differential_correction(latents, s, t, x_t, order=self.scheduler.solver_order, r=r, 
    #                                                         model_s_output = model_s_output, model_r_output = model_r_output, text_embeddings=text_embeddings)
    #                         torch.set_grad_enabled(False)
    #         else:
    #             pass

    #     return latents
    
    def differential_correction(self, y_tjm1, y_tj, t_j, t_jm1, r=None, order=2, n_iter=100, lr=0.1, th=1e-6, model_s_output=None, model_r_output=None, text_embeddings=None):
        if order==1:
            import copy
            model = copy.deepcopy(self.unet)
            input = y_tjm1.clone()
            x_t = y_tj.clone()
            text_embeddings = text_embeddings.clone()

            input.requires_grad_(True)
            loss_function = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.SGD([input], lr=lr)

            for i in range(n_iter):
                model_output = model(input, t_jm1, encoder_hidden_states=text_embeddings).sample # estimated noise
                model_output = self.scheduler.step(model_output, t_jm1, input).prev_sample.detach()

                x_t_pred = self.scheduler.dpm_solver_first_order_update(model_output, t_j, t_jm1, input)

                loss = loss_function(x_t_pred, x_t)
                #print(f"t: {t}, Iteration {i}, Loss: {loss.item():.6f}")
                if loss.item() < th:
                    break             
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return input            
        elif order==2:
            assert r is not None
            # only for multistep
            import copy
            model = copy.deepcopy(self.unet)
            input = y_tjm1.clone()
            x_t = y_tj.clone()   
            text_embeddings = text_embeddings.clone()

            input.requires_grad_(True)
            loss_function = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.SGD([input], lr=lr)
            
            # for 2nd order correction
            model_t_output = model(x_t, t_jm1, encoder_hidden_states=text_embeddings).sample.detach()
            model_t_output2 = self.scheduler.step(model_t_output, t_jm1, x_t).prev_sample.detach()

            lambda_tj = self.scheduler.lambda_t[t_j]
            lambda_tjm1 = self.scheduler.lambda_t[t_jm1]
            lambda_r = self.scheduler.lambda_t[r]

            sigma_tj = self.scheduler.sigma_t[t_j]
            sigma_tjm1 = self.scheduler.sigma_t[t_jm1]
            
            alpha_tj = self.scheduler.alpha_t[t_j]

            h_0 = lambda_tj - lambda_tjm1
            h = lambda_r - lambda_tj
            r0 = h_0 / h
            phi_1 = torch.expm1(-h)

            for i in range(n_iter):
                model_output = model(input, t_jm1, encoder_hidden_states=text_embeddings).sample # estimated noise
                model_output = self.scheduler.step(model_output, t_jm1, input).prev_sample.detach()
                # next line may not be necessary
                x_t_pred = self.scheduler.dpm_solver_first_order_update(model_output, t_jm1, t_j, input)
                # 2nd order correction..
                # diff = (1. / r0) * (model_t_output - model_output)
                if model_s_output is not None and model_r_output is not None:
                    diff =  (1./ r0) * (model_s_output - model_r_output)
                else:
                    diff = 1. * (model_t_output2 - model_output)
                x_t_pred = x_t_pred - 0.5 * alpha_tj * phi_1 * diff

                loss = loss_function(x_t_pred, x_t)

                # make_dot(loss).render("nostep", format="png")
                #print(f"t: {t:.3f}, Iteration {i}, Loss: {loss.item():.6f}")
                if loss.item() < th:
                    break             
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return input
        else:
            raise NotImplementedError

    def edcorrector(self, x):
        """
        edcorrector calculates latents z of the image x by solving optimization problem ||E(x)-z||,
        not by directly encoding with VAE encoder. "Decoder inversion"

        INPUT
        x : image data (1, 3, 512, 512) -> given data
        OUTPUT
        z : modified latent data (1, 4, 64, 64)

        Goal : minimize norm(e(x)-z), working on adding regularizer
        """
        print(f"decoder inversion started..", end="\t")
        input = x.clone().float()

        z = self.get_image_latents(x).clone().float() # initial z
        z.requires_grad_(True)

        # Loss를 계산할 때 무언가를 가져와야 한다
        loss_function = torch.nn.MSELoss(reduction='sum')

        ## Adjusting Adam
        optimizer = torch.optim.Adam([z], lr=1e-2)

        for i in range(1000):
            x_pred = self.decode_image_for_gradient_float(z)

            loss = loss_function(x_pred, input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del x
        print(f"decoder inversion ended!")
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
