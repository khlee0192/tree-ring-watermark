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
    def forward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: True = False,
        inverse_opt=True,
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
        # To make it a reverse
        reverse_process = True

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
            
            # ddim - original
            # alpha_prod_t = self.scheduler.alphas_cumprod[t]
            # alpha_prod_t_prev = (
            #     self.scheduler.alphas_cumprod[prev_timestep]
            #     if prev_timestep >= 0
            #     else self.scheduler.final_alpha_cumprod
            # )
            # if reverse_process:
            #     alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            # latents = backward_ddim(
            #     x_t=latents,
            #     alpha_t=alpha_prod_t,
            #     alpha_tm1=alpha_prod_t_prev,
            #     eps_xt=noise_pred,
            # )

            # Our Algorithm

            # Algorithm 1
            if self.scheduler.solver_order == 1:
                with torch.no_grad():
                    if (i + 2 < len(timesteps_tensor)):
                        s = timesteps_tensor[i + 1]
                        r = timesteps_tensor[i + 2]
                        
                        lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                        sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                        h = lambda_t - lambda_s
                        alphas, alphat = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                        phi_1 = torch.expm1(-h)
                        model_s = self.unet(latent_model_input, s, encoder_hidden_states=text_embeddings).sample

                        x_t = latents

                        latents = (sigma_s / sigma_t * latents + sigma_s / sigma_t * alpha_t * phi_1 * model_s)

                        if (inverse_opt):
                            torch.set_grad_enabled(True)
                            latents = self.differential_correction(latents, s, t, x_t, order=self.scheduler.solver_order, r=r, text_embeddings=text_embeddings)
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
            elif self.scheduler.solver_order == 2:
                with torch.no_grad():
                    if (i + 2 < len(timesteps_tensor)):
                        y = latents.clone()
                        for step in range(i, i+2, 1):
                            s = timesteps_tensor[i]
                            t = timesteps_tensor[i+1]
                            r = timesteps_tensor[i+2]

                            lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                            sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                            h = lambda_t - lambda_s
                            alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                            phi_1 = torch.expm1(-h)
                            model_s = self.unet(latent_model_input, s, encoder_hidden_states=text_embeddings).sample
                            model_s = self.scheduler.step(model_s, s, latent_model_input).prev_sample

                            y_t = y

                            y = (sigma_s / sigma_t * y + sigma_s / sigma_t * alpha_t * phi_1 * model_s)
                            
                            if inverse_opt:
                                torch.set_grad_enabled(True)
                                y = self.differential_correction(y, s, t, y_t, order=self.scheduler.solver_order, r=r, text_embeddings=text_embeddings)
                                torch.set_grad_enabled(False)
                        
                        # outer step
                        s = timesteps_tensor[i+1]
                        t = timesteps_tensor[i+2]

                        lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                        sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                        h = lambda_t - lambda_s
                        alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                        phi_1 = torch.expm1(-h)
                        model_s = self.unet(latent_model_input, s, encoder_hidden_states=text_embeddings).sample
                        model_s = self.scheduler.step(model_s, s, latent_model_input).prev_sample

                        x_t = latents
                        if not inverse_opt:
                            # naive DDIM inversion
                            latents = (
                                sigma_s / sigma_t * latents
                                + sigma_s / sigma_t * alpha_t * phi_1 * model_s
                            )

                        if inverse_opt:
                            y_t_model_input = (torch.cat([y_t] * 2) if do_classifier_free_guidance else y_t)
                            y_t_model_input = self.scheduler.scale_model_input(y_t, s)
                            model_s_output = self.unet(y_t_model_input, s, encoder_hidden_states=text_embeddings).sample
                            model_s_output = self.scheduler.step(model_s_output, s, latent_model_input).prev_sample

                            y_model_input = (torch.cat([y] * 2) if do_classifier_free_guidance else y)
                            y_model_input = self.scheduler.scale_model_input(y, r)
                            model_r_output = self.unet(y_model_input, r, encoder_hidden_states=text_embeddings).sample
                            model_r_output = self.scheduler.step(model_r_output, s, latent_model_input).prev_sample
                        
                            # not naive DDIM inversion
                            latents = (
                                sigma_s / sigma_t * latents
                                + sigma_s / sigma_t * alpha_t * phi_1 * model_s_output
                            )
                            torch.set_grad_enabled(True)
                            # latents stay the same for now
                            latents = self.differential_correction(latents, s, t, x_t, order=self.scheduler.solver_order, r=r, 
                                                            model_s_output = model_s_output, model_r_output = model_r_output, text_embeddings=text_embeddings)
                            torch.set_grad_enabled(False)
            else:
                pass

        return latents    

    @torch.inference_mode
    def inversion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process=True,
        inverse_opt=True,
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
            
            # Our Algorithm

            # Algorithm 1
            if self.scheduler.solver_order == 1:
                with torch.no_grad():
                    if (i + 2 < len(timesteps_tensor)):
                        s = timesteps_tensor[i + 1]
                        r = timesteps_tensor[i + 2]
                        
                        lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                        sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                        h = lambda_t - lambda_s
                        alphas, alphat = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                        phi_1 = torch.expm1(-h)
                        model_s = noise_pred

                        x_t = latents

                        latents = (sigma_s / sigma_t * latents + sigma_s / sigma_t * alpha_t * phi_1 * model_s)

                        if (inverse_opt):
                            torch.set_grad_enabled(True)
                            latents = self.differential_correction(latents, s, t, x_t, order=self.scheduler.solver_order, r=r, text_embeddings=text_embeddings)
                            torch.set_grad_enabled(False)

                    elif (i + 2 == len(timesteps_tensor)):
                        s = timesteps_tensor[i + 1]

                        lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                        sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                        h = lambda_t - lambda_s
                        alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                        phi_1 = torch.expm1(-h)
                        model_s = noise_pred

                        x_t = latents

                        latents = (sigma_s / sigma_t * latents + sigma_s / sigma_t * alpha_t * phi_1 * model_s)

                        if (inverse_opt):
                            torch.set_grad_enabled(True)
                            latents = self.differential_correction(latents, s, t, x_t, order=1, text_embeddings=text_embeddings)
                            torch.set_grad_enabled(False)

            # Algorithm 2
            elif self.scheduler.solver_order == 2:
                with torch.no_grad():
                    if (i + 2 < len(timesteps_tensor)):
                        y = latents.clone()
                        for step in range(i, i+2, 1):
                            s = timesteps_tensor[i+1]
                            r = timesteps_tensor[i+2]

                            lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                            sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                            h = lambda_t - lambda_s
                            alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                            phi_1 = torch.expm1(-h)
                            model_s = noise_pred

                            y_t = y

                            y = (sigma_s / sigma_t * y + sigma_s / sigma_t * alpha_t * phi_1 * model_s)
                            
                            if inverse_opt:
                                torch.set_grad_enabled(True)
                                y = self.differential_correction(y, s, t, y_t, order=self.scheduler.solver_order, r=r, text_embeddings=text_embeddings)
                                torch.set_grad_enabled(False)
                        
                        # outer step
                        s = timesteps_tensor[i+1]
                        r = timesteps_tensor[i+2]

                        lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                        sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                        h = lambda_t - lambda_s
                        alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                        phi_1 = torch.expm1(-h)
                        model_s = noise_pred

                        x_t = latents
                        if not inverse_opt:
                            # naive DDIM inversion
                            latents = (
                                sigma_s / sigma_t * latents
                                + sigma_s / sigma_t * alpha_t * phi_1 * model_s
                            )

                        if inverse_opt:
                            y_t_model_input = (torch.cat([y_t] * 2) if do_classifier_free_guidance else y_t)
                            y_t_model_input = self.scheduler.scale_model_input(y_t, s)
                            model_s_output = self.unet(y_t_model_input, s, encoder_hidden_states=text_embeddings).sample

                            y_model_input = (torch.cat([y] * 2) if do_classifier_free_guidance else y)
                            y_model_input = self.scheduler.scale_model_input(y, r)
                            model_r_output = self.unet(y_model_input, r, encoder_hidden_states=text_embeddings).sample
                        
                            # not naive DDIM inversion
                            latents = (
                                sigma_s / sigma_t * latents
                                + sigma_s / sigma_t * alpha_t * phi_1 * model_s_output
                            )
                            torch.set_grad_enabled(True)
                            latents = self.differential_correction(latents, s, t, x_t, order=self.scheduler.solver_order, r=r, 
                                                            model_s_output = model_s_output, model_r_output = model_r_output, text_embeddings=text_embeddings)
                            torch.set_grad_enabled(False)
            else:
                pass

        return latents
    
    def differential_correction(self, x, s, t, x_t, r=None, order=1, use_float=False, n_iter=100, lr=0.001, th=1e-6, model_s_output=None, model_r_output=None, text_embeddings=None):
        if order==1:
            import copy
            model = copy.deepcopy(self.unet)
            input = x.clone()
            x_t = x_t.clone()

            ## Use Float
            # model = model.float()
            # input = input.float()
            # x_t = x_t.float()

            input.requires_grad_(True)
            loss_function = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.SGD([input], lr=lr)
            for i in range(n_iter):
                model_output = model(input, s, encoder_hidden_states=text_embeddings).sample # estimated noise
                x_t_pred = self.scheduler.dpm_solver_first_order_update(model_output, t, s, input)
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
            input = x.clone()

            x_t = x_t.clone()   

            ## Use Float
            # model = model.float()
            # input = input.float()
            # x_t = x_t.float()

            input.requires_grad_(True)
            loss_function = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.SGD([input], lr=lr)
            
            # for 2nd order correction
            model_t_output = model(x_t, t, encoder_hidden_states=text_embeddings).sample.detach()
            lambda_prev_1, lambda_prev_0, lambda_t = self.scheduler.lambda_t[r], self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
            sigma_prev_0, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
            alpha_t = self.scheduler.alpha_t[t]
            h_0 = lambda_prev_0 - lambda_prev_1
            h = lambda_t - lambda_prev_0
            r0 = h_0 / h
            phi_1 = torch.expm1(-h)

            for i in range(n_iter):
                model_output = model(input, s, encoder_hidden_states=text_embeddings).sample # estimated noise
                x_t_pred = self.scheduler.dpm_solver_first_order_update(model_output, t, s, input)
                # 2nd order correction
                # diff = (1. / r0) * (model_t_output - model_output)
                if model_s_output is not None and model_r_output is not None:
                    diff =  (1./ r0) * (model_s_output - model_r_output)
                else:
                    diff = 1. * (model_t_output - model_output)
                x_t_pred = x_t_pred - 0.5 * alpha_t * phi_1 * diff

                loss = loss_function(x_t_pred, x_t)
                #print(f"t: {t:.3f}, Iteration {i}, Loss: {loss.item():.6f}")
                if loss.item() < th:
                    break             
                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
            return input
        else:
            raise NotImplementedError

    def edcorrector(self, x, encoder_hidden_states):
        """
        edcorrector calculates latents z of the image x by solving optimization problem ||E(x)-z||,
        not by directly encoding with VAE encoder. "Decoder inversion"

        INPUT
        x : image data (1, 3, 512, 512) -> given data
        OUTPUT
        z : modified latent data (1, 4, 64, 64)

        Goal : minimize norm(e(x)-z), working on adding regularizer
        """
        input = x.clone().float()

        z = self.get_image_latents(x).clone().float() # initial z
        z.requires_grad_(True)

        # Loss를 계산할 때 무언가를 가져와야 한다
        loss_function = torch.nn.MSELoss(reduction='sum')
        losses = []

        ## SGD : improvement with 63.75% accuracy
        #optimizer = torch.optim.SGD([z], lr=1e-3, momentum=0.9)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 1000, 3000], gamma=0.1)
        
        ## Current best
        #optimizer = torch.optim.Adam([z], lr=1e-3)

        ## Adjusting Adam
        optimizer = torch.optim.Adam([z], lr=1e-3)

        t =self.scheduler.timesteps[-1]
        scheduler = copy.deepcopy(self.scheduler)
        unet = copy.deepcopy(self.unet).float()

        lam = 1

        """
        z_output = copy.deepcopy(z)
        z_output = scheduler.convert_model_output(z_output, t, z)
        z_output = scheduler.step(z_output, t, z).prev_sample
        """

        for i in range(10):
            x_pred = self.decode_image_for_gradient_float(z)

            #if, without regularizer
            loss = loss_function(x_pred, input)

            if i%1000==0:
                print(f"t: {0}, Iteration {i}, Loss: {loss.item()}")
            

            """
            #if, with unet regularizer
            with torch.no_grad():
                noise = unet(z, t, encoder_hidden_states=encoder_hidden_states.float()).sample
            #noise = noise.detach()

            loss = loss_function(x_pred, input) + lam * torch.sum(torch.abs(noise))
            loss1 = loss_function(x_pred, input).detach()
            loss2 = torch.sum(torch.abs(noise)).detach()

            if i%1000==0:
                print(f"t: {0}, Iteration {i}, Loss1: {loss1.item()}, Loss2: {loss2.item()}")
            """
                
            """ if, with ddim regularizer
            with torch.no_grad():
                noise = unet(z, t, encoder_hidden_states=encoder_hidden_states.float()).sample
                sample = scheduler.step(noise, t, z).prev_sample
            sample = sample.detach()

            loss = loss_function(x_pred, input) + lam * loss_function(z, sample)
            loss1 = loss_function(x_pred, input).detach()
            loss2 = loss_function(z, sample).detach()

            #losses.append(loss.detach().item())
            if i%1000==0:
                print(f"t: {0}, Iteration {i}, Loss1: {loss1.item()}, Loss2: {loss2.item()}")
            """
            
            """#if, with z regularizer
            
            loss = loss_function(x_pred, input) + torch.norm(z)
            loss1 = loss_function(x_pred, input).detach()
            loss2 = torch.norm(z)

            if i%1000==0:
                print(f"t: {0}, Iteration {i}, Loss1: {loss1.item()}, Loss2: {loss2.item()}")
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            
        #plt.plot(losses)
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
