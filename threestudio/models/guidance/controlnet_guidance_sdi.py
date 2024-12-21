import os
from dataclasses import dataclass

import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from controlnet_aux import CannyDetector, NormalBaeDetector
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline, DDIMInverseScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("controlnet-depth-guidance-sdi")
class ControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "stablediffusionapi/realistic-vision-51"
        ddim_scheduler_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        control_type: str = "normal"  # normal/canny

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 20

        use_nfsd: bool = False
        use_dsd: bool = False
        edit_image: bool = False

        # Canny threshold
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100
        
        # sdi related:
        enable_sdi: bool = True
        inversion_guidance_scale: float = -7.5
        inversion_n_steps: int = 10
        inversion_eta: float = 0.3
        t_anneal: bool = True
        # n_ddim_steps: int = 50 # same as diffusion steps
        trainer_max_steps: int = 10000
        view_dependent_prompting: bool = True

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading ControlNet ...")

        controlnet_name_or_path: str
        if self.cfg.control_type == "normal":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_normalbae"
        elif self.cfg.control_type == "canny":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_canny"
        elif self.cfg.control_type == "depth":
            controlnet_name_or_path = "lllyasviel/control_v11f1p_sd15_depth"
        elif self.cfg.control_type == "openpose":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_openpose"
        elif self.cfg.control_type == "mediapipe":
            controlnet_name_or_path = "CrucibleAI/ControlNetMediaPipeFace"

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        if self.cfg.control_type == "mediapipe":
            if self.cfg.pretrained_model_name_or_path in ["stablediffusionapi/realistic-vision-51",
                                                          "runwayml/stable-diffusion-v1-5"]:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_name_or_path,
                    subfolder="diffusion_sd15",
                    torch_dtype=self.weights_dtype,
                    cache_dir=self.cfg.cache_dir,
                )
            else:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_name_or_path,
                    torch_dtype=self.weights_dtype,
                    cache_dir=self.cfg.cache_dir,
                )

        else:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_name_or_path,
                torch_dtype=self.weights_dtype,
                cache_dir=self.cfg.cache_dir,
            )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, controlnet=controlnet, **pipe_kwargs
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()

        if self.cfg.control_type == "normal":
            self.preprocessor = NormalBaeDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
            self.preprocessor.model.to(self.device)
        elif self.cfg.control_type == "canny":
            self.preprocessor = CannyDetector()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=self.device)
        self.scheduler.set_timesteps(self.cfg.diffusion_steps, device=self.device)
        
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.inverse_scheduler.set_timesteps(self.cfg.inversion_n_steps, device=self.device)
        self.inverse_scheduler.alphas_cumprod = self.inverse_scheduler.alphas_cumprod.to(device=self.device)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded ControlNet!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
            self,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            image_cond: Float[Tensor, "..."],
            condition_scale: float,
            encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        # print('controlnet latents.shape: ', latents.shape)
        # print('controlnet t.shape: ', t.shape)
        # print('controlnet encoder_hidden_states.shape: ', encoder_hidden_states.shape)
        # print('controlnet image_cond.shape: ', image_cond.shape)
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
            self,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            encoder_hidden_states: Float[Tensor, "..."],
            cross_attention_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
            self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
            self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
            self,
            latents: Float[Tensor, "B 4 H W"],
            latent_height: int = 64,
            latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def predict_noise(
        self,
        latents_noisy: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        guidance_scale: float = 1.0,
        text_embeddings: Optional[Float[Tensor, "..."]] = None,
        image_cond = None
    ):
        
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
        

            with torch.no_grad():
                latent_model_input = torch.cat([latents_noisy] * 4)
                image_cond_input = torch.cat([image_cond] * 4)
                down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                    image_cond=image_cond_input,
                    condition_scale=self.cfg.condition_scale,
                )

                noise_pred = self.forward_control_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            # print('noise_pred_text.shape', noise_pred_text.shape)
            # print('noise_pred_uncond.shape', noise_pred_uncond.shape)
            # print('noise_pred_neg.shape', noise_pred_neg.shape)

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ).to(e_i_neg.device) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            
            if text_embeddings is None:
                text_embeddings = prompt_utils.get_text_embeddings(
                    elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
                )
            
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 3)
                image_cond_input = torch.cat([image_cond] * 3)

                _t = torch.cat([t] * 3)
                # print(_t.shape, latent_model_input.shape, image_cond_input.shape)
                down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                    latent_model_input,
                    _t,
                    encoder_hidden_states=text_embeddings,
                    image_cond=image_cond_input,
                    condition_scale=self.cfg.condition_scale,
                )

                noise_pred = self.forward_control_unet(
                    latent_model_input,
                    _t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

            # BUG?
            # noise_pred_text, noise_pred_uncond, noise_pred_null = noise_pred.chunk(3)
            # noise_pred = noise_pred_text + guidance_scale * (
            #     noise_pred_text - noise_pred_uncond
            # )
            noise_pred_text, _, noise_pred_null = noise_pred.chunk(3)
            noise_pred = noise_pred_null + guidance_scale * (
                noise_pred_text - noise_pred_null
            )
        
        return noise_pred, neg_guidance_weights, text_embeddings

    def ddim_inversion_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        # 1. compute alphas, betas
        # change original implementation to exactly match noise levels for analogous forward process
        alpha_prod_t = self.inverse_scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.inverse_scheduler.initial_alpha_cumprod
        alpha_prod_t_prev = self.inverse_scheduler.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.inverse_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.inverse_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.inverse_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.inverse_scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        # 3. Clip or threshold "predicted x_0"
        if self.inverse_scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.inverse_scheduler.config.clip_sample_range, self.inverse_scheduler.config.clip_sample_range
            )
        # 4. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 5. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        # 6. Add noise to the sample
        variance = self.scheduler._get_variance(prev_timestep, timestep) ** (0.5)
        prev_sample += self.cfg.inversion_eta * torch.randn_like(prev_sample) * variance
        
        return prev_sample
    
    def get_inversion_timesteps(self, invert_to_t, B):
        n_training_steps = self.inverse_scheduler.config.num_train_timesteps
        effective_n_inversion_steps = self.cfg.inversion_n_steps #int((n_training_steps / invert_to_t) * self.cfg.inversion_n_steps)

        if self.inverse_scheduler.config.timestep_spacing == "leading":
            step_ratio = n_training_steps // effective_n_inversion_steps
            timesteps = (np.arange(0, effective_n_inversion_steps) * step_ratio).round().copy().astype(np.int64)
            timesteps += self.inverse_scheduler.config.steps_offset
        elif self.inverse_scheduler.config.timestep_spacing == "trailing":
            step_ratio = n_training_steps / effective_n_inversion_steps
            timesteps = np.round(np.arange(n_training_steps, 0, -step_ratio)[::-1]).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )
        # use only timesteps before invert_to_t
        timesteps = timesteps[timesteps < int(invert_to_t)]
        
        # Roll timesteps array by one to reflect reversed origin and destination semantics for each step
        timesteps = np.concatenate([[int(timesteps[0] - step_ratio)], timesteps])
        timesteps = torch.from_numpy(timesteps).to(self.device)
        
        # Add the last step
        delta_t = int(random.random() * self.inverse_scheduler.config.num_train_timesteps // self.cfg.inversion_n_steps)
        last_t = torch.tensor(
                    min(  #timesteps[-1] + self.inverse_scheduler.config.num_train_timesteps // self.inverse_scheduler.num_inference_steps,
                        invert_to_t + delta_t,
                        self.inverse_scheduler.config.num_train_timesteps - 1
                    )
                , device=self.device)
        timesteps = torch.cat([timesteps, last_t.repeat([B])])
        return timesteps
    
    @torch.no_grad()
    def invert_noise(self, start_latents, invert_to_t, prompt_utils, elevation, azimuth, camera_distances, image_cond):
        latents = start_latents.clone()
        B = start_latents.shape[0]
        
        timesteps = self.get_inversion_timesteps(invert_to_t, B)
        for t, next_t in zip(timesteps[:-1], timesteps[1:]):
            noise_pred, _, _ =  self.predict_noise(latents, t.repeat([B]), prompt_utils, elevation, azimuth, camera_distances,
                                                    guidance_scale=self.cfg.inversion_guidance_scale, image_cond=image_cond)
            latents = self.ddim_inversion_step(noise_pred, t, next_t, latents)

        # remap the noise from t+delta_t to t
        found_noise = self.get_noise_from_target(start_latents, latents, next_t)

        return latents, found_noise
    
    def get_noise_from_target(self, target, cur_xt, t):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        noise = (cur_xt - target * alpha_prod_t ** (0.5)) / (beta_prod_t ** (0.5))
        return noise
    
    def get_x0(self, original_samples, noise_pred, t):
        step_results = self.scheduler.step(noise_pred, t[0], original_samples, return_dict=True)
        if "pred_original_sample" in step_results:
            return step_results["pred_original_sample"]
        elif "denoised" in step_results:
            return step_results["denoised"]
        raise ValueError("Looks like the scheduler does not compute x0")
    
    @torch.no_grad()
    def compute_grad_sdi(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        image_cond,
        call_with_defined_noise: Optional[Float[Tensor, "B 4 64 64"]] = None,
    ):
        if call_with_defined_noise is not None:
            noise = call_with_defined_noise.clone()
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
        elif self.cfg.enable_sdi:
            latents_noisy, noise = self.invert_noise(latents, t[0], prompt_utils, elevation, azimuth, camera_distances, image_cond)
        else:
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
        
        noise_pred, neg_guidance_weights, text_embeddings = self.predict_noise(
            latents_noisy,
            t,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            image_cond=image_cond,
            guidance_scale=self.cfg.guidance_scale
        )

        latents_denoised = self.get_x0(latents_noisy, noise_pred, t).detach() # (latents_noisy - sigma * noise_pred) / alpha

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": camera_distances,
        }

        return latents_denoised, latents_noisy, noise, guidance_eval_utils




    def edit_latents(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 64 64"]:
        # self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.config.num_train_timesteps = 1000
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            image_cond_input = torch.cat([image_cond] * 2)
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            # threestudio.debug("Start editing...")
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    latent_model_input = torch.cat([latents] * 2)
                    (
                        down_block_res_samples,
                        mid_block_res_sample,
                    ) = self.forward_controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        image_cond=image_cond_input,
                        condition_scale=self.cfg.condition_scale,
                    )

                    noise_pred = self.forward_control_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )
                # perform classifier-free guidance
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )
                # get previous sample, continue loop
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            # threestudio.debug("Editing finished.")
        return latents

    def compute_grad_sds(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
    ):
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)
            image_cond_input = torch.cat([image_cond] * 3)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond_input,
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_neg, noise_pred_null = noise_pred.chunk(3)
        noise_pred = noise_pred_null + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_null
        )
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad

    def compute_grad_nfsd(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
    ):
        batch_size = latents.shape[0]
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)
            image_cond_input = torch.cat([image_cond] * 3)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond_input,
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_neg, noise_pred_null = noise_pred.chunk(3)
        # Eq.6 in Noise-free Score Distillation, Katzir et al., arXiv preprint arXiv:2310.17590, 2023.
        delta_c = self.cfg.guidance_scale * (noise_pred_text - noise_pred_null)
        mask = (t < 200).int().view(batch_size, 1, 1, 1)
        if self.cfg.use_dsd:
            delta_d = mask * noise_pred_null + (1 - mask) * (noise_pred_null + (noise_pred_null - noise_pred_neg))
        else:
            delta_d = mask * noise_pred_null + (1 - mask) * (noise_pred_null - noise_pred_neg)

        # noise_pred = noise_pred_text + self.cfg.guidance_scale * (
        #     noise_pred_text - noise_pred_uncond
        # )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (delta_c + delta_d)
        return grad

    def __call__(
            self,
            rgb: Float[Tensor, "B H W C"],
            control_image: Float[Tensor, "B H W C"],
            prompt_utils: PromptProcessorOutput,
            elevation: Float[Tensor, "B"],
            azimuth: Float[Tensor, "B"],
            camera_distances: Float[Tensor, "B"],
            rgb_as_latents=False,
            call_with_defined_noise=None,
            **kwargs,
    ):
        batch_size = rgb.shape[0]
        # assert batch_size == 1

        rgb_BCHW = rgb
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # image_cond = control_image
        image_cond = F.interpolate(
            control_image, (512, 512), mode="bilinear", align_corners=False
        )

        # temp = torch.zeros(batch_size).to(rgb.device)
        # text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, True
        )
        # text_embeddings = text_embeddings[:batch_size * 2]
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.edit_image:
            edit_latents = self.edit_latents(text_embeddings, latents, image_cond, t)
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (512, 512), mode="bilinear")
            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

        if self.cfg.use_nfsd or self.cfg.use_dsd:
            grad = self.compute_grad_nfsd(text_embeddings, latents, image_cond, t)
            target = (latents - grad).detach()
        elif self.cfg.enable_sdi:
            target, noisy_img, noise, guidance_eval_utils = self.compute_grad_sdi(
                latents,
                t,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                image_cond,
                call_with_defined_noise=call_with_defined_noise,
            )
            grad = latents - target
        else:
            grad = self.compute_grad_sds(text_embeddings, latents, image_cond, t)
            target = (latents - grad).detach()


        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        loss_sds = 0.5 * F.mse_loss(latents, target.detach(), reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.t_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            )  # progress percentage
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/experimental/controlnet-normal.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )

    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = cv2.resize(rgb_image, (512, 512))
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (guidance_out["edit_images"][0].detach().cpu().clip(0, 1).numpy() * 255)
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)
