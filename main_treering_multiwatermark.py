import sys, os
import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import to_pil_image, rgb_to_grayscale

from inverse_stable_diffusion_fixed import InversableStableDiffusionPipeline2
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
import torch

def main(args):
    table = None
    args.with_tracking=True
    
    wandb.init(entity='khlee0192', project='New-multiwatermark', name=args.run_name, tags=['tree_ring_watermark'])
    wandb.config.update(args)
    #table = wandb.Table(columns=['gen_no_w', 'gen_w1', 'gen_w2', 'reverse_no_w', 'reverse_w1', 'reverse_w2', 'prompt', 'no_w_metric', 'w_metric11', 'w_metric22', 'w_metric12', 'w_metric21'])
    #table = wandb.Table(columns=['prompt', 'no_w_metric1', 'no_w_metric2', 'w_metric11', 'w_metric22', 'w_metric12', 'w_metric21'])

    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    scheduler = DPMSolverMultistepScheduler(
        beta_end = 0.012,
        beta_schedule = 'scaled_linear', #squaredcos_cap_v2
        beta_start = 0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset = 1, #CHECK
        trained_betas = None,
        solver_order = args.solver_order,
        # set_alpha_to_one = False,
        # skip_prk_steps = True,
        # clip_sample = False,
        # algorithm_type = 'dpmsolver++',
        # solver_type = 'midpoint',
        # lower_order_final = True,
        # lambda_min_clipped = -5.1,
        # timestep_spacing = 'linspace',
        # steps_offset = 1, # not sure
        # prediction_type, thresholding, use_karras_sigmas, variance_type
        )

    pipe = InversableStableDiffusionPipeline2.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        #revision='fp16',
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)


    # Create patchs
    gt_patches = []
    for i in range(args.target_num):
        gt_patches.append(get_watermarking_pattern(pipe, args, device, option=i))

    w_metrics = [[] for _ in range(args.target_num)]

    image_dir = "./images/"+ args.run_name +"/"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    ind = 0
    for i in tqdm(range(args.start, args.end)):
        print(f"Image number : {ind+1}")

        # Plot part
        plot_name = image_dir + str(ind+1) + ".png"

        if ind == args.length:
            break
        
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]
        
        if args.prompt_reuse:
            text_embeddings = pipe.get_text_embedding(current_prompt)
            text_embeddings = pipe._encode_prompt(
                    current_prompt, 'cuda', 1, True, None)

        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()
        init_latents_w = copy.deepcopy(init_latents_no_w)

        # First generation
        outputs_no_w, latents_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )
        orig_image_no_w = outputs_no_w.images[0]

        # get watermarking mask
        watermarking_masks = []
        for i in range(args.target_num):
            watermarking_masks.append(get_watermarking_mask(init_latents_no_w, args, device, option=i))

        # inject watermark
        init_latents_ws = []
        ffts = []
        for i in range(args.target_num):
            temp_init_latents_w, temp_fft = inject_watermark(init_latents_w, watermarking_masks[i], gt_patches[i], args)
            init_latents_ws.append(temp_init_latents_w)
            ffts.append(temp_fft)

        # Create image
        orig_image_ws = []
        for i in range(args.target_num):
            temp_outputs_w, temp_latents_w = pipe(
                current_prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
            )
            orig_image_ws.append(temp_outputs_w.images[0])

        ### test watermark
        # distortion
        orig_image_no_w_auged = image_distortion_single(orig_image_no_w, seed, args)
        orig_image_w_augeds = []
        for i in range(args.target_num):
            orig_image_w_augeds.append(image_distortion_single(orig_image_ws[i], seed, args))

        ## SECTION : No Watermark
        # reverse img without watermarking (NO watermark)
        img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)

        # When we are only interested in w_metric
        if args.edcorrector:
            image_latents_no_w = pipe.edcorrector(img_no_w)
        else:    
            image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        # forward_diffusion -> inversion
        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.test_num_inference_steps,
            inverse_opt=not args.inv_naive,
            inv_order=args.inv_order
        )

        ## SECTION : With Watermark, repeated with args.target_num times
        # reverse img with watermarking (With watermark, first)
        img_ws = []
        image_latents_ws = []
        reversed_latents_ws = []
        for i in range(args.target_num):
            img_w = transform_img(orig_image_w_augeds[i]).unsqueeze(0).to(text_embeddings.dtype).to(device)

            if args.edcorrector:
                image_latents_w = pipe.edcorrector(img_w)
            else:    
                image_latents_w = pipe.get_image_latents(img_w, sample=False)
        
            # forward_diffusion -> inversion
            reversed_latents_w = pipe.forward_diffusion(
                latents=image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.test_num_inference_steps,
                inverse_opt=not args.inv_naive,
                inv_order=args.inv_order,
            )
            img_ws.append(img_w)
            image_latents_ws.append(image_latents_w)
            reversed_latents_ws.append(reversed_latents_w)

        ## Evaluation Period

        # First will check no_w_metric
        no_w_metric, _ = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_masks[0], gt_patches[0])
        # Second check w_metric of n*n cases
        for i in range(args.target_num): # i is index of original watermark
            for j in range(args.target_num): # j is index of checking watermark
                w_metrics[i][j], _ = eval_watermark(reversed_latents_no_w, reversed_latents_w[i], watermarking_masks[j], gt_patches[j])

        """
        results.append({
            'no_w1_metric': no_w_metric1,
            'no_w2_metric': no_w_metric2,
            'w_metric11': w_metric11,
            'w_metric22': w_metric22,
            'w_metric12': w_metric12,
            'w_metric21': w_metric21,
        })

        no_w_metrics1.append(no_w_metric1)
        no_w_metrics2.append(no_w_metric2)
        w_metrics11.append(w_metric11)
        w_metrics22.append(w_metric22)
        w_metrics12.append(w_metric12)
        w_metrics21.append(w_metric21)

        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                table.add_data(current_prompt,
                               no_w_metric1, no_w_metric2, w_metric11, w_metric22, w_metric12, w_metric21,
                               )
            else:
                table.add_data(current_prompt,
                               no_w_metric1, no_w_metric2, w_metric11, w_metric22, w_metric12, w_metric21,
                            )
        """

        plt.figure()
        plt.imshow(orig_image_no_w)
        plt.show()

        plt.figure(figsize=(6*args.target_num, 12))
        for i in range(0, args.target_num):
            img_title = "Image generated with watermark of " + str("A"+i)
            noise_title = "Noise reconstructed with watermark of " + str("A"+i)
            plt.subplot(2, args.target_num, 2*i)
            plt.imshow(orig_image_ws[i])
            plt.title(img_title)
            plt.subplot(2, args.target_num, 2*i+1)
            plt.imshow(torch.abs(ffts[i])[0][args.w_channel].detach().cpu())
            plt.title(img_title)
        plt.show()
        ind = ind + 1

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.finish()
    
    print('done')

    """ Below is ordinary accuracy check part, just skip since our current interst is metric
    # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [1] * len(no_w_metrics) + [0] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'w_metric_mean' : mean(w_metrics), 'w_metric_std' : stdev(w_metrics),
                   'w_metric_min' : min(w_metrics), 'w_metric_max' : max(w_metrics),
                   'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                   'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                   'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
        wandb.finish()
    
    print(f'clip_score_mean: {mean(clip_scores)}')
    print(f'w_clip_score_mean: {mean(clip_scores_w)}')
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1000, type=int)
    parser.add_argument('--length', default=2, type=int)
    parser.add_argument('--target_num', default=1, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=3.0, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=6, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    parser.add_argument("--w_crosscheck", action="store_true", default=False)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)
    
    # experiment
    parser.add_argument("--solver_order", default=1, type=int, help='1:DDIM, 2:DPM') 
    parser.add_argument("--edcorrector", action="store_true", default=False)
    parser.add_argument("--inv_naive", action='store_true', default=False, help="Naive DDIM of inversion")
    parser.add_argument("--inv_order", type=int, default=None, help="order of inversion, default:same as sampling")
    parser.add_argument("--prompt_reuse", action='store_true', default=False, help="use the same prompt for inversion")

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    if args.inv_order is None:
        if args.inv_naive:
            args.inv_order = 1
        else:
            args.inv_order = args.solver_order
    
    main(args)