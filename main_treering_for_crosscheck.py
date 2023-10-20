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


def compare_latents(z, z_comp):
    """
    parameters
    z : latent variables after calculation
    z_comp : latent vatiables for comparison

    returns norm(z-z_comp)/norm(z_comp)
    """
    diff = z - z_comp
    return torch.norm(diff)/torch.norm(z_comp)

def plot_compare(latent, modified_latent, pipe, title):
    # Latent, pipe -> draw image, maybe good to make clean code
        # check on image_latents_w, pipe.get_image_latents(pipe.decode_image(image_latents_w)), image_latents_w_modified
    img = (pipe.decode_image(latent)/2+0.5).clamp(0, 1)
    img_wo_correction = (pipe.decode_image(pipe.get_image_latents(pipe.decode_image(latent)))/2+0.5).clamp(0, 1)
    img_w_correction = (pipe.decode_image(modified_latent)/2+0.5).clamp(0, 1) # modified

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(to_pil_image(img[0]))
    plt.title("z")
    plt.subplot(1,3,2)
    plt.imshow(to_pil_image(img_wo_correction[0]))
    plt.title("E(D(z))")
    plt.subplot(1,3,3)
    plt.imshow(to_pil_image(img_w_correction[0]))
    plt.title("E*(D(z))")
    plt.savefig(title)
    #plt.show()

def plot_compare_errormap(latent, modified_latent, pipe, title):
    # Latent, pipe -> draw image, maybe good to make clean code
        # check on image_latents_w, pipe.get_image_latents(pipe.decode_image(image_latents_w)), image_latents_w_modified
    img = (pipe.decode_image(latent)/2+0.5).clamp(0, 1)
    img_wo_correction = (pipe.decode_image(pipe.get_image_latents(pipe.decode_image(latent)))/2+0.5).clamp(0, 1)
    img_w_correction = (pipe.decode_image(modified_latent)/2+0.5).clamp(0, 1)

    img = img[0]
    error1 = (img_wo_correction[0]-img)
    error1norm = torch.sqrt(torch.abs(error1[0])**2 + torch.abs(error1[1])**2 + torch.abs(error1[2])**2)
    error1norm = torch.flip(error1norm, [0])
    error2 = (img_w_correction[0]-img)
    error2norm = torch.sqrt(torch.abs(error2[0])**2 + torch.abs(error2[1])**2 + torch.abs(error2[2])**2)
    error2norm = torch.flip(error2norm, [0])
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(rgb_to_grayscale(to_pil_image(img[0])))
    plt.title("z")
    plt.subplot(1,3,2)
    plt.pcolor(error1norm.cpu())
    plt.title("E(D(z))")
    plt.subplot(1,3,3)
    plt.pcolor(error2norm.cpu())
    plt.title("E*(D(z))")
    plt.colorbar()
    #plt.savefig(title)
    plt.show()


def main(args):
    table = None
    args.with_tracking=True
    
    wandb.init(entity='khlee0192', project='Crosscheck', name=args.run_name, tags=['tree_ring_watermark'])
    wandb.config.update(args)
    #table = wandb.Table(columns=['gen_no_w', 'gen_w1', 'w_clip_score', 'reverse_no_w', 'reverse_w', 'prompt', 'no_w_metric', 'w_metric'])
    table = wandb.Table(columns=['gen_no_w', 'gen_w1', 'gen_w2', 'reverse_no_w', 'reverse_w1', 'reverse_w2', 'prompt', 'no_w_metric', 'w_metric11', 'w_metric22', 'w_metric12', 'w_metric21'])

    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    scheduler = DPMSolverMultistepScheduler(
        beta_end = 0.012,
        beta_schedule = 'scaled_linear', #squaredcos_cap_v2
        beta_start = 0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        #steps_offset = 1, #CHECK
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


    # ground-truth patch
    gt_patch1 = get_watermarking_pattern(pipe, args, device)

    # set new watermark
    args2 = args
    args2.w_radius = 15
    gt_patch2 = get_watermarking_pattern(pipe, args2, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics11 = []
    w_metrics22 = []
    w_metrics12 = []
    w_metrics21 = []

    ind = 0
    for i in tqdm(range(args.start, args.end)):
        print(f"Image number : {ind+1}")
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
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask1 = get_watermarking_mask(init_latents_w, args, device)
        watermarking_mask2 = get_watermarking_mask(init_latents_w, args2, device)

        # inject watermark
        init_latents_w1, _ = inject_watermark(init_latents_w, watermarking_mask1, gt_patch1, args)
        init_latents_w2, _ = inject_watermark(init_latents_w, watermarking_mask2, gt_patch2, args2)

        # Create image
        outputs_w1, latents_w1 = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w1,
            )
        orig_image_w1 = outputs_w1.images[0]

        outputs_w2, latents_w2 = pipe(
            current_prompt,
            num_images_per_prompt=args2.num_images,
            guidance_scale=args2.guidance_scale,
            num_inference_steps=args2.num_inference_steps,
            height=args2.image_length,
            width=args2.image_length,
            latents=init_latents_w2,
            )
        orig_image_w2 = outputs_w2.images[0]

        ### test watermark
        # distortion
        orig_image_no_w_auged, orig_image_w_auged1 = image_distortion(orig_image_no_w, orig_image_w1, seed, args)
        orig_image_w_auged2 = image_distortion_single(orig_image_w2, seed, args2)

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
        
        reversed_image_no_w = to_pil_image(((pipe.decode_image(reversed_latents_no_w)/2+0.5).clamp(0,1))[0])

        # reverse img with watermarking (With watermark, first)
        img_w1 = transform_img(orig_image_w_auged1).unsqueeze(0).to(text_embeddings.dtype).to(device)

        if args.edcorrector:
            image_latents_w1 = pipe.edcorrector(img_w1)
        else:    
            image_latents_w1 = pipe.get_image_latents(img_w1, sample=False)
        
        # forward_diffusion -> inversion
        reversed_latents_w1 = pipe.forward_diffusion(
            latents=image_latents_w1,
            text_embeddings=text_embeddings,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.test_num_inference_steps,
            inverse_opt=not args.inv_naive,
            inv_order=args.inv_order,
        )
        reversed_image_w1 = to_pil_image(((pipe.decode_image(reversed_latents_w1)/2+0.5).clamp(0,1))[0])
        
        # reverse img with watermarking (With watermark, second)
        img_w2 = transform_img(orig_image_w_auged2).unsqueeze(0).to(text_embeddings.dtype).to(device)

        if args.edcorrector:
            image_latents_w2 = pipe.edcorrector(img_w2)
        else:    
            image_latents_w2 = pipe.get_image_latents(img_w2, sample=False)
        
        # forward_diffusion -> inversion
        reversed_latents_w2 = pipe.forward_diffusion(
            latents=image_latents_w2,
            text_embeddings=text_embeddings,
            guidance_scale=args2.guidance_scale,
            num_inference_steps=args2.test_num_inference_steps,
            inverse_opt=not args2.inv_naive,
            inv_order=args2.inv_order,
        )
        reversed_image_w2 = to_pil_image(((pipe.decode_image(reversed_latents_w2)/2+0.5).clamp(0,1))[0])


        # eval
        # A checks with watermarked image from A
        no_w_metric, w_metric11 = eval_watermark(reversed_latents_no_w, reversed_latents_w1, watermarking_mask1, gt_patch1, args)
        # B checks with watermarked image from B
        _, w_metric22 = eval_watermark(reversed_latents_no_w, reversed_latents_w2, watermarking_mask2, gt_patch2, args2)
        # A checks with wateermarked image from B
        _, w_metric12 = eval_watermark(reversed_latents_no_w, reversed_latents_w2, watermarking_mask1, gt_patch1, args)
        # B checks with wateermarked image from A
        _, w_metric21 = eval_watermark(reversed_latents_no_w, reversed_latents_w1, watermarking_mask2, gt_patch2, args2)

        """
        if args.reference_model is not None:
            sims = measure_similarity([orig_image_no_w, orig_image_w], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
            w_no_sim = sims[0].item()
            w_sim = sims[1].item()
        else:
            w_no_sim = 0
            w_sim = 0
        """

        results.append({
            'no_w_metric': no_w_metric,
            'w_metric11': w_metric11,
            'w_metric22': w_metric22,
            'w_metric12': w_metric12,
            'w_metric21': w_metric21,
        })

        no_w_metrics.append(no_w_metric)
        w_metrics11.append(w_metric11)
        w_metrics22.append(w_metric22)
        w_metrics12.append(w_metric12)
        w_metrics21.append(w_metric21)

        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                table.add_data(wandb.Image(orig_image_no_w), wandb.Image(orig_image_w1), wandb.Image(orig_image_w2), 
                               wandb.Image(reversed_image_no_w), wandb.Image(reversed_image_w1), wandb.Image(reversed_image_w2),
                               current_prompt,
                               no_w_metric, w_metric11, w_metric22, w_metric12, w_metric21,
                               )
            else:
                table.add_data(None, None, None,
                               None, None, None,
                               current_prompt,
                               no_w_metric, w_metric11, w_metric22, w_metric12, w_metric21,
                               )
        
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
    parser.add_argument('--w_radius', default=10, type=int)
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