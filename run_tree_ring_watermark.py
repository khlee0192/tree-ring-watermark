import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
from torchvision.transforms.functional import to_pil_image, rgb_to_grayscale

#from inverse_stable_diffusion_smh import InversableStableDiffusionPipeline
from inverse_stable_diffusion import InversableStableDiffusionPipeline
#from inverse_stable_diffusion_smh import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *

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
    img_w_correction = (pipe.decode_image(latent)/2+0.5).clamp(0, 1)

    plt.figure(figsize=(18, 6))
    plt.subplot(1,3,1)
    plt.imshow(to_pil_image(img[0]))
    plt.title("Original")
    plt.tick_params(axis='both', which='both', labelsize=8)
    plt.subplot(1,3,2)
    plt.imshow(to_pil_image(img_wo_correction[0]))
    plt.title("Encoder")
    plt.tick_params(axis='both', which='both', labelsize=8)
    plt.subplot(1,3,3)
    plt.imshow(to_pil_image(img_w_correction[0]))
    plt.title("Optimization")
    plt.tick_params(axis='both', which='both', labelsize=8)
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

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19.5, 6))

    axes[0].imshow(rgb_to_grayscale(to_pil_image(img[0])))
    axes[0].set_title("Original") 

    im2 = axes[1].pcolor(error1norm.cpu())
    axes[1].set_aspect('equal')
    axes[1].set_title("Encoder")

    im3 = axes[2].pcolor(error2norm.cpu())
    axes[2].set_aspect('equal')
    axes[2].set_title("Optimization(Ours)")

    fig.colorbar(im3, ax=axes.ravel().tolist())
    
    plt.savefig(title)
    #plt.show()

def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
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
    gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []

    accuracy = []
    accuracy_dist = []

    datadir = "./data/"
    errordatadir = "./errordata/"
    distortiondatadir = "./distortiondata/"
    distortionerrordatadir = "./distortionerrordata/"

    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(errordatadir):
        os.makedirs(errordatadir)
    if not os.path.exists(distortiondatadir):
        os.makedirs(distortiondatadir)
    if not os.path.exists(distortionerrordatadir):
        os.makedirs(distortionerrordatadir)

    ind = 0
    for i in tqdm(range(args.start, args.end)):
        if ind== 5: #Test optimization on 10 images
            break

        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents() # setting random x without WM
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )
        orig_image_no_w = outputs_no_w.images[0] # image generated at the first, without WM

        # Starting latent analysis with generated image
        orig_image = transform_img(orig_image_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
        test = pipe.get_image_latents(orig_image, sample=False)

        image_latents_w_modified = pipe.edcorrector(pipe.decode_image(test), text_embeddings) # input as the image
        original_error = compare_latents(test, pipe.get_image_latents(pipe.decode_image(test)))
        corrected_error = compare_latents(test, image_latents_w_modified)
        single_improvement = (original_error-corrected_error)/original_error*100

        print(f"compare error : {original_error}")
        print(f"error after optimization : {corrected_error}")
        print(f"Improvement : {single_improvement}%")
        
        accuracy.append(single_improvement)

        plot_name = datadir+str(i)+'.png'
        errorplot_name = errordatadir+str(i)+'.png'
        #plot_compare(test, image_latents_w_modified, pipe, plot_name)
        #plot_compare_errormap(test, image_latents_w_modified, pipe, errorplot_name)

        # Test on image distortion
        orig_image_auged = image_distortion_single(orig_image_no_w, seed, args)

        img_no_w = transform_img(orig_image_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        test = pipe.get_image_latents(img_no_w, sample=False)

        image_latents_w_modified = pipe.edcorrector(pipe.decode_image(test), text_embeddings) # input as the image
        original_error = compare_latents(test, pipe.get_image_latents(pipe.decode_image(test)))
        corrected_error = compare_latents(test, image_latents_w_modified)
        single_improvement = (original_error-corrected_error)/original_error*100

        print(f"compare error_dist : {original_error}")
        print(f"error after optimization_dist : {corrected_error}")
        print(f"Improvement_dist : {single_improvement}%")

        accuracy_dist.append(single_improvement)

        plot_name = distortiondatadir+str(i)+'.png'
        errorplot_name = distortionerrordatadir+str(i)+'.png'
        #plot_compare(test, image_latents_w_modified, pipe, plot_name)
        #plot_compare_errormap(test, image_latents_w_modified, pipe, errorplot_name)
    
    
        # generation with watermark
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device) # Get initial 

        # inject watermark, injection occurs on frequency domain
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args) # Saves latent with WM

        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            ) # Not a single tensor, a pipeline structure containing "images", "nsfw_content_detected", "init_latents"
        orig_image_w = outputs_w.images[0]

        ### test watermark
        # distortion
        orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)

        # reverse img without watermarking
        img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # reverse img with watermarking
        img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )
        
        ## Evaluation on quality of the watermark
        # eval
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)

        if args.reference_model is not None:
            sims = measure_similarity([orig_image_no_w, orig_image_w], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
            w_no_sim = sims[0].item()
            w_sim = sims[1].item()
        else:
            w_no_sim = 0
            w_sim = 0

        results.append({
            'no_w_metric': no_w_metric, 'w_metric': w_metric, 'w_no_sim': w_no_sim, 'w_sim': w_sim,
        })

        no_w_metrics.append(no_w_metric)
        w_metrics.append(w_metric)

        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                # log images when we use reference_model
                table.add_data(wandb.Image(orig_image_no_w), w_no_sim, wandb.Image(orig_image_w), w_sim, current_prompt, no_w_metric, w_metric)
            else:
                table.add_data(None, w_no_sim, None, w_sim, current_prompt, no_w_metric, w_metric)

            clip_scores.append(w_no_sim)
            clip_scores_w.append(w_sim)
    
        ind = ind + 1

    # LKH : Show results of optimization
    accuracy_list = []
    for i in accuracy:
        accuracy_list.append(i.cpu().detach().numpy())
    accuracy_list = np.array(accuracy_list, dtype=np.float64)
    
    print(f"accuracy mean : {accuracy_list.mean()}")
    print(f"accuracy min : {accuracy_list.min()}")
    print(f"accuracy max : {accuracy_list.max()}")
    print(f"accuracy var : {accuracy_list.var()}")

    accuracy_list_dist = []
    for i in accuracy_dist:
        accuracy_list_dist.append(i.cpu().detach().numpy())
    accuracy_list_dist = np.array(accuracy_list_dist, dtype=np.float64)
    
    print(f"accuracy mean : {accuracy_list_dist.mean()}")
    print(f"accuracy min : {accuracy_list_dist.min()}")
    print(f"accuracy max : {accuracy_list_dist.max()}")
    print(f"accuracy var : {accuracy_list_dist.var()}")
    
    # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [1] * len(no_w_metrics) + [0] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                   'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                   'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
    
    print(f'clip_score_mean: {mean(clip_scores)}')
    print(f'w_clip_score_mean: {mean(clip_scores_w)}')
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
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
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)