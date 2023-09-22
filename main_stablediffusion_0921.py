import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import to_pil_image, rgb_to_grayscale
import torchvision.transforms as transforms

from inverse_stable_diffusion import InversableStableDiffusionPipeline
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

def evaluate(t1,t2,t3,t4):
    recon_err_T0T = []
    recon_err_0T0 = [] 
    for i in range(len(t1)):
        recon_err_T0T.append( (((t1[i]-t3[i]).norm()/(t1[i].norm())).item())**2 )
        recon_err_0T0.append( (((t2[i]-t4[i]).norm()/(t2[i].norm())).item())**2 )
    import statistics
    data_T0T = recon_err_T0T
    mean_T0T = statistics.mean(data_T0T)
    std_T0T = statistics.stdev(data_T0T)

    # 결과 출력
    print("T0T")
    print("평균(mean):", mean_T0T)
    print("표준 편차(std):", std_T0T)
    data_0T0 = recon_err_0T0
    mean_0T0 = statistics.mean(data_0T0)
    std_0T0 = statistics.stdev(data_0T0)

    # 결과 출력
    print("0T0")
    print("평균(mean):", mean_0T0)
    print("표준 편차(std):", std_0T0)

    return mean_T0T, std_T0T, mean_0T0, std_0T0

def main(args):
    table = None
    args.with_tracking = True
    if args.with_tracking:
        print("wandb initialized")
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['image', 'recon_latent', 'recon_image' 'prompt'])
    
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

    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)


    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)


    x_T_first = []
    x_0_second = []
    x_T_third = []
    x_0_fourth = []

    test = 2
    ind = 0
    for i in tqdm(range(args.start, args.end)):
        if ind == test:
             break
        print(f"{ind+1}/{test} iteration")
        
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        if args.prompt_reuse:
            text_embeddings = pipe.get_text_embedding(current_prompt)

        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents = pipe.get_random_latents()
        x_T_first.append(init_latents.clone())

        outputs = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents,
            )
        orig_image = outputs.images[0]
        x_0_second.append(transforms.ToTensor()(orig_image).to(torch.float16))
        
        
        img = transform_img(orig_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
        
        if args.edcorrector:
            image_latents = pipe.edcorrector(img)
        else:    
            image_latents = pipe.get_image_latents(img, sample=False)

        ##forward_diffusion -> inversion
        # reversed_latents = pipe.backward_diffusion(
        #     latents=image_latents,
        #     text_embeddings=text_embeddings,
        #     guidance_scale=1,
        #     num_inference_steps=args.test_num_inference_steps,
        #     inverse_opt=not args.inv_naive,
        #     inv_order=args.inv_order,
        #     reverse_process=True,
        # )
        reversed_latents = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
            inverse_opt=not args.inv_naive,
            inv_order=args.inv_order
        )

        x_T_third.append(reversed_latents.clone())
            
        #reversed_image_no_w = to_pil_image(((pipe.decode_image(reversed_latents)/2+0.5).clamp(0,1))[0])

        reconstructed_outputs = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=reversed_latents,
            )
        reconstruted_image = reconstructed_outputs.images[0]
        x_0_fourth.append(transforms.ToTensor()(reconstruted_image).to(torch.float16))

        
        if args.with_tracking:
            table.add_data(wandb.Image(orig_image),wandb.Image(reconstruted_image), current_prompt)
        
        ind = ind + 1


    mean_T0T, std_T0T, mean_0T0, std_0T0 = evaluate(x_T_first,x_0_second,x_T_third,x_0_fourth)

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'mean_T0T' : mean_T0T,'std_T0T' : std_T0T,'mean_0T0' : mean_0T0,'std_0T0' : std_0T0})



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true', default=True)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=50, type=int)
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
    
    # experiment
    parser.add_argument("--solver_order", default=1, type=int, help='1:DDIM, 2:DPM') 
    parser.add_argument("--edcorrector", action="store_true", default=False)
    parser.add_argument("--inv_naive", action='store_true', default=False, help="Naive DDIM of inversion")
    parser.add_argument("--inv_order", type=int, default=None, help="order of inversion, default:same as sampling")
    parser.add_argument("--prompt_reuse", action='store_true', default=True, help="use the same prompt for inversion")
    parser.add_argument("--baseline", default=False, type=str)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    if args.inv_order is None:
        if args.inv_naive:
            args.inv_order = 1
        else:
            args.inv_order = args.solver_order
    
    main(args)