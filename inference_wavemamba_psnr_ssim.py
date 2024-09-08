import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.wavemamba_arch import WaveMamba
from basicsr.utils.download_util import load_file_from_url

import torch

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

#from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import peak_signal_noise_ratio as psnr

from comput_psnr_ssim import calculate_ssim as ssim_gray
from comput_psnr_ssim import calculate_psnr as psnr_gray

import torch.nn.functional as F

def check_image_size(x,window_size=128):
    _, _, h, w = x.size()
    mod_pad_h = (window_size  - h % (window_size)) % (
                window_size )
    mod_pad_w = (window_size  - w % (window_size)) % (
                window_size)
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
    return x

def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/home/ywp/zwb/LOLv1/eval15/low',
                        help='Input image or folder')

    parser.add_argument('-g', '--gt', type=str,
                        default='/home/ywp/zwb/LOLv1/eval15/high',
                        help='groundtruth image')
    parser.add_argument('-w', '--weight', type=str,
                        default='/home/zwb/code/WaveMamba/ckpt/WaveMamba_LOLv1.pth',
                        help='path for model weights')

    parser.add_argument('-o', '--output', type=str, default='results/WaveMamba_UHDLL', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600,
                        help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enhance_weight_path = args.weight

    EnhanceNet = WaveMamba(in_chn=3,
                 wf=32,
                 n_l_blocks=[1,2,4],
                 n_h_blocks=[1,1,2], 
                 ffn_scale=2.0).to(device)
                 
    EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=False)
    EnhanceNet.eval()
    print_network(EnhanceNet)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    ssim_all = 0
    psnr_all = 0
    lpips_all = 0
    num_img = 0
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path) 
        pbar.set_description(f'Test {img_name}')

        gt_path = args.gt 
        file_name = path.split('/')[-1]

        gt_img = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED) 

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
        img_tensor = img2tensor(img).to(device) / 255. 
        img_tensor = img_tensor.unsqueeze(0) 
        b, c, h, w = img_tensor.size()
        print('b, c, h, w = img_tensor.size()', img_tensor.size())
        img_tensor = check_image_size(img_tensor) 

        with torch.no_grad():
            output = EnhanceNet.restoration_network(img_tensor) 
        output = output

        output = output[:, :, :h, :w]
        output_img = tensor2img(output) 
        gray = True

        ssim = ssim_gray(output_img, gt_img) 
        psnr = psnr_gray(output_img, gt_img) 
        lpips_value = lpips(2 * torch.clip(img2tensor(output_img).unsqueeze(0) / 255.0, 0, 1) - 1,
                            2 * img2tensor(gt_img).unsqueeze(0) / 255.0 - 1) 
        ssim_all += ssim
        psnr_all += psnr
        lpips_all += lpips_value
        num_img += 1
        print('num_img', num_img)
        print('ssim', ssim)
        print('psnr', psnr)
        print('lpips_value', lpips_value)
        save_path = os.path.join(args.output, f'{img_name}')
        imwrite(output_img, save_path) 

        pbar.update(1)
    pbar.close()
    print('avg_ssim:%f' % (ssim_all / num_img))
    print('avg_psnr:%f' % (psnr_all / num_img))
    print('avg_lpips:%f' % (lpips_all / num_img))


if __name__ == '__main__':
    main()
