import sys
sys.path.append('SegGPT/SegGPT_inference')

import os, json
import argparse
import torch
import numpy as np, cv2
import torch.nn.functional as F
import torch as T
from tqdm import tqdm
from SegGPT.SegGPT_inference.models_seggpt import seggpt_vit_large_patch16_input896x448
from PIL import Image
from utils import *
import shutil

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

COLOR_MAP = np.array([
    (0, 0,  0), # Background
    (255,  255, 255), # Sea, lake, & pond
])

#Evaluatoin criteria
CATCH_THRESHOLD = 0.3
RESP_RATE_THRESHOLD = 0.000239


def calculate_iou(pred:np.array, gt:np.array):
    pred_total = (pred != 0)
    gt_total = (gt != 0)
    intersection = (pred_total & gt_total).sum()
    union = pred_total.sum() + gt_total.sum() - intersection
    
    return intersection/union

def calculate_correct_yield(pred:np.array):
    pred_total = (pred != 0).sum()
    image_size = pred.shape[0]*pred.shape[1]

    return pred_total/image_size

@torch.no_grad()  
def run_one_image(img, tgt, model, device, mask=None):
    x = torch.tensor(img)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = torch.einsum('nhwc->nchw', tgt)

    if mask is None:
        bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
        bool_masked_pos[model.patch_embed.num_patches//2:] = 1
        bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    else:
        bool_masked_pos = torch.tensor(mask).unsqueeze(dim=0)
    valid = torch.ones_like(tgt)

    seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :] 

    output = torch.clip((output * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255)

    mask = mask[:, :, None].repeat(1, 1, model.patch_size**2 * 3)
    mask = model.unpatchify(mask)
    mask = mask.permute(0, 2, 3, 1)
    mask = mask[0, mask.shape[1]//2:, :, :]
    mask = mask.cpu().float()

    return output, mask

def inference_image_with_crop(model, device, img_path, img2_paths, tgt2_paths, outdir, split=2):
    res, hres = 448, 448

    full_image = Image.open(img_path).convert("RGB").resize((1024, 1024))
    row_size = full_image.size[0] // split
    col_size = full_image.size[1] // split
    
    h, w = full_image.size
    final_out_color = np.zeros((h, w, 3))
    final_out_label = np.zeros((h, w))
    final_out_image = np.zeros((h, w, 3))

    for row in range(split):
        for col in range(split):
            image = full_image.crop((row * row_size, col * col_size, (row + 1) * row_size, (col + 1) * col_size))
            input_image = np.array(image)
            image = np.array(image.resize((res, hres))) / 255.

            image_batch, target_batch = [], []
            for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
                full_img2 = Image.open(img2_path).convert("RGB").resize((1024, 1024))
                full_tgt2 = Image.open(tgt2_path).convert("RGB").resize((1024, 1024), Image.NEAREST)

                for i_row in range(split):
                    for i_col in range(split):
                        img2 = full_img2.crop((i_row * row_size, i_col * col_size, (i_row + 1) * row_size, (i_col + 1) * col_size))
                        tgt2 = full_tgt2.crop((i_row * row_size, i_col * col_size, (i_row + 1) * row_size, (i_col + 1) * col_size))

                        img2 = img2.resize((res, hres))
                        img2 = np.array(img2) / 255.

                        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
                        tgt2 = np.array(tgt2) / 255.

                        tgt = tgt2  # tgt is not available
                        tgt = np.concatenate((tgt2, tgt), axis=0)
                        img = np.concatenate((img2, image), axis=0)
                    
                        assert img.shape == (2*res, res, 3), f'{img.shape}'
                        # normalize by ImageNet mean and std
                        img = img - IMAGENET_MEAN
                        img = img / IMAGENET_STD

                        assert tgt.shape == (2*res, res, 3), f'{tgt.shape}'
                        # normalize by ImageNet mean and std
                        tgt = tgt - IMAGENET_MEAN
                        tgt = tgt / IMAGENET_STD

                        image_batch.append(img)
                        target_batch.append(tgt)
            
            img = np.stack(image_batch, axis=0)
            tgt = np.stack(target_batch, axis=0)
            torch.manual_seed(2)
            output, _ = run_one_image(img, tgt, model, device)
            output = F.interpolate(
                output[None, ...].permute(0, 3, 1, 2), 
                size=[row_size, col_size], 
                mode='nearest',
            ).permute(0, 2, 3, 1)
            
            output, label = cmap_to_lbl(output, torch.tensor(COLOR_MAP, device=output.device, dtype=output.dtype).unsqueeze(0))
            output = output[0].numpy()
            label = label[0].numpy()
            final_out_color[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = output
            final_out_label[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = label
            final_out_image[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = input_image

    concat = np.concatenate((final_out_image, final_out_color), axis=1)
    final_out_color = Image.fromarray((final_out_color).astype(np.uint8))
    concat = Image.fromarray((concat).astype(np.uint8))
    final_out_label = Image.fromarray((final_out_label).astype(np.uint8))

    filename = os.path.basename(img_path)
    os.makedirs(os.path.join(outdir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'concat'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'label'), exist_ok=True)

    final_out_color.save(os.path.join(outdir, 'color', filename))
    final_out_label.save(os.path.join(outdir, 'label', filename))
    concat.save(os.path.join(outdir, 'concat', filename))

    #calculate IOU
    temp = img_path.split('/')
    temp[-2] = 'labels'
    gt_path = '/'.join(temp) 
    gt = np.array(Image.open(gt_path).convert("RGB").resize((1024, 1024)))[:,:,0]
    pred_ = np.array(final_out_label)
    iou = calculate_iou(pred_, gt)
    print('Iou: ',iou)
    if iou >= CATCH_THRESHOLD:
        print("Catch: good catch")
    else:
        print("Catch: pass")

    #calculate correct_yield
    correct_yield = calculate_correct_yield(pred_)
    if correct_yield < RESP_RATE_THRESHOLD:
        print("correct_yield: correct yield")
    else:
        print("correct_yield: overkill")

    return iou, correct_yield

def inference_stitch(model, device, img_path, tgt_path, lbl_path, img2_paths, tgt2_paths, outdir, split=2, width=4):
    # run after inference_image_with_crop
    # only works for split = 2
    res, hres = 448, 448

    full_image = Image.open(img_path).convert('RGB').resize((1024, 1024))
    full_tgt = Image.open(tgt_path).convert('RGB').resize((1024, 1024), Image.NEAREST)
    full_lbl = Image.open(lbl_path).convert('L').resize((1024, 1024), Image.NEAREST)
    col_size = full_image.size[0] // split
    row_size = full_image.size[1] // split
    
    w, h = full_image.size
    final_out_color = np.array(full_tgt)
    final_out_label = np.array(full_lbl)

    crop_params = [
        [(w // 4, 0, 3 * w // 4, h // 2), 0], # top middle
        [(w // 4, h // 2, 3 * w // 4, h), 0], # bottom middle
        [(0, h // 4, w // 2, 3 * h // 4), 1], # left middle
        [(w // 2, h // 4, w, 3 * h // 4), 1], # right middle
        [(w // 4, h // 4, 3 * w // 4, 3 * h // 4), 2] # center
    ]

    for crop_param, stitch_type in crop_params:
        j1, i1, j2, i2 = crop_param
        assert j2 - j1 == col_size and i2 - i1 == row_size

        cropped_image = full_image.crop(crop_param).resize((res, hres))
        cropped_tgt = full_tgt.crop(crop_param).resize((res, hres), Image.NEAREST)

        cropped_image = np.array(cropped_image.resize((res, hres))) / 255.
        cropped_tgt = np.array(cropped_tgt) / 255.

        image_batch, target_batch = [], []
        for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
            full_img2 = Image.open(img2_path).convert('RGB').resize((1024, 1024))
            full_tgt2 = Image.open(tgt2_path).convert('RGB').resize((1024, 1024), Image.NEAREST)

            for i_row in range(split):
                for i_col in range(split):
                    img2 = full_img2.crop((i_row * row_size, i_col * col_size, (i_row + 1) * row_size, (i_col + 1) * col_size))
                    tgt2 = full_tgt2.crop((i_row * row_size, i_col * col_size, (i_row + 1) * row_size, (i_col + 1) * col_size))

                    img2 = img2.resize((res, hres))
                    img2 = np.array(img2) / 255.

                    tgt2 = tgt2.resize((res, hres), Image.NEAREST)
                    tgt2 = np.array(tgt2) / 255.

                    tgt = cropped_tgt
                    tgt = np.concatenate((tgt2, tgt), axis=0)
                    img = np.concatenate((img2, cropped_image), axis=0)

                    assert img.shape == (2*res, res, 3), f'{img.shape}'
                    # normalize by ImageNet mean and std
                    img = img - IMAGENET_MEAN
                    img = img / IMAGENET_STD

                    assert tgt.shape == (2*res, res, 3), f'{img.shape}'
                    # normalize by ImageNet mean and std
                    tgt = tgt - IMAGENET_MEAN
                    tgt = tgt / IMAGENET_STD

                    image_batch.append(img)
                    target_batch.append(tgt)
        
        img = np.stack(image_batch, axis=0)
        tgt = np.stack(target_batch, axis=0)
        torch.manual_seed(2)
        hstitch_mask = create_stitch_mask(28, 28, stitch_type, width)
        output, mask = run_one_image(img, tgt, model, device, hstitch_mask)
        output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2), 
            size=[row_size, col_size], 
            mode='nearest',
        ).permute(0, 2, 3, 1)
        mask = F.interpolate(
            mask[None, ...].permute(0, 3, 1, 2), 
            size=[row_size, col_size], 
            mode='nearest',
        ).permute(0, 2, 3, 1)
        
        output, label = cmap_to_lbl(output, torch.tensor(COLOR_MAP, device=output.device, dtype=output.dtype).unsqueeze(0))
        output = output[0].numpy()
        label = label[0].numpy()
        mask = mask[0].numpy()

        final_out_color[i1:i2, j1:j2] = output * mask + final_out_color[i1:i2, j1:j2] * (1 - mask)
        final_out_label[i1:i2, j1:j2] = label * mask[:, :, 0] + final_out_label[i1:i2, j1:j2] * (1 - mask[:, :, 0])

    concat = np.concatenate((np.array(full_image), np.array(full_tgt), final_out_color), axis=1)
    final_out_color = Image.fromarray((final_out_color).astype(np.uint8))
    final_out_label = Image.fromarray((final_out_label).astype(np.uint8))
    concat = Image.fromarray((concat).astype(np.uint8))

    filename = os.path.basename(img_path)
    os.makedirs(os.path.join(outdir, 'stitch', 'color'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'stitch', 'concat'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'stitch', 'label'), exist_ok=True)

    final_out_color.save(os.path.join(outdir, 'stitch', 'color', filename))
    final_out_label.save(os.path.join(outdir, 'stitch', 'label', filename))
    concat.save(os.path.join(outdir, 'stitch', 'concat', filename))

def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--model-path', type=str, help='path to ckpt', required=True)
    parser.add_argument('--prompt-img-dir', type=str, help='path to prompt image directory', default='/shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/dataset/train_dataset/train/images')
    parser.add_argument('--prompt-label-dir', type=str, help='path to prompt colored label directory', default='/shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/dataset/train_dataset/train/labels')
    parser.add_argument('--dataset-dir', type=str, help='path to input image dir to be tested', default='/shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/dataset/train_dataset/val/images')
    parser.add_argument('--mapping', type=str, help='path to mapping of query and prompt list', default="mappings/mapping_vit_filtered.json")
    parser.add_argument('--split', type=int, help='how many to image split into (each dim)', default=2)
    parser.add_argument('--stitch-width', type=int, help='width of the stitching', default=4)
    parser.add_argument('--top-k', type=int, help='top-k prompts to use', default=2)
    parser.add_argument('--device', type=str, help='cuda or cpu', default='cuda')
    parser.add_argument('--outdir', type=str, help='path to output directory', default='/shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/output')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_parser()
    print(args)

    model = seggpt_vit_large_patch16_input896x448()
    ckpt = T.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print('Checkpoint loaded')

    model = model.to(args.device)
    model.eval()

    mapping = json.load(open(args.mapping))

    try: #디렉토리가 존재하지 않으면 pass, 존재하면 지우기
        shutil.rmtree(args.outdir)
    except:
        pass
    os.mkdir(args.outdir)
    #open iou file with write mode
    iou_file = open(os.path.join(args.outdir, "info.txt"), 'w')
    total_iou = 0
    good_catch_total = 0
    correct_yield_total = 0
    for idx, input_image in enumerate(tqdm(mapping)):
        input = os.path.join(args.dataset_dir, input_image)
        prompt = [os.path.join(args.prompt_img_dir, file) for file in mapping[input_image][:args.top_k]]
        prompt_target = [os.path.join(args.prompt_label_dir, file) for file in mapping[input_image][:args.top_k]]

        iou, correct_yield = inference_image_with_crop(model, args.device, input, prompt, prompt_target, args.outdir, split=args.split)

        
        if iou >= CATCH_THRESHOLD:
            catch_result = "good catch"
            good_catch_total+=1
        else:
            catch_result = "escape" 

        if correct_yield < RESP_RATE_THRESHOLD:
            yield_result = "correct yield"
            correct_yield_total+=1
        else:
            yield_result = "overkill"

        iou_file.write(f"Iou, good catch, correct_yield of [{idx+1} input image] : {iou}("+catch_result+"), "+f", {correct_yield}("+yield_result+")"+"\n")
        total_iou += iou

        if args.split == 2:
            tgt_path = os.path.join(args.outdir, 'color', input_image)
            lbl_path = os.path.join(args.outdir, 'label', input_image)
            inference_stitch(model, args.device, input, tgt_path, lbl_path, prompt, prompt_target, args.outdir, split=args.split, width=args.stitch_width)
    
    mIou = total_iou/len(mapping)
    catch_rate = good_catch_total/len(mapping)
    yield_rate = correct_yield_total/len(mapping)
    pes = 0.5*catch_rate + 0.5*yield_rate
    print('mIou: ',mIou)
    iou_file.write(f"---------------mIou---------------\n mIou : {mIou}\n")
    iou_file.write(f"---------------catch_rate---------------\n catch_rate : {catch_rate}\n")
    iou_file.write(f"---------------yield_rate---------------\n yield_rate : {yield_rate}\n")
    iou_file.write(f"---------------PES---------------\n pes : {pes}\n")
    iou_file.close()
"""
python inference.py --ckpt_path /home/steve/SegGPT-FineTune/logs/1710148218/weights/epoch15_loss0.7601_metric0.0000.pt --output_dir submission

python seggpt_inference.py --ckpt_path ../../../tuning.pt --input_image /home/steve/Datasets/OpenEarthMap-FSS/valset/images/tonga_64.tif --prompt_image /home/steve/Datasets/OpenEarthMap-FSS/valset/images/christchurch_39.tif /home/steve/Datasets/OpenEarthMap-FSS/valset/images/sechura_37.tif /home/steve/Datasets/OpenEarthMap-FSS/valset/images/kitsap_22.tif /home/steve/Datasets/OpenEarthMap-FSS/valset/images/duesseldorf_15.tif /home/steve/Datasets/OpenEarthMap-FSS/valset/images/sechura_11.tif --prompt_target /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/christchurch_39.png /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/sechura_37.png /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/kitsap_22.png /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/duesseldorf_15.png /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/sechura_11.png --output_dir tuning
"""