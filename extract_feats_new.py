import torch
from PIL import Image
import torchvision.transforms as T
import torchvision
from torchvision.transforms.functional import InterpolationMode, to_pil_image, resize, to_tensor
from sklearn.decomposition import PCA
import numpy as np
import imageio 
import math
from itertools import product
from torch.nn import functional as F
import glob
import os
import pickle
import time


def generate_crop_boxes_quadratic(
    im_size, n_layers: int, overlap_ratio: float, num_crops_l0=2
):
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    # crop_boxes.append([0, 0, im_w, im_h])
    crop_boxes.append([
        int((im_w/2)-(short_side/2)), 
        int((im_h/2)-(short_side/2)),
        int((im_w/2)+(short_side/2)),
        int((im_h/2)+(short_side/2))])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))
    
    def reverse_overlap(orig_len, n_crops, crop):
        return int((crop * n_crops - orig_len)/(n_crops - 1))

    for i_layer in range(n_layers):
        n_crops_per_side_w = num_crops_l0 ** (i_layer + 1) + 1 ** (i_layer)
        n_crops_per_side_h = num_crops_l0 ** (i_layer + 1)

        overlap_w = int(overlap_ratio * im_w * (2 / n_crops_per_side_w))
        overlap_h = int(overlap_ratio * im_h * (2 / n_crops_per_side_h))

        crop_w = crop_len(im_w, n_crops_per_side_w, overlap_w)
        crop_h = crop_len(im_h, n_crops_per_side_h, overlap_h)
        crop = max(crop_w, crop_h)

        if im_w > im_h:
            overlap_h = reverse_overlap(im_h, n_crops_per_side_h, crop)
        else:
            overlap_w = reverse_overlap(im_w, n_crops_per_side_w, crop)

        crop_box_x0 = [int((crop - overlap_w) * i) for i in range(n_crops_per_side_w)]
        crop_box_y0 = [int((crop - overlap_h) * i) for i in range(n_crops_per_side_h)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop, im_w), min(y0 + crop, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def generate_im_feats(
        image: np.ndarray,
        model,
        transforms,
        output_size=(180, 320),
        num_crops_l0=4,
        crop_n_layers=2,
        model_input_size=896,
        crop_overlap_ratio=512/1500,
        embedding_dim=384,
        device="cuda:0"):

    orig_size = image.shape[:2]

    crop_boxes, layer_idxs = generate_crop_boxes_quadratic(
        orig_size, crop_n_layers, crop_overlap_ratio, num_crops_l0=num_crops_l0
        )

    if output_size is None:
        output_size = orig_size
        scale_h = 1
        scale_w = 1
    else:
        scale_h = output_size[0]/orig_size[0]
        scale_w = output_size[1]/orig_size[1]
        
    image_features = torch.zeros(1, embedding_dim, output_size[0], output_size[1]).to(device)
    image_features_sum = torch.zeros(1, 1, output_size[0], output_size[1]).to(device)

    i = 0
    for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        # get image features
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        transformed_im = preprocess(cropped_im, model_input_size)
        transformed_im_size = (transformed_im.shape[2], transformed_im.shape[3])
        crop_feat = predict(cropped_im, transforms, model, device)
        if model_input_size == 224:
            crop_feat = crop_feat.reshape(crop_feat.shape[0], 16, 16, crop_feat.shape[2]).permute(0, 3, 1, 2)
        else:
            crop_feat = crop_feat.reshape(crop_feat.shape[0], 64, 64, crop_feat.shape[2]).permute(0, 3, 1, 2)

        if scale_h != 1:
            scaled_size = (
                int(cropped_im_size[0]*scale_h),
                int(cropped_im_size[1]*scale_w))
        else:
            scaled_size = cropped_im_size

        crop_feat = postprocess_masks(
            crop_feat,
            transformed_im_size,
            scaled_size,
            model_input_size)

        # add features, upscaled embedding and mask data
        x0, y0, x1, y1 = crop_box
        if scale_h != 1:
            y0, x0 = int(scale_h*y0), int(scale_w*x0)
            y1, x1 = y0+scaled_size[0], x0+scaled_size[1]

        image_features[:, :, y0:y1, x0:x1] += crop_feat
        image_features_sum[:, :, y0:y1, x0:x1] += 1
        i += 1
    
    image_features = image_features / image_features_sum
    return image_features.cpu()


def postprocess_masks(
        feats: torch.Tensor,
        input_size,
        original_size,
        img_size
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        feats = F.interpolate(
            feats,
            img_size,
            mode="bilinear",
            align_corners=False,
        )

        feats = feats[:, :, : input_size[0], : input_size[1]]
        feats = F.interpolate(feats, original_size, mode="bilinear", align_corners=False)
        return feats


def preprocess(x, model_input_size) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    target_size = get_preprocess_shape(x.shape[0], x.shape[1], model_input_size)
    x = np.array(resize(to_pil_image(x), target_size))
    x = torch.as_tensor(x)
    x = x.permute(2, 0, 1).contiguous()[None, :, :, :]
    # Pad
    h, w = x.shape[-2:]
    padh = model_input_size - h
    padw = model_input_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def predict(img, transforms, model, device):
    img = to_pil_image(img)
    img = transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.forward_features(img)["x_norm_patchtokens"]
    return features

if __name__ == "__main__":
    do_pca = False
    save_feats = True
    dataset = 'DAVIS'
    dataset = 'iphone'
    # dataset = 'jono'
    # dataset = 'rgb_stacking'
    num_crops_l0 = 4
    crop_n_layers = 1
    device = 'cuda:7'
    num_dims = 32
    just_first = False

    # paths
    save_dir = '/data3/jseidens'
    input_dir = '/data3/jseidens' # '/home/jseidens/Documents/DynoSplatTAM/data'

    # get input paths
    if dataset == 'jono':
        input_path = f'{input_dir}/data/'
        # seqs = ['softball/ims/27', 'juggle/ims/14', 'boxes/ims/27', 'basketball/ims/21', 'football/ims/18', 'tennis/ims/8']
        seqs = ['softball/ims/8', 'juggle/ims/23', 'boxes/ims/23', 'basketball/ims/24', 'football/ims/3', 'tennis/ims/27']
        seqs = ['/'.join(p.split('/')[-3:]) for p in glob.glob(f'{input_path}/*/*/*') if 'ims' in p]
        just_first = True

    elif dataset == 'rgb_stacking':
        input_path = f'{input_dir}/tapvid_rgb_stacking'
        with open(f'{input_path}/tapvid_rgb_stacking.pkl', 'rb') as f:
            data = pickle.load(f)
        seqs = list(range(len(data)))
        _p = f'{input_dir}/tapvid_rgb_stacking/feats'

    elif dataset == 'iphone':
        input_path = f'{input_dir}/iphone2'
        seqs = ['/'.join(p.split('/')[-3:]) for p in glob.glob(f'{input_path}/*/rgb/2x')]
    else:
        input_path = f'{input_dir}/DAVIS/JPEGImages/480p'
        seqs = os.listdir(input_path)


    # model and transforms
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

    embedding_dim = 384
    model_input_size = 896 # 224
    transforms = T.Compose([
        T.Resize(model_input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(model_input_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # iterate over seqs
    for j, seq in enumerate(seqs):
        if 'teddy' not in seq:
            continue
        print(f"Processing Sequence {seq} ({j}/{len(seqs)})...")
        if dataset == 'jono':
            path = f'{input_path}/{seq}/*.jpg'
            paths = glob.glob(path)
            output_size = (360, 640)
            output_size = (180, 320)
        elif dataset == 'rgb_stacking':
            paths = [(seq, i) for i in range(len(data[seq]['video']))]
            output_size = (256, 256)
            output_size = (128, 128)
        elif dataset == 'iphone':
            path = f'{input_path}/{seq}/*.png'
            paths = glob.glob(path)
            # output_size = (480, 360) # 720 × 960
            output_size = (240, 180)
        else:
            path = f'{input_path}/{seq}/*.jpg'
            paths = glob.glob(path)
            # output_size = (480, 910)
            output_size = (240, 455)
            # output_size = (120, 225)

        initial_scale = torchvision.transforms.Resize(
            output_size, InterpolationMode.BILINEAR)
        pca = None
        times = list()
        
        for i, p in enumerate(sorted(paths)):
            if i%20 == 0:
                print(f"{i}/{len(paths)} of {seq}")
                if i != 0:
                    print(e-s)
            s = time.time()
            if dataset == 'iphone':
                img = to_tensor(Image.open(p))[:3]
            elif dataset != 'rgb_stacking':
                img = to_tensor(Image.open(p))
            else:
                img = torch.from_numpy(data[p[0]]['video'][p[1]]).permute(2, 0, 1)
            
            img = initial_scale(img).permute(1, 2, 0).numpy()

            features = generate_im_feats(
                img,
                model,
                transforms,
                output_size=output_size,
                model_input_size=model_input_size,
                num_crops_l0=num_crops_l0,
                crop_n_layers=crop_n_layers,
                embedding_dim=embedding_dim,
                device=device)

            features = features.permute(0, 2, 3, 1)
            features = features.cpu().squeeze().numpy()
            if features.shape[-1] != num_dims:
                shape = features.shape
                features = features.reshape(-1, shape[2])
                if pca is None:
                    pca = PCA(n_components=num_dims)
                    pca.fit(features)
                features = pca.transform(features)
                features = features.reshape(shape[0], shape[1], num_dims)

            if do_pca:
                shape = features.shape
                features = features.reshape( -1, shape[2])
                pca = PCA(n_components=3)
                pca.fit(features)

                pca_features = pca.transform(features)
                pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
                pca_features = pca_features * 255
                pca_features = pca_features.reshape(shape[0], shape[1], 3).astype(np.uint8)
                # pca_features = pca_features.reshape(shape[1], shape[2], 3).astype(np.uint8)
                _seq = str(seq).replace('/', '_')
                imageio.imwrite(f'test_{_seq}_{num_crops_l0}_{crop_n_layers}_{num_dims}_{model_input_size}.png', pca_features)
                quit()

            if save_feats:
                if dataset == 'rgb_stacking':
                    p = '{}/{:04d}/{:04d}.jpg'.format(_p, p[0], p[1])
                if model_input_size == 224:
                    path = os.path.join(
                            os.path.dirname(p).replace('ims', 'feats'),
                            os.path.basename(p)[:-4] + f'dino_img_quat_{num_crops_l0}_{crop_n_layers}_{num_dims}_{model_input_size}_{output_size[0]}_{output_size[1]}.npy')
                else:
                    path = os.path.join(
                            os.path.dirname(p).replace('ims', 'feats').replace('rgb', 'feats'),
                            os.path.basename(p)[:-4] + f'dino_img_quat_{num_crops_l0}_{crop_n_layers}_{num_dims}_{output_size[0]}_{output_size[1]}.npy')
                if i == 0:
                    print(path)
                
                path = path.replace('JPEGImages', 'FEATS')
                path = path.replace(input_dir, save_dir)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, features.squeeze())
            e = time.time()
            times.append(e-s)
            print(e-s) 
            if just_first:
                break
        print('times', sum(times)/len(times))
