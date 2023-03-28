import os
import h5py
import pickle
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from imgviz import label_colormap
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T

import multiprocessing
import torch
from utils.ray_utils import *
from utils.render_util import *


class ReplicaDatasetDMNeRF(Dataset):
    def __init__(self, datadir, split='train', near=0.05, far=47, scene_bbox_stretch=5.5,
                 downsample=1.0, is_stack=False,
                 use_tps_dataset=False,
                 gen_rays=True, use_sem=True, load_colored_sem=False,
                 dino_feature_path=None, sem_interval=1):
        self.split = split
        if use_tps_dataset:
            self.root_dir = os.path.join(datadir, f'{self.split}_dataset')
        else:
            self.root_dir = datadir
        self.dino_feature_path = dino_feature_path

        self.is_stack = is_stack
        self.use_tps_dataset = use_tps_dataset
        self.gen_rays = gen_rays
        self.use_sem = use_sem
        self.load_colored_sem = load_colored_sem

        img_w, img_h = 640, 480
        self.img_wh = (int(img_w / downsample), int(img_h / downsample))
        self.define_transforms()  # tensor transforms

        self.img_total_num = len(glob(os.path.join(self.root_dir, "rgb", "rgb_*.png")))

        # replica near_far
        self.near_far = [near, far]  # used in sample_ray(tensorBase.py) for clipping samples, near must be 0.1 ?
        self.scene_bbox_stretch = scene_bbox_stretch

        self.sem_interval = sem_interval
        self.read_meta()
        if gen_rays:
            self.define_proj_mat()

        self.white_bg = True
        self.downsample = downsample

    def read_meta(self):
        w, h = self.img_wh

        hfov = 90
        self.focal_x = 0.5 * w / np.tan(0.5 * np.radians(hfov))  # w ?
        self.focal_y = self.focal_x
        cx = (w - 1.) / 2
        cy = (h - 1.) / 2

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x, self.focal_y])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x, 0, cx], [0, self.focal_y, cy], [0, 0, 1]]).float().cpu()

        # load c2w for all images in the video
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        # self.image_paths = []
        # self.sem_paths = []
        # self.colored_sem_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_sems = []
        self.all_colored_sems = []
        self.all_masks = []
        self.all_depth = []
        self.downsample = 1.0
        # for semantic labels remapping ?
        self.sem_samples = {}
        self.sem_samples["sem_img"] = []
        self.sem_samples["label_ins_map"] = {}
        self.sem_samples["ins_label_map"] = {}

        if self.use_tps_dataset:
            self.idxs = list(range(0, self.img_total_num))
        else:
            img_eval_interval = 5
            if self.split == "train":
                self.idxs = list(range(0, self.img_total_num, img_eval_interval))
            elif self.split == "test":
                self.idxs = list(range(img_eval_interval // 2, self.img_total_num, img_eval_interval))

        for i in tqdm(self.idxs, desc=f'Loading data {self.split} ({len(self.idxs)})'):  # img_list:#
            c2w = torch.FloatTensor(self.Ts_full[i])
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, "rgb", f"rgb_{i}.png")
            img = Image.open(image_path)
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w), normalized
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
            self.all_rgbs += [img]

            if self.use_sem:
                sem_image_path = os.path.join(self.root_dir, 'semantic_instance', f"semantic_instance_{i}.png")
                sem_img = Image.open(sem_image_path)  # type: Image
                self.sem_samples["sem_img"].append(np.array(sem_img))
                if self.downsample != 1.0:
                    sem_img = sem_img.resize(self.img_wh, Image.LANCZOS)
                sem_img = self.transform(sem_img)
                sem_img = sem_img.view(-1, h*w).permute(1, 0)  # (h*w, 1)
                self.all_sems += [sem_img]

            if self.load_colored_sem:
                colored_sem_img_path = os.path.join(self.root_dir, "sem", f"{i:03d}.png")
                # self.colored_sem_paths += colored_sem_img_path
                colored_sem_img = Image.open(colored_sem_img_path)
                if self.downsample != 1.0:
                    colored_sem_img = colored_sem_img.resize(self.img_wh, Image.LANCZOS)
                colored_sem_img = self.transform(colored_sem_img)
                colored_sem_img = colored_sem_img.view(3, -1).permute(1, 0)  # (h*w, 3)
                self.all_colored_sems += [colored_sem_img]

            if self.gen_rays:
                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            # pixel-wise in training
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (num_imgs*h*w, 3)
        else:
            # image-wise in testing
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)  # (num_imgs,h,w,3)

        if self.use_sem:
            if not self.is_stack:
                self.all_sems = torch.cat(self.all_sems, 0)  # (num_imgs*h*w, 1)
            else:
                self.all_sems = torch.stack(self.all_sems, 0).reshape(-1, *self.img_wh[::-1], 1)  # (num_imgs,h,w,1)

        """load"""
        if self.load_colored_sem:
            if not self.is_stack:
                self.all_colored_sems = torch.cat(self.all_colored_sems, 0)  # (num_imgs*h*w, 3)
            else:
                self.all_colored_sems = torch.stack(self.all_colored_sems, 0).reshape(-1, *self.img_wh[::-1], 3)  # (num_imgs,h,w,3)
        """change from sem_maps"""

        if self.gen_rays:
            # used in train.py for aabb
            # adaptive scene_bbox
            all_rays_o = torch.stack(self.all_rays)[..., :3]  # for all images, (N_imgs, h*w, 3)
            all_rays_o = all_rays_o.reshape(-1, 3)

            scene_min = torch.min(all_rays_o, 0)[0] - self.scene_bbox_stretch
            scene_max = torch.max(all_rays_o, 0)[0] + self.scene_bbox_stretch

            self.scene_bbox = torch.stack([scene_min, scene_max]).reshape(-1, 3)

            self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
            self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

            self.poses = torch.stack(self.poses)

            if not self.is_stack:
                self.all_rays = torch.cat(self.all_rays, 0)  # (num_imgs*h*w, 3)
            else:
                self.all_rays = torch.stack(self.all_rays, 0)  # (num_imgs,h*w, 3)

            # render images from new view-points
            center = torch.mean(self.scene_bbox, dim=0)
            radius = torch.norm(self.scene_bbox[1]-center)*0.1
            up = torch.mean(self.poses[:, :3, 1], dim=0).tolist()
            pos_gen = circle(radius=radius, h=-0.2*up[1], axis='y')
            self.render_path = gen_path(pos_gen, up=up, frames=200).float().cpu()
            self.render_path[:, :3, 3] += center

    def save_c2ws(self, c2w_save_path):
        for i in self.idxs:
            c2w_array = self.Ts_full[i].reshape(1, -1)
            os.makedirs(c2w_save_path, exist_ok=True)

            if self.split == "train":
                with open(f'{c2w_save_path}/traj_w_c.txt', 'ab') as f:
                    np.savetxt(f, c2w_array, delimiter=" ")
            elif self.split == "test":
                with open(f'{c2w_save_path}/traj_w_c.txt', 'ab') as f:
                    np.savetxt(f, c2w_array, delimiter=" ")

    def remap_sem_gt_label(self, train_sem_imgs=None, test_sem_imgs=None, sem_info_path=None,
                           save_map=False,
                           load_map=False, load_ins2label_path=None):
        self.sem_samples["sem_img"] = np.asarray(self.sem_samples["sem_img"])
        self.sem_samples["sem_remap"] = self.sem_samples["sem_img"].copy()

        if load_map:
            assert load_ins2label_path, "map file path must be provided"
            with open(load_ins2label_path, "rb") as f:
                ins2label = pickle.load(f)
            self.num_semantic_class = len(ins2label.keys())
            self.num_valid_semantic_class = self.num_semantic_class

            for ins, label in ins2label.items():
                self.sem_samples["sem_remap"][self.sem_samples["sem_img"] == ins] = label
        else:
            self.semantic_classes = np.unique(np.concatenate((np.unique(train_sem_imgs), np.unique(test_sem_imgs))).astype(np.uint8))
            self.num_semantic_class = self.semantic_classes.shape[0]
            self.num_valid_semantic_class = self.num_semantic_class

            for i in range(self.num_semantic_class):
                self.sem_samples["sem_remap"][self.sem_samples["sem_img"] == self.semantic_classes[i]] = i
                self.sem_samples["label_ins_map"][i] = self.semantic_classes[i]
                self.sem_samples["ins_label_map"][self.semantic_classes[i]] = i

            if save_map:
                with open(self.root_dir+"semantic_instance/label2ins_map.pkl", "wb") as f:
                    pickle.dump(self.sem_samples["label_ins_map"], f)

                with open(self.root_dir+"semantic_instance/ins2label_map.pkl", "wb") as f:
                    pickle.dump(self.sem_samples["ins_label_map"], f)

    def select_sems(self):
        assert self.is_stack
        self.selected_rays = self.all_rays[::self.sem_interval, ...]
        self.selected_rgbs = self.all_rgbs[::self.sem_interval, ...]
        self.selected_sems = torch.tensor(self.sem_samples["sem_remap"][::self.sem_interval, ...]).unsqueeze(-1)

    def set_label_colour_map(self, sem_info_path, label2color_path=None):
        if label2color_path:  # label to assigned color
            color_f = os.path.join(label2color_path)
        else:
            color_f = os.path.join(self.root_dir, 'ins_rgb.hdf5')

        with h5py.File(color_f, 'r') as f:
            ins_rgbs = f['datasets'][:]  # ndarray
        f.close()

        def label_color_map(sem_map):
            color_map = np.zeros(shape=(int(self.img_wh[0] * self.img_wh[1]), 3))
            for label in np.unique(sem_map):
                valid_label_list = list(range(0, ins_rgbs.shape[0]))
                if label in valid_label_list:
                    color_map[sem_map == label] = ins_rgbs[label]
            return color_map

        self.label_color_map = label_color_map

    def inv_map_sem_gt_label(self,
                             sem_map  # (H*W)
                             ):
        # todo: check before tps
        gt_sem_map = np.zeros_like(sem_map)
        for remap_sem_value in self.sem_samples["label_ins_map"].keys():
            if remap_sem_value in sem_map:
                gt_sem_map[sem_map == remap_sem_value] = self.sem_samples["label_ins_map"][remap_sem_value]

        return gt_sem_map

    def load_dino_features(self, normalized=True, resize=True):
        features = torch.load(self.dino_feature_path)
        n_channels, feamap_h, feamap_w,  = features[list(features.keys())[0]].shape

        # reshape dino features
        if resize:
            for k in features:
                features[k] = torch.nn.functional.interpolate(features[k][None], size=(self.img_wh[1], self.img_wh[0]), mode="nearest")[0]
            n_channels, feamap_h, feamap_w = features[k].shape

        if normalized:
            for k in features:
                features[k] = torch.nn.functional.normalize(features[k], dim=0)

        features_array = np.zeros(shape=(len(self.idxs), n_channels, feamap_h, feamap_w))
        for i in tqdm(range(len(self.idxs)), desc='Loading DINO features: '):
            fn_idx = self.idxs[i]
            features_array[i] = features[f'rgb_{fn_idx}.png'].numpy()
        # features = np.stack([features[f'rgb_{fn}.png'].permute(1, 2, 0).numpy() for fn in self.idxs], axis=-1)  # (h, w, feature_channels, num_imgs)
        # features = np.moveaxis(features, -1, 0)  # (num_imgs, h, w, feature_channels)

        assert n_channels in [3, 64, 96]  # either rgb or pca features
        # features_array = features_array[:, None]  # (N, 1, H, W, C)
        # features_array = np.transpose(features_array, [0, 2, 3, 1, 4])  # (N, H, W, 1, C)
        features_array = np.moveaxis(features_array, 1, -1)
        features_array = np.reshape(features_array, [-1, n_channels]).astype(np.float32)  # (N*H*W, C)

        self.all_semfeas = torch.Tensor(features_array)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def get_sem_loss(self, sem_map, sem_train):
        sem_loss_fun = torch.nn.CrossEntropyLoss()
        sem_loss = lambda logit, label: sem_loss_fun(logit, label)

        return sem_loss(sem_map, sem_train.squeeze().long())

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        img = self.all_rgbs[idx]
        rays = self.all_rays[idx]
        sems = self.all_sems[idx]

        sample = {'rays': rays,
                  'rgbs': img,
                  'sems': sems}

        return sample

