from email.mime import base
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import numpy as np
import torch
import pickle as pkl
import os
from glob import glob
from pprint import pprint
import random
# from HeadNeRFOptions import BaseOptions


# Summary: input: four codes -> output: img

# Preprocess: create input list and output list; setup train set and validation set
def input_output_lists(root_dir, seed):
    train_code_list = []
    train_img_list = []
    train_mask_list = []
    train_semantic_list = []
    train_gazeMask_list = []
    test_code_list = []
    test_img_list = []
    test_mask_list = []
    test_semantic_list = []
    test_gazeMask_list = []

    frame_list = [x for x in os.listdir(root_dir) if "frame" in x]
    frame_list.sort()
    frame_list = frame_list[:200]
    random.seed(seed)
    random.shuffle(frame_list)
    train_frames = frame_list[:190]
    test_frames = frame_list[190:]
    for frame in train_frames:
        frame_path = os.path.join(root_dir, frame)
        code_path = sorted([x for x in glob(f"{frame_path}/*.pkl")])[:12]
        mask_path = sorted([x for x in glob(f"{frame_path}/*_mask.png")])[:12]
        semantic_path = sorted(
            [x for x in glob(f"{frame_path}/*_semantic.png")])[:12]
        gazeMask_path = sorted(
            [x for x in glob(f"{frame_path}/*_gazeMask.png")])[:12]
        img_path = sorted([x for x in glob(
            f"{frame_path}/*.png") if x not in (mask_path+semantic_path+gazeMask_path)])[:12]
        train_code_list = train_code_list + code_path
        train_img_list = train_img_list + img_path
        train_mask_list = train_mask_list + mask_path
        train_semantic_list = train_semantic_list + semantic_path
        train_gazeMask_list = train_gazeMask_list + gazeMask_path
    for frame in test_frames:
        frame_path = os.path.join(root_dir, frame)
        code_path = [x for x in glob(f"{frame_path}/*.pkl")][:12]
        mask_path = [x for x in glob(f"{frame_path}/*_mask.png")][:12]
        semantic_path = sorted(
            [x for x in glob(f"{frame_path}/*_semantic.png")])[:12]
        gazeMask_path = sorted(
            [x for x in glob(f"{frame_path}/*_gazeMask.png")])[:12]
        img_path = [x for x in glob(
            f"{frame_path}/*.png") if x not in (mask_path+semantic_path+gazeMask_path)][:12]
        test_code_list = test_code_list + code_path
        test_img_list = test_img_list + img_path
        test_mask_list = test_mask_list + mask_path
        test_semantic_list = test_semantic_list + semantic_path
        train_gazeMask_list = train_gazeMask_list + gazeMask_path

    train_code_list.sort()
    train_img_list.sort()
    train_mask_list.sort()
    train_semantic_list.sort()
    train_gazeMask_list.sort()
    test_code_list.sort()
    test_img_list.sort()
    test_mask_list.sort()
    test_semantic_list.sort()
    test_gazeMask_list.sort()

    train_set = {"code_list": train_code_list,
                 "img_list": train_img_list, "mask_list": train_mask_list, "semantic_list": train_semantic_list, "gazeMask_list": train_gazeMask_list}
    test_set = {"code_list": test_code_list,
                "img_list": test_img_list, "mask_list": test_mask_list, "semantic_list": test_semantic_list, "gazeMask_list": test_gazeMask_list}

    with open(f"{root_dir}/train_set.pkl", "wb") as f:
        pkl.dump(train_set, f)
    with open(f"{root_dir}/test_set.pkl", "wb") as f:
        pkl.dump(test_set, f)

    return train_set, test_set

# Dataset
# transfer img to img_tensor


def data_loader(img_path, mask_path, semantic_path, gazeMask_path, para_3dmm_path, opt):

    # process imgs
    img_size = (opt.pred_img_size, opt.pred_img_size)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0

    gt_img_size = img.shape[0]
    if gt_img_size != opt.pred_img_size:
        img = cv2.resize(img, dsize=img_size, fx=0, fy=0,
                         interpolation=cv2.INTER_LINEAR)

    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    if mask_img.shape[0] != opt.pred_img_size:
        mask_img = cv2.resize(mask_img, dsize=img_size,
                              fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    img[mask_img < 0.5] = 1.0

    semantic_img = cv2.imread(
        semantic_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_BGR2RGB)
    if semantic_img.shape[0] != opt.pred_img_size:
        semantic_img = cv2.resize(semantic_img, dsize=img_size, fx=0, fy=0,
                                  interpolation=cv2.INTER_LINEAR)
    gazeMask_img = cv2.imread(
        gazeMask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)

    img_tensor = (torch.from_numpy(img).permute(2, 0, 1))  # CHW
    mask_tensor = torch.from_numpy(mask_img[None, :, :])
    semantic_tensor = (torch.from_numpy(semantic_img).permute(2, 0, 1))
    eye_mask_tensor = torch.from_numpy(gazeMask_img[None, :, :])

    # load init codes from the results generated by solving 3DMM rendering opt.
    with open(para_3dmm_path, "rb") as f:
        nl3dmm_para_dict = pkl.load(f)
    base_code = nl3dmm_para_dict["code"].detach()
    featmap_size = opt.featmap_size

    base_iden = base_code[:opt.iden_code_dims]
    base_expr = base_code[opt.iden_code_dims:opt.iden_code_dims +
                          opt.expr_code_dims]
    base_text = base_code[opt.iden_code_dims + opt.expr_code_dims:opt.iden_code_dims
                          + opt.expr_code_dims + opt.text_code_dims]
    base_illu = base_code[opt.iden_code_dims +
                          opt.expr_code_dims + opt.text_code_dims:-2]
    base_gaze = base_code[-2:]

    base_code_info = {
        "base_iden": base_iden,
        "base_expr": base_expr,
        "base_text": base_text,
        "base_gaze": base_gaze,
        "base_illu": base_illu
    }

    base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"].detach()
    base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"].detach().unsqueeze(-1)
    base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].detach()
    base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].detach().unsqueeze(-1)

    temp_inmat = nl3dmm_para_dict["inmat"].detach()
    temp_inmat[:2, :] *= (featmap_size / gt_img_size)

    temp_inv_inmat = torch.zeros_like(temp_inmat)
    temp_inv_inmat[0, 0] = 1.0 / temp_inmat[0, 0]
    temp_inv_inmat[1, 1] = 1.0 / temp_inmat[1, 1]
    temp_inv_inmat[0, 2] = -(temp_inmat[0, 2] / temp_inmat[0, 0])
    temp_inv_inmat[1, 2] = -(temp_inmat[1, 2] / temp_inmat[1, 1])
    temp_inv_inmat[2, 2] = 1.0

    cam_info = {
        "batch_Rmats": base_c2w_Rmat,
        "batch_Tvecs": base_c2w_Tvec,
        "batch_inv_inmats": temp_inv_inmat
    }

    return img_tensor, mask_tensor, semantic_tensor, eye_mask_tensor, cam_info, base_code_info


class trainset(Dataset):
    def __init__(self, train_set, opt, loader=data_loader):
        super().__init__()
        self.opt = opt
        # 定义好 image 的路径
        self.code_list = train_set["code_list"]
        self.img_list = train_set["img_list"]
        self.mask_list = train_set["mask_list"]
        self.semantic_list = train_set["semantic_list"]
        self.gazeMask_list = train_set["gazeMask_list"]
        self.loader = loader

    def __getitem__(self, index):
        img = self.img_list[index]
        mask = self.mask_list[index]
        code = self.code_list[index]
        semantic = self.semantic_list[index]
        gazeMask = self.gazeMask_list[index]
        img_tensor, mask_tensor, semantic_tensor, eye_mask_tensor, cam_info, base_code_info = self.loader(
            img, mask, semantic, gazeMask, code, self.opt)
        return img_tensor, mask_tensor, semantic_tensor, eye_mask_tensor, cam_info, base_code_info

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    root_dir = "/home/hengfei/Desktop/research/headnerf/train_data/subject0000"
    train_set, test_set = input_output_lists(root_dir, 1)
    pprint(train_set["code_list"][-20:])
    pprint(train_set["img_list"][-20:])
    pprint(train_set["mask_list"][-20:])
    pprint(train_set["gazeMask_list"][-20:])

    # train_dataset = trainset(train_set, BaseOptions())
    # train_dataloader = DataLoader(train_dataset, batch_size=4, shulffle=True)
    # img_tensor, mask_tensor, cam_info, base_code_info = next(iter(train_dataloader))f
    # print(cam_info["batch_inv_inmats"].shape)
