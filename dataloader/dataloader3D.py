import os
import torch
import numpy as np
import nibabel as nib
import torchvision.transforms.functional as F1
from torch.utils.data import Dataset

import nibabel as nib
import numpy as np
from skimage import filters, feature, morphology
from scipy import ndimage
import os

import torch
import nibabel as nib
from skimage import filters, feature, morphology
import numpy as np
from scipy.ndimage import distance_transform_edt


def extract_boundaries_and_save_nii(input_path, output_path=None, min_slice_mean=0.005):
    """
    提取NIfTI文件中不同区域的分界线并保存为二值NIfTI文件

    参数:
    input_path: 输入NIfTI文件路径
    output_path: 输出NIfTI文件路径(可选)
    min_slice_mean: 切片平均值的最小阈值，低于此值的切片将被跳过
    """
    # 加载NIfTI文件
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine

    # 初始化输出数组
    boundary_volume = np.zeros_like(data)

    # 处理每个切片
    for slice_idx in range(data.shape[2]):
        # 获取当前切片
        slice_data = data[:, :, slice_idx]

        # 跳过平均值小于阈值的切片
        if np.mean(slice_data) < min_slice_mean:
            continue

        # 归一化切片
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)

        if slice_max > slice_min:
            slice_normalized = ((slice_data - slice_min) /
                                (slice_max - slice_min) * 255).astype(np.uint8)

            # 方法1: 使用Sobel边缘检测
            sobel_edges = filters.sobel(slice_normalized)

            # 方法2: 使用Canny边缘检测（通常效果更好）
            # 自动确定Canny阈值
            low_threshold = np.percentile(slice_normalized, 10)
            high_threshold = np.percentile(slice_normalized, 80)
            canny_edges = feature.canny(slice_normalized,
                                        sigma=1.0,  # 控制平滑程度
                                        low_threshold=low_threshold,
                                        high_threshold=high_threshold)

            # 方法3: 使用局部二值化捕捉纹理变化
            # 应用局部自适应阈值（对纹理变化更敏感）
            adaptive_thresh = filters.threshold_local(slice_normalized, block_size=1, offset=10)
            binary_adaptive = slice_normalized > adaptive_thresh

            # 提取二值化结果的边缘
            adaptive_edges = filters.sobel(binary_adaptive.astype(float))

            # 组合多种边缘检测结果
            combined_edges = np.maximum.reduce([
                sobel_edges / np.max(sobel_edges) if np.max(sobel_edges) > 0 else sobel_edges,
                canny_edges.astype(float),
                adaptive_edges / np.max(adaptive_edges) if np.max(adaptive_edges) > 0 else adaptive_edges
            ])

            # 应用阈值将边缘转换为二值图像
            edge_threshold = 0.3  # 可调整此阈值
            binary_edges = (combined_edges > edge_threshold).astype(np.uint8)

            # 细化边缘（可选）
            binary_edges = morphology.skeletonize(binary_edges)

            # 存储结果
            boundary_volume[:, :, slice_idx] = binary_edges

    return boundary_volume


def generate_weighted_tensor(input_tensor, max_weight=1.0, min_weight=0.5):
    # 确保输入是一个0和1的张量
    if not np.all(np.isin(input_tensor, [0, 1])):
        raise ValueError("Input tensor must contain only 0s and 1s.")

    # 计算距离变换，获取每个点到最近1的距离
    distance_map = distance_transform_edt(input_tensor == 0)

    # 根据距离设置权重
    weights = np.zeros_like(distance_map, dtype=float)

    # 根据距离设置权重，距离越近权重越大
    for i in range(distance_map.shape[0]):
        for j in range(distance_map.shape[1]):
            for k in range(distance_map.shape[2]):
                dist = distance_map[i, j, k]
                if dist == 0:
                    weights[i, j, k] = max_weight  # 本身是1的地方权重最大
                elif dist == 1:
                    weights[i, j, k] = 0.9
                elif dist == 2:
                    weights[i, j, k] = 0.8
                elif dist == 3:
                    weights[i, j, k] = 0.7
                elif dist == 4:
                    weights[i, j, k] = 0.6
                else:
                    weights[i, j, k] = min_weight  # 超过一定距离的最小权重

    return weights


def get_real_attention(path):
    save_path = path.replace("MRI2pet", "attention")
    save_path = save_path.replace(".nii", ".nii.gz")
    ba = extract_boundaries_and_save_nii(path)
    ba = torch.from_numpy(ba.astype(np.float32))
    weight_map = generate_weighted_tensor(ba)
    nii_file = nib.load(path)
    nii_file = nii_file.get_fdata()
    maska = (nii_file != 0).astype(float)
    mask = (weight_map == 0).astype(float)
    mask = 0.5 * mask
    weight_map += mask
    weight_map *= maska
    # data = nib.Nifti1Image(weight_map, np.eye(4))
    # nib.save(data, save_path)
    return weight_map


def get_all_paths_var(data_dir, var_dir, mode):
    mode_path = os.path.join(data_dir, mode)
    var_path = os.path.join(var_dir, mode)
    mri_paths = []
    pet_paths = []
    var_conditionpath = []
    for i in os.listdir(os.path.join(mode_path, 'MRI2pet')):
        mri_path = os.path.join(mode_path, 'MRI2pet', i)
        pet_path = os.path.join(mode_path, 'PET', i)
        mri_paths.append(mri_path)
        pet_paths.append(pet_path)
        pth_name = i.split('.')[0] + '.pth'
        var_conditionpath.append(os.path.join(var_path, pth_name))
    return mri_paths, pet_paths, var_conditionpath


def get_all_paths_var_base(data_dir, mode,modality):
    mode_path = os.path.join(data_dir, mode)
    mri_paths = []
    pet_paths = []
    for i in os.listdir(os.path.join(mode_path, modality)):
        mri_path = os.path.join(mode_path, modality, i)
        pet_path = os.path.join(mode_path, 'PET', i)
        mri_paths.append(mri_path)
        pet_paths.append(pet_path)
    return mri_paths, pet_paths


def load_mri_PET_MC_3D(mri_paths, pet_paths, varconditionpath, xiaobo_paths):
    mris = []
    pets = []
    wenlis = []
    xiaobo_pets = []
    varconditions = []
    for i in range(len(mri_paths)):
        mri_path, pet_path, xiaobo_path = mri_paths[i], pet_paths[i], xiaobo_paths[i]
        mri = nib.load(mri_path).get_fdata()
        pet = nib.load(pet_path).get_fdata()
        wenli_path = pet_path.replace("PET", "wenli")
        wenli = nib.load(wenli_path).get_fdata()
        xiaobo_pet = nib.load(xiaobo_path).get_fdata()
        # var_condition = torch.load(varconditionpath[i])
        var_condition = torch.load(varconditionpath[i], map_location=torch.device('cuda:2'))

        mris.append(mri)
        pets.append(pet)
        wenlis.append(wenli)
        xiaobo_pets.append(xiaobo_pet)
        varconditions.append(var_condition)

    return mris, pets, varconditions, xiaobo_pets, wenlis


def load_data(mri_paths, pet_paths, varconditionpath):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    mri_path, pet_path = mri_paths, pet_paths
    mri_wenli_path = pet_path.replace("PET", "wenli")
    mri = nib.load(mri_path).get_fdata()
    pet = nib.load(pet_path).get_fdata()
    wenli = nib.load(mri_wenli_path).get_fdata()

    var_condition = torch.load(varconditionpath, map_location=device)
    mri = torch.tensor(mri, dtype=torch.float32).to(device)
    pet = torch.tensor(pet, dtype=torch.float32).to(device)
    wenli = torch.tensor(wenli, dtype=torch.float32).to(device)

    return mri, pet, var_condition, wenli

def load_data_base(mri_paths, pet_paths):
    mri_path, pet_path = mri_paths, pet_paths
    mri = nib.load(mri_path).get_fdata()
    pet = nib.load(pet_path).get_fdata()
    mri = torch.tensor(mri, dtype=torch.float32)
    pet = torch.tensor(pet, dtype=torch.float32)
    return mri, pet



class myDataset3Dbefore(Dataset):
    def __init__(self, data_dir, var_dir, mode):
        mri_path, pet_path, var_conditionpath, xiaobo_paths = get_all_paths_var(data_dir, var_dir, mode)
        self.mris, self.pets, self.varconditions, self.xiaobo_pets, self.wenlis = load_mri_PET_MC_3D(mri_path, pet_path,
                                                                                                     var_conditionpath,
                                                                                                     xiaobo_paths)

    def __len__(self):
        return len(self.mris)

    def __getitem__(self, idx):
        mri = self.mris[idx].astype(np.float32)
        pet = self.pets[idx].astype(np.float32)
        wenli = self.wenlis[idx].astype(np.float32)
        varcondition = self.varconditions[idx]

        # 归一化函数
        def normalize(data):
            data_min = np.min(data)
            data_max = np.max(data)
            # 避免除零错误
            denominator = data_max - data_min + 1e-8
            return (data - data_min) / denominator

        mri = mri.transpose((2, 0, 1))
        pet = pet.transpose((2, 0, 1))
        wenli = wenli.transpose((2, 0, 1))

        mri = F1.resize(torch.from_numpy(mri), size=(256, 256))
        pet = F1.resize(torch.from_numpy(pet), size=(256, 256))

        mri = normalize(mri.numpy())
        pet = normalize(pet.numpy())

        mri_tensor = torch.from_numpy(mri).unsqueeze(0)
        pet_tensor = torch.from_numpy(pet).unsqueeze(0)
        wenli_pet = torch.from_numpy(wenli).unsqueeze(0)

        return mri_tensor, pet_tensor, varcondition, wenli_pet


def get_all_MRIwenliattention_paths(data_dir, var_dir, mode):
    mri_paths, pet_paths, _ = get_all_paths_var(data_dir, var_dir, mode)
    for i in mri_paths:
        wenli_path = i.replace("MRI2pet", "wenli")
        data = extract_boundaries_and_save_nii(i)
        data = generate_weighted_tensor(data)
        nib.save(nib.Nifti1Image(data, np.eye(4)), wenli_path)


class myDataset3D(Dataset):
    def __init__(self, data_dir, var_dir, mode):
        mri_path, pet_path, var_conditionpath = get_all_paths_var(data_dir, var_dir, mode)
        # if mode == 'testy':
        #     get_all_MRIwenliattention_paths(data_dir, var_dir, mode)
        self.mri_path = mri_path
        self.pet_path = pet_path
        self.var_conditionpath = var_conditionpath

    def __len__(self):
        return len(self.mri_path)

    def __getitem__(self, idx):
        mri_path = self.mri_path[idx]
        name = os.path.basename(mri_path)
        name = name.split('.')[0] + '.nii.gz'
        pet_path = self.pet_path[idx]
        var_conditionpath = self.var_conditionpath[idx]
        mri, pet, varcondition, mri_wenli = load_data(mri_path, pet_path, var_conditionpath)

        wenli = extract_boundaries_and_save_nii(pet_path)
        wenli = generate_weighted_tensor(wenli)
        wenli = wenli.astype(np.float32)
        wenli = torch.from_numpy(wenli).to(torch.float32).to(varcondition.device)

        # 归一化函数
        def normalize(data):
            data_min = data.min()
            data_max = data.max()
            # 避免除零错误
            denominator = data_max - data_min + 1e-8
            return (data - data_min) / denominator

        mri = mri.permute((2, 0, 1))
        pet = pet.permute((2, 0, 1))
        wenli = wenli.permute((2, 0, 1))
        mri_wenli = mri_wenli.permute((2, 0, 1))

        mri = normalize(mri)
        pet = normalize(pet)

        mri_tensor = mri.unsqueeze(0)
        pet_tensor = pet.unsqueeze(0)
        wenli_pet = wenli.unsqueeze(0)
        mri_wenli = mri_wenli.unsqueeze(0)

        return mri_tensor, pet_tensor, varcondition, wenli_pet, mri_wenli, name



class myDataset3Dbase(Dataset):
    def __init__(self, data_dir, mode,modality):
        mri_path, pet_path = get_all_paths_var_base(data_dir, mode,modality)
        self.mri_path = mri_path
        self.pet_path = pet_path

    def __len__(self):
        return len(self.mri_path)

    def __getitem__(self, idx):
        mri_path = self.mri_path[idx]
        name = os.path.basename(mri_path)
        name = name.split('.')[0] + '.nii.gz'
        pet_path = self.pet_path[idx]

        mri, pet= load_data_base(mri_path, pet_path)

        def normalize(data):
            data_min = data.min()
            data_max = data.max()
            denominator = data_max - data_min + 1e-8
            return (data - data_min) / denominator

        mri = mri.permute((2, 0, 1))
        pet = pet.permute((2, 0, 1))

        mri = normalize(mri)
        pet = normalize(pet)

        mri_tensor = mri.unsqueeze(0)
        pet_tensor = pet.unsqueeze(0)

        return mri_tensor, pet_tensor, name


