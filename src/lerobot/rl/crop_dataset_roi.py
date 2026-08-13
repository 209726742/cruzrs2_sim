#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# 【文件整体功能说明】
# 本脚本用于对 LeRobot 数据集的图像观测数据进行 手动/配置文件 矩形区域裁剪 + 尺寸缩放
# 核心用途：裁剪机器人视觉数据中无关区域（如背景、墙壁），保留关键目标区域，提升模型训练效率
# 支持功能：
#   1. 交互式窗口手动框选 ROI（感兴趣区域）
#   2. 从 JSON 配置文件加载预设裁剪参数
#   3. 批量处理数据集所有帧，裁剪并缩放图像
#   4. 生成新的裁剪后 LeRobot 数据集，支持推送到 Hugging Face Hub
#   5. 自动保存裁剪参数到新数据集，方便复现
#
# 【使用示例命令】
# 1. 交互式手动裁剪（推荐首次使用）
# python crop_lerobot_dataset.py --repo-id lerobot/pushT --root ./data --task "push T-shaped block"
#
# 2. 从 JSON 裁剪参数文件批量裁剪（无需交互）
# python crop_lerobot_dataset.py --repo-id lerobot/pushT --root ./data --crop-params-path ./crop_params.json --task "push T-shaped block"
#
# 3. 裁剪后自动推送到 Hugging Face Hub
# python crop_lerobot_dataset.py --repo-id lerobot/pushT --root ./data --push-to-hub --task "push T-shaped block"
#
# 4. 指定新数据集名称
# python crop_lerobot_dataset.py --repo-id lerobot/pushT --new-repo-id my_user/pushT_cropped --task "push T-shaped block"
# =============================================================================

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F  # type: ignore  # noqa: N812
from tqdm import tqdm  # type: ignore

from src.lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.lerobot.utils.constants import DONE, REWARD


def select_rect_roi(img):
    """
    【功能】在单张图像上开启交互式窗口，允许用户鼠标拖动框选矩形 ROI（感兴趣区域）
    【操作说明】
        - 点击拖动：绘制矩形框
        - c：确认选择
        - r：重新绘制
        - ESC：取消选择
    【输入参数】
        img: np.ndarray —— 待框选的 OpenCV 图像（RGB/BGR 格式均可）
    【返回值】
        tuple(top, left, height, width) / None —— 成功返回裁剪坐标，取消返回 None
            top: 裁剪区域顶部y坐标
            left: 裁剪区域左侧x坐标
            height: 裁剪高度
            width: 裁剪宽度
    """
    # 创建图像副本，避免修改原图
    clone = img.copy()
    working_img = clone.copy()

    roi = None  # 存储最终 ROI 坐标 (top, left, height, width)
    drawing = False  # 是否正在拖动绘制
    index_x, index_y = -1, -1  # 鼠标按下的起始坐标

    # 鼠标事件回调：处理按下、移动、松开
    def mouse_callback(event, x, y, flags, param):
        nonlocal index_x, index_y, drawing, roi, working_img

        # 鼠标左键按下：开始绘制
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            index_x, index_y = x, y

        # 鼠标拖动：实时显示矩形框
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                top = min(index_y, y)
                left = min(index_x, x)
                bottom = max(index_y, y)
                right = max(index_x, x)
                temp = working_img.copy()
                cv2.rectangle(temp, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.imshow("Select ROI", temp)

        # 鼠标左键松开：完成绘制，固定矩形
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            top = min(index_y, y)
            left = min(index_x, x)
            bottom = max(index_y, y)
            right = max(index_x, x)
            height = bottom - top
            width = right - left
            roi = (top, left, height, width)
            # 在图像上绘制最终选区
            working_img = clone.copy()
            cv2.rectangle(working_img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Select ROI", working_img)

    # 创建窗口并绑定鼠标回调
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)
    cv2.imshow("Select ROI", working_img)

    # 打印操作提示
    print("Instructions for ROI selection:")
    print("  - Click and drag to draw a rectangular ROI.")
    print("  - Press 'c' to confirm the selection.")
    print("  - Press 'r' to reset and draw again.")
    print("  - Press ESC to cancel the selection.")

    # 等待用户按键确认/重置/取消
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and roi is not None:  # 确认
            break
        elif key == ord("r"):  # 重置
            working_img = clone.copy()
            roi = None
            cv2.imshow("Select ROI", working_img)
        elif key == 27:  # ESC 取消
            roi = None
            break

    cv2.destroyWindow("Select ROI")
    return roi


def select_square_roi_for_images(images: dict) -> dict:
    """
    【功能】批量对字典中的多张图像依次打开窗口，让用户选择 ROI
    【输入参数】
        images: dict —— key=图像键名（如camera、wrist_image），value=OpenCV图像数组
    【返回值】
        dict —— key=图像键名，value=对应裁剪坐标 (top, left, height, width)
    """
    selected_rois = {}

    for key, img in images.items():
        if img is None:
            print(f"Image for key '{key}' is None, skipping.")
            continue

        print(f"\nSelect rectangular ROI for image with key: '{key}'")
        roi = select_rect_roi(img)

        if roi is None:
            print(f"No valid ROI selected for '{key}'.")
        else:
            selected_rois[key] = roi
            print(f"ROI for '{key}': {roi}")

    return selected_rois


def get_image_from_lerobot_dataset(dataset: LeRobotDataset):
    """
    【功能】从 LeRobot 数据集中提取第一帧的所有图像数据（用于预览框选）
    【输入参数】
        dataset: LeRobotDataset —— 加载好的 LeRobot 数据集对象
    【返回值】
        dict —— key=图像键名，value=张量格式的图像数据（CHW）
    """
    row = dataset[0]
    image_dict = {}
    for k in row:
        if "image" in k:
            image_dict[k] = deepcopy(row[k])
    return image_dict


def convert_lerobot_dataset_to_cropped_lerobot_dataset(
    original_dataset: LeRobotDataset,
    crop_params_dict: dict[str, tuple[int, int, int, int]],
    new_repo_id: str,
    new_dataset_root: str,
    resize_size: tuple[int, int] = (128, 128),
    push_to_hub: bool = False,
    task: str = "",
) -> LeRobotDataset:
    """
    【功能】核心函数：遍历原始数据集所有帧，对指定图像键执行裁剪+缩放，生成新数据集
    【输入参数】
        original_dataset: LeRobotDataset —— 原始数据集
        crop_params_dict: dict —— 裁剪参数字典，key=图像键，value=(top, left, height, width)
        new_repo_id: str —— 新数据集的 Hugging Face 仓库ID
        new_dataset_root: str/Path —— 新数据集本地保存路径
        resize_size: tuple —— 裁剪后缩放尺寸 (height, width)，默认(128,128)
        push_to_hub: bool —— 是否自动推送到 HF Hub
        task: str —— 任务描述文本（会写入每帧）
    【返回值】
        LeRobotDataset —— 裁剪完成的新数据集对象
    """
    # 1. 创建空的新数据集（继承原始数据集配置）
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=int(original_dataset.fps),
        root=new_dataset_root,
        robot_type=original_dataset.meta.robot_type,
        features=original_dataset.meta.info["features"],
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )

    # 2. 更新元数据：将裁剪后图像的尺寸写入数据集配置
    for key in crop_params_dict:
        if key in new_dataset.meta.info["features"]:
            new_dataset.meta.info["features"][key]["shape"] = [3] + list(resize_size)

    # 3. 遍历所有帧，执行裁剪+缩放
    prev_episode_index = 0
    for frame_idx in tqdm(range(len(original_dataset))):
        frame = original_dataset[frame_idx]

        new_frame = {}
        for key, value in frame.items():
            # 跳过索引类字段
            if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
                continue
            # 对 reward/done 做维度扩展（适配数据集格式）
            if key in (DONE, REWARD):
                value = value.unsqueeze(0)

            # 核心：对需要裁剪的图像执行 crop + resize
            if key in crop_params_dict:
                top, left, height, width = crop_params_dict[key]
                cropped = F.crop(value, top, left, height, width)
                value = F.resize(cropped, resize_size)
                value = value.clamp(0, 1)  # 限制数值在合法范围

            # 补充信息维度对齐
            if key.startswith("complementary_info") and isinstance(value, torch.Tensor) and value.dim() == 0:
                value = value.unsqueeze(0)

            new_frame[key] = value

        # 写入任务描述并添加帧到新数据集
        new_frame["task"] = task
        new_dataset.add_frame(new_frame)

        # 章节切换时保存章节
        if frame["episode_index"].item() != prev_episode_index:
            new_dataset.save_episode()
            prev_episode_index = frame["episode_index"].item()

    # 保存最后一个章节
    new_dataset.save_episode()

    # 推送到 Hub
    if push_to_hub:
        new_dataset.push_to_hub()

    return new_dataset


if __name__ == "__main__":
    # ====================== 命令行参数解析 ======================
    parser = argparse.ArgumentParser(description="Crop rectangular ROIs from a LeRobot dataset.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot",
        help="原始 LeRobot 数据集的 HF 仓库ID（必填）",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="原始数据集本地根路径（不指定则自动使用默认路径）",
    )
    parser.add_argument(
        "--crop-params-path",
        type=str,
        default=None,
        help="裁剪参数 JSON 路径（指定则跳过交互式框选）",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="处理完成后自动推送到 Hugging Face Hub",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="数据集任务描述（推荐填写，用于模型训练）",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        default=None,
        help="新数据集仓库ID（不指定则自动在原ID后加 _cropped_resized）",
    )
    args = parser.parse_args()

    # ====================== 加载原始数据集 ======================
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, video_backend="pyav")

    # ====================== 提取第一帧图像用于预览 ======================
    # 从数据集取图像 → 张量转numpy → 归一化转0-255像素值
    images = get_image_from_lerobot_dataset(dataset)
    images = {k: v.cpu().permute(1, 2, 0).numpy() for k, v in images.items()}
    images = {k: (v * 255).astype("uint8") for k, v in images.items()}

    # ====================== 获取裁剪参数（交互/文件） ======================
    if args.crop_params_path is None:
        # 无配置文件 → 交互式手动框选
        rois = select_square_roi_for_images(images)
    else:
        # 有配置文件 → 直接加载
        with open(args.crop_params_path) as f:
            rois = json.load(f)

    # 打印最终裁剪参数
    print("\nSelected Rectangular Regions of Interest (top, left, height, width):")
    for key, roi in rois.items():
        print(f"{key}: {roi}")

    # ====================== 配置新数据集路径 ======================
    new_repo_id = args.new_repo_id if args.new_repo_id else args.repo_id + "_cropped_resized"

    if args.new_repo_id:
        new_dataset_name = args.new_repo_id.split("/")[-1]
        new_dataset_root = dataset.root.parent.parent / new_dataset_name
    else:
        new_dataset_root = Path(str(dataset.root) + "_cropped_resized")

    # ====================== 执行批量裁剪并生成新数据集 ======================
    cropped_resized_dataset = convert_lerobot_dataset_to_cropped_lerobot_dataset(
        original_dataset=dataset,
        crop_params_dict=rois,
        new_repo_id=new_repo_id,
        new_dataset_root=new_dataset_root,
        resize_size=(128, 128),
        push_to_hub=args.push_to_hub,
        task=args.task,
    )

    # ====================== 保存裁剪参数到新数据集 ======================
    meta_dir = new_dataset_root / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "crop_params.json", "w") as f:
        json.dump(rois, f, indent=4)

    print(f"\n✅ 裁剪完成！新数据集保存在：{new_dataset_root}")
    print(f"✅ 裁剪参数已保存至：{meta_dir}/crop_params.json")
