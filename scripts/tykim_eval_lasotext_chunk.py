import argparse
import os
import os.path as osp
import gc
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor

# 색상 정의
color = [(255, 0, 0)]

def load_lasot_gt(gt_path):
    """Ground truth 파일에서 각 프레임별 bounding box를 불러옴"""
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(int, line.split(','))
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def split_list(full_list, num_chunks):
    chunk_size = len(full_list) // num_chunks
    return [full_list[i:i+chunk_size] for i in range(0, len(full_list), chunk_size)]

def inference_chunk(chunk_videos, args):
    checkpoint = f"sam2/checkpoints/sam2.1_hiera_{args.model_name}.pt"
    model_cfg = (
        "configs/samurai/sam2.1_hiera_b+.yaml"
        if args.model_name == "base_plus"
        else f"configs/samurai/sam2.1_hiera_{args.model_name[0]}.yaml"
    )

    pred_folder = osp.join(args.result_folder, args.exp_name, f"{args.exp_name}_{args.model_name}")
    if args.save_to_video:
        vis_folder = osp.join(args.vis_folder, args.exp_name, args.model_name)
        os.makedirs(vis_folder, exist_ok=True)

    for vid, video in enumerate(chunk_videos):
        cat_name = video.split('-')[0]
        frame_folder = osp.join(args.video_folder, cat_name, video, "img")
        gt_path = osp.join(args.video_folder, cat_name, video, "groundtruth.txt")
        num_frames = len(os.listdir(frame_folder))

        print(f"\033[91m[GPU {args.device}] Running video [{vid+1}/{len(chunk_videos)}]: {video} with {num_frames} frames\033[0m")

        height, width = cv2.imread(osp.join(frame_folder, "00000001.jpg")).shape[:2]
        predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=args.device)
        predictions = []

        if args.save_to_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = osp.join(vis_folder, f'{video}.mp4')
            out = cv2.VideoWriter(out_path, fourcc, 30, (width, height))

        with torch.inference_mode(), torch.autocast(args.device, dtype=torch.float16):
            state = predictor.init_state(
                frame_folder,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True
            )

            prompts = load_lasot_gt(gt_path)
            bbox, track_label = prompts[0]
            _, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                mask_to_vis = {}
                bbox_to_vis = {}

                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy() > 0.0
                    non_zero_indices = np.argwhere(mask)

                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                    bbox_to_vis[obj_id] = bbox
                    mask_to_vis[obj_id] = mask

                if args.save_to_video:
                    img_path = osp.join(frame_folder, f"{frame_idx+1:08d}.jpg")
                    img = cv2.imread(img_path)
                    if img is None:
                        break

                    for obj_id, mask in mask_to_vis.items():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = color[(obj_id + 1) % len(color)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.75, 0)

                    for obj_id, bbox in bbox_to_vis.items():
                        x, y, w, h = bbox
                        cv2.rectangle(img, (x, y), (x + w, y + h), color[obj_id % len(color)], 2)

                    if frame_idx in prompts:
                        x1, y1, x2, y2 = prompts[frame_idx][0]
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    out.write(img)

                predictions.append(bbox_to_vis)

        os.makedirs(pred_folder, exist_ok=True)
        with open(osp.join(pred_folder, f'{video}.txt'), 'w') as f:
            for pred in predictions:
                x, y, w, h = pred[0]
                f.write(f"{x},{y},{w},{h}\n")

        if args.save_to_video:
            out.release()

        del predictor
        del state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing_set", type=str, default="data/testing_set.txt")
    parser.add_argument("--video_folder", type=str, default="/home/tykim/Documents/dataset/LaSOT-Ext/LaSOT-Extension")
    parser.add_argument("--exp_name", type=str, default="samurai")
    parser.add_argument("--model_name", type=str, default="large")
    parser.add_argument("--result_folder", type=str, default="results")
    parser.add_argument("--vis_folder", type=str, default="visualization")
    parser.add_argument("--save_to_video", action='store_true')
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    with open(args.testing_set, 'r') as f:
        test_videos = sorted([line.strip() for line in f.readlines()])

    chunks = split_list(test_videos, args.num_chunks)
    chunk_videos = chunks[args.chunk_idx]

    inference_chunk(chunk_videos, args)

if __name__ == "__main__":
    main()
