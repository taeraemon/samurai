# ----------------
# note
# 
# export CUDA_VISIBLE_DEVICES=0,1
# 
# python3 scripts/tykim_eval_lsotbtir_chunk.py --num_chunks 2 --chunk_idx 0 --device cuda:0 --save_to_video
# python3 scripts/tykim_eval_lsotbtir_chunk.py --num_chunks 2 --chunk_idx 1 --device cuda:1 --save_to_video
# ----------------

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

def load_gt_from_txt(gt_path):
    """Ground truth 파일에서 각 프레임별 bounding box를 불러옴 (x,y,w,h -> x1,y1,x2,y2)"""
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    
    prompts = {}
    for fid, line in enumerate(gt):
        # LSOTB-TIR의 groundtruth_rect.txt도 콤마 또는 탭으로 구분된 x,y,w,h 형식
        parts = line.strip().replace('\t', ',').split(',')
        if len(parts) == 4:
            x, y, w, h = map(float, parts)
            x, y, w, h = int(x), int(y), int(w), int(h)
            prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def split_list(full_list, num_chunks):
    """리스트를 지정된 개수의 청크로 나눕니다."""
    # 청크가 비디오 수보다 많을 경우를 대비하여 조정
    if not full_list:
        return []
    if num_chunks > len(full_list):
        num_chunks = len(full_list)
    chunk_size = len(full_list) // num_chunks
    remainder = len(full_list) % num_chunks
    
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(full_list[start:end])
        start = end
    return chunks

def inference_chunk(chunk_videos, args):
    """할당된 비디오 청크에 대해 추론을 수행합니다."""
    # ### 에러 수정: 현재 프로세스가 사용할 GPU 장치를 명시적으로 설정 ###
    # 이는 멀티 GPU 환경에서 발생할 수 있는 'illegal memory access' 오류를 방지합니다.
    torch.cuda.set_device(args.device)

    # 모델 및 체크포인트 경로 설정
    checkpoint = f"sam2/checkpoints/sam2.1_hiera_{args.model_name}.pt"
    model_cfg = (
        "configs/samurai/sam2.1_hiera_b+.yaml"
        if args.model_name == "base_plus"
        else f"configs/samurai/sam2.1_hiera_{args.model_name[0]}.yaml"
    )

    # 결과 및 시각화 폴더 설정
    pred_folder = osp.join(args.result_folder, args.exp_name, f"{args.exp_name}_{args.model_name}")
    if args.save_to_video:
        vis_folder = osp.join(args.vis_folder, args.exp_name, args.model_name)
        os.makedirs(vis_folder, exist_ok=True)
    
    # LSOTB-TIR 데이터셋 시퀀스 폴더 경로
    video_sequence_folder = osp.join(args.video_folder, "LSOTB-TIR-EvaluationData")

    # 할당된 비디오들에 대해 순차적으로 추론 수행
    for vid, video in enumerate(chunk_videos):
        frame_folder = osp.join(video_sequence_folder, video, "img")
        gt_path = osp.join(video_sequence_folder, video, "groundtruth_rect.txt")

        if not osp.isdir(frame_folder):
            print(f"Warning: Image folder not found for {video}, skipping.")
            continue
        
        num_frames = len(os.listdir(frame_folder))

        print(f"\033[91m[GPU {args.device}] Running video [{vid+1}/{len(chunk_videos)}]: {video} with {num_frames} frames\033[0m")

        first_frame_path = osp.join(frame_folder, "0001.jpg")
        if not osp.exists(first_frame_path):
            print(f"Warning: First frame '0001.jpg' not found in {frame_folder}, skipping video.")
            continue
        height, width = cv2.imread(first_frame_path).shape[:2]

        predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=args.device)
        predictions = []

        if args.save_to_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = osp.join(vis_folder, f'{video}.mp4')
            out = cv2.VideoWriter(out_path, fourcc, 30, (width, height))

        with torch.inference_mode(), torch.autocast(args.device.split(':')[0], dtype=torch.float16):
            state = predictor.init_state(
                frame_folder,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True
            )

            if not osp.exists(gt_path):
                print(f"Warning: Ground truth file not found at {gt_path}, skipping video.")
                if args.save_to_video: out.release()
                continue

            prompts = load_gt_from_txt(gt_path)
            if 0 not in prompts:
                print(f"Warning: No ground truth found for the first frame in {gt_path}, skipping video.")
                if args.save_to_video: out.release()
                continue
                
            bbox, track_label = prompts[0]
            _, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                mask_to_vis = {}
                bbox_to_vis = {}

                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy() > 0.0
                    non_zero_indices = np.argwhere(mask)

                    if len(non_zero_indices) == 0:
                        bbox_pred = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox_pred = [x_min, y_min, x_max - x_min, y_max - y_min]

                    bbox_to_vis[obj_id] = bbox_pred
                    mask_to_vis[obj_id] = mask

                if args.save_to_video:
                    img_path = osp.join(frame_folder, f"{frame_idx+1:04d}.jpg")
                    img = cv2.imread(img_path)
                    if img is None: break

                    for obj_id, mask in mask_to_vis.items():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = color[(obj_id + 1) % len(color)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.75, 0)

                    for obj_id, bbox_pred in bbox_to_vis.items():
                        x, y, w, h = bbox_pred
                        cv2.rectangle(img, (x, y), (x + w, y + h), color[obj_id % len(color)], 2)

                    if frame_idx in prompts:
                        x1, y1, x2, y2 = prompts[frame_idx][0]
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    out.write(img)

                if 0 in bbox_to_vis:
                    predictions.append(bbox_to_vis[0])
                else:
                    predictions.append([0,0,0,0])

        os.makedirs(pred_folder, exist_ok=True)
        with open(osp.join(pred_folder, f'{video}.txt'), 'w') as f:
            for pred in predictions:
                x, y, w, h = pred
                f.write(f"{x},{y},{w},{h}\n")

        if args.save_to_video:
            out.release()

        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    # --- LSOTB-TIR에 맞게 기본값 및 도움말 수정 ---
    parser.add_argument("--video_folder", type=str, default="/home/tykim/Documents/dataset/LSOTB-TIR/", help="LSOTB-TIR 데이터셋의 루트 폴더")
    parser.add_argument("--exp_name", type=str, default="samurai_TIR", help="실험 이름")
    parser.add_argument("--model_name", type=str, default="base_plus", help="사용할 모델 이름 (e.g., base_plus, large)")
    parser.add_argument("--result_folder", type=str, default="results", help="결과 .txt 파일 저장 폴더")
    parser.add_argument("--vis_folder", type=str, default="visualization", help="시각화 비디오 저장 폴더")
    parser.add_argument("--save_to_video", action='store_true', help="시각화 비디오 저장 여부")
    # --- 멀티 GPU 실행을 위한 인자 ---
    parser.add_argument("--chunk_idx", type=int, default=0, help="처리할 비디오 청크의 인덱스")
    parser.add_argument("--num_chunks", type=int, default=1, help="전체 비디오를 나눌 청크의 개수")
    parser.add_argument("--device", type=str, default="cuda:0", help="사용할 GPU 장치 (e.g., cuda:0, cuda:1)")
    args = parser.parse_args()

    # LSOTB-TIR 데이터셋 구조에 맞게 비디오 목록 로드
    video_sequence_folder = osp.join(args.video_folder, "LSOTB-TIR-EvaluationData")
    if not osp.isdir(video_sequence_folder):
        raise NotADirectoryError(f"Video sequence folder not found: {video_sequence_folder}")
    
    test_videos = sorted([d for d in os.listdir(video_sequence_folder) if osp.isdir(osp.join(video_sequence_folder, d))])

    # 비디오 목록을 청크로 분할
    chunks = split_list(test_videos, args.num_chunks)
    if args.chunk_idx >= len(chunks):
        print(f"Chunk index {args.chunk_idx} is out of bounds. No videos to process for this chunk.")
        return
        
    chunk_videos = chunks[args.chunk_idx]

    # 할당된 청크에 대해 추론 실행
    inference_chunk(chunk_videos, args)

if __name__ == "__main__":
    main()
