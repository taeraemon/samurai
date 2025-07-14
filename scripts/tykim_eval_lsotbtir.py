import os
import os.path as osp
import gc
import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor

# ==================== 파라미터 설정 ====================
# ### LSOTB-TIR 변경 사항 1: 데이터셋 루트 폴더 경로 수정 ###
# 데이터셋 루트 폴더
video_folder = "/home/tykim/Documents/dataset/LSOTB-TIR/"

# ### LSOTB-TIR 변경 사항 2: testing_set.txt 대신 디렉토리에서 직접 비디오 목록 로드 ###
# LSOTB-TIR 데이터셋의 비디오 시퀀스 폴더
video_sequence_folder = osp.join(video_folder, "LSOTB-TIR-EvaluationData")
# 해당 폴더 내의 모든 디렉토리(비디오 시퀀스)를 테스트 비디오로 사용
test_videos = sorted([d for d in os.listdir(video_sequence_folder) if osp.isdir(osp.join(video_sequence_folder, d))])


# 실험 이름과 모델 이름
exp_name = "samurai_TIR" # 결과 저장 폴더명을 구분하기 위해 TIR 추가
model_name = "tiny"
# model_name = "small"
# model_name = "base_plus"
# model_name = "large"

# 결과 저장 폴더
pred_folder = f"results/{exp_name}/{exp_name}_{model_name}"

# 시각화 여부 및 폴더
save_to_video = True
if save_to_video:
    vis_folder = f"visualization/{exp_name}/{model_name}"
    os.makedirs(vis_folder, exist_ok=True)

# 색상 정의 (객체 ID 별 시각화 색)
color = [(255, 0, 0)]

# ==================== 유틸 함수 ====================
### LSOTB-TIR 변경 사항 3: 함수 이름을 보다 범용적으로 변경 (기능은 동일) ###
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

# ==================== 메인 추론 루프 ====================
# 체크포인트 및 설정 파일
checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
model_cfg = (
    "configs/samurai/sam2.1_hiera_b+.yaml"
    if model_name == "base_plus"
    else f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"
)

# ==================== 비디오별 추론 ====================
for vid, video in enumerate(test_videos):

    ### LSOTB-TIR 변경 사항 4: 프레임 폴더 경로 구성 방식 변경 ###
    # LaSOT-Ext의 '카테고리/시퀀스' 구조가 아닌 '시퀀스' 구조에 맞게 경로 수정
    frame_folder = osp.join(video_sequence_folder, video, "img")
    
    if not osp.isdir(frame_folder):
        print(f"Warning: Image folder not found for {video}, skipping.")
        continue
        
    num_frames = len(os.listdir(frame_folder))

    print(f"\033[91mRunning video [{vid+1}/{len(test_videos)}]: {video} with {num_frames} frames\033[0m")

    ### LSOTB-TIR 변경 사항 5: 이미지 파일명 형식에 맞게 첫 프레임 경로 수정 ###
    # LaSOT-Ext: 00000001.jpg, LSOTB-TIR: 0001.jpg
    first_frame_path = osp.join(frame_folder, "0001.jpg")
    if not osp.exists(first_frame_path):
        print(f"Warning: First frame '0001.jpg' not found in {frame_folder}, skipping video.")
        continue
    height, width = cv2.imread(first_frame_path).shape[:2]

    # SAM2 predictor 로드
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

    predictions = []

    # 시각화 비디오 저장 초기화
    if save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = osp.join(vis_folder, f'{video}.mp4')
        out = cv2.VideoWriter(out_path, fourcc, 30, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # 초기 상태 세팅
        # state = predictor.init_state(
        #     frame_folder,
        #     img_name_format="%04d.jpg", # ### LSOTB-TIR 변경 사항 6: 이미지 파일 포맷 지정 ###
        #     offload_video_to_cpu=True,
        #     offload_state_to_cpu=True,
        #     async_loading_frames=True
        # )
        state = predictor.init_state(
            frame_folder,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True
        )

        ### LSOTB-TIR 변경 사항 7: Ground Truth 파일 경로 및 이름 변경 ###
        # LaSOT-Ext: groundtruth.txt, LSOTB-TIR: groundtruth_rect.txt
        gt_path = osp.join(video_sequence_folder, video, "groundtruth_rect.txt")
        if not osp.exists(gt_path):
            print(f"Warning: Ground truth file not found at {gt_path}, skipping video.")
            continue

        prompts = load_gt_from_txt(gt_path)
        if 0 not in prompts:
            print(f"Warning: No ground truth found for the first frame in {gt_path}, skipping video.")
            continue
            
        bbox, track_label = prompts[0]
        _, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        # 프레임별 추론
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

            # 시각화 및 저장
            if save_to_video:
                ### LSOTB-TIR 변경 사항 8: 이미지 파일명 형식 변경 (8자리 -> 4자리) ###
                img_path = osp.join(frame_folder, f"{frame_idx+1:04d}.jpg")
                img = cv2.imread(img_path)
                if img is None:
                    break

                # 마스크 색칠
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.75, 0)

                # 예측 BBox 그리기
                for obj_id, bbox in bbox_to_vis.items():
                    x, y, w, h = bbox
                    cv2.rectangle(img, (x, y), (x + w, y + h), color[obj_id % len(color)], 2)

                # GT BBox (녹색) 그리기
                if frame_idx in prompts:
                    x1, y1, x2, y2 = prompts[frame_idx][0]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                out.write(img)

            # 예측 결과 저장
            # LSOTB-TIR은 보통 객체가 하나이므로 pred[0]을 사용
            if 0 in bbox_to_vis:
                predictions.append(bbox_to_vis[0])
            else:
                predictions.append([0,0,0,0]) # 객체를 못 찾은 경우

    # 예측 결과 텍스트 저장
    os.makedirs(pred_folder, exist_ok=True)
    with open(osp.join(pred_folder, f'{video}.txt'), 'w') as f:
        for pred in predictions:
            x, y, w, h = pred
            f.write(f"{x},{y},{w},{h}\n")

    # 비디오 저장 종료
    if save_to_video:
        out.release()

    # 메모리 정리
    del predictor
    del state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
