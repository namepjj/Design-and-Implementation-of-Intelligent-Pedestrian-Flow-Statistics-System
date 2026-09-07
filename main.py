"""阶段1：基于 YOLO 的视频人员检测。

流程：
    视频 -> OpenCV 读取 -> YOLO 检测 -> 只保留 person -> 绘制框/置信度/人数 -> 窗口显示

运行：
    python main.py                     # 自动使用 videos/ 下第一个 mp4
    python main.py path/to/video.mp4   # 指定视频
    python main.py --conf 0.4          # 调整置信度阈值

按 q 或 ESC 退出，直接关闭窗口也可退出。
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# 项目根目录（main.py 所在目录）
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"

DEFAULT_MODEL = "yolo11n.pt"   # 轻量模型，首次运行自动下载到 models/
PERSON_CLASS_ID = 0            # COCO 数据集中 person 的类别 ID
WINDOW_NAME = "YOLO Person Detection"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="YOLO 视频人员检测（阶段1）")
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="待检测的 MP4 视频路径；不传则自动使用 videos/ 目录下第一个视频",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO 模型文件名（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值，0~1（默认 0.25）",
    )
    return parser.parse_args()


def resolve_video_path(arg_path: str | None) -> Path | None:
    """确定要处理的视频路径。

    优先使用命令行传入的路径；否则取 videos/ 目录下第一个 mp4 文件。
    找不到时返回 None。
    """
    if arg_path:
        p = Path(arg_path)
        return p if p.is_file() else None

    if VIDEOS_DIR.is_dir():
        videos = sorted(VIDEOS_DIR.glob("*.mp4"))
        if videos:
            return videos[0]
    return None


def pick_device() -> tuple[str, object]:
    """检测 CUDA 是否可用，返回 (用于打印的设备名, 传给 YOLO 的 device 参数)。

    CUDA 可用 -> GPU；否则自动回退 CPU。
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return f"cuda (GPU: {gpu_name})", 0
    return "cpu", "cpu"


def draw_detections(frame, result, person_count: int, device_label: str) -> None:
    """在帧上绘制检测框、置信度、当前人数和设备信息（原地修改 frame）。"""
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            # 绿色检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 置信度标签
            label = f"person {conf:.2f}"
            cv2.putText(
                frame, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

    # 左上角统计信息：人数 + 设备
    cv2.putText(
        frame, f"Persons: {person_count}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
    )
    cv2.putText(
        frame, f"device: {device_label}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )


def window_closed() -> bool:
    """判断显示窗口是否被用户手动关闭。"""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main() -> int:
    args = parse_args()

    # 1. 选择设备并打印
    device_label, device_arg = pick_device()
    print(f"[INFO] 当前 device: {device_label}")

    # 2. 确定视频路径
    video_path = resolve_video_path(args.video)
    if video_path is None:
        print("[ERROR] 未找到视频。请把 MP4 放入 videos/ 目录，或用参数指定：")
        print("        python main.py path/to/video.avi")
        return 1
    print(f"[INFO] 视频文件: {video_path}")

    # 3. 加载模型（不存在时自动下载到 models/）
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / args.model
    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(str(model_path))

    # 4. 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] 视频 FPS={fps:.1f}, 总帧数={total}")
    print("[INFO] 开始检测，按 q 或 ESC 退出。")

    # 5. 逐帧检测（非阻塞显示）
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 视频结束。")
            break

        # 只检测 person 类别
        results = model.predict(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=args.conf,
            device=device_arg,
            verbose=False,
        )
        result = results[0]
        person_count = 0 if result.boxes is None else len(result.boxes)

        draw_detections(frame, result, person_count, device_label)

        cv2.imshow(WINDOW_NAME, frame)

        # 非阻塞按键：q / ESC 退出
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("[INFO] 用户退出。")
            break
        # 窗口被手动关闭时退出
        if window_closed():
            print("[INFO] 窗口已关闭，退出。")
            break

    # 6. 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
