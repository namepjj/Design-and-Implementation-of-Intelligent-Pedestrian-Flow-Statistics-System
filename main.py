"""阶段3：YOLO + ByteTrack + 虚拟计数线的进出人数统计。

流程：
    视频 -> OpenCV 读取 -> YOLO 检测(只保留 person) -> ByteTrack 跟踪(Track ID)
         -> 计数线穿越判断(进入/离开) -> 绘制框/ID/计数线/统计 -> 窗口显示

运行：
    python main.py                     # 自动使用 videos/ 下第一个视频(mp4/avi)
    python main.py path/to/video.avi   # 指定视频
    python main.py --conf 0.4          # 调整置信度阈值
    python main.py --line-ratio 0.6    # 调整计数线位置(占画面高度比例，默认0.5)

按 q 或 ESC 退出，直接关闭窗口也可退出。
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from core.counter import LineCounter

# 项目根目录（main.py 所在目录）
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"

DEFAULT_MODEL = "yolo11n.pt"   # 轻量模型，首次运行自动下载到 models/
PERSON_CLASS_ID = 0            # COCO 数据集中 person 的类别 ID
TRACKER = "bytetrack.yaml"     # Ultralytics 原生 ByteTrack 跟踪配置
WINDOW_NAME = "YOLO People Counting"

LINE_RATIO = 0.5               # 计数线默认位置：画面高度的 50%
LINE_COLOR = (0, 0, 255)       # 计数线颜色(BGR，红色)

# 中文字体候选（cv2.putText 不支持中文，统计面板改用 PIL 渲染）
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]
_FONT_CACHE: dict[int, object] = {}

# 按 Track ID 循环取色，便于肉眼确认同一个人 ID 是否稳定
ID_COLORS = [
    (0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (128, 255, 0), (0, 0, 255), (255, 255, 0),
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack 进出人数统计（阶段3）")
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="待检测的视频路径(mp4/avi)；不传则自动使用 videos/ 目录下第一个视频",
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
    parser.add_argument(
        "--line-ratio",
        type=float,
        default=LINE_RATIO,
        help=f"计数线位置，占画面高度比例 0~1（默认 {LINE_RATIO}）",
    )
    return parser.parse_args()


def resolve_video_path(arg_path: str | None) -> Path | None:
    """确定要处理的视频路径。

    优先使用命令行传入的路径；否则取 videos/ 目录下第一个视频文件(mp4/avi/mov/mkv)。
    找不到时返回 None。
    """
    if arg_path:
        p = Path(arg_path)
        return p if p.is_file() else None

    if VIDEOS_DIR.is_dir():
        videos = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            videos.extend(VIDEOS_DIR.glob(ext))
        if videos:
            return sorted(videos)[0]
    return None


def pick_device() -> tuple[str, object]:
    """检测 CUDA 是否可用，返回 (用于打印的设备名, 传给 YOLO 的 device 参数)。

    CUDA 可用 -> GPU；否则自动回退 CPU。
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return f"cuda (GPU: {gpu_name})", 0
    return "cpu", "cpu"


def _get_font(size: int):
    """按字号加载中文字体（带缓存）；全部失败返回 None。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    _FONT_CACHE[size] = font
    return font


def put_texts_cn(frame, items) -> None:
    """在帧上绘制中文文本（原地修改 frame）。

    items: [(text, (x, y), bgr_color, font_size), ...]
    字体不可用时回退到 cv2.putText（中文可能显示为方块，但不会崩溃）。
    """
    if any(_get_font(size) is None for _, _, _, size in items):
        for text, pos, color, size in items:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 24.0, color, 2)
        return

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(img)
    for text, pos, color, size in items:
        drawer.text(pos, text, font=_get_font(size),
                    fill=(color[2], color[1], color[0]))
    np.copyto(frame, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))


def center_y_of(xyxy) -> float:
    """返回检测框的中心点 y 坐标。"""
    return (float(xyxy[1]) + float(xyxy[3])) / 2.0


def update_counts(result, counter: LineCounter):
    """遍历跟踪框，用中心点 y 更新计数线穿越统计。

    :return: (本帧检测到的人数, 事件列表[(track_id, 'enter'/'exit'), ...])
    """
    events = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0, events

    track_ids = boxes.id
    person_count = 0
    for i, box in enumerate(boxes):
        person_count += 1
        if track_ids is None:
            continue
        tid = int(track_ids[i])
        cy = center_y_of(box.xyxy[0].tolist())
        ev = counter.update(tid, cy)
        if ev:
            events.append((tid, ev))
    return person_count, events


def draw_frame(frame, result, counter: LineCounter,
               device_label: str, person_count: int) -> None:
    """绘制计数线、检测框/Track ID/中心点，以及中文统计面板（原地修改）。"""
    h, w = frame.shape[:2]

    # 1. 虚拟计数线
    cv2.line(frame, (0, counter.line_y), (w, counter.line_y), LINE_COLOR, 2)
    cv2.putText(frame, "COUNT LINE", (max(0, w - 150), counter.line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_COLOR, 2)

    # 2. 检测框 + Track ID + 中心点
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        track_ids = boxes.id
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            if track_ids is not None:
                tid = int(track_ids[i])
                color = ID_COLORS[tid % len(ID_COLORS)]
                label = f"ID: {tid}  {conf:.2f}"
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 3, color, -1)
            else:
                color = (0, 255, 0)
                label = f"person  {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 3. 中文统计面板
    snap = counter.snapshot()
    put_texts_cn(frame, [
        (f"进入：{snap['enter']}", (10, 10), (0, 200, 0), 26),
        (f"离开：{snap['exit']}", (10, 44), (0, 128, 255), 26),
        (f"当前人数：{snap['current']}", (10, 78), (0, 0, 255), 28),
        (f"检测到人：{person_count}    device：{device_label}",
         (10, 116), (255, 255, 255), 18),
    ])


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
        print("[ERROR] 未找到视频。请把视频(mp4/avi)放入 videos/ 目录，或用参数指定：")
        print("        python main.py path/to/video.avi")
        return 1
    print(f"[INFO] 视频文件: {video_path}")

    # 3. 加载模型（不存在时自动下载到 models/）
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / args.model
    print(f"[INFO] 加载模型: {model_path} | 跟踪器: {TRACKER}")
    model = YOLO(str(model_path))

    # 4. 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] 视频 FPS={fps:.1f}, 总帧数={total}")
    print("[INFO] 开始统计，按 q 或 ESC 退出。")

    # 5. 计数线：按画面高度比例确定 y（首帧拿到尺寸后初始化）
    counter: LineCounter | None = None

    # 6. 逐帧：检测 + ByteTrack 跟踪 + 计数线穿越统计（非阻塞显示）
    #    persist=True 是关键：跨帧复用同一个跟踪器，Track ID 才不会每帧重置
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 视频结束。")
            break

        # 首帧初始化计数线与死区（死区随画面高度自适应）
        if counter is None:
            frame_h = frame.shape[0]
            line_y = int(frame_h * args.line_ratio)
            margin = max(6, int(frame_h * 0.015))
            counter = LineCounter(line_y, margin=margin)
            print(f"[INFO] 计数线 line_y={line_y}（画面高 {frame_h} 的 "
                  f"{args.line_ratio:.0%}），死区 margin={margin}")

        # 只跟踪 person 类别
        results = model.track(
            source=frame,
            tracker=TRACKER,
            persist=True,
            classes=[PERSON_CLASS_ID],
            conf=args.conf,
            device=device_arg,
            verbose=False,
        )
        result = results[0]

        # 更新进出统计（中心点穿越计数线）
        person_count, events = update_counts(result, counter)
        for tid, ev in events:
            cn = "进入" if ev == "enter" else "离开"
            print(f"[计数] ID {tid} {cn} | 进入={counter.enter_count} "
                  f"离开={counter.exit_count} 当前={counter.current_count}")

        draw_frame(frame, result, counter, device_label, person_count)

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
