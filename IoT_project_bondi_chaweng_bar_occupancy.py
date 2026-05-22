#!/usr/bin/env python3
# Generated from bondi_chaweng_bar_activity_retention_assignment.ipynb

"""
Bondi Chaweng Bar Activity Analytics from a Public Livestream
Postgraduate Program in IoT & Big Data - Data Extraction from Livestreams

What this notebook does:
1. Opens a public livestream
2. Processes frames with OpenCV and YOLO object detection.
3. Extracts useful information:
   - public/bar activity index
   - number of people
   - number of vehicles
   - number of bags/luggage-like objects
   - estimated people entering/stopping at the bar only after the lower-body reference point crosses a manually drawn entry line
   - estimated people passing by the bar by counting stable unique tracked people visible in the video
   - estimated people sitting in the bar using a manually drawn 4-point bar polygon/rectangle
   - bar retention/conversion rate: bar entries divided by total unique people detected by the camera
   - estimated bar revenue per day using 7 EUR per person entering the bar
4. Generates structured events:
   - HIGH_ACTIVITY
   - CROWD_DETECTED
   - BAR_ENTRY_DETECTED
   - BAR_PASSERBY_DETECTED
   - BAR_OCCUPANCY_HIGH
   - HIGH_BAR_RETENTION
   - LOW_BAR_RETENTION
   - VEHICLE_DENSITY
5. Stores data in CSV and SQLite.
6. Generates simple analysis charts and a text summary report.

Typical usage in the notebook:
1) Run the install/import cells.
2) Run calibration once to create rois_bondi_bar_visible_passersby.json. Only the 4-point bar area and bar entry line are calibrated.
3) Run live extraction.
4) Run analysis.

Notes:
- Webpage livestreams may need yt-dlp to resolve the real video stream URL.
- If the webpage does not open, pass a direct .m3u8 livestream URL, a YouTube live URL,
  a local video path, or 0 for your webcam.
- This is an educational aggregate analytics application. It does not do face recognition,
  identity recognition, or store personal images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import ssl
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
BAG_CLASSES = {"backpack", "handbag", "suitcase"}

DEFAULT_SOURCE = "https://webcams24.live/webcam/bondi-chaweng-samui-webcam"
DEFAULT_ROI_FILE = "rois_bondi_bar_visible_passersby.json"
DEFAULT_OUTPUT_DIR = "outputs_bondi_bar_fullbox_hourly"
DEFAULT_MODEL = "yolov8s.pt"

def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def box_centroid(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_bottom_center(box: Box) -> Point:
    """Point near the feet/lower body. Better than the centroid for testing whether
    a person belongs to a ground/terrace ROI in an oblique camera view.
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_aspect_ratio(box: Box) -> float:
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return w / h


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if not polygon or len(polygon) < 3:
        return False
    contour = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def normalize_points(points: Sequence[Point], width: int, height: int) -> List[List[float]]:
    return [[float(x) / float(width), float(y) / float(height)] for x, y in points]


def denormalize_points(points: Sequence[Sequence[float]], width: int, height: int) -> List[Point]:
    return [(float(x) * float(width), float(y) * float(height)) for x, y in points]


def line_side(point: Point, a: Point, b: Point) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def crossed_line(prev: Point, curr: Point, line: Sequence[Point]) -> bool:
    if len(line) != 2:
        return False
    a, b = line
    s1 = line_side(prev, a, b)
    s2 = line_side(curr, a, b)
    if s1 == 0 or s2 == 0:
        return False
    return (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0)


def line_signed_distance(point: Point, a: Point, b: Point) -> float:
    """Signed pixel distance of a point from an oriented line.

    Positive and negative values indicate the two sides of the line. Dividing
    by line length makes the value interpretable in pixels, which lets us use
    a small margin to avoid jitter at the counting line.
    """
    length = max(1e-6, distance(a, b))
    return line_side(point, a, b) / length


def box_corners(box: Box) -> List[Point]:
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def full_box_side_of_line(box: Box, line: Sequence[Point], margin_pixels: float = 0.0) -> int:
    """Return the side of a line only when the full bounding box is across it.

    Returns:
        1  -> every corner of the box is on the positive side of the line
       -1  -> every corner of the box is on the negative side of the line
        0  -> the box intersects/touches the line, so the person is not fully across

    This fixes the bias where a person was counted as entering when only the
    head/centroid crossed the line.
    """
    if len(line) != 2:
        return 0
    a, b = line
    dists = [line_signed_distance(pt, a, b) for pt in box_corners(box)]
    margin = max(0.0, float(margin_pixels))
    if min(dists) > margin:
        return 1
    if max(dists) < -margin:
        return -1
    return 0


def polygon_centroid(points: Sequence[Point]) -> Point:
    if not points:
        return (0.0, 0.0)
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def target_side_from_bar_polygon(line: Sequence[Point], bar_polygon: Sequence[Point]) -> int:
    """Infer which side of the entry line is the bar side using the bar polygon."""
    if len(line) != 2 or not bar_polygon:
        return 0
    a, b = line
    d = line_signed_distance(polygon_centroid(bar_polygon), a, b)
    if d > 0:
        return 1
    if d < 0:
        return -1
    return 0


def update_full_box_entry_crossing(track: "Track", line: Sequence[Point], target_side: int, margin_pixels: float) -> bool:
    """Deprecated strict full-box crossing helper kept for reference.

    The active code now uses update_entry_point_crossing(), because strict
    full-box crossing often misses real entries in webcam footage with
    occlusion and perspective distortion.
    """
    current_side = full_box_side_of_line(track.box, line, margin_pixels)
    if current_side == 0:
        # The box is still intersecting the line: do not count yet and do not
        # overwrite the last full-side state.
        return False

    previous_side = track.bar_entry_reference_side
    crossed = previous_side != 0 and previous_side != current_side
    entered_target_side = target_side == 0 or current_side == target_side
    track.bar_entry_reference_side = current_side
    return bool(crossed and entered_target_side)




def entry_reference_point(box: Box, y_ratio: float = 0.90) -> Point:
    """Return a lower-body point used for reliable entry counting.

    y_ratio=0.0 means the top of the person box, 0.5 means centroid height,
    and 1.0 means bottom/feet. A value around 0.85-0.95 avoids counting a
    person when only the head crosses the line, but it is less strict than
    demanding the full bounding box to clear the line.
    """
    x1, y1, x2, y2 = box
    ratio = clamp(float(y_ratio), 0.0, 1.0)
    return ((x1 + x2) / 2.0, y1 + ratio * (y2 - y1))


def point_side_of_line(point: Point, line: Sequence[Point], margin_pixels: float = 0.0) -> int:
    """Return the side of a line for one point, with a small no-count margin."""
    if len(line) != 2:
        return 0
    a, b = line
    d = line_signed_distance(point, a, b)
    margin = max(0.0, float(margin_pixels))
    if d > margin:
        return 1
    if d < -margin:
        return -1
    return 0


def update_entry_point_crossing(
    track: "Track",
    line: Sequence[Point],
    target_side: int,
    margin_pixels: float,
    y_ratio: float,
) -> bool:
    """Return True when the lower-body point crosses into the bar side.

    This fixes two practical issues:
    1. A head/centroid crossing is too early and can overcount.
    2. Requiring the complete bounding box to clear the line can be too strict
       and often misses real entries in low-resolution or occluded webcam video.
    """
    ref_point = entry_reference_point(track.box, y_ratio)
    current_side = point_side_of_line(ref_point, line, margin_pixels)
    if current_side == 0:
        return False

    previous_side = track.bar_entry_reference_side
    crossed = previous_side != 0 and previous_side != current_side
    entered_target_side = target_side == 0 or current_side == target_side
    track.bar_entry_reference_side = current_side
    return bool(crossed and entered_target_side)

def resize_if_needed(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_label(frame: np.ndarray, text: str, org: Tuple[int, int]) -> None:
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - th - baseline - 6), (x + tw + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_polygon(frame: np.ndarray, points: Sequence[Point], label: str, color: Tuple[int, int, int]) -> None:
    if not points:
        return
    pts = np.array(points, dtype=np.int32)
    if len(points) >= 3:
        cv2.polylines(frame, [pts], True, color, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    for p in points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 4, color, -1)
    if label:
        draw_label(frame, label, (int(points[0][0]), int(points[0][1]) - 8))


def draw_line(frame: np.ndarray, points: Sequence[Point], label: str, color: Tuple[int, int, int]) -> None:
    if len(points) != 2:
        return
    p1 = (int(points[0][0]), int(points[0][1]))
    p2 = (int(points[1][0]), int(points[1][1]))
    cv2.line(frame, p1, p2, color, 3)
    cv2.circle(frame, p1, 5, color, -1)
    cv2.circle(frame, p2, 5, color, -1)
    if label:
        draw_label(frame, label, (p1[0], p1[1] - 8))

def resolve_stream_source(source: str) -> Union[str, int]:
    """Return a value usable by cv2.VideoCapture.

    OpenCV can directly open webcam indexes.
    """
    source = str(source).strip()
    if source.isdigit():
        return int(source)

    lower = source.lower()
    direct_indicators = (
        ".m3u8",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        "rtsp://",
        "rtmp://",
    )
    if os.path.exists(source):
        return source
    if lower.startswith(("rtsp://", "rtmp://")) or any(ind in lower for ind in direct_indicators):
        return source

    def _set_certifi_environment() -> None:
        try:
            import certifi  # type: ignore
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except Exception:
            pass

    def _extract_with_ytdlp(ignore_ssl_errors: bool = False) -> Optional[str]:
        try:
            import yt_dlp  # type: ignore
        except Exception as exc:
            print(f"[warning] yt-dlp is not installed or cannot be imported: {exc}")
            return None

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[ext=mp4]/best[protocol*=m3u8]/best",
            "noplaylist": True,
        }
        if ignore_ssl_errors:
            # Same effect as yt-dlp command-line option --no-check-certificates.
            # Use only as a fallback for a trusted public livestream in this assignment.
            ydl_opts["nocheckcertificate"] = True

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=False)
            if not info:
                return None
            if "url" in info and info["url"]:
                return str(info["url"])

            formats = info.get("formats") or []
            playable = []
            for fmt in formats:
                url = fmt.get("url")
                vcodec = fmt.get("vcodec")
                protocol = str(fmt.get("protocol") or "")
                if url and vcodec and vcodec != "none" and not protocol.startswith("mhtml"):
                    playable.append(fmt)
            if playable:
                playable.sort(key=lambda f: int(f.get("height") or 0), reverse=True)
                return str(playable[0]["url"])
            return None

    def _extract_direct_video_url_from_webpage(ignore_ssl_errors: bool = False) -> Optional[str]:
        if not lower.startswith(("http://", "https://")):
            return None

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        request = urllib.request.Request(source, headers=headers)
        try:
            if ignore_ssl_errors:
                context = ssl._create_unverified_context()
            else:
                try:
                    import certifi  # type: ignore
                    context = ssl.create_default_context(cafile=certifi.where())
                except Exception:
                    context = ssl.create_default_context()
            with urllib.request.urlopen(request, timeout=20, context=context) as response:
                html = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"[warning] Could not fetch webpage to search for video URL: {exc}")
            return None

        # Decode escaped URLs sometimes found inside JavaScript strings.
        html = html.replace("\\/", "/")
        html = bytes(html, "utf-8").decode("unicode_escape", errors="ignore")

        # Look for direct HLS/video URLs in page source.
        candidates = re.findall(r"https?://[^'\"\\s<>]+?(?:\\.m3u8|\\.mp4)(?:[^'\"\\s<>]*)?", html, flags=re.IGNORECASE)
        # Prefer HLS streams when present.
        candidates = sorted(set(candidates), key=lambda u: (".m3u8" not in u.lower(), len(u)))
        if candidates:
            print(f"[info] Found direct video URL in webpage: {candidates[0][:120]}...")
            return candidates[0]
        return None

    if lower.startswith("http://") or lower.startswith("https://"):
        _set_certifi_environment()

        try:
            resolved = _extract_with_ytdlp(ignore_ssl_errors=False)
            if resolved:
                return resolved
        except Exception as exc:
            print(f"[warning] yt-dlp extraction failed: {exc}")

        resolved = _extract_direct_video_url_from_webpage(ignore_ssl_errors=False)
        if resolved:
            return resolved

        print("[warning] Retrying stream resolution with certificate checking disabled for this public-stream demo.")
        try:
            resolved = _extract_with_ytdlp(ignore_ssl_errors=True)
            if resolved:
                return resolved
        except Exception as exc:
            print(f"[warning] yt-dlp fallback also failed: {exc}")

        resolved = _extract_direct_video_url_from_webpage(ignore_ssl_errors=True)
        if resolved:
            return resolved

        print("[warning] Trying the original URL directly with OpenCV.")
        return source

    return source


def open_capture(source: str) -> cv2.VideoCapture:
    resolved = resolve_stream_source(source)
    print(f"[info] Opening video source with OpenCV: {resolved}")
    cap = cv2.VideoCapture(resolved)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video source. Use a direct .m3u8 URL, a YouTube live URL, "
            "a local video path, or 0 for your webcam. For generic webcam webpages, "
            "the page may not expose a stable direct stream URL to OpenCV."
        )
    return cap


def read_first_frame(source: str, target_width: int) -> np.ndarray:
    cap = open_capture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read the first frame from the stream.")
    return resize_if_needed(frame, target_width)

def select_four_point_polygon(window_name: str, frame: np.ndarray, instruction: str, label: str = "Bar sitting area") -> List[Point]:
    """
    Select the bar sitting rectangle by clicking four corners.

    This is intentionally not a two-corner axis-aligned rectangle.
    The user clicks the four visible corners of the bar/terrace area,
    like in the original Grand-Place version. This works better with
    perspective because the seating area may look like a trapezoid.

    Important: click the four points in clockwise or counter-clockwise order.
    """
    points: List[Point] = []
    mouse_pos: Optional[Point] = None
    clone = frame.copy()

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal mouse_pos
        mouse_pos = (float(x), float(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((float(x), float(y)))
            else:
                # After four points, a new left click starts the selection again.
                points[:] = [(float(x), float(y))]
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = clone.copy()
        draw_label(display, instruction, (20, 35))
        draw_label(display, "Click 4 corners in order | Right click: undo | ENTER: save | q: quit", (20, 65))

        for idx, p in enumerate(points, start=1):
            cv2.circle(display, (int(p[0]), int(p[1])), 5, (0, 180, 255), -1)
            draw_label(display, str(idx), (int(p[0]) + 7, int(p[1]) - 7))

        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(
                    display,
                    (int(points[i][0]), int(points[i][1])),
                    (int(points[i + 1][0]), int(points[i + 1][1])),
                    (0, 180, 255),
                    2,
                )

        if 0 < len(points) < 4 and mouse_pos is not None:
            cv2.line(
                display,
                (int(points[-1][0]), int(points[-1][1])),
                (int(mouse_pos[0]), int(mouse_pos[1])),
                (0, 180, 255),
                1,
            )

        if len(points) == 4:
            cv2.line(
                display,
                (int(points[3][0]), int(points[3][1])),
                (int(points[0][0]), int(points[0][1])),
                (0, 180, 255),
                2,
            )
            draw_polygon(display, points, label, (0, 180, 255))

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if len(points) == 4:
                area = abs(cv2.contourArea(np.array(points, dtype=np.float32)))
                if area > 20:
                    break
                print("The selected bar rectangle is too small. Right click to adjust points.")
            else:
                print("Select exactly 4 points for the bar sitting rectangle.")
        elif key == ord("q") or key == 27:
            raise SystemExit("Calibration cancelled.")
    cv2.destroyWindow(window_name)
    return points

def select_line(window_name: str, frame: np.ndarray, instruction: str, label: str = "Counting line", color: Tuple[int, int, int] = (255, 0, 255)) -> List[Point]:
    points: List[Point] = []
    clone = frame.copy()

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((float(x), float(y)))
            else:
                points[:] = [(float(x), float(y))]
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = clone.copy()
        draw_label(display, instruction, (20, 35))
        draw_label(display, "Click 2 points for the line | Right click: undo | ENTER: save | q: quit", (20, 65))
        for p in points:
            cv2.circle(display, (int(p[0]), int(p[1])), 5, color, -1)
        if len(points) == 2:
            cv2.line(display, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), color, 2)
            draw_label(display, label, (int(points[0][0]), int(points[0][1]) - 8))
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if len(points) == 2:
                break
            print("Select exactly 2 points for a line.")
        elif key == ord("q") or key == 27:
            raise SystemExit("Calibration cancelled.")
    cv2.destroyWindow(window_name)
    return points


def calibrate_rois(source: str, roi_file: Union[str, Path], target_width: int) -> None:
    frame = read_first_frame(source, target_width)
    h, w = frame.shape[:2]

    print("Calibration step 1/2: select the bar sitting rectangle with 4 corner clicks.")
    bar_seating_rect = select_four_point_polygon(
        "Calibration - Bar Sitting Area",
        frame,
        "Click the 4 visible corners of the bar/terrace sitting rectangle in clockwise or counter-clockwise order.",
        label="Bar sitting area",
    )

    print("Calibration step 2/2: select the bar entry counting line.")
    bar_entry_line = select_line(
        "Calibration - Bar Entry Line",
        frame,
        "Draw the line customers cross when getting into/stopping at the bar.",
        label="Bar entry line",
        color=(0, 180, 255),
    )

    data = {
        "frame_width": w,
        "frame_height": h,
        "bar_seating_rect_norm": normalize_points(bar_seating_rect, w, h),
        "bar_entry_line_norm": normalize_points(bar_entry_line, w, h),
        "passerby_counting_method": "total_unique_detected_people_from_camera",
    }
    with open(roi_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved ROI calibration to {roi_file}")
    print("Passers-by are counted automatically as the total unique detected people in the camera view; no pass-by line is required.")


def load_rois(roi_file: Union[str, Path], width: int, height: int) -> Dict[str, List[Point]]:
    path = Path(roi_file)
    if not path.exists():
        raise FileNotFoundError(
            f"ROI file {roi_file} not found. Run calibration first. This version needs only a 4-point bar sitting rectangle and a bar entry line."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bar_rect_norm = data.get("bar_seating_rect_norm", data.get("bar_polygon_norm", []))
    bar_entry_line_norm = data.get("bar_entry_line_norm", [])

    if not bar_rect_norm or len(bar_rect_norm) != 4:
        raise ValueError("Your ROI file must contain exactly 4 points in bar_seating_rect_norm. Run calibration again.")
    if not bar_entry_line_norm or len(bar_entry_line_norm) != 2:
        raise ValueError("Your ROI file does not contain a valid bar_entry_line_norm. Run calibration again.")

    return {
        "bar_seating_rect": denormalize_points(bar_rect_norm, width, height),
        "bar_entry_line": denormalize_points(bar_entry_line_norm, width, height),
    }

@dataclass
class Detection:
    class_name: str
    confidence: float
    box: Box

    @property
    def centroid(self) -> Point:
        return box_centroid(self.box)


@dataclass
class Track:
    track_id: int
    class_name: str
    box: Box
    centroid: Point
    first_seen: float
    last_seen: float
    missed: int = 0
    hits: int = 1
    counted_bar_entry: bool = False
    counted_bar_passerby: bool = False
    counted_bar_seated: bool = False
    # Last side where the lower-body reference point was seen relative to the bar entry line.
    # 0 means unknown or too close to the line.
    bar_entry_reference_side: int = 0
    history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=120))

    def update(self, detection: Detection, now: float) -> None:
        self.class_name = detection.class_name
        self.box = detection.box
        self.centroid = detection.centroid
        self.last_seen = now
        self.missed = 0
        self.hits += 1
        self.history.append((self.centroid[0], self.centroid[1], now))

    def mark_missed(self) -> None:
        self.missed += 1


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_missed: int = 10) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: Sequence[Detection], now: float) -> Dict[int, Track]:
        if not self.tracks:
            for det in detections:
                self._create_track(det, now)
            return self.active_tracks()

        track_ids = list(self.tracks.keys())
        det_indices = list(range(len(detections)))
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(det_indices)

        if detections:
            pairs: List[Tuple[float, int, int]] = []
            for tid in track_ids:
                for di, det in enumerate(detections):
                    dist = distance(self.tracks[tid].centroid, det.centroid)
                    pairs.append((dist, tid, di))
            pairs.sort(key=lambda x: x[0])

            for dist, tid, di in pairs:
                if dist > self.max_distance:
                    continue
                if tid not in unmatched_tracks or di not in unmatched_detections:
                    continue
                self.tracks[tid].update(detections[di], now)
                unmatched_tracks.remove(tid)
                unmatched_detections.remove(di)

        for tid in list(unmatched_tracks):
            self.tracks[tid].mark_missed()
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]

        for di in sorted(unmatched_detections):
            self._create_track(detections[di], now)

        return self.active_tracks()

    def _create_track(self, detection: Detection, now: float) -> None:
        tid = self.next_id
        self.next_id += 1
        tr = Track(
            track_id=tid,
            class_name=detection.class_name,
            box=detection.box,
            centroid=detection.centroid,
            first_seen=now,
            last_seen=now,
        )
        tr.history.append((tr.centroid[0], tr.centroid[1], now))
        self.tracks[tid] = tr

    def active_tracks(self) -> Dict[int, Track]:
        return {tid: tr for tid, tr in self.tracks.items() if tr.missed <= self.max_missed}

MEASUREMENT_FIELDS = [
    "timestamp",
    "stream_name",
    "frame_id",
    "elapsed_seconds",
    "people_count",
    "vehicle_count",
    "bag_count",
    "activity_index",
    "bar_entries_total",
    "bar_entries_per_hour_estimate",
    "bar_passersby_total",
    "bar_retention_total_percent",
    "bar_seated_count",
    "observed_bar_revenue_eur",
    "projected_daily_bar_revenue_eur",
    "events_in_sample",
]

EVENT_FIELDS = [
    "timestamp",
    "stream_name",
    "event_type",
    "severity",
    "value",
    "description",
]


class DataStore:
    def __init__(self, output_dir: Union[str, Path]) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.measurements_path = self.output_dir / "measurements.csv"
        self.events_path = self.output_dir / "events.csv"
        self.sqlite_path = self.output_dir / "livestream_data.sqlite"

        self.measurement_file = open(self.measurements_path, "a", newline="", encoding="utf-8")
        self.event_file = open(self.events_path, "a", newline="", encoding="utf-8")
        self.measurement_writer = csv.DictWriter(self.measurement_file, fieldnames=MEASUREMENT_FIELDS)
        self.event_writer = csv.DictWriter(self.event_file, fieldnames=EVENT_FIELDS)

        if self.measurements_path.stat().st_size == 0:
            self.measurement_writer.writeheader()
        if self.events_path.stat().st_size == 0:
            self.event_writer.writeheader()

        self.conn = sqlite3.connect(self.sqlite_path)
        self._create_or_update_table("measurements", MEASUREMENT_FIELDS)
        self._create_or_update_table("events", EVENT_FIELDS)

    def _create_or_update_table(self, table_name: str, fields: Sequence[str]) -> None:
        cols = ", ".join([f"{name} TEXT" for name in fields])
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols})")
        existing = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        for field in fields:
            if field not in existing:
                self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {field} TEXT")
        self.conn.commit()

    def write_measurement(self, row: Dict[str, Any]) -> None:
        clean = {field: row.get(field, "") for field in MEASUREMENT_FIELDS}
        self.measurement_writer.writerow(clean)
        placeholders = ", ".join(["?"] * len(MEASUREMENT_FIELDS))
        self.conn.execute(
            f"INSERT INTO measurements ({', '.join(MEASUREMENT_FIELDS)}) VALUES ({placeholders})",
            [str(clean[field]) for field in MEASUREMENT_FIELDS],
        )
        self.conn.commit()
        self.measurement_file.flush()

    def write_event(self, row: Dict[str, Any]) -> None:
        clean = {field: row.get(field, "") for field in EVENT_FIELDS}
        self.event_writer.writerow(clean)
        placeholders = ", ".join(["?"] * len(EVENT_FIELDS))
        self.conn.execute(
            f"INSERT INTO events ({', '.join(EVENT_FIELDS)}) VALUES ({placeholders})",
            [str(clean[field]) for field in EVENT_FIELDS],
        )
        self.conn.commit()
        self.event_file.flush()

    def close(self) -> None:
        self.measurement_file.close()
        self.event_file.close()
        self.conn.close()


class EventGate:
    def __init__(self) -> None:
        self.last_emitted: Dict[str, float] = {}

    def allowed(self, event_type: str, now: float, cooldown_seconds: float) -> bool:
        last = self.last_emitted.get(event_type, 0.0)
        if now - last >= cooldown_seconds:
            self.last_emitted[event_type] = now
            return True
        return False

def run_yolo(model: Any, frame: np.ndarray, conf: float, imgsz: int) -> List[Detection]:
    results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return []
    result = results[0]
    detections: List[Detection] = []
    names = getattr(model, "names", {})

    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf_value = float(box.conf[0].item())
        class_name = str(names.get(cls_id, cls_id))
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        detections.append(Detection(class_name=class_name, confidence=conf_value, box=(x1, y1, x2, y2)))
    return detections


def retention_rate_percent(entries: int, passersby: int) -> float:
    """Percentage of nearby passers-by who enter/stop at the bar.

    Formula: entries / passersby * 100. The result is capped at 100 to avoid impossible
    percentages caused by imperfect ROI calibration or tracker errors.
    """
    if passersby <= 0:
        return 0.0
    return round(clamp((float(entries) / float(passersby)) * 100.0, 0.0, 100.0), 2)


def activity_index(people: int, vehicles: int, bar_entries_per_hour: float, bar_seated: int) -> float:
    """Explainable public/bar activity score from 0 to 100.

    The index uses current visible activity plus the estimated bar entries per hour:
    - people currently visible in the scene
    - vehicles currently visible
    - estimated bar entries per hour
    - current estimated bar occupancy

    It intentionally does not use passers-by-per-hour, because passers-by are
    estimated from unique visible people and the user requested that this hourly
    estimate be removed.
    """
    people_score = clamp(people / 60.0, 0.0, 1.0)
    vehicle_score = clamp(vehicles / 12.0, 0.0, 1.0)
    bar_entry_score = clamp(float(bar_entries_per_hour) / 80.0, 0.0, 1.0)
    bar_seated_score = clamp(bar_seated / 20.0, 0.0, 1.0)
    score = (
        0.45 * people_score
        + 0.15 * vehicle_score
        + 0.25 * bar_entry_score
        + 0.15 * bar_seated_score
    )
    return round(100.0 * score, 2)


def is_stationary_or_sitting(track: Track, now: float, seconds: float, max_pixels: float, aspect_threshold: float) -> bool:
    # Shape proxy: seated people often have a wider, shorter bounding box than standing people.
    # This is only a proxy and should be explained as an estimate in the report.
    if box_aspect_ratio(track.box) >= aspect_threshold:
        return True

    recent = [(x, y, t) for x, y, t in track.history if now - t <= seconds]
    if len(recent) < 3:
        return False
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return spread <= max_pixels


def track_roi_point(track: Track, mode: str = "bottom") -> Point:
    """Return the point used to decide whether a track is inside the bar ROI.

    - bottom: usually better for terrace/ground polygons because the person
      belongs to the place where their feet/lower body are located.
    - centroid: useful if the ROI is drawn around the full body area.
    """
    if mode == "centroid":
        return track.centroid
    return box_bottom_center(track.box)


def track_age_seconds(track: Track) -> float:
    return max(0.0, track.last_seen - track.first_seen)


def track_dedup_point(track: Track) -> Point:
    """Point used to decide whether two short-lived tracks are probably the same person."""
    return box_bottom_center(track.box)


def prune_passerby_memory(memory: Dict[int, Dict[str, float]], now: float, keep_seconds: float) -> None:
    """Remove old counted-person signatures from the de-duplication memory."""
    for tid in list(memory.keys()):
        if now - memory[tid].get("last_seen", 0.0) > keep_seconds:
            del memory[tid]


def update_passerby_memory(memory: Dict[int, Dict[str, float]], tracks: Dict[int, Track], now: float) -> None:
    """Keep the latest position/size of already-counted passers-by.

    If a person is briefly lost and then receives a new track ID, this memory lets
    the code recognize the new track as probably the same physical person.
    """
    for tid, tr in tracks.items():
        if tr.counted_bar_passerby:
            x, y = track_dedup_point(tr)
            memory[tid] = {
                "x": float(x),
                "y": float(y),
                "area": float(max(1.0, box_area(tr.box))),
                "last_seen": float(now),
            }


def is_probable_duplicate_passerby(track: Track, memory: Dict[int, Dict[str, float]], now: float, args: argparse.Namespace) -> bool:
    """Return True when a new track is probably the same person as a recent counted track.

    This is a simple re-identification heuristic. It checks recent location, time,
    and bounding-box size. It reduces double counting caused by YOLO/tracker ID
    fragmentation without requiring a pass-by line.
    """
    if not memory:
        return False

    current_x, current_y = track_dedup_point(track)
    current_area = float(max(1.0, box_area(track.box)))
    max_seconds = float(getattr(args, "passerby_dedup_seconds", 8.0))
    max_distance = float(getattr(args, "passerby_dedup_distance", 130.0))
    min_ratio = float(getattr(args, "passerby_dedup_box_ratio_min", 0.4))
    max_ratio = float(getattr(args, "passerby_dedup_box_ratio_max", 2.5))

    for sig in memory.values():
        if now - sig.get("last_seen", 0.0) > max_seconds:
            continue
        d = distance((current_x, current_y), (sig.get("x", current_x), sig.get("y", current_y)))
        if d > max_distance:
            continue
        ratio = current_area / float(max(1.0, sig.get("area", current_area)))
        if min_ratio <= ratio <= max_ratio:
            return True
    return False


def should_count_new_passerby(track: Track, memory: Dict[int, Dict[str, float]], now: float, args: argparse.Namespace) -> Tuple[bool, bool]:
    """Decide whether a track should increment the passer-by counter.

    Returns (should_increment_counter, is_duplicate). If is_duplicate is True,
    the track is marked as already counted but the total is not incremented.
    """
    if track.counted_bar_passerby:
        return False, False

    min_age = float(getattr(args, "passerby_min_track_age_seconds", 1.0))
    min_hits = int(getattr(args, "passerby_min_hits", 2))
    if track_age_seconds(track) < min_age or track.hits < min_hits:
        return False, False

    if is_probable_duplicate_passerby(track, memory, now, args):
        return False, True

    return True, False


def update_entry_memory(memory: Dict[int, Dict[str, float]], tracks: Dict[int, Track], now: float) -> None:
    """Keep latest signatures of counted bar entries to avoid duplicate entry events.

    If YOLO loses a person around the entrance line and creates a new track ID,
    the same physical person could otherwise be counted as a second bar entry.
    """
    for tid, tr in tracks.items():
        if tr.counted_bar_entry:
            x, y = track_dedup_point(tr)
            memory[tid] = {
                "x": float(x),
                "y": float(y),
                "area": float(max(1.0, box_area(tr.box))),
                "last_seen": float(now),
            }


def is_probable_duplicate_entry(track: Track, memory: Dict[int, Dict[str, float]], now: float, args: argparse.Namespace) -> bool:
    """Return True when a candidate entry is probably the same physical person
    as a recently counted entry. This prevents one person from generating
    multiple BAR_ENTRY_DETECTED events after short tracker ID losses.
    """
    if not memory:
        return False

    current_x, current_y = track_dedup_point(track)
    current_area = float(max(1.0, box_area(track.box)))
    max_seconds = float(getattr(args, "entry_dedup_seconds", getattr(args, "passerby_dedup_seconds", 8.0)))
    max_distance = float(getattr(args, "entry_dedup_distance", getattr(args, "passerby_dedup_distance", 130.0)))
    min_ratio = float(getattr(args, "entry_dedup_box_ratio_min", getattr(args, "passerby_dedup_box_ratio_min", 0.4)))
    max_ratio = float(getattr(args, "entry_dedup_box_ratio_max", getattr(args, "passerby_dedup_box_ratio_max", 2.5)))

    for sig in memory.values():
        if now - sig.get("last_seen", 0.0) > max_seconds:
            continue
        d = distance((current_x, current_y), (sig.get("x", current_x), sig.get("y", current_y)))
        if d > max_distance:
            continue
        ratio = current_area / float(max(1.0, sig.get("area", current_area)))
        if min_ratio <= ratio <= max_ratio:
            return True
    return False


def count_visible_person_if_needed(
    track: Track,
    memory: Dict[int, Dict[str, float]],
    now: float,
    args: argparse.Namespace,
    current_total: int,
) -> Tuple[int, bool]:
    """Ensure a physical person is counted once in the camera-total denominator.

    This is used both during normal visible-person counting and when a person
    is counted as a bar entry. An entry is also a detected camera person, so
    the denominator must include that person. This prevents impossible values
    such as more bar entries than total detected people.

    Returns (new_total, incremented_counter).
    """
    if track.counted_bar_passerby:
        return current_total, False

    if is_probable_duplicate_passerby(track, memory, now, args):
        track.counted_bar_passerby = True
        update_passerby_memory(memory, {track.track_id: track}, now)
        return current_total, False

    track.counted_bar_passerby = True
    current_total += 1
    update_passerby_memory(memory, {track.track_id: track}, now)
    return current_total, True


def qualifies_as_bar_seated_or_present(track: Track, now: float, args: argparse.Namespace, rois: Dict[str, List[Point]]) -> bool:
    """Decide whether a tracked person should contribute to current bar occupancy.

    The old strict mode used a stationary/sitting proxy. That is often too strict
    in real bar webcams because tables, chairs, umbrellas, low resolution and
    partial occlusion make seated people difficult to classify. The default
    roi_occupancy mode counts any stable person whose selected ROI point is inside
    the 4-point bar area.
    """
    roi_point = track_roi_point(track, getattr(args, "bar_roi_point", "bottom"))
    in_bar_area = point_in_polygon(roi_point, rois["bar_seating_rect"])
    if not in_bar_area:
        return False

    min_age = float(getattr(args, "bar_min_track_age_seconds", 0.5))
    if track.last_seen - track.first_seen < min_age:
        return False

    mode = getattr(args, "bar_seated_mode", "roi_occupancy")
    if mode == "strict_sitting":
        return is_stationary_or_sitting(
            track,
            now,
            seconds=args.stationary_seconds,
            max_pixels=args.stationary_pixels,
            aspect_threshold=args.sitting_aspect_ratio,
        )

    # Default: more realistic occupancy count for webcam/bar scenes.
    return True


def estimate_per_hour(count: int, elapsed_seconds: float) -> float:
    """Estimate hourly flow from the observed count and elapsed runtime.

    Example: 10 entries observed in 15 minutes -> 40 entries/hour.
    This replaces the old 15-minute / 30-minute metric.
    """
    if elapsed_seconds <= 0:
        return 0.0
    return round((float(count) * 3600.0) / float(elapsed_seconds), 2)

def run_live(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install ultralytics first: pip install ultralytics") from exc

    cap = open_capture(args.source)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Could not read from stream after opening it.")

    frame = resize_if_needed(frame, args.width)
    h, w = frame.shape[:2]
    rois = load_rois(args.roi_file, w, h)
    bar_entry_target_side = target_side_from_bar_polygon(rois["bar_entry_line"], rois["bar_seating_rect"])
    if getattr(args, "flip_entry_side", False):
        bar_entry_target_side *= -1
    if bar_entry_target_side == 0:
        print("[warning] Could not infer which side of the entry line is the bar side. Entry counting will count lower-body crossings in either direction.")
    else:
        print(f"Entry target side relative to your line: {bar_entry_target_side}. Use --flip-entry-side if entries are not counted because the side is reversed.")

    model = YOLO(args.model)
    tracker = CentroidTracker(max_distance=args.track_max_distance, max_missed=args.track_max_missed)

    # Create a clean output folder for this specific live run.
    # This prevents old CSV rows from previous tests being mixed with the new analysis.
    base_output_dir = Path(args.output_dir)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_output_dir = ensure_dir(base_output_dir / run_id)
    args.output_dir = str(run_output_dir)
    store = DataStore(args.output_dir)
    gate = EventGate()

    window_name = "Bondi Chaweng bar analytics"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_sample_time = 0.0
    start_time = time.time()
    frame_id = 0

    bar_entries_total = 0
    bar_passersby_total = 0
    observed_bar_revenue = 0.0
    # Recent counted-person signatures used to avoid double-counting the same passer-by
    # when YOLO briefly loses a person and the tracker assigns a new ID.
    counted_passerby_memory: Dict[int, Dict[str, float]] = {}
    counted_entry_memory: Dict[int, Dict[str, float]] = {}

    last_people_count = 0
    last_vehicle_count = 0
    last_bag_count = 0
    last_activity = 0.0
    events_in_current_sample: List[str] = []
    latest_detections: List[Detection] = []
    latest_tracks: Dict[int, Track] = {}

    print("Running live extraction. Press q in the video window to quit.")
    print("Bar setup: 4-point bar area = current bar occupancy area; entry line = lower-body crossing into the bar side; passers-by = total unique people detected by the camera with duplicate-ID filtering.")
    print(f"Revenue model: each counted bar entry is assumed to spend EUR {args.spend_per_person:.2f}.")

    try:
        while True:
            if frame_id > 0:
                ok, frame_raw = cap.read()
                if not ok or frame_raw is None:
                    print("[warning] Stream frame not available. Trying again...")
                    time.sleep(0.5)
                    continue
                frame = resize_if_needed(frame_raw, args.width)

            frame_id += 1
            now = time.time()
            display = frame.copy()

            if frame_id % args.process_every == 0:
                latest_detections = run_yolo(model, frame, args.confidence, args.imgsz)
                person_detections = [d for d in latest_detections if d.class_name in PERSON_CLASSES]
                latest_tracks = tracker.update(person_detections, now)

                vehicle_detections = [d for d in latest_detections if d.class_name in VEHICLE_CLASSES]
                bag_detections = [d for d in latest_detections if d.class_name in BAG_CLASSES]

                last_people_count = len(person_detections)
                last_vehicle_count = len(vehicle_detections)
                last_bag_count = len(bag_detections)

                prune_passerby_memory(counted_passerby_memory, now, args.passerby_dedup_seconds)
                update_passerby_memory(counted_passerby_memory, latest_tracks, now)
                prune_passerby_memory(counted_entry_memory, now, float(getattr(args, "entry_dedup_seconds", args.passerby_dedup_seconds)))
                update_entry_memory(counted_entry_memory, latest_tracks, now)

                for tid, tr in list(latest_tracks.items()):
                    # People passing by = total unique people detected by the camera.
                    # We still wait until a track is stable and use duplicate-ID filtering
                    # so that the same physical person is not counted repeatedly.
                    should_increment, is_duplicate = should_count_new_passerby(tr, counted_passerby_memory, now, args)
                    if is_duplicate:
                        tr.counted_bar_passerby = True
                        update_passerby_memory(counted_passerby_memory, {tid: tr}, now)
                    elif should_increment:
                        bar_passersby_total, incremented = count_visible_person_if_needed(
                            tr, counted_passerby_memory, now, args, bar_passersby_total
                        )
                        if incremented:
                            event = {
                                "timestamp": local_timestamp(),
                                "stream_name": args.stream_name,
                                "event_type": "BAR_PASSERBY_DETECTED",
                                "severity": "low",
                                "value": bar_passersby_total,
                                "description": "A unique detected person was counted once as part of the total people detected by the camera / potential customers.",
                            }
                            store.write_event(event)
                            events_in_current_sample.append("BAR_PASSERBY_DETECTED")

                    # Bar entry/stopping count: unique tracked person whose lower-body point
                    # crosses the entry line into the bar side. This avoids head-only
                    # counting but is less strict than requiring the whole YOLO box to clear the line.
                    if not tr.counted_bar_entry:
                        stable_for_entry = (
                            track_age_seconds(tr) >= float(getattr(args, "entry_min_track_age_seconds", 0.5))
                            and tr.hits >= int(getattr(args, "entry_min_hits", 2))
                        )
                        if stable_for_entry and update_entry_point_crossing(
                            tr,
                            rois["bar_entry_line"],
                            bar_entry_target_side,
                            float(getattr(args, "entry_line_margin_pixels", 2.0)),
                            float(getattr(args, "entry_point_y_ratio", 0.90)),
                        ):
                            if is_probable_duplicate_entry(tr, counted_entry_memory, now, args):
                                # Same physical person already generated a recent entry event.
                                tr.counted_bar_entry = True
                                update_entry_memory(counted_entry_memory, {tid: tr}, now)
                            else:
                                # An entrant is also a detected camera person. Count them in
                                # the denominator if they were not already counted as a visible person.
                                bar_passersby_total, passer_incremented_from_entry = count_visible_person_if_needed(
                                    tr, counted_passerby_memory, now, args, bar_passersby_total
                                )
                                if passer_incremented_from_entry:
                                    passer_event = {
                                        "timestamp": local_timestamp(),
                                        "stream_name": args.stream_name,
                                        "event_type": "BAR_PASSERBY_DETECTED",
                                        "severity": "low",
                                        "value": bar_passersby_total,
                                        "description": "A bar entrant was also counted as part of the total people detected by the camera.",
                                    }
                                    store.write_event(passer_event)
                                    events_in_current_sample.append("BAR_PASSERBY_DETECTED")

                                tr.counted_bar_entry = True
                                bar_entries_total += 1
                                # Safety rule: entries are a subset of detected camera people.
                                # This keeps retention logically valid even when tracking is noisy.
                                if bar_passersby_total < bar_entries_total:
                                    bar_passersby_total = bar_entries_total
                                observed_bar_revenue += args.spend_per_person
                                update_entry_memory(counted_entry_memory, {tid: tr}, now)
                                event = {
                                    "timestamp": local_timestamp(),
                                    "stream_name": args.stream_name,
                                    "event_type": "BAR_ENTRY_DETECTED",
                                    "severity": "medium",
                                    "value": bar_entries_total,
                                    "description": "A tracked person was counted as entering after the lower-body reference point crossed the selected bar entry line into the bar side.",
                                }
                                store.write_event(event)
                                events_in_current_sample.append("BAR_ENTRY_DETECTED")

                    # Bar occupancy/seated estimate is stored as a measurement, not as
                    # a per-person event. In earlier versions, BAR_SEATED_DETECTED was
                    # written each time a tracker ID first appeared inside the bar ROI.
                    # That made the event count look too high when YOLO lost/recreated IDs.
                    # The reliable value is bar_seated_count below: current occupancy at
                    # the sampling moment.

            # Current estimated bar occupancy/seated count in the 4-point bar area.
            raw_bar_occupancy = 0
            for tr in latest_tracks.values():
                if qualifies_as_bar_seated_or_present(tr, now, args, rois):
                    raw_bar_occupancy += 1
            bar_seated_count = int(round(raw_bar_occupancy * float(getattr(args, "bar_occupancy_correction", 1.0))))

            elapsed_seconds = max(0.0, now - start_time)
            # Entries must be a subset of the total detected people from the camera.
            # If tracker noise ever makes entries higher, correct the denominator.
            if bar_passersby_total < bar_entries_total:
                bar_passersby_total = bar_entries_total
            bar_entries_per_hour_estimate = estimate_per_hour(bar_entries_total, elapsed_seconds)
            bar_retention_total_percent = retention_rate_percent(bar_entries_total, bar_passersby_total)

            last_activity = activity_index(
                last_people_count,
                last_vehicle_count,
                bar_entries_per_hour_estimate,
                bar_seated_count,
            )

            # Daily projection from the observed entries-per-hour estimate.
            projected_daily_revenue = bar_entries_per_hour_estimate * args.business_hours_per_day * args.spend_per_person

            # Rule-based events with cooldowns.
            # We generate an event only when bar occupancy is meaningfully high, not
            # for every person detected inside the bar area. The exact occupancy is
            # already stored continuously in measurements.csv as bar_seated_count.
            if bar_seated_count >= args.bar_occupancy_high_threshold and gate.allowed("BAR_OCCUPANCY_HIGH", now, args.event_cooldown_seconds):
                event = {
                    "timestamp": local_timestamp(),
                    "stream_name": args.stream_name,
                    "event_type": "BAR_OCCUPANCY_HIGH",
                    "severity": "medium",
                    "value": bar_seated_count,
                    "description": "Estimated current bar occupancy exceeded the configured threshold.",
                }
                store.write_event(event)
                events_in_current_sample.append("BAR_OCCUPANCY_HIGH")

            if last_activity >= args.high_activity_threshold and gate.allowed("HIGH_ACTIVITY", now, args.event_cooldown_seconds):
                event = {
                    "timestamp": local_timestamp(),
                    "stream_name": args.stream_name,
                    "event_type": "HIGH_ACTIVITY",
                    "severity": "high",
                    "value": last_activity,
                    "description": f"Activity index reached {last_activity}.",
                }
                store.write_event(event)
                events_in_current_sample.append("HIGH_ACTIVITY")

            if last_people_count >= args.crowd_threshold and gate.allowed("CROWD_DETECTED", now, args.event_cooldown_seconds):
                event = {
                    "timestamp": local_timestamp(),
                    "stream_name": args.stream_name,
                    "event_type": "CROWD_DETECTED",
                    "severity": "high",
                    "value": last_people_count,
                    "description": f"Detected {last_people_count} people in the scene.",
                }
                store.write_event(event)
                events_in_current_sample.append("CROWD_DETECTED")

            if last_vehicle_count >= args.vehicle_threshold and gate.allowed("VEHICLE_DENSITY", now, args.event_cooldown_seconds):
                event = {
                    "timestamp": local_timestamp(),
                    "stream_name": args.stream_name,
                    "event_type": "VEHICLE_DENSITY",
                    "severity": "medium",
                    "value": last_vehicle_count,
                    "description": f"Detected {last_vehicle_count} vehicles in the scene.",
                }
                store.write_event(event)
                events_in_current_sample.append("VEHICLE_DENSITY")

            if (
                bar_passersby_total >= args.min_passersby_for_retention_event
                and bar_retention_total_percent >= args.high_retention_threshold
                and gate.allowed("HIGH_BAR_RETENTION", now, args.event_cooldown_seconds)
            ):
                event = {
                    "timestamp": local_timestamp(),
                    "stream_name": args.stream_name,
                    "event_type": "HIGH_BAR_RETENTION",
                    "severity": "medium",
                    "value": bar_retention_total_percent,
                    "description": f"Overall bar retention reached {bar_retention_total_percent:.2f}%.",
                }
                store.write_event(event)
                events_in_current_sample.append("HIGH_BAR_RETENTION")

            if (
                bar_passersby_total >= args.min_passersby_for_retention_event
                and bar_retention_total_percent <= args.low_retention_threshold
                and gate.allowed("LOW_BAR_RETENTION", now, args.event_cooldown_seconds)
            ):
                event = {
                    "timestamp": local_timestamp(),
                    "stream_name": args.stream_name,
                    "event_type": "LOW_BAR_RETENTION",
                    "severity": "medium",
                    "value": bar_retention_total_percent,
                    "description": f"Overall bar retention is {bar_retention_total_percent:.2f}%.",
                }
                store.write_event(event)
                events_in_current_sample.append("LOW_BAR_RETENTION")

            # Periodic measurement row.
            if now - last_sample_time >= args.sample_interval_seconds:
                last_sample_time = now
                row = {
                    "timestamp": local_timestamp(),
                    "stream_name": args.stream_name,
                    "frame_id": frame_id,
                    "elapsed_seconds": round(elapsed_seconds, 2),
                    "people_count": last_people_count,
                    "vehicle_count": last_vehicle_count,
                    "bag_count": last_bag_count,
                    "activity_index": last_activity,
                    "bar_entries_total": bar_entries_total,
                    "bar_entries_per_hour_estimate": bar_entries_per_hour_estimate,
                    "bar_passersby_total": bar_passersby_total,
                    "bar_retention_total_percent": bar_retention_total_percent,
                    "bar_seated_count": bar_seated_count,
                    "observed_bar_revenue_eur": round(observed_bar_revenue, 2),
                    "projected_daily_bar_revenue_eur": round(projected_daily_revenue, 2),
                    "events_in_sample": "|".join(sorted(set(events_in_current_sample))),
                }
                store.write_measurement(row)
                print(row)
                events_in_current_sample = []

            # Draw detections.
            for det in latest_detections:
                x1, y1, x2, y2 = [int(v) for v in det.box]
                if det.class_name in PERSON_CLASSES:
                    color = (0, 255, 0)
                elif det.class_name in VEHICLE_CLASSES:
                    color = (255, 160, 0)
                elif det.class_name in BAG_CLASSES:
                    color = (0, 255, 255)
                else:
                    continue
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                draw_label(display, f"{det.class_name} {det.confidence:.2f}", (x1, max(20, y1 - 4)))

            # Draw track IDs.
            for tid, tr in latest_tracks.items():
                cx, cy = int(tr.centroid[0]), int(tr.centroid[1])
                cv2.circle(display, (cx, cy), 4, (255, 255, 255), -1)
                draw_label(display, f"ID {tid}", (cx + 5, cy - 5))

            # Draw calibrated ROIs.
            draw_polygon(display, rois["bar_seating_rect"], "Bar sitting rectangle (4 pts)", (0, 180, 255))
            draw_line(display, rois["bar_entry_line"], "Bar entry line", (0, 180, 255))

            # Overlay metrics.
            overlay_lines = [
                f"People: {last_people_count} | Vehicles: {last_vehicle_count} | Bags: {last_bag_count}",
                f"Activity index: {last_activity:.1f}",
                f"Bar entries: {bar_entries_total} | Est/hour: {bar_entries_per_hour_estimate:.1f}",
                f"Total people detected: {bar_passersby_total}",
                f"Retention: {bar_retention_total_percent:.1f}% total",
                f"Estimated bar occupancy now: {bar_seated_count}",
                f"Observed revenue: EUR {observed_bar_revenue:.2f} | Projected daily: EUR {projected_daily_revenue:.2f}",
                "Press q to quit",
            ]
            y = 28
            for line in overlay_lines:
                draw_label(display, line, (20, y))
                y += 28

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            # If the user closes the OpenCV window with the X button, stop the run cleanly.
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if args.max_seconds > 0 and now - start_time >= args.max_seconds:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        store.close()
        print(f"Data saved to: {args.output_dir}")

def run_analysis(output_dir: Union[str, Path], business_hours_per_day: float = 12.0) -> None:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install pandas first: pip install pandas") from exc

    matplotlib_available = True
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        matplotlib_available = False
        plt = None
        print("[warning] Matplotlib is not available. SVG charts will be created instead.")

    output_path = ensure_dir(output_dir)
    measurements_csv = output_path / "measurements.csv"
    events_csv = output_path / "events.csv"

    if not measurements_csv.exists() or measurements_csv.stat().st_size == 0:
        raise FileNotFoundError(f"No measurement data found at {measurements_csv}. Run live extraction first.")

    df = pd.read_csv(measurements_csv)
    if df.empty:
        raise RuntimeError("The measurements CSV is empty.")

    numeric_cols = [
        "frame_id",
        "elapsed_seconds",
        "people_count",
        "vehicle_count",
        "bag_count",
        "activity_index",
        "bar_entries_total",
        "bar_entries_per_hour_estimate",
        "bar_passersby_total",
        "bar_retention_total_percent",
        "bar_seated_count",
        "observed_bar_revenue_eur",
        "projected_daily_bar_revenue_eur",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # Sort the selected run by timestamp and recalculate retention from the stored counts.
    # This avoids inconsistent summaries such as 3 entries, 9 passers-by, but 50% retention.
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["timestamp_dt", "frame_id"], na_position="last").reset_index(drop=True)

    # In the business definition used here, people passing by means the total
    # unique people detected by the camera during the run. Bar entries are a
    # subset of those people, so the denominator must never be lower than entries.
    df["bar_passersby_total"] = df[["bar_passersby_total", "bar_entries_total"]].max(axis=1)
    df["bar_retention_total_percent"] = df.apply(
        lambda row: retention_rate_percent(int(row["bar_entries_total"]), int(row["bar_passersby_total"])),
        axis=1,
    )

    def _save_svg_line_chart(dataframe, cols, title, y_label, filename):
        width, height = 1100, 500
        margin_left, margin_right, margin_top, margin_bottom = 80, 30, 50, 90
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        valid_cols = [c for c in cols if c in dataframe.columns]
        if not valid_cols:
            return
        values = []
        for c in valid_cols:
            values.extend([float(v) for v in dataframe[c].tolist() if not pd.isna(v)])
        if not values:
            return
        v_min, v_max = min(values), max(values)
        if abs(v_max - v_min) < 1e-9:
            v_max = v_min + 1.0

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        def x_pos(i):
            if len(dataframe) <= 1:
                return margin_left
            return margin_left + i * plot_w / (len(dataframe) - 1)

        def y_pos(v):
            return margin_top + plot_h - ((float(v) - v_min) / (v_max - v_min)) * plot_h

        svg = []
        svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        svg.append('<rect width="100%" height="100%" fill="white"/>')
        svg.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>')
        svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="black"/>')
        svg.append(f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="black"/>')
        svg.append(f'<text x="18" y="{margin_top+plot_h/2}" transform="rotate(-90, 18, {margin_top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>')
        svg.append(f'<text x="{margin_left}" y="{margin_top+plot_h+38}" font-family="Arial" font-size="12">{dataframe["timestamp"].iloc[0]}</text>')
        svg.append(f'<text x="{margin_left+plot_w}" y="{margin_top+plot_h+38}" text-anchor="end" font-family="Arial" font-size="12">{dataframe["timestamp"].iloc[-1]}</text>')
        svg.append(f'<text x="{margin_left-8}" y="{margin_top+5}" text-anchor="end" font-family="Arial" font-size="12">{v_max:.1f}</text>')
        svg.append(f'<text x="{margin_left-8}" y="{margin_top+plot_h}" text-anchor="end" font-family="Arial" font-size="12">{v_min:.1f}</text>')

        legend_y = 55
        for idx, c in enumerate(valid_cols):
            points = []
            for i, val in enumerate(dataframe[c].tolist()):
                try:
                    if pd.isna(val):
                        continue
                    points.append(f"{x_pos(i):.1f},{y_pos(val):.1f}")
                except Exception:
                    continue
            if not points:
                continue
            color = colors[idx % len(colors)]
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>')
            svg.append(f'<line x1="{margin_left+20+idx*250}" y1="{legend_y}" x2="{margin_left+50+idx*250}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
            svg.append(f'<text x="{margin_left+56+idx*250}" y="{legend_y+4}" font-family="Arial" font-size="13">{c}</text>')
        svg.append('</svg>')
        (output_path / filename).write_text("\n".join(svg), encoding="utf-8")

    def _save_svg_bar_chart(counts, title, filename):
        width, height = 1000, 450
        margin_left, margin_right, margin_top, margin_bottom = 80, 30, 45, 130
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        items = list(counts.items())
        if not items:
            return
        max_val = max(v for _, v in items) or 1
        bar_w = max(8, plot_w / len(items) * 0.7)
        gap = plot_w / len(items) * 0.3
        svg = []
        svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        svg.append('<rect width="100%" height="100%" fill="white"/>')
        svg.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>')
        svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="black"/>')
        svg.append(f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="black"/>')
        svg.append(f'<text x="{margin_left-8}" y="{margin_top+5}" text-anchor="end" font-family="Arial" font-size="12">{max_val}</text>')
        for i, (name, val) in enumerate(items):
            x = margin_left + i * (bar_w + gap) + gap / 2
            h = (val / max_val) * plot_h
            y = margin_top + plot_h - h
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#1f77b4"/>')
            svg.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{val}</text>')
            label = str(name)[:24]
            svg.append(f'<text x="{x+bar_w/2:.1f}" y="{margin_top+plot_h+18}" transform="rotate(35,{x+bar_w/2:.1f},{margin_top+plot_h+18})" font-family="Arial" font-size="11">{label}</text>')
        svg.append('</svg>')
        (output_path / filename).write_text("\n".join(svg), encoding="utf-8")

    # Charts.
    if matplotlib_available:
        plt.figure(figsize=(12, 5))
        plt.plot(df["timestamp"], df["activity_index"])
        plt.title("Bar/public activity index over time")
        plt.xlabel("Time")
        plt.ylabel("Activity index, 0-100")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path / "activity_index_over_time.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(df["timestamp"], df["people_count"], label="People in scene")
        plt.plot(df["timestamp"], df["bar_seated_count"], label="Estimated seated in bar")
        plt.plot(df["timestamp"], df["bar_entries_total"], label="Bar entries total")
        plt.plot(df["timestamp"], df["bar_passersby_total"], label="Passers-by total")
        plt.title("People count, bar seated occupancy, entries, and total detected people")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path / "people_bar_seated_entries_passersby.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(df["timestamp"], df["bar_retention_total_percent"], label="Total retention %")
        plt.title("Bar retention rate: bar entries compared to unique total detected people")
        plt.xlabel("Time")
        plt.ylabel("Retention %")
        plt.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path / "bar_retention_rate.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(df["timestamp"], df["bar_entries_per_hour_estimate"], label="Estimated bar entries/hour")
        plt.title("Estimated bar entries per hour")
        plt.xlabel("Time")
        plt.ylabel("People per hour")
        plt.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path / "bar_entries_per_hour_estimate.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(df["timestamp"], df["observed_bar_revenue_eur"], label="Observed accumulated revenue")
        plt.plot(df["timestamp"], df["projected_daily_bar_revenue_eur"], label="Projected daily revenue")
        plt.title("Bar revenue estimate")
        plt.xlabel("Time")
        plt.ylabel("EUR")
        plt.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path / "bar_revenue_estimate.png", dpi=150)
        plt.close()
    else:
        _save_svg_line_chart(df, ["activity_index"], "Bar/public activity index over time", "Activity index, 0-100", "activity_index_over_time.svg")
        _save_svg_line_chart(df, ["people_count", "bar_seated_count", "bar_entries_total", "bar_passersby_total"], "People count, bar seated occupancy, entries, and total detected people", "Count", "people_bar_seated_entries_passersby.svg")
        _save_svg_line_chart(df, ["bar_retention_total_percent"], "Bar retention rate: bar entries compared to unique total detected people", "Retention %", "bar_retention_rate.svg")
        _save_svg_line_chart(df, ["bar_entries_per_hour_estimate"], "Estimated bar entries per hour", "People per hour", "bar_entries_per_hour_estimate.svg")
        _save_svg_line_chart(df, ["observed_bar_revenue_eur", "projected_daily_bar_revenue_eur"], "Bar revenue estimate", "EUR", "bar_revenue_estimate.svg")

    # Event summary and chart.
    event_summary = "No events file found."
    if events_csv.exists() and events_csv.stat().st_size > 0:
        ev = pd.read_csv(events_csv)
        if not ev.empty and "event_type" in ev.columns:
            counts = ev["event_type"].value_counts()
            if matplotlib_available:
                plt.figure(figsize=(10, 5))
                counts.plot(kind="bar")
                plt.title("Generated event counts")
                plt.xlabel("Event type")
                plt.ylabel("Count")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.savefig(output_path / "event_counts.png", dpi=150)
                plt.close()
            else:
                _save_svg_bar_chart(counts.to_dict(), "Generated event counts", "event_counts.svg")
            event_summary = counts.to_string()

    # Use one consistent set of counts for the final summary.
    # Retention is always: bar entries detected / total detected people counted * 100.
    final_bar_entries = int(df["bar_entries_total"].max())
    final_passersby = max(int(df["bar_passersby_total"].max()), final_bar_entries)
    final_retention = retention_rate_percent(final_bar_entries, final_passersby)

    timestamp_series = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.Series([], dtype="datetime64[ns]")
    valid_times = timestamp_series.dropna()
    if len(valid_times) >= 2:
        elapsed_hours_for_summary = max((valid_times.max() - valid_times.min()).total_seconds() / 3600.0, 0.0)
    else:
        elapsed_hours_for_summary = float(df["elapsed_seconds"].max()) / 3600.0 if df["elapsed_seconds"].max() > 0 else 0.0

    if elapsed_hours_for_summary > 0:
        entries_per_hour_summary = round(final_bar_entries / elapsed_hours_for_summary, 2)
    else:
        entries_per_hour_summary = 0.0

    observed_revenue_summary = float(df["observed_bar_revenue_eur"].max())
    if final_bar_entries > 0 and observed_revenue_summary > 0:
        spend_per_person_summary = observed_revenue_summary / float(final_bar_entries)
    else:
        spend_per_person_summary = 7.0
    projected_daily_revenue_summary = round(entries_per_hour_summary * float(business_hours_per_day) * spend_per_person_summary, 2)

    summary = []
    summary.append("Bondi Chaweng Bar Livestream Analytics - Summary")
    summary.append("=" * 58)
    summary.append(f"Number of measurement rows: {len(df)}")
    summary.append(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    summary.append(f"Average people count: {df['people_count'].mean():.2f}")
    summary.append(f"Maximum people count: {df['people_count'].max():.0f}")
    summary.append(f"Average activity index: {df['activity_index'].mean():.2f}")
    summary.append(f"Maximum activity index: {df['activity_index'].max():.2f}")
    summary.append(f"Bar entries detected: {final_bar_entries}")
    summary.append(f"Total people detected by camera: {final_passersby}")
    summary.append(f"Final total bar retention: {final_retention:.2f}%")
    summary.append(f"Estimated bar entries per hour: {entries_per_hour_summary:.2f}")
    summary.append(f"Maximum estimated seated people in the bar: {df['bar_seated_count'].max():.0f}")
    summary.append(f"Observed bar revenue: EUR {observed_revenue_summary:.2f}")
    summary.append(f"Projected daily bar revenue: EUR {projected_daily_revenue_summary:.2f}")
    summary.append("")
    summary.append("Event summary:")
    summary.append(event_summary)
    summary.append("")
    summary.append("Generated chart files:")
    if matplotlib_available:
        summary.append("- activity_index_over_time.png")
        summary.append("- people_bar_seated_entries_passersby.png")
        summary.append("- bar_retention_rate.png")
        summary.append("- bar_entries_per_hour_estimate.png")
        summary.append("- bar_revenue_estimate.png")
        if events_csv.exists():
            summary.append("- event_counts.png")
    else:
        summary.append("- activity_index_over_time.svg")
        summary.append("- people_bar_seated_entries_passersby.svg")
        summary.append("- bar_retention_rate.svg")
        summary.append("- bar_entries_per_hour_estimate.svg")
        summary.append("- bar_revenue_estimate.svg")
        if events_csv.exists():
            summary.append("- event_counts.svg")
    summary.append("")
    summary.append("Big Data context:")
    summary.append(
        "The application converts raw video frames into compact structured time-series rows and event records. "
        "In a larger IoT pipeline, many camera nodes could publish the same schema through MQTT or Kafka. "
        "The data could then be stored in a data lake, processed by Spark/Flink, and visualized in dashboards "
        "for tourism activity monitoring, venue capacity analysis, footfall-to-customer conversion, and local business planning."
    )
    summary.append("")
    summary.append("Privacy note:")
    summary.append(
        "The system stores aggregate counts and events, not face identities or raw personal images. "
        "This is more privacy-preserving than storing continuous video."
    )

    report_path = output_path / "summary_report.txt"
    report_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"Analysis written to {output_path}")
    print(report_path.read_text(encoding="utf-8"))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Public livestream bar activity extraction using Python, OpenCV, YOLO, CSV, and SQLite.",
        allow_abbrev=False,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Public stream URL, YouTube live URL, local video path, or webcam index 0.")
    parser.add_argument("--stream-name", default="bondi_chaweng_samui", help="Name stored in CSV/SQLite.")
    parser.add_argument("--roi-file", default=DEFAULT_ROI_FILE, help="ROI calibration JSON file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for CSV, SQLite, charts, and report.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model file/name, e.g. yolov8s.pt, yolo11s.pt, or yolov8m.pt.")
    parser.add_argument("--calibrate", action="store_true", help="Open the stream and select bar ROIs.")
    parser.add_argument("--analyze", action="store_true", help="Analyze collected CSV data and generate charts/report.")

    parser.add_argument("--width", type=int, default=1600, help="Resize video width for speed. Use 0 for original size. Larger width helps small/distant people.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size. Larger values help detect small people but run slower.")
    parser.add_argument("--confidence", type=float, default=0.20, help="YOLO confidence threshold. Lower values find more people but can add false positives.")
    parser.add_argument("--process-every", type=int, default=3, help="Run YOLO every N frames. Lower values improve tracking but run slower.")
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0, help="Write one measurement row every N seconds.")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Stop after N seconds. 0 means run until q is pressed.")

    parser.add_argument("--track-max-distance", type=float, default=120.0, help="Max pixel distance for matching person tracks. Increase if the same person gets new IDs.")
    parser.add_argument("--track-max-missed", type=int, default=30, help="Remove track after N missed detection updates. Increase to bridge short occlusions.")
    parser.add_argument("--passerby-min-track-age-seconds", type=float, default=1.0, help="A person must be tracked for this long before being counted as a passer-by.")
    parser.add_argument("--passerby-min-hits", type=int, default=2, help="Minimum YOLO/tracker updates before a person is counted as a passer-by.")
    parser.add_argument("--passerby-dedup-seconds", type=float, default=8.0, help="How long to remember counted passers-by for duplicate-ID filtering.")
    parser.add_argument("--passerby-dedup-distance", type=float, default=130.0, help="Max pixel distance for treating a new track as the same recent passer-by.")
    parser.add_argument("--passerby-dedup-box-ratio-min", type=float, default=0.4, help="Min box-area ratio for duplicate passer-by matching.")
    parser.add_argument("--passerby-dedup-box-ratio-max", type=float, default=2.5, help="Max box-area ratio for duplicate passer-by matching.")

    parser.add_argument("--entry-min-track-age-seconds", type=float, default=0.5, help="A person must be tracked this long before lower-body entry counting can trigger.")
    parser.add_argument("--entry-min-hits", type=int, default=2, help="Minimum detection/tracker updates before lower-body entry counting can trigger.")
    parser.add_argument("--entry-line-margin-pixels", "--entry-line-box-margin-pixels", dest="entry_line_margin_pixels", type=float, default=2.0, help="Small pixel margin around the entry line for the lower-body entry point. Higher values reduce jitter.")
    parser.add_argument("--entry-point-y-ratio", type=float, default=0.90, help="Vertical point inside the person box used for entry counting. 0.5=center, 0.9=lower body, 1.0=feet/bottom.")
    parser.add_argument("--entry-dedup-seconds", type=float, default=8.0, help="How long to remember counted bar entries for duplicate-ID filtering.")
    parser.add_argument("--entry-dedup-distance", type=float, default=130.0, help="Max pixel distance for treating a new entry track as the same recent entry.")
    parser.add_argument("--flip-entry-side", action="store_true", help="Flip the inferred bar side of the entry line if entries are not being counted.")
    parser.add_argument("--spend-per-person", type=float, default=7.0, help="EUR spent per person entering the bar.")
    parser.add_argument("--business-hours-per-day", type=float, default=12.0, help="Assumed bar opening hours per day for projection.")
    parser.add_argument("--bar-seated-mode", choices=["roi_occupancy", "strict_sitting"], default="roi_occupancy", help="roi_occupancy counts stable detected people inside the bar polygon. strict_sitting also requires stationary/sitting posture.")
    parser.add_argument("--bar-roi-point", choices=["bottom", "centroid"], default="bottom", help="Point used to test whether a person is inside the bar polygon. bottom is usually better for terrace/ground ROIs.")
    parser.add_argument("--bar-min-track-age-seconds", type=float, default=0.5, help="Minimum time a detected person must be tracked before counting as bar occupancy.")
    parser.add_argument("--bar-occupancy-correction", type=float, default=1.0, help="Optional multiplier after manual validation, e.g. 1.25 if YOLO misses about 20 percent of visible customers.")
    parser.add_argument("--stationary-seconds", type=float, default=20.0, help="Only used with --bar-seated-mode strict_sitting.")
    parser.add_argument("--stationary-pixels", type=float, default=30.0, help="Only used with --bar-seated-mode strict_sitting.")
    parser.add_argument("--sitting-aspect-ratio", type=float, default=0.65, help="Only used with --bar-seated-mode strict_sitting.")

    parser.add_argument("--high-activity-threshold", type=float, default=70.0, help="Event threshold for activity index.")
    parser.add_argument("--crowd-threshold", type=int, default=40, help="Event threshold for people count.")
    parser.add_argument("--vehicle-threshold", type=int, default=8, help="Event threshold for vehicle count.")
    parser.add_argument("--bar-occupancy-high-threshold", type=int, default=10, help="Current estimated bar occupancy that triggers BAR_OCCUPANCY_HIGH.")
    parser.add_argument("--low-retention-threshold", type=float, default=5.0, help="Low retention event threshold in percent.")
    parser.add_argument("--high-retention-threshold", type=float, default=25.0, help="High retention event threshold in percent.")
    parser.add_argument("--min-passersby-for-retention-event", type=int, default=5, help="Minimum total total detected people before retention events are generated.")
    parser.add_argument("--event-cooldown-seconds", type=float, default=60.0, help="Cooldown for repeated non-instant events.")

    return parser


def _clean_notebook_argv(argv: Sequence[str]) -> List[str]:
    """Remove Jupyter kernel arguments such as -f kernel.json.

    This is necessary because options like -f can otherwise be interpreted as
    abbreviations for --flip-entry-side by argparse on Windows/Jupyter.
    """
    cleaned: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-f", "--f"):
            i += 2
            continue
        if arg.startswith("--f="):
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned


def main() -> None:
    parser = build_parser()
    running_in_notebook = "ipykernel" in sys.modules or any(
        arg in ("-f", "--f") or arg.startswith("--f=") for arg in sys.argv
    )
    if running_in_notebook:
        args, unknown = parser.parse_known_args(_clean_notebook_argv(sys.argv[1:]))
        if unknown:
            print(f"[warning] Ignored unknown notebook arguments: {unknown}")
    else:
        args = parser.parse_args()

if args.analyze:
    output_path = Path(args.output_dir)

    # If the base output folder does not contain measurements.csv,
    # automatically find the latest run_* folder.
    if not (output_path / "measurements.csv").exists():
        run_folders = [
            p for p in output_path.glob("run_*")
            if p.is_dir() and (p / "measurements.csv").exists()
        ]

        if not run_folders:
            raise FileNotFoundError(
                f"No measurements.csv found in {output_path} or in any run_* subfolder. "
                "Run live extraction first."
            )

        latest_run = max(run_folders, key=lambda p: p.stat().st_mtime)
        print(f"[info] Automatically analyzing latest run folder: {latest_run}")
        output_path = latest_run

    run_analysis(output_path, business_hours_per_day=args.business_hours_per_day)
    return

    if args.calibrate:
        calibrate_rois(args.source, args.roi_file, args.width)
        return

    run_live(args)


# In this notebook, use the cells below to calibrate, run, or analyze.


if __name__ == "__main__":
    main()
