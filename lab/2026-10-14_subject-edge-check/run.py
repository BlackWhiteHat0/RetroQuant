"""主體偵測（DeepLabV3）+ AI 線條偵測（DexiNed）效果檢查。

每張圖輸出到 outputs/<名稱>/：
  original.jpg      原圖
  subject.jpg       主體遮罩疊在原圖上（紅色 = 模型認為是主體）
  lines_all.jpg     DexiNed 在整張圖偵測到的線條（黑線白底，越黑越確定）
  lines_target.jpg  實際要銳利化的線條：有主體 → 只留主體範圍內（含外輪廓）；沒有主體 → 全圖
  sobel.jpg         對照組：SPEC 原本的 Sobel 邊緣圖 E
並把每張圖的統計寫進 outputs/summary.json。

用法：python run.py
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from kornia.models.dexined import DexiNed
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
OUT = HERE / "outputs"

MAX_SIDE = 768            # 統一縮到長邊 768（Kodak 原本就是 768×512）
SUBJECT_THRESHOLD = 0.5   # S = 1 − P(背景) 大於這個值才算主體
MIN_SUBJECT_RATIO = 0.01  # 主體面積小於全圖 1% 視為「沒有主體」
SUBJECT_DILATE_PX = 6     # 遮罩往外擴幾個像素，讓外輪廓的線條也被包含進來
MIN_CLASS_RATIO = 0.005   # 類別面積超過 0.5% 才列在偵測結果裡


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    # DexiNed 內部會下採樣 4 次，寬高要是 16 的倍數
    nh, nw = (round(h * s / 16) * 16, round(w * s / 16) * 16)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


class Subject:
    def __init__(self) -> None:
        self.weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.model = deeplabv3_mobilenet_v3_large(weights=self.weights).eval()
        self.tf = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    @torch.inference_mode()
    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = torch.from_numpy(img).permute(2, 0, 1)
        x = self.tf(t).unsqueeze(0)
        logits = self.model(x)["out"]
        logits = torch.nn.functional.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
        prob = logits.softmax(1)[0]
        s = (1 - prob[0]).clamp(0, 1).numpy().astype(np.float32)
        cls = prob.argmax(0).numpy().astype(np.uint8)
        return s, cls


class Lines:
    def __init__(self) -> None:
        self.model = DexiNed(pretrained=True).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - self.mean) * 255.0     # 和 kornia EdgeDetectorBuilder 的前處理一致
        return torch.sigmoid(self.model(x))[0, 0].numpy().astype(np.float32)


def sobel_map(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return np.clip(g / (np.percentile(g, 99) + 1e-8), 0, 1)


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    red = np.array([230, 40, 40], np.float32)
    out[mask] = out[mask] * 0.45 + red * 0.55
    # 描出遮罩邊界
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = (255, 255, 0)
    return out.astype(np.uint8)


def ink(m: np.ndarray) -> np.ndarray:
    """0～1 的線條強度 → 黑線白底灰階圖。"""
    return (255 - np.clip(m, 0, 1) * 255).astype(np.uint8)


def save(path: Path, img: np.ndarray) -> None:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    subject, lines = Subject(), Lines()
    sources = json.loads((IMAGES / "SOURCES.json").read_text())
    summary = []

    for name in sorted(sources):
        img = load_rgb(IMAGES / name)
        stem = Path(name).stem
        d = OUT / stem
        d.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        s, cls = subject(img)
        t1 = time.perf_counter()
        line = lines(img)
        t2 = time.perf_counter()

        mask = s > SUBJECT_THRESHOLD
        ratio = float(mask.mean())
        has_subject = ratio >= MIN_SUBJECT_RATIO
        if has_subject:
            k = 2 * SUBJECT_DILATE_PX + 1
            region = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
            target = np.where(region, line, 0)
        else:
            target = line

        ids, counts = np.unique(cls, return_counts=True)
        found = [
            {"class": subject.categories[i], "ratio": round(float(c) / cls.size, 4)}
            for i, c in zip(ids, counts)
            if i != 0 and c / cls.size >= MIN_CLASS_RATIO
        ]

        save(d / "original.jpg", img)
        save(d / "subject.jpg", overlay(img, mask))
        save(d / "lines_all.jpg", ink(line))
        save(d / "lines_target.jpg", ink(target))
        save(d / "sobel.jpg", ink(sobel_map(img)))

        row = {
            "name": stem,
            "source": sources[name],
            "size": [img.shape[1], img.shape[0]],
            "has_subject": has_subject,
            "subject_ratio": round(ratio, 4),
            "classes": sorted(found, key=lambda r: -r["ratio"]),
            "line_pixels_ratio": round(float((line > 0.5).mean()), 4),
            "sec_deeplab": round(t1 - t0, 3),
            "sec_dexined": round(t2 - t1, 3),
        }
        summary.append(row)
        cls_txt = ", ".join(f"{r['class']} {r['ratio']:.0%}" for r in row["classes"]) or "—"
        print(f"{stem:12s} 主體 {ratio:6.1%}  {'有主體' if has_subject else '無主體'}  [{cls_txt}]  "
              f"deeplab {row['sec_deeplab']:.2f}s  dexined {row['sec_dexined']:.2f}s")

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    n = sum(r["has_subject"] for r in summary)
    print(f"\n共 {len(summary)} 張，有主體 {n} 張，無主體 {len(summary) - n} 張")


if __name__ == "__main__":
    main()
