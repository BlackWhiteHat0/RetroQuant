# RetroQuant — 架構與模組介面

> 本文件是三人平行開發的「介面合約」。**函式簽名、陣列形狀、資料型別一旦在里程碑 M1 凍結，之後要改必須另外兩人同意。**

目錄
1. 分層原則
2. Repo 結構
3. 資料流與資料型別
4. 模組介面（虛擬碼簽名）
5. 設定檔
6. Pipeline 組裝
7. 工作室（展示層）架構
8. 實驗層架構
9. 假模組策略（平行開發的關鍵）

---

## 1. 分層原則

```
┌──────────────────────────────────────────────────────────┐
│  app/（Streamlit 工作室）       experiments/（實驗腳本）   │  ← 展示層／實驗層：只呼叫 pipeline，不寫演算法
├──────────────────────────────────────────────────────────┤
│  retroquant/pipeline.py   retroquant/cli.py              │  ← 組裝層：把核心模組串起來
├──────────────────────────────────────────────────────────┤
│  retroquant/core/*   retroquant/eval/*                   │  ← 核心層：純函式，NumPy 進、NumPy 出
├──────────────────────────────────────────────────────────┤
│  PyTorch/torchvision · OpenCV · NumPy · Numba · skimage  │  ← 第三方：只提供特徵與執行環境
└──────────────────────────────────────────────────────────┘
```

**規則**
1. **依賴只能往下**：`core` 不可以 import `app` 或 `experiments`。
2. **核心是純函式**：輸入 NumPy 陣列，輸出 NumPy 陣列或 dataclass；不讀檔、不 print、不畫圖。
3. **研究核心自己寫**：權重、VQ、抖色、編解碼不可以呼叫現成黑盒 API（sklearn 只能出現在 `tests/`）。
4. **torch 只出現在 `segmenter.py`**，其他模組都不 import torch，方便測試和換模型。

---

## 2. Repo 結構

```
retroquant/                         # git repo 根目錄
├─ README.md
├─ pyproject.toml                   # 依賴、ruff、pytest 設定
├─ uv.lock                          # （或 environment.yml）
├─ .pre-commit-config.yaml
├─ .github/
│  ├─ workflows/ci.yml              # 每個 PR 自動跑 lint + 測試
│  ├─ PULL_REQUEST_TEMPLATE.md
│  └─ ISSUE_TEMPLATE/
├─ docs/                            # 本系列文件
│
├─ src/retroquant/
│  ├─ __init__.py
│  ├─ types.py                      # 全員共用的 dataclass        ← M1 凍結
│  ├─ config.py                     # RQConfig + YAML 讀寫        ← M1 凍結
│  ├─ core/
│  │  ├─ segmenter.py               # [A] AI 語意圖 S
│  │  ├─ edge.py                    # [A] Sobel 邊緣圖 E
│  │  ├─ fusion.py                  # [A] W = 正規化(1 + αS + βE)
│  │  ├─ weighted_vq.py             # [B] 加權 k-means++ 與 Lloyd 迭代
│  │  ├─ dither.py                  # [B] FS 抖色 + 邊緣保護（Numba）
│  │  └─ rq_codec.py                # [C] .rq 編碼／解碼／位元打包
│  ├─ eval/
│  │  ├─ metrics.py                 # [A] PSNR、ROI-PSNR、SSIM、ΔE2000、邊緣 F1
│  │  └─ timing.py                  # [A] 計時與記憶體工具
│  ├─ pipeline.py                   # [C] run(image, cfg) -> RQResult
│  ├─ viz.py                        # [C] 熱圖、色票色塊、並排比較圖
│  └─ cli.py                        # [C] encode / decode / eval 指令
│
├─ app/
│  ├─ studio.py                     # [C] Streamlit 進入點
│  └─ components/                   # [C] 介面元件
│
├─ experiments/
│  ├─ configs/                      # 每個實驗一份 YAML
│  ├─ run_ablation.py               # [A] A～E 組 × K × 資料集
│  ├─ run_sensitivity.py            # [A] 在驗證集上做 α/β 網格搜尋
│  ├─ plot_results.py               # [A] 率失真曲線、熱圖、長條圖
│  └─ results/                      # CSV + PNG（除了總結檔以外都不進 git）
│
├─ datasets/
│  ├─ MANIFEST.md                   # 來源、授權、資料切分
│  ├─ fetch.py                      # 下載腳本
│  ├─ make_synthetic.py             # [B] 合成影像 + 標準答案遮罩
│  ├─ splits.json                   # 驗證集／測試集清單（固定不變）
│  └─ roi_masks/                    # [A] 人工標註的 ROI（若採用）
│
└─ tests/
   ├─ test_edge.py  test_fusion.py  test_weighted_vq.py
   ├─ test_dither.py  test_rq_codec.py  test_metrics.py
   └─ test_pipeline.py              # 端到端冒煙測試
```

> 採用 `src/` 結構，避免「自己電腦能跑、別人電腦不能跑」的 import 問題。

---

## 3. 資料流與資料型別

### 3.1 全專案慣例

| 項目 | 慣例 |
|---|---|
| 影像輸入輸出 | `uint8`，形狀 `(H, W, 3)`，**RGB 順序**（OpenCV 讀進來是 BGR，一定要轉！） |
| 計算用影像 | `float32`，範圍 **0～1** |
| 單通道圖 S/E/W | `float32`，形狀 `(H, W)` |
| 色票 | 運算中用 `float32 (K,3)`、範圍 0～1；輸出與編碼用 `uint8 (K,3)` |
| 索引圖 | `uint8`，形狀 `(H, W)`（K ≤ 256） |
| 亂數 | 一律用 `numpy.random.default_rng(seed)`，不要用全域的 `np.random.seed` |

### 3.2 共用 dataclass（`types.py`，M1 凍結）

```text
@dataclass(frozen=True)
class Maps:
    S: ndarray | None        # (H,W) float32；關閉語意分支時為 None
    E: ndarray | None        # (H,W) float32；關閉邊緣分支時為 None
    W: ndarray               # (H,W) float32；平均值 = 1

@dataclass(frozen=True)
class VQResult:
    palette: ndarray         # (K,3) uint8（已四捨五入）
    labels: ndarray          # (H,W) uint8
    history: list[float]     # 每輪的 J
    n_iter: int

@dataclass(frozen=True)
class RQResult:
    original: ndarray        # (H,W,3) uint8
    recon: ndarray           # (H,W,3) uint8 = palette[labels]
    maps: Maps
    vq: VQResult
    rq_bytes: bytes          # 編碼後的檔案內容
    bpp: float
    timings_ms: dict[str, float]   # {"segment":..,"edge":..,"vq":..,"dither":..,"encode":..}
    config: RQConfig
```

### 3.3 資料流

```
uint8 影像 I ──► segmenter.semantic_map ──► S ┐
             └─► edge.edge_map ────────────► E ┼─► fusion.weight_map ─► W
                                               │
I（浮點）+ W ──► weighted_vq.fit ─► 色票（uint8）、索引圖
                     │
                     └─►（開抖色？）dither.error_diffuse(I, 色票, E, γ) ─► 新索引圖
索引圖、色票 ──► rq_codec.encode ─► 位元組 ─► rq_codec.decode ─► 重建圖（檢查 == 色票[索引]）
重建圖、I、ROI ──► eval.metrics.* ─► 各項數值
```

---

## 4. 模組介面

> 只寫簽名與行為合約，實作方式由負責人決定。

### 4.1 `core/segmenter.py` — 負責人 A

```text
class Segmenter:
    def __init__(self, device: str = "cpu", weights: str = "DEFAULT"): ...
        # 只載入一次 torchvision 的 deeplabv3_mobilenet_v3_large；設為 model.eval()

    def semantic_map(self, img_u8: ndarray) -> ndarray:
        # 輸入：(H,W,3) uint8 RGB
        # 輸出：(H,W) float32，範圍 0～1，S = 1 - P(背景)
        # 必須縮放回 (H,W)；在 inference_mode 下結果固定

    def class_map(self, img_u8) -> ndarray:      # 選用，給視覺化用
        # 輸出：(H,W) uint8，每個像素機率最大的類別編號
```

### 4.2 `core/edge.py` — 負責人 A

```text
def edge_map(img_u8: ndarray, *, blur_sigma: float = 0.0,
             percentile: float = 99.0, mode: str = "gray") -> ndarray:
    # 輸出：(H,W) float32，範圍 0～1
    # mode："gray"（灰階）或 "max_channel"（三通道取最大），見 DISCUSSION 第 5 題
```

### 4.3 `core/fusion.py` — 負責人 A

```text
def weight_map(S: ndarray | None, E: ndarray | None,
               alpha: float, beta: float) -> ndarray:
    # None 當作全 0；W = (1 + αS + βE) / 平均值
    # 保證：W > 0、mean(W) == 1（誤差 1e-5 內）、alpha=beta=0 時全為 1
```

### 4.4 `core/weighted_vq.py` — 負責人 B

```text
def fit(X: ndarray, w: ndarray, K: int, *, max_iter=50, tol=1e-4,
        seed=0, n_fit=200_000, init="kmeans++") -> tuple[ndarray, list[float]]:
    # X：(N,3) float32，0～1；w：(N,) float32
    # 回傳：浮點色票 (K,3)、每輪 J 的紀錄

def assign(X: ndarray, palette_u8: ndarray, chunk=262_144) -> ndarray:
    # 每個像素找最近的色票索引，分塊計算以控制記憶體 -> (N,) uint8

def quantize(img_f: ndarray, W: ndarray, K: int, **kw) -> VQResult:
    # 便利函式：攤平 -> fit -> 色票轉 uint8 -> assign -> 還原形狀
```

記憶體提示：N=393k、K=256 時，N×K 距離矩陣大約 400 MB（float32），所以 `assign` 一定要分塊計算。

### 4.5 `core/dither.py` — 負責人 B

```text
def error_diffuse(img_f: ndarray, palette_u8: ndarray,
                  E: ndarray | None, gamma: float = 0.0,
                  serpentine: bool = False) -> ndarray:
    # 輸出：索引圖 (H,W) uint8
    # gamma=0 -> 一般 Floyd–Steinberg；E 為 None 時當作全 0
    # 內層迴圈是 @numba.njit 函式，輸入連續記憶體的 float32 陣列
```

### 4.6 `core/rq_codec.py` — 負責人 C

```text
HEADER_FMT = "<2sBBHHBB"     # 魔術字、版本、K-1、寬、高、位元數、旗標（共 10 位元組）

def encode(labels: ndarray, palette_u8: ndarray, *, flags: int = 0) -> bytes
def decode(data: bytes) -> tuple[ndarray, ndarray, dict]   # 索引圖 (H,W)、色票 (K,3)、檔頭資訊
def pack_indices(labels_flat: ndarray, bits: int) -> bytes
def unpack_indices(data: bytes, bits: int, n: int) -> ndarray
def actual_bpp(n_bytes: int, h: int, w: int) -> float
def reconstruct(labels, palette_u8) -> ndarray             # palette[labels] -> (H,W,3) uint8
```

### 4.7 `eval/metrics.py` — 負責人 A

```text
def psnr(orig_u8, recon_u8, mask=None) -> float          # mask=None 為全圖；有 mask 時為 ROI-PSNR
def ssim(orig_u8, recon_u8) -> float
def delta_e2000(orig_u8, recon_u8, mask=None) -> float   # CIEDE2000 平均值
def edge_f1(orig_u8, recon_u8, mask=None, tol=1) -> float
def all_metrics(orig_u8, recon_u8, roi_mask=None) -> dict[str, float]
```

### 4.8 `pipeline.py` — 負責人 C

```text
def run(img_u8: ndarray, cfg: RQConfig, *, segmenter: Segmenter | None = None,
        cached_S: ndarray | None = None) -> RQResult:
    # 1. S = cached_S 或 segmenter.semantic_map(img)   （cfg.use_semantic 時）
    # 2. E = edge_map(img, ...)                        （cfg.use_edge 或 cfg.dither_gamma > 0 時）
    # 3. W = weight_map(S, E, cfg.alpha, cfg.beta)
    # 4. vq = quantize(img/255, W, cfg.K, seed=cfg.seed, ...)
    # 5. labels = 開抖色時 error_diffuse(...)，否則 vq.labels
    # 6. data = encode(labels, vq.palette, flags=...)
    # 7. 除錯模式下檢查 decode(data) 能完全還原
    # 8. return RQResult(..., timings_ms=...)
```

`cached_S` 讓工作室調整 α/β/K 時不必重跑模型。

---

## 5. 設定檔

```text
@dataclass
class RQConfig:
    K: int = 16
    use_semantic: bool = True
    use_edge: bool = True
    alpha: float = 1.0
    beta: float = 0.5
    dither: bool = False
    dither_gamma: float = 0.5        # 邊緣保護強度
    serpentine: bool = False         # 蛇形掃描
    color_space: str = "rgb"         # "rgb" 或 "lab"（延伸項目 X2）
    edge_mode: str = "gray"
    edge_blur_sigma: float = 0.0
    n_fit: int = 200_000
    max_iter: int = 50
    tol: float = 1e-4
    seed: int = 0
```

- 消融實驗的每一組 = 一組 `RQConfig` 覆寫值；例如 B 組：`{use_semantic: false, use_edge: true, alpha: 0}`。
- 每筆實驗結果的 CSV 會把整份設定攤平寫入（`asdict(cfg)`），任何一列都能重跑（NFR-01）。
- YAML 範例見 `EXPERIMENTS.md`。

---

## 6. Pipeline 組裝圖

```
           cfg（RQConfig）
               │
img_u8 ──► pipeline.run ──────────────────────────────────────────────┐
               ├─► Segmenter.semantic_map  (A)  ─► S                  │
               ├─► edge_map                (A)  ─► E                  │
               ├─► weight_map              (A)  ─► W                  │
               ├─► weighted_vq.quantize    (B)  ─► 色票、索引圖        │
               ├─► dither.error_diffuse    (B)  ─► 新索引圖            │
               └─► rq_codec.encode         (C)  ─► 位元組、bpp         │
                                                                      ▼
                                                                  RQResult
         ┌────────────────────────────┬───────────────────────────────┐
         ▼                            ▼                               ▼
   app/studio.py (C)        experiments/run_*.py (A)            cli.py (C)
```

---

## 7. 工作室架構（負責人 C）

### 7.1 畫面配置

```
┌──────────────── 側欄 ──────────────────┐┌──────────────── 主畫面 ───────────────────────────┐
│ [上傳] / [拍照]                          ││  前後比較滑桿（基準組 ⟷ RetroQuant）               │
│ K： (8)(16)(32)(64)                     ││                                                   │
│ α 語意權重 ───●──── 1.0                  ││  分頁：[比較] [AI 圖 S] [邊緣 E] [權重 W]          │
│ β 邊緣權重 ──●───── 0.5                  ││        [色票] [指標]                               │
│ 抖色 [ ]；γ ───●── 0.5                   ││                                                   │
│ 預設組：[基準][邊緣][語意][完整方法]      ││  原圖 1.18 MB → .rq 192 KB · bpp 4.00             │
│                                          ││  [下載 .rq] [匯出色票] [下載 PNG]                  │
└──────────────────────────────────────────┘└───────────────────────────────────────────────────┘
```

### 7.2 關鍵技術點

| 需求 | 做法（搜尋關鍵字） |
|---|---|
| 模型只載入一次 | `@st.cache_resource def load_segmenter()` |
| 同一張圖的 S 不重算 | `@st.cache_data`，以影像雜湊值為 key 快取 S、E |
| 上傳／拍照 | `st.file_uploader`、`st.camera_input` |
| 滑桿 | `st.slider`；K 用 `st.segmented_control` 或 `st.radio` |
| 左右比較 | 第三方元件 `streamlit-image-comparison`，或用 `st.columns` 分兩欄 |
| 熱圖 | matplotlib 色彩對照表 → `st.image` |
| 下載 | `st.download_button(data=rq_bytes, file_name="out.rq")` |
| 隱私 | 全部在記憶體處理，不寫檔（NFR-04） |
| 同時算基準組 | 同一個 `pipeline.run`，設定改成 α=β=0、關抖色 |

### 7.3 展場流程（2～3 分鐘）

上傳或拍照 → 用滑桿比較基準組和 RetroQuant → 按「AI 圖」看 S → 把 K 調到 8 讓差異更明顯 → 調 α/β → 看 bpp 和檔案大小 → 下載 `.rq` 和色票。

---

## 8. 實驗層架構（負責人 A）

```text
run_ablation.py --config experiments/configs/ablation.yaml
    for image in 資料集(split):
        S_cache = segmenter.semantic_map(image)        # 每張圖只算一次，各組各 K 共用
        for group in config.groups:                    # A～E
            for K in config.K_list:
                cfg  = 基本設定 | 組別覆寫值 | {K}
                res  = pipeline.run(image, cfg, cached_S=S_cache)
                m    = all_metrics(image, res.recon, roi_mask(image))
                rows.append({**asdict(cfg), **m, "image": 編號, "group": 組名,
                             "bpp": res.bpp, **res.timings_ms, "git_sha": 目前 commit})
    寫入 CSV：results/<實驗名稱>/<時間戳>.csv
    在 CSV 旁邊存一份設定檔副本
```

- 每份結果都附 `git_sha` 和設定檔副本，可以追溯。
- `plot_results.py` 只讀 CSV、不重新運算，畫圖和跑實驗分開。

---

## 9. 假模組策略

M1 凍結介面後，每個人的**第一個 PR 就是假模組版本**，讓 pipeline 從第一週就能端到端跑起來：

| 模組 | 假模組的行為 | 由誰換成真版本 |
|---|---|---|
| `Segmenter.semantic_map` | 回傳中央橢圓 = 1、其他 = 0 的假前景 | A |
| `edge_map` | `np.zeros` | A |
| `weighted_vq.quantize` | 暫時呼叫 PIL 的 `Image.quantize(K)`（只限假模組！） | B |
| `error_diffuse` | 直接回傳最近色分配的結果 | B |
| `rq_codec` | 暫時用 `np.save` 存成位元組，bpp 照公式算 | C |

> 好處：C 可以馬上開始做工作室、A 可以馬上寫實驗腳本，不用等 B 的 VQ 完成。
