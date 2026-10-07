# RetroQuant

> 語意引導之極低色數影像量化與壓縮系統

讓 AI 決定「哪些像素值得保留」，再把有限的色票資源交給加權向量量化（Weighted VQ）分配。

---

## 1. 一句話說明

給定固定色數 K（例如 16），傳統 K-Means 只看 RGB 距離，大面積背景容易搶走色票。RetroQuant 用「語意分割（AI）＋邊緣（Sobel）」產生每個像素的權重 W，讓權重影響群中心（centroid）的更新，把色票優先留給主體與輪廓；最後封裝成自訂的 `.rq` 檔，並用 Streamlit 網頁工作室展示。

---

## 2. 整體形狀（先看這張）

```
              ┌──────────┐
RGB 影像 ────►│ AI 語意   │──► S（H×W，0~1）┐
 (H×W×3)      └──────────┘                  │   ┌────────┐   ┌──────────┐   ┌────────┐   ┌──────────┐
              ┌──────────┐                  ├──►│ 融合    │──►│ 加權 VQ   │──►│ 抖色    │──►│ .rq 編碼 │──► 檔案 + 實際 bpp
         ────►│ 邊緣 Sobel│──► E（H×W，0~1）┘   │ 權重圖 W│   │ K 色色票  │   │（可選） │   │ 位元打包 │
              └──────────┘                      └────────┘   └──────────┘   └────────┘   └──────────┘
                                                                                              │
                      experiments/（消融實驗、率失真、敏感度分析） ◄──────── 指標 ◄───────────┤
                      app/（Streamlit 工作室：前後比較、AI 圖、色票、下載） ◄─────────────────┘
```

四個核心貢獻：

| 編號 | 內容 |
|---|---|
| 01 | 語意先驗：AI 分割 → 像素重要性 |
| 02 | 加權 VQ：依重要性分配色票 |
| 03 | `.rq` 編解碼器：色票 + 打包索引 + 解碼器，可算出實際 bpp |
| 04 | 互動工作室：視覺證據 + 實際 bpp + 色票匯出 |

---

## 3. 文件地圖

建議閱讀順序（由上而下）：

| 檔案 | 內容 | 誰最需要讀 |
|---|---|---|
| `README.md` | 本檔：全貌、文件地圖、快速開始 | 全員 |
| `docs/SPEC.md` | **規格**：要做什麼、演算法、`.rq` 格式、指標定義、驗收標準 | 全員 |
| `docs/ARCHITECTURE.md` | **架構**：repo 結構、模組介面（函式簽名、陣列形狀、資料型別）、設定檔 | 全員，寫程式前必讀 |
| `docs/TECH_STACK.md` | 技術棧、版本、搜尋關鍵字、官方文件連結 | 全員 |
| `docs/TEAM_PLAN.md` | **分工**、任務拆解、里程碑、每週節奏 | 全員 |
| `docs/EXPERIMENTS.md` | 消融實驗、資料集、率失真、敏感度分析流程 | 負責實驗者 + 全員 |
| `docs/CONTRIBUTING.md` | Git 流程、PR、程式風格、測試 | 全員 |
| `docs/DISCUSSION.md` | **待三人決定的問題**（第一次開會用） | 全員，開會前讀 |
| `docs/DECISIONS.md` | 決策紀錄（空白範本，開會後填） | 會議記錄者 |

---

## 4. 三人分工速覽

（詳細見 `docs/TEAM_PLAN.md`）

| 角色 | 負責的模組 | 橫向職責 |
|---|---|---|
| **A — AI 與評估** | `core/segmenter.py`、`core/edge.py`、`core/fusion.py`、`eval/metrics.py` | 資料集與 ROI 標準答案、實驗腳本 |
| **B — 核心演算法** | `core/weighted_vq.py`、`core/dither.py` | 數學正確性、效能（Numba）、單元測試 |
| **C — 編解碼與產品** | `core/rq_codec.py`、`app/`、`pipeline.py` | Repo、CI、環境、Demo、整合 |

> 原則：**一人一個主模組 + 一個橫向職責**；介面先定好，再用假模組（mock）平行開發。

---

## 5. 快速開始（規劃中的指令）

> 以下是預計的使用方式，repo 建好後再更新成實際指令。

```bash
# 1. 環境（見 TECH_STACK.md）
uv sync                      # 或：conda env create -f environment.yml

# 2. 下載資料（資料不放進 git）
python datasets/fetch.py --set kodak

# 3. 單張影像
python -m retroquant encode  input.png  -o out.rq  --K 16 --alpha 1.0 --beta 0.5 --dither off
python -m retroquant decode  out.rq     -o recon.png

# 4. 實驗
python experiments/run_ablation.py --config experiments/configs/ablation_k16.yaml

# 5. 展示
streamlit run app/studio.py
```

---

## 6. 範圍與誠實限制

- **不宣稱**首次把 AI 與 VQ 結合（已有 SQ-GAN 2025 等研究）；差異化在於「經典、可解釋的加權 VQ + 顯式的語意／邊緣權重圖 + 自訂編解碼器 + 互動展示 + 消融實驗驗證」。
- **不宣稱**取代 JPEG / PNG；`.rq` 是固定色票低色數編碼的最小可解碼封裝。
- 分割錯誤會傳到權重上；極低 K 仍可能出現色帶（banding）；效果必須用實驗量化。
- 非醫療、非臨床、無受試者介入；展示時上傳的影像只在記憶體處理，不會存檔。

---

## 7. 參考資料

1. Torchvision — DeepLabV3-MobileNetV3-Large：<https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html>
2. Pezone、Barbarossa、Caire（2025）。SQ-GAN。arXiv:2502.09520 — <https://arxiv.org/abs/2502.09520>
3. scikit-image `deltaE_ciede2000`：<https://scikit-image.org/docs/stable/api/skimage.color.html>
4. USC-SIPI 影像資料庫（版權說明）：<https://sipi.usc.edu/database/>
5. Kodak 無損彩色影像集：<https://r0k.us/graphics/kodak/>
