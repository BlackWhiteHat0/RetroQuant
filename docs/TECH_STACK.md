# RetroQuant — 技術棧、搜尋關鍵字與官方文件

> 每一項都附上「要查的關鍵字」和官方文件連結，方便自己查、自己學。版本號只是建議，開工時用 `pip freeze` 產生的 `Requirement/requirements_lock.txt` 鎖定。

---

## 1. 一覽表

| 層級 | 技術 | 用途 | 主要使用者 |
|---|---|---|---|
| 語言 | Python 3.11 | 全專案 | 全員 |
| 環境 | **pip** + venv | 依賴管理、鎖定檔 | C 維護 |
| AI | PyTorch + torchvision | DeepLabV3-MobileNetV3-Large 推論 | A |
| 影像 | OpenCV（`opencv-python-headless`） | 讀寫、Sobel、Canny、縮放 | A、B |
| 數值 | NumPy | 所有核心運算 | 全員 |
| 加速 | Numba | 抖色的逐像素迴圈、（選用）VQ 分配 | B |
| 評估 | scikit-image | PSNR、SSIM、ΔE2000、rgb2lab | A |
| 對照 | scikit-learn | **只用於測試**：KMeans `sample_weight` 對照 | B |
| 資料 | pandas | 彙整實驗結果 CSV | A |
| 繪圖 | matplotlib | 率失真曲線、熱圖、色票色塊 | A、C |
| 展示 | Streamlit | 網頁工作室 | C |
| 標註 | LabelMe 或 CVAT | ROI 標準答案（如果需要） | A（全員分攤） |
| 品質 | ruff、pytest、pre-commit | 程式風格檢查與格式化、測試、提交前檢查 | 全員 |
| 協作 | Git + GitHub（Issues、Projects、Actions） | 版本控制、任務板、持續整合 | 全員 |
| 文件 | Markdown（本系列）、選用 MkDocs | 文件 | 全員 |

---

## 2. 逐項說明

### 2.1 Python 環境 — pip

- 全隊統一使用 Python 內建的 `venv` 建虛擬環境、用 **pip** 安裝套件；依賴清單放在 `Requirement/requirements_*.txt`，用 `pip freeze` 產生鎖定檔，確保三人環境一致。**全隊只用這一種方式**，不混用 conda 或其他工具。
- 關鍵字：`python -m venv`、`pip install -r`、`pip install -e .`、`pip freeze`、`--index-url`、requirements file、src layout
- 文件：<https://pip.pypa.io/>、<https://docs.python.org/3/library/venv.html>
- PyTorch CPU 版安裝方式（避免下載很大的 GPU 版）：<https://pytorch.org/get-started/locally/>

### 2.2 PyTorch / torchvision — 語意分割

- 只做推論、不訓練。前處理用官方權重附的 `transforms()`，確保和訓練時一致。
- 關鍵字：
  `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large`、
  `DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT`、`weights.transforms()`、`weights.meta["categories"]`、
  `model.eval()`、`torch.inference_mode()`、`torch.nn.functional.softmax`、`torch.nn.functional.interpolate`（`mode="bilinear"`）
- 文件：
  - 模型頁：<https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html>
  - 分割模型總覽：<https://pytorch.org/vision/stable/models.html#semantic-segmentation>
  - 背景論文：DeepLabV3（Chen 等人，arXiv:1706.05587）、MobileNetV3（Howard 等人，arXiv:1905.02244）
- 輸出重點：`model(x)["out"]` 的形狀是 `[N, 21, h, w]`，第 0 類是背景。

### 2.3 OpenCV — Sobel、Canny、讀寫

- `cv2.imread` 讀進來是 **BGR**，要用 `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` 轉成 RGB。Sobel 輸出深度請用 `cv2.CV_32F`，避免溢位被截斷。
- 關鍵字：`cv2.Sobel`、`cv2.magnitude`、`cv2.GaussianBlur`、`cv2.Canny`、`cv2.dilate`、`cv2.cvtColor`、`cv2.resize`（`INTER_AREA` / `INTER_LINEAR`）
- 文件：
  - Sobel 教學：<https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html>
  - Canny 教學：<https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html>
- 套件選 `opencv-python-headless`（沒有圖形介面，伺服器和 CI 比較好裝）。

### 2.4 NumPy — 核心運算與位元打包

- 關鍵字：廣播（broadcasting）、`np.einsum` 或「‖x‖² − 2x·μ + ‖μ‖²」距離技巧、`np.argmin(axis=1)`、`np.bincount(weights=...)`（快速算加權群中心！）、`np.packbits`、`np.unpackbits`、`np.random.default_rng`、`np.percentile`、`np.frombuffer`
- 文件：
  - <https://numpy.org/doc/stable/>
  - `packbits`：<https://numpy.org/doc/stable/reference/generated/numpy.packbits.html>
  - `bincount`：<https://numpy.org/doc/stable/reference/generated/numpy.bincount.html>
  - 亂數產生器：<https://numpy.org/doc/stable/reference/random/generator.html>
- 小提示：加權群中心更新可以寫成 `np.bincount(labels, weights=w*X[:,ch], minlength=K) / np.bincount(labels, weights=w, minlength=K)`，三個通道各算一次，不必對 K 跑迴圈。

### 2.5 Numba — 抖色加速

- Floyd–Steinberg 每個像素都依賴前面像素的結果，無法向量化；Numba 的 `@njit` 可以把 Python 迴圈編譯成機器碼，大約快上百倍。
- 關鍵字：`@numba.njit(cache=True)`、nopython 模式、連續記憶體陣列（`np.ascontiguousarray`）、`float32`、第一次呼叫的編譯時間（要先暖機）
- 文件：<https://numba.readthedocs.io/en/stable/user/5minguide.html>
- 背景：Floyd–Steinberg 抖色 — <https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering>

### 2.6 scikit-image — 評估指標

- 關鍵字：`skimage.metrics.peak_signal_noise_ratio`、`skimage.metrics.structural_similarity`（`channel_axis=-1, data_range=255`）、`skimage.color.rgb2lab`、`skimage.color.deltaE_ciede2000`
- 文件：
  - 指標：<https://scikit-image.org/docs/stable/api/skimage.metrics.html>
  - 色彩：<https://scikit-image.org/docs/stable/api/skimage.color.html>
- 背景論文：Sharma、Wu、Dalal（2005）CIEDE2000 實作說明；Wang 等人（2004）SSIM。

### 2.7 scikit-learn — 只在測試中當對照組

- `KMeans(n_clusters=K, init=<我們的初始點>, n_init=1).fit(X, sample_weight=w)` 可以驗證我們的加權 Lloyd 是否正確。**不可以**放進 `src/`。
- 文件：<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>
- 背景論文：Arthur 與 Vassilvitskii（2007）k-means++；Lloyd（1982）最小平方量化。

### 2.8 Streamlit — 網頁工作室

- 關鍵字：`st.file_uploader`、`st.camera_input`、`st.slider`、`st.segmented_control`、`st.tabs`、`st.columns`、`st.image`、`st.download_button`、`st.cache_resource`、`st.cache_data`、`st.session_state`、重新執行模型（每次互動整支腳本會從頭跑一次）
- 文件：
  - API 總覽：<https://docs.streamlit.io/develop/api-reference>
  - 快取：<https://docs.streamlit.io/develop/concepts/architecture/caching>
  - 執行模型：<https://docs.streamlit.io/develop/concepts/architecture/run-your-app>
- 第三方元件：`streamlit-image-comparison`（左右滑桿比較），到 PyPI 或 GitHub 搜尋這個名稱，確認還有在維護再用。

### 2.9 pandas + matplotlib — 結果分析

- 關鍵字：`DataFrame.groupby(["group","K"]).agg(["mean","std"])`、`pivot_table`（α×β 熱圖）、`matplotlib.pyplot.errorbar`、`imshow` + `colorbar`
- 文件：<https://pandas.pydata.org/docs/> ・ <https://matplotlib.org/stable/>

### 2.10 品質工具

| 工具 | 用途 | 關鍵字 | 文件 |
|---|---|---|---|
| ruff | 程式風格檢查與格式化（取代 flake8、black、isort） | `ruff check --fix`、`ruff format` | <https://docs.astral.sh/ruff/> |
| pytest | 測試 | fixture、`parametrize`、`tmp_path`、`pytest.approx` | <https://docs.pytest.org/> |
| pre-commit | 提交前自動跑 ruff | `.pre-commit-config.yaml`、`pre-commit install` | <https://pre-commit.com/> |
| GitHub Actions | PR 自動測試 | workflow、`on: pull_request`、matrix | <https://docs.github.com/actions> |
| GitHub Projects | 任務板 | 看板檢視、迭代、自訂欄位 | <https://docs.github.com/issues/planning-and-tracking-with-projects> |
| Hypothesis（選用） | 性質測試：編解碼還原 | `@given`、strategies | <https://hypothesis.readthedocs.io/> |

### 2.11 資料集

| 資料 | 用途 | 連結 | 注意 |
|---|---|---|---|
| Kodak（24 張，768×512） | 主要色彩測試集 | <https://r0k.us/graphics/kodak/> | 簡報中寫明來源和使用條款，不要籠統寫「開源」 |
| 合成影像（自製） | 可控制難度、有完美的 ROI 標準答案 | `datasets/make_synthetic.py` | 自己產生，沒有授權問題 |
| USC-SIPI（精選） | 經典測試影像 | <https://sipi.usc.edu/database/> | 很多影像版權狀態不明，公開發表前逐張確認 |
| PASCAL VOC 2012 驗證集（候選） | 有標準分割答案 → 客觀的 ROI | <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/> | 見 `DISCUSSION.md` 第 4 題 |
| ECSSD / DUTS（候選） | 顯著性標準答案 → ROI | 搜尋「ECSSD dataset」、「DUTS saliency dataset」 | 確認授權 |

### 2.12 相關研究（口試備查）

- SQ-GAN（2025）：<https://arxiv.org/abs/2502.09520>
- 找更多文獻的關鍵字：`saliency-guided color quantization`、`semantic-aware image compression`、`edge-preserving color quantization`、`weighted k-means color palette`、`ROI-based image coding`

---

## 3. 依賴檔（`Requirement/`）

| 檔案 | 內容 | 誰要裝 |
|---|---|---|
| `requirements_torch_cpu.txt` | torch、torchvision（走 PyTorch CPU 索引，要單獨安裝） | 全員 |
| `requirements_core.txt` | 核心執行期依賴 | 全員 |
| `requirements_dev.txt` | 測試與程式碼檢查 | 全員 |
| `requirements_all.txt` | core + dev 的合集 | 全員 |
| `requirements_optional.txt` | 選用與延伸 | 開會決定後才裝 |
| `requirements_tools.txt` | 獨立 GUI 工具（不裝進專案環境） | 需要時 |

```bash
python -m venv .venv && source .venv/bin/activate   # Windows：.venv\Scripts\activate
pip install -r Requirement/requirements_torch_cpu.txt
pip install -r Requirement/requirements_all.txt
pip install -e .                                     # pyproject.toml 只放套件資訊與 ruff、pytest 設定
pip freeze --exclude-editable > Requirement/requirements_lock.txt
```

> 依賴檔裡不寫死版本；環境裝好、測試通過後用 `pip freeze` 產生 `requirements_lock.txt` 並提交進 git，其他人用 `pip install -r Requirement/requirements_lock.txt` 重現同一個環境（torch 仍需先從 CPU 索引安裝）。除非遇到相容性問題，不要手動寫死版本。

---

## 4. 硬體需求

- 全部可以在 CPU 筆電上跑（模型 1,100 萬參數、10.45 GFLOPs），不需要 GPU。如果有人有 NVIDIA 顯卡，`Segmenter(device="cuda")` 可以加速實驗，但**報告裡的時間數字以 CPU 為準**，或者明確寫出使用的硬體。
- 跑實驗時把每張圖的 S 快取成 `.npy`，避免每一組消融實驗都重跑模型。
