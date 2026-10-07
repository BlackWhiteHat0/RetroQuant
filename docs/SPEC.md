# RetroQuant — 規格書

> 版本：v0.1（供小組討論的草稿）
> 本文件定義「要做什麼」以及「做到什麼程度算完成」；程式怎麼組織請看 `ARCHITECTURE.md`。

目錄
1. 研究問題與假設
2. 系統範圍
3. 功能需求
4. 演算法規格（S → E → W → VQ → 抖色）
5. `.rq` 檔案格式
6. 評估指標定義
7. 非功能需求
8. 驗收標準
9. 名詞表

---

## 1. 研究問題與假設

**問題：** 當色票數 K 固定且很小（8～64）時，要怎麼把有限的色票優先配置給「語意重要」與「結構明顯」的像素？

**假設：** 把 AI 語意圖 S 和 Sobel 邊緣圖 E 融合成權重 W，放進加權 K-Means 的目標函數，可以在**相同 K、相同 bpp** 下，穩定提升主體區域（ROI）的色彩精度與邊緣保留，而全圖品質只會小幅下降。

**要驗證的不是「有沒有用 AI」，而是：** AI 產生的重要性圖是否「穩定地」改善重要區域？這由 A～E 組的消融實驗回答，見 `EXPERIMENTS.md`。

---

## 2. 系統範圍

### 2.1 第一版必做

| 編號 | 項目 |
|---|---|
| S1 | 預訓練語意分割 → 前景機率圖 S |
| S2 | Sobel 梯度 → 正規化邊緣圖 E |
| S3 | 融合權重圖 W（α、β 可調） |
| S4 | **自行實作**加權 K-Means（含加權 k-means++ 初始化） |
| S5 | Floyd–Steinberg 誤差擴散 + 邊緣保護（可開關） |
| S6 | `.rq` 編碼器與解碼器，可算出實際 bpp |
| S7 | 評估：PSNR、ROI-PSNR、SSIM、ΔE2000、邊緣保留率、執行時間 |
| S8 | 消融實驗 A～E、K ∈ {8,16,32,64,128,256}、α/β 網格搜尋 |
| S9 | Streamlit 網頁工作室 |

### 2.2 第一版不做

- 訓練或微調大型分割模型
- 用小型模型直接預測 α、β（列為延伸）
- 宣稱比 JPEG / WebP / AVIF 這類通用壓縮更好
- 醫療影像、臨床資料
- 手機 App（Flutter 等），統一用 Streamlit

### 2.3 延伸項目（有餘力才做）

| 編號 | 項目 |
|---|---|
| X1 | 對索引做熵編碼（zlib/DEFLATE 或區間編碼），讓 bpp 隨影像內容變化 |
| X2 | 改在 Lab 色彩空間做 VQ |
| X3 | 小型 α/β 預測模型 |
| X4 | 改用不分類別的顯著性模型（如 U²-Net）當 S |
| X5 | 在相同 K 下與 PNG-8 / GIF 比較檔案大小 |

---

## 3. 功能需求

| 編號 | 需求 | 優先度 |
|---|---|---|
| FR-01 | 輸入任意尺寸的 RGB 影像（PNG/JPG），輸出同尺寸的量化影像 | 必要 |
| FR-02 | K 可選，範圍 2～256 | 必要 |
| FR-03 | α、β ≥ 0 可調；α=β=0 時必須完全退化成標準 K-Means（基準組） | 必要 |
| FR-04 | 語意、邊緣、抖色三個開關可以各自控制 | 必要 |
| FR-05 | 固定亂數種子時，結果必須逐位元相同 | 必要 |
| FR-06 | `.rq` 編碼再解碼後，必須和編碼端的量化影像完全一致 | 必要 |
| FR-07 | 回報實際 bpp = 8 × 檔案位元組數 / (H × W) | 必要 |
| FR-08 | 匯出色票（色塊 PNG + JSON/HEX 色碼） | 應該 |
| FR-09 | 可視化 S、E、W 熱圖 | 必要 |
| FR-10 | 批次實驗自動輸出 CSV 與圖表 | 必要 |
| FR-11 | 命令列工具：`encode` / `decode` / `eval` | 應該 |
| FR-12 | 工作室：上傳或拍照、前後比較、調整 K/α/β、下載 `.rq` | 必要 |

---

## 4. 演算法規格

> 這一節是三人共同的「數學合約」。實作細節可以變，但**輸入輸出和公式不能各自改**；要改請先在 `DISCUSSION.md` 提出，再走 PR。

### 4.0 記號

| 符號 | 意義 |
|---|---|
| I | 輸入影像，H×W×3，uint8 |
| xᵢ | 第 i 個像素的顏色（float32；**全專案統一用 0～1 範圍**） |
| N | 像素總數 H×W |
| K | 色票數 |
| μₖ | 第 k 個群中心（也就是色票顏色） |
| cᵢ | 像素 i 被分到的群編號 |
| Sᵢ、Eᵢ、wᵢ | 語意權重、邊緣權重、最終權重 |

### 4.1 語意圖 S

使用 torchvision 預訓練的 `deeplabv3_mobilenet_v3_large`（COCO 搭配 VOC 標籤的權重，共 21 類，含背景）。做 softmax 後取背景類（編號 0）的機率：

```
Sᵢ = 1 − P(背景 | xᵢ)        ∈ [0, 1]
```

```text
function semantic_map(I):                       # I：H×W×3 uint8
    t      = weights.transforms()(to_tensor(I))  # 官方前處理（縮放 + 標準化）
    logits = model(t.unsqueeze(0))["out"]        # 形狀 1×21×h×w（h、w 是縮放後尺寸）
    prob   = softmax(logits, dim=1)
    S_low  = 1 - prob[0, 0]                      # 背景是第 0 類
    S      = bilinear_resize(S_low, to=(H, W))   # 一定要縮放回原尺寸！
    return clip(S, 0, 1).astype(float32)         # H×W
```

注意：
- 推論時用 `model.eval()` 和 `torch.inference_mode()`；模型只載入一次（Streamlit 用 `st.cache_resource`）。
- **已知風險：** VOC 只有 20 類（人、車、貓、狗等）。風景類影像（Kodak 裡很多）的 S 可能幾乎全是 0，方法就會退化成只有邊緣。這一定要在報告中呈現並討論（見 `DISCUSSION.md` 第 3 題）。

### 4.2 邊緣圖 E

```
G  = ‖∇I‖ = sqrt(Gx² + Gy²)          # 對灰階做 Sobel（或對三通道取最大，見 DISCUSSION 第 5 題）
Eᵢ = clip( Gᵢ / (P₉₉(G) + ε), 0, 1 ) # 用第 99 百分位數正規化，ε = 1e-8
```

```text
function edge_map(I):
    gray = to_gray_float(I)                     # 0～1
    gx   = sobel(gray, dx=1, dy=0, ksize=3)
    gy   = sobel(gray, dx=0, dy=1, ksize=3)
    G    = sqrt(gx² + gy²)
    p99  = percentile(G, 99)
    return clip(G / (p99 + 1e-8), 0, 1)         # H×W float32
```

- 用 P99 而不是最大值：避免少數極端像素把其他邊緣壓得太小。
- 可選：做 Sobel 前先輕微高斯模糊（σ≈1）降雜訊，當作設定參數。

### 4.3 融合 → 權重圖 W

```
w̃ᵢ = 1 + α·Sᵢ + β·Eᵢ
wᵢ  = w̃ᵢ / mean(w̃)               # 正規化：權重平均值 = 1
```

- 「1 +」保證每個像素都有基本權重，背景不會被完全忽略。
- **正規化定義為除以平均值**（使 Σwᵢ = N），這樣不同 α、β 下目標函數 J 的大小才能比較。
- α = β = 0 ⇒ wᵢ 全是 1 ⇒ 完全等於標準 K-Means（FR-03）。

### 4.4 加權 K-Means（核心）

**目標函數**

```
J(μ, c) = Σᵢ wᵢ · ‖xᵢ − μ_{cᵢ}‖²
```

**交替最佳化（Lloyd 演算法）**

```
分配步驟：  cᵢ  ← argminₖ ‖xᵢ − μₖ‖²                 # wᵢ 對每個 k 都一樣 → 在 argmin 中抵消
更新步驟：  μₖ  ← Σ_{i∈Cₖ} wᵢ xᵢ / Σ_{i∈Cₖ} wᵢ        # 權重真正發揮作用的地方
```

**重要觀念（口試會問）：**
群中心固定時，wᵢ 對每一群都相同，所以分配步驟不會直接受權重影響；權重改變的是**加權群中心的位置**——高權重像素把群中心拉向自己，下一輪的分配才因此改變。另外**初始化**（加權 k-means++）也受權重影響，讓高權重區域更容易拿到初始群中心。我們**不宣稱**高權重區域一定能拿到專屬的色票，這要由消融實驗驗證。

**虛擬碼**

```text
function weighted_kmeans(X, w, K, max_iter=50, tol=1e-4, seed=0, n_fit=200_000):
    # X：N×3 float32，範圍 0～1；w：長度 N 的 float32，平均為 1
    rng = RNG(seed)

    # (1) 為了速度先抽樣——均勻抽樣，並保留權重
    #     （不要「依權重抽樣」又「保留權重」，那會把權重算兩次）
    idx  = rng.choice(N, size=min(N, n_fit), replace=False)
    Xf, wf = X[idx], w[idx]

    # (2) 加權 k-means++ 初始化
    μ[0] = Xf[ rng.choice(len(Xf), p = wf / sum(wf)) ]
    for k in 1..K-1:
        D2   = 到目前所有 μ[j]（j<k）的最小平方距離
        p    = wf * D2 ;  p /= sum(p)
        μ[k] = Xf[ rng.choice(len(Xf), p = p) ]

    # (3) Lloyd 迭代
    J_prev = +inf
    for t in 1..max_iter:
        c   = argmin_k ‖Xf − μ_k‖²                         # 向量化：N×K 距離矩陣，或分塊計算
        for k in 0..K-1:
            m = (c == k)
            if sum(wf[m]) == 0:  μ[k] = reseed_empty_cluster(...)   # 例如改種在最遠的高權重點
            else:                μ[k] = Σ wf[m]·Xf[m] / Σ wf[m]
        J = Σ wf · ‖Xf − μ_c‖²
        if (J_prev − J) / J_prev < tol: break
        J_prev = J

    # (4) 對「全部」像素做最後一次分配
    labels = argmin_k ‖X − μ_k‖²          # 長度 N
    return palette=μ (K×3), labels, history=[每輪的 J]
```

合約要點：
- 必須定義空群的處理方式（建議：重新種在 wᵢ·D²ᵢ 最大的像素）。
- 要回傳 `history`（每輪的 J），用來畫收斂圖和寫單元測試（J 必須單調不增）。
- 參考對照實作：`sklearn.cluster.KMeans(..., sample_weight=w)`，**只能用在測試**中比對結果，不能當核心實作。
- 輸出的色票最後要轉回 uint8（×255 後四捨五入），**之後的抖色與編碼都用 uint8 色票**，確保編碼端和解碼端一致。

### 4.5 誤差擴散 + 邊緣保護

**Floyd–Steinberg 擴散核**（掃描方向左→右、上→下；可選蛇形掃描）

```
        *    7/16
3/16  5/16   1/16
```

**邊緣保護：** 在高梯度區降低誤差擴散量，避免輪廓被雜點破壞。

```
eᵢ        = xᵢ − palette[cᵢ]                  # 量化誤差（浮點數、RGB）
e_spreadᵢ = (1 − γ · Eᵢ) · eᵢ                 # γ ∈ [0,1]；γ=0 → 一般 FS；γ=1 → 強邊緣處完全不擴散
```

```text
function dither(I_float, palette_u8, E, gamma, serpentine=False):
    P   = palette_u8 / 255
    buf = copy(I_float)                          # H×W×3
    lab = zeros(H, W, uint8)
    for y in 0..H-1:
        xs = 0..W-1；若開蛇形掃描且 y 為奇數則反向
        for x in xs:
            old       = buf[y,x]
            k         = nearest(P, old)          # argmin ‖old − P_k‖²
            lab[y,x]  = k
            err       = (old − P[k]) * (1 − gamma * E[y,x])
            把 err 依 7、3、5、1 / 16 的比例分給鄰居（蛇形時左右鏡射）
    return lab                                   # 索引圖 H×W
```

- 這是逐像素的序列迴圈，Python 會很慢，**必須用 Numba `@njit`** 加速（見 TECH_STACK）。
- 抖色只改變**索引的分佈**，不改變色票；在固定位元打包下 bpp 不變。（若做 X1 熵編碼，抖色通常會讓檔案變大，這也是可以報告的取捨。）

### 4.6 不開抖色時

`labels = argmin_k ‖xᵢ − palette_k‖²`（用 uint8 色票重新分配一次，確保和解碼端一致）。

---

## 5. `.rq` 檔案格式（v1）

### 5.1 結構

```
┌────────── 檔頭（10 位元組）──────────┬──── 色票（3K 位元組）────┬──── 打包索引（⌈N·b/8⌉ 位元組）────┐
```

| 偏移 | 大小 | 欄位 | 說明 |
|---|---|---|---|
| 0 | 2 B | `magic` | ASCII 字元 `"RQ"`（0x52 0x51） |
| 2 | 1 B | `version` | `1` |
| 3 | 1 B | `k_minus_1` | 存 K − 1（這樣 K=256 才放得進 1 個位元組） |
| 4 | 2 B | `width` | uint16，小端序 |
| 6 | 2 B | `height` | uint16，小端序 |
| 8 | 1 B | `bits` | b = ⌈log₂ K⌉（K=1 時定為 1） |
| 9 | 1 B | `flags` | 第 0 位：有無抖色；第 1 位：有無熵編碼（X1）；其餘保留為 0 |
| 10 | 3K B | `palette` | K 個 RGB uint8，依索引順序 |
| 10+3K | ⌈N·b/8⌉ B | `indices` | 逐列排列，高位元在前的位元流，最後一個位元組不足補 0 |

- 尺寸上限 65535×65535（uint16）。
- 全部用小端序；建議用 Python `struct` 的格式字串 `"<2sBBHHBB"`。

### 5.2 位元打包

```text
function pack(labels（長度 N，uint8）, b):
    if b == 8:  return labels.tobytes()
    bits = unpackbits(labels[:, None], axis=1)[:, 8-b:]    # 只留低 b 位，高位在前 → N×b
    return packbits(bits.reshape(-1))                       # 最後一個位元組自動補 0

function unpack(data, b, N):
    bits = unpackbits(frombuffer(data, uint8))[: N*b].reshape(N, b)
    pad  = zeros(N, 8-b)
    return packbits(concat([pad, bits], axis=1), axis=1).reshape(N)
```

### 5.3 大小計算

```
檔案位元組數 = 10 + 3K + ⌈N·b/8⌉
實際 bpp     = 8 · 檔案位元組數 / N
```

範例：Kodak 768×512（N = 393,216），K = 16 → b = 4
檔案位元組數 = 10 + 48 + 196,608 = 196,666 B ≈ 192 KB → 實際 bpp ≈ **4.0012**

| K | b | 只算索引時，相對 24 位元 RGB 的縮減比例 |
|---|---|---|
| 8 | 3 | 87.5% |
| 16 | 4 | 83.3% |
| 32 | 5 | 79.2% |
| 64 | 6 | 75.0% |

> **注意（口試）：** 「83.3%」只是索引部分（每像素從 24 位元變 4 位元）。實際檔案還要加檔頭和色票，所以報告一律用**實際 bpp**。

> **另一個重要觀察：** v1 是固定位元打包，**同一個 K 下所有方法的 bpp 完全相同**，率失真曲線上不同方法只會在「品質軸」上分開。若要讓 bpp 隨內容變化（真正的率失真取捨），需要 X1 熵編碼。見 `DISCUSSION.md` 第 6 題。

### 5.4 編解碼正確性要求

- 對所有 K（2～256）以及各種奇數寬高，`decode(encode(labels, palette))` 都必須逐位元還原 `(labels, palette)`。
- 魔術字、版本或長度不符時，要丟出清楚的例外。

---

## 6. 評估指標定義

全部用 **uint8 RGB 重建影像對原圖** 計算。

| 指標 | 定義 | 工具 |
|---|---|---|
| 全圖 PSNR | 10·log₁₀(255² / MSE)，全圖 RGB | `skimage.metrics.peak_signal_noise_ratio(data_range=255)` |
| ROI-PSNR | 只在 ROI 遮罩內算 MSE | 自己寫：`mse = mean((x−y)²[mask])` |
| SSIM | 結構相似度（彩色） | `skimage.metrics.structural_similarity(channel_axis=-1, data_range=255)` |
| ΔE2000 | 轉成 Lab 後算 CIEDE2000 的平均（全圖與 ROI 各一個） | `skimage.color.rgb2lab` + `deltaE_ciede2000` |
| 邊緣保留率 | 原圖與重建圖 Canny 邊緣的 F1 分數（容忍 1 像素，用膨脹實作），ROI 內與全圖各一個 | `cv2.Canny` + `cv2.dilate` |
| 實際 bpp | 8 × 檔案位元組數 / N | 自己寫 |
| 編碼／解碼時間 | 牆鐘毫秒，排除模型第一次載入；取 3 次的中位數 | `time.perf_counter` |
| 記憶體峰值 | 處理單張影像時的最高用量 | `tracemalloc` 或 `psutil` |

### 6.1 ROI 遮罩從哪裡來（非常重要）

- **不可以**用同一個模型輸出的 S 當 ROI 來評估 → 這叫循環評估（自己出題自己改）。
- ROI 必須來自**獨立的標準答案**：
  1. 有標準分割答案的公開資料（如 PASCAL VOC val 的分割遮罩，或顯著性資料集 ECSSD / DUTS 的標準答案），或
  2. 人工標註 Kodak 24 張的主體遮罩（LabelMe / CVAT），標註規則寫成文件，兩人交叉檢查。
- 合成影像的 ROI 由生成程式直接給出，是完美的標準答案。
- 決定方式見 `DISCUSSION.md` 第 4 題。

### 6.2 邊緣保留率細節

```text
function edge_f1(orig, recon, roi=None, tol=1):
    Eo = canny(gray(orig)); Er = canny(gray(recon))           # 兩邊用相同門檻
    if roi: Eo &= roi; Er &= roi
    precision = |Er ∧ dilate(Eo, tol)| / |Er|                  # 精確率
    recall    = |Eo ∧ dilate(Er, tol)| / |Eo|                  # 召回率
    return 2PR / (P + R)
```

---

## 7. 非功能需求

| 編號 | 需求 | 目標 |
|---|---|---|
| NFR-01 | 可重現：每筆結果都記錄亂數種子、資料切分、設定檔 | CSV 任一列都能用它的設定重跑出相同數字 |
| NFR-02 | 效能（CPU 筆電、768×512、K=16） | 分割 < 2 秒；VQ < 3 秒；抖色（Numba）< 1 秒；解碼 < 0.2 秒（初估，實測後修正） |
| NFR-03 | 展示反應速度 | 改 K/α/β 後 5 秒內出結果（模型輸出的 S 要快取，不重算） |
| NFR-04 | 隱私 | 上傳影像只存在記憶體，不寫入磁碟、不記錄 |
| NFR-05 | 可測試 | `core/` 主要函式都有單元測試，CI 全部通過才可以合併 |
| NFR-06 | 授權 | 資料集來源與條款寫在 `datasets/MANIFEST.md`；SIPI 影像公開發表前逐張確認 |

---

## 8. 驗收標準

### 8.1 每個模組的完成定義

- [ ] 符合 `ARCHITECTURE.md` 的函式簽名、陣列形狀、資料型別
- [ ] 有單元測試，CI 通過
- [ ] 有說明文件字串（輸入、輸出、數值範圍）
- [ ] 經另一位組員審查程式碼

### 8.2 系統層級

| 編號 | 驗收條件 |
|---|---|
| AC-1 | α=β=0、不抖色時，與 sklearn KMeans（相同初始化）的 J 相差 < 1% |
| AC-2 | 所有 K 下 `.rq` 編碼再解碼逐位元一致 |
| AC-3 | 消融 A～E × K ∈ {8,16,32,64} × 整個測試集跑完，並產出 CSV 與圖表 |
| AC-4 | α/β 在驗證集決定後鎖定，測試集只跑一次 |
| AC-5 | 報告誠實呈現結果，**包含方法沒有比較好的情況**（例如沒有 VOC 物件的影像） |
| AC-6 | 工作室完成展場流程：上傳/拍照 → 比較 → 顯示 AI 圖 → 調 K/α/β → 看 bpp → 下載 `.rq` 與色票 |

---

## 9. 名詞表

| 名詞 | 說明 |
|---|---|
| 色票 / 調色盤 / 碼本 | K 個代表色 |
| 向量量化（VQ） | 把每個像素對應到最近的代表色 |
| 加權 K-Means | 目標函數的每一項都乘上權重 wᵢ |
| 語意分割 | 對每個像素做分類 |
| 誤差擴散 / 抖色 | 把量化誤差推給周圍還沒處理的像素 |
| 位元率（bpp） | 每個像素平均用幾個位元 |
| 率失真曲線 | bpp 對品質的曲線 |
| 消融實驗 | 一次只改一個變因的比較實驗 |
| 感興趣區域（ROI） | 畫面中的主體區域 |
| 循環評估 | 用模型自己的輸出來評估模型，結論不可信 |
| 色帶（banding） | 平滑漸層被量化成明顯的一階一階 |
