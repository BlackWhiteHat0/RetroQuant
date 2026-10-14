# 2026-10-14 主體偵測與 AI 線條偵測：效果檢查

| 項目 | 內容 |
|---|---|
| 狀態 | 完成 |
| 結論 | 可行。主體與線條都偵測得到，效果和預期差不多；主體佔比過低的情況需要另外處理 |
| 結果網頁 | <https://claude.ai/artifact/Dsgv4vL7SrBqQ36HTPKAyQ>（本機：`outputs/gallery.html`） |

---

## 1. 目的

我們想做的效果是「只關心主體，把主體的線條銳利化」，規劃的流程如下：

```
影像 ─► ① 找出主體範圍 ─► ② 用 AI 找出線條 ─► ③ 把主體範圍內（含外輪廓）的線條銳利化
                                              沒有主體時 → 全圖銳利化
```

這次只驗證 **① 和 ②**：模型能不能用、找出來的主體和線條長什麼樣子。**③ 銳利化還沒有做。**

## 2. 方法

### 2.1 模型

| 步驟 | 模型 | 說明 | 權重大小 |
|---|---|---|---|
| ① 主體範圍 | DeepLabV3-MobileNetV3-Large（torchvision） | 語意分割，認得 20 種 VOC 類別（人、動物、交通工具、部分家具），其他都算背景。即 SPEC 指定的模型 | 43 MB |
| ② AI 線條 | DexiNed（kornia 移植版，BIPED 資料集訓練） | 學習式邊緣偵測，輸出每個像素「是線條」的機率 | 135 MB |
| 對照組 | Sobel | SPEC 原本的邊緣圖 E，不是 AI | — |

權重在第一次執行時自動下載到 `~/.cache/torch/hub/checkpoints/`。

### 2.2 處理規則

| 規則 | 設定 |
|---|---|
| 圖片尺寸 | 縮到長邊 768 px，寬高取 16 的倍數（DexiNed 的需求） |
| 主體像素 | S = 1 − P(背景) > 0.5 |
| 有沒有主體 | 主體像素佔全圖 ≥ 1% 算有主體，否則視為無主體 |
| 外輪廓 | 主體遮罩往外擴 6 px，讓主體邊界上的線條也算在內 |
| 要銳利化的線條 | 有主體：DexiNed 線條只保留主體範圍內的部分；無主體：保留全圖線條 |

參數都寫在 `run.py` 開頭。

### 2.3 測試圖片（30 張）

| 來源 | 張數 | 用途 |
|---|---|---|
| Kodak Lossless True Color Image Suite（<https://r0k.us/graphics/kodak/>） | 24 | 專案的主要測試集，人物、風景、建築都有 |
| Wikimedia Commons | 6 | 確保有 VOC 物件：貓、狗、馬、車、鳥、牛 |

Wikimedia 圖片的作者與授權記錄在 `images/SOURCES.json`（CC BY-SA、GFDL、公有領域）。原圖不進 git，用 `fetch_images.py` 重新下載。

## 3. 結果

### 3.1 數字

| 項目 | 結果 |
|---|---|
| 偵測到主體 | 21 / 30 張（Kodak 16 / 24，Wikimedia 6 / 6） |
| 無主體，走全圖路線 | 9 張（全部來自 Kodak） |
| 偵測到的類別 | person、boat、bird、motorbike、aeroplane、pottedplant、cat、dog、horse、car、cow |
| 速度（CPU，每張） | DeepLabV3 約 0.11 秒、DexiNed 約 0.58 秒，合計不到 1 秒 |

主體佔比低的圖片（< 15%）：

| 圖片 | 主體佔比 | 判定 | 實際內容 |
|---|---|---|---|
| kodim01、13、16、22、24 | 0.03%～0.7% | 無主體 | 零星誤判的小碎片，被 1% 門檻擋掉 |
| **kodim19** | **1.2%** | **有主體** | 燈塔頂端和救生圈的誤判碎片，剛好超過門檻 |
| kodim12 | 7.1% | 有主體 | 海灘上的兩個人，主體真的很小 |
| kodim06、09、11 | 10～11% | 有主體 | 遠處的船 |
| voc_bird | 14.2% | 有主體 | 樹枝上的鳥 |

### 3.2 觀察

1. **主體找得到，範圍準確。** 人、動物、車、船、飛機的遮罩都和物體貼合得不錯，外擴 6 px 後外輪廓也有包進來。
2. **AI 線條乾淨，明顯優於 Sobel。** DexiNed 抓到的是物體的輪廓和結構線（例如五官、窗框、機車零件、貓的鬍鬚），幾乎不受草地、水面、毛髮這類紋理影響。Sobel 會把所有紋理都當成邊緣，整張圖都是雜點（對照 voc_dog、kodim13）。
3. **把線條限制在主體內的效果符合預期。** 例如 voc_bird 只留下鳥、排除樹枝；kodim12 只留下人、排除海岸線。
4. **主體佔比過低時會出問題。** kodim19 的誤判碎片只佔 1.2%，卻被當成「有主體」，結果幾乎沒有線條會被銳利化，效果比直接全圖銳利化還差。kodim13（0.7%）也只是剛好沒有超過門檻。用「總面積 1%」當門檻不夠可靠。

## 4. 結論

這次的測試結果不錯：主體和線條都有找到，效果也和預想的差不多，**DeepLabV3 + DexiNed 的組合可以繼續往下做**。

需要處理的是**主體佔比過低**的情況。目前只看主體的總面積，零星的誤判碎片加起來超過門檻，就會被當成有主體，導致幾乎不做銳利化。

## 5. 下一步

- [ ] 改善主體佔比過低的判斷，可以比較以下做法：
  - 改看**最大連通區塊**的面積，而不是所有碎片加總
  - 先去掉太小的碎片，再算面積
  - 參考模型的信心程度（例如主體區域的平均機率）
- [ ] 用 kodim19、kodim13、kodim12 等主體佔比低的圖片驗證新規則
- [ ] 實作步驟 ③ 銳利化，比較銳利化前後的效果

## 6. 重現方式

環境：Python 3.12.14、torch 2.14.1+cpu、torchvision 0.29.1+cpu、opencv-python-headless 5.0.0、kornia 0.8.3，只用 CPU。

```bash
# 在 repo 根目錄
~/.local/python3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r Requirement/requirements_torch_cpu.txt
.venv/bin/pip install -r lab/2026-10-14_subject-edge-check/requirements.txt

cd lab/2026-10-14_subject-edge-check
../../.venv/bin/python fetch_images.py    # 下載圖片到 images/
../../.venv/bin/python run.py             # 產生 outputs/<圖名>/*.jpg 與 outputs/summary.json
../../.venv/bin/python make_gallery.py    # 整理成 outputs/gallery.html
```

| 檔案 | 內容 |
|---|---|
| `fetch_images.py` | 下載 Kodak 與 Wikimedia 測試圖片，記錄來源與授權 |
| `run.py` | 跑兩個模型，輸出每張圖的五張結果圖與 `summary.json` |
| `make_gallery.py` | 把結果整理成單檔網頁，圖片內嵌 |
| `outputs/<圖名>/` | `original`、`subject`、`lines_all`、`lines_target`、`sobel` 五張結果圖 |
