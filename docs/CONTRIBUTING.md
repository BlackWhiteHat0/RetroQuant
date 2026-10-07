# RetroQuant — 協作規範

> 三個人、一個 repo，目標是「不互相踩到、隨時都能跑」。

---

## 1. Git 分支策略（GitHub Flow）

```
main  ──●──────●──────────●──────●───►     受保護：只能透過 PR 合併，持續整合必須通過
         \    /  \       /        \
          feat/vq-lloyd   feat/rq-codec  ...
```

- `main` 永遠可以執行。開啟**分支保護**：需要 1 人核准 + 持續整合通過。
- 分支命名：`<類型>/<簡短描述>`（分支名稱只能用英文和連字號）
  - `feat/weighted-kmeanspp`、`fix/codec-odd-width`、`exp/sensitivity-grid`、`docs/spec-roi`、`chore/ci-cache`
- 小步快跑：**一個 PR 只做一件事**，差異最好少於 400 行，分支最多存在 3～4 天。
- 合併前先 `git pull --rebase origin main`（或在 GitHub 上按「Update branch」），解完衝突再請別人審查。

文件：GitHub Flow <https://docs.github.com/get-started/using-github/github-flow> ・ 分支保護 <https://docs.github.com/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches>

---

## 2. 提交訊息格式（Conventional Commits）

```
<類型>(<範圍>): <摘要>

feat(vq): 加入加權 k-means++ 初始化
fix(codec): 處理寬×高×位元數不是 8 的倍數的情況
test(dither): 新增 gamma=1 時邊緣不擴散的測試
exp(ablation): 率失真掃描設定加入 K=128、256
docs(spec): 定義 ROI 標準答案的來源
```

類型：`feat`（新功能）、`fix`（修錯）、`test`（測試）、`refactor`（重構）、`perf`（效能）、`exp`（實驗）、`docs`（文件）、`chore`（雜項）
範圍：`seg`、`edge`、`fusion`、`vq`、`dither`、`codec`、`pipeline`、`app`、`eval`、`exp`、`ci`

文件：<https://www.conventionalcommits.org/>

---

## 3. Issue 與任務板

- 每個工作分解任務（`TEAM_PLAN.md` §3）開一個 Issue，標題帶編號：`[3.3] 加權 Lloyd 迭代`。
- 標籤：`module:*`（seg／vq／codec…）、`type:*`（功能／錯誤／實驗／文件）、`priority:high`、`blocked`、`discussion`。
- GitHub Projects 欄位：`待辦清單 → 本週要做 → 進行中 → 審查中 → 完成`。
- 每個 Issue 都要指定**負責人**和**里程碑**（M1～M6）。
- PR 描述寫 `Closes #12`，合併後會自動關閉對應的 Issue。

**Issue 範本（建議）**

```markdown
### 目標
### 完成定義
- [ ] ...
### 介面影響（無／會改 types.py → 需要另外兩人都核准）
### 參考（SPEC §x.x）
```

---

## 4. Pull Request 規則

**PR 範本**（`.github/PULL_REQUEST_TEMPLATE.md`）

```markdown
## 做了什麼
## 為什麼（Closes #__）
## 怎麼驗證
- [ ] 新增或更新單元測試
- [ ] 本機 `uv run pytest` 通過
- [ ] （如果會影響畫質）附上前後比較圖
## 有沒有改介面？
- [ ] 沒有
- [ ] 有 → 已更新 ARCHITECTURE.md，並且需要另外兩人都核准
```

**審查檢查項目**
- 是否符合 `ARCHITECTURE.md` 的函式簽名、陣列形狀、資料型別？
- 有沒有測試？邊界情況（K=2、K=256、奇數尺寸、全黑影像）有沒有測到？
- `core/` 裡有沒有讀寫檔、print、使用全域亂數？
- 命名清楚嗎？有沒有說明文件字串？
- 收到審查請求後 **24 小時內**回應（考試週可以放寬，在群組說一聲）。

---

## 5. 程式風格

- **ruff** 負責風格檢查與格式化；執行 `pre-commit install` 後，每次提交都會自動跑。
- 公開函式要寫型別提示；說明文件字串用 NumPy 格式，寫清楚陣列形狀、資料型別、數值範圍：

```text
def edge_map(img_u8, *, percentile=99.0):
    """正規化後的 Sobel 梯度強度。

    參數
    ----
    img_u8 : ndarray，形狀 (H, W, 3)，uint8，RGB
    percentile : float
        正規化用的百分位數（依 SPEC §4.2 為 P99）。

    回傳
    ----
    E : ndarray，形狀 (H, W)，float32，範圍 0～1
    """
```

- 不要提交：資料集、`.npy` 快取、`results/` 裡的大檔、模型權重、`.venv`（全部寫進 `.gitignore`）。
- 常數（例如檔頭格式）集中定義，不要到處寫寫死的數字。

---

## 6. 測試

| 層級 | 內容 | 範例 |
|---|---|---|
| 單元測試 | 每個核心函式 | `fusion`：α=β=0 時全為 1；`mean(W)≈1` |
| 性質測試 | 對任意輸入都要成立 | 編解碼還原（用 hypothesis 隨機產生 K 和尺寸）；J 單調不增 |
| 對照測試 | 和可信的實作比較 | VQ 對照 sklearn 的 `sample_weight`（AC-1） |
| 已知答案測試 | 可以手算的結果 | 4 個像素分 2 群的加權群中心；合成影像只有 K 種顏色時應該完美還原 |
| 端到端冒煙測試 | pipeline 跑一張小圖不會出錯 | `test_pipeline.py`（分割模型可以用假模組加速 CI） |

- 持續整合上**不下載**分割模型權重（太慢），改用假的分割模組；真模型的測試標記為 `@pytest.mark.slow`，在本機跑。

---

## 7. 資料與大檔案

- 資料只放在本機 `datasets/raw/`（不進 git），用 `fetch.py` 下載；`MANIFEST.md` 記錄來源、授權、檢查碼。
- ROI 遮罩（小 PNG）可以提交。
- 實驗結果：只提交 `summary.csv` 和最終圖表；完整 CSV 放在三人約定好的共用雲端資料夾。

---

## 8. 溝通管道

| 管道 | 用途 |
|---|---|
| GitHub Issue／PR | 和程式碼有關的討論（可以追溯） |
| 通訊軟體群組 | 快速問題、週報、約時間 |
| 每週會議 | 做決策、調整分工 |
| `docs/DECISIONS.md` | 所有重要決策的最終紀錄 |

> 原則：**在群組做的決定，一定要寫回 Issue 或 DECISIONS.md**，否則就等於沒有決定。
