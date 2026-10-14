"""把 outputs/ 的結果圖整理成一個單檔網頁 outputs/gallery.html（圖片內嵌，可以直接分享）。

用法：python make_gallery.py   （先跑 run.py）
"""

import base64
import html
import json
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PANEL_WIDTH = 560

PANELS = [
    ("original", "原圖", "縮到長邊 768"),
    ("subject", "主體範圍", "DeepLabV3，紅色＝主體，黃線＝邊界"),
    ("lines_all", "AI 偵測到的線條（全圖）", "DexiNed，越黑越確定"),
    ("lines_target", "要銳利化的線條", "有主體→只留主體內與外輪廓；無主體→全圖"),
    ("sobel", "對照：Sobel", "SPEC 原本的邊緣圖 E"),
]

# 看過結果後的人工觀察
NOTES = {
    "kodim19": ("warn", "燈塔頂端和救生圈被零星誤判成主體（共 1.2%），剛好超過 1% 門檻，被當成「有主體」。結果幾乎沒有線條會被銳利化，比當成無主體還糟。"),
    "kodim07": ("warn", "花叢被判成 pottedplant（盆栽），窗戶不算主體。這算不算我們要的主體，需要討論。"),
    "kodim13": ("ok", "右下角石頭被誤判 0.7%，低於 1% 門檻，正確當成無主體。"),
    "kodim01": ("ok", "左下角有 0.2% 誤判，低於門檻，正確當成無主體。"),
    "kodim04": ("info", "主體佔 86%，幾乎等於全圖，結果和全圖銳利化差不多。"),
    "kodim12": ("info", "人只佔 7%，海岸線和沙灘上的線條都不會被銳利化。"),
    "kodim05": ("info", "機車和騎士都認得出來，金屬零件的細線保留得很好。"),
    "voc_bird": ("info", "樹枝不屬於 VOC 類別，被排除在外，只留下鳥。"),
    "voc_cat": ("info", "鬍鬚這種細線也偵測得到。畫面上方的紅色欄杆被包進主體範圍。"),
    "kodim14": ("info", "人和橡皮艇一起被當成主體，水花大部分被排除。"),
}


def b64(path: Path) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    img = cv2.resize(img, (PANEL_WIDTH, round(h * PANEL_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def card(row: dict) -> str:
    name = row["name"]
    has = row["has_subject"]
    note = NOTES.get(name)
    flags = ["subject" if has else "nosubject"] + (["warn"] if note and note[0] == "warn" else [])
    classes = "".join(
        f'<span class="tag">{html.escape(c["class"])} {c["ratio"]:.0%}</span>' for c in row["classes"]
    ) or '<span class="tag tag-muted">沒有認得的類別</span>'
    src = row["source"]
    if src["set"] == "kodak":
        credit = 'Kodak Lossless True Color Image Suite（<a href="https://r0k.us/graphics/kodak/">r0k.us</a>）'
    else:
        credit = (f'{html.escape(src["artist"][:60])} · {html.escape(src["license"])} · '
                  f'<a href="{html.escape(src["page"])}">Wikimedia Commons</a>')
    note_html = ""
    if note:
        note_html = f'<p class="note note-{note[0]}">{html.escape(note[1])}</p>'
    figs = "".join(
        f'<figure><img src="{b64(OUT / name / (key + ".jpg"))}" alt="{name} {label}" loading="lazy">'
        f'<figcaption>{label}</figcaption></figure>'
        for key, label, _ in PANELS
    )
    state = '<span class="state state-yes">有主體</span>' if has else '<span class="state state-no">無主體 → 全圖</span>'
    return f"""
<article class="card" data-flags="{' '.join(flags)}">
  <header class="card-head">
    <h3>{name}</h3>
    {state}
    <span class="ratio">主體面積 {row['subject_ratio']:.1%}</span>
    <div class="tags">{classes}</div>
  </header>
  {note_html}
  <div class="strip">{figs}</div>
  <p class="credit">{credit} · {row['size'][0]}×{row['size'][1]} · DeepLabV3 {row['sec_deeplab']:.2f}s · DexiNed {row['sec_dexined']:.2f}s</p>
</article>"""


def main() -> None:
    rows = json.loads((OUT / "summary.json").read_text())
    n = len(rows)
    n_sub = sum(r["has_subject"] for r in rows)
    n_warn = sum(1 for r in rows if NOTES.get(r["name"], ("",))[0] == "warn")
    t_dl = sum(r["sec_deeplab"] for r in rows) / n
    t_dx = sum(r["sec_dexined"] for r in rows) / n
    kodak = [r for r in rows if r["source"]["set"] == "kodak"]
    kodak_none = sum(not r["has_subject"] for r in kodak)

    legend = "".join(
        f'<li><b>{label}</b><span>{desc}</span></li>' for _, label, desc in PANELS
    )
    cards = "".join(card(r) for r in rows)
    page = TEMPLATE.format(
        n=n, n_sub=n_sub, n_none=n - n_sub, n_warn=n_warn, t_dl=t_dl, t_dx=t_dx,
        kodak_none=kodak_none, kodak_n=len(kodak), legend=legend, cards=cards,
    )
    (OUT / "gallery.html").write_text(page)
    print(f"寫入 {OUT / 'gallery.html'}（{len(page) / 1e6:.1f} MB）")


TEMPLATE = """<title>主體線條檢查</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=Noto+Sans+TC:wght@400;500;700&display=swap">
<style>
/* 版面：實驗紀錄本。上方摘要與觀察，下方每張圖一列五格的對照條 */
:root {{
  --paper: #f6f7f8;
  --sheet: #ffffff;
  --ink: #1d2328;
  --ink-soft: #5a6670;
  --rule: #dde2e6;
  --accent: #0f6e78;
  --mask: #d7392f;
  --ok: #2f7d4f;
  --warn: #a8550c;
  --warn-bg: #fdf1e4;
  --ok-bg: #e9f4ed;
  --info-bg: #eef3f5;
  --font-body: "IBM Plex Sans", "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif;
  --font-head: "Noto Sans TC", "IBM Plex Sans", "PingFang TC", "Microsoft JhengHei", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --paper: #14181b; --sheet: #1c2226; --ink: #e4e9ec; --ink-soft: #9aa7b0; --rule: #2e373d;
    --accent: #5cc3cc; --mask: #f07a70; --ok: #7cc79a; --warn: #f0a560;
    --warn-bg: #33261a; --ok-bg: #1b2c22; --info-bg: #1f2a30;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --paper: #14181b; --sheet: #1c2226; --ink: #e4e9ec; --ink-soft: #9aa7b0; --rule: #2e373d;
  --accent: #5cc3cc; --mask: #f07a70; --ok: #7cc79a; --warn: #f0a560;
  --warn-bg: #33261a; --ok-bg: #1b2c22; --info-bg: #1f2a30;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--paper); color: var(--ink); font-family: var(--font-body); line-height: 1.6; }}
.wrap {{ max-width: 1440px; margin: 0 auto; padding-inline: clamp(16px, 3vw, 40px); padding-block: 32px 64px; display: grid; gap: 32px; }}
a {{ color: var(--accent); }}
h1, h2, h3 {{ font-family: var(--font-head); text-wrap: balance; margin: 0; }}
h1 {{ font-size: clamp(1.6rem, 3vw, 2.2rem); font-weight: 700; }}
h2 {{ font-size: 1.15rem; font-weight: 700; }}
.eyebrow {{ font-family: var(--font-mono); font-size: .78rem; letter-spacing: .06em; color: var(--ink-soft); text-transform: uppercase; }}
.lede {{ max-width: 68ch; color: var(--ink-soft); margin: 8px 0 0; }}
.pipeline {{ font-family: var(--font-mono); font-size: .85rem; background: var(--sheet); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 16px; overflow-x: auto; white-space: pre; margin: 0; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }}
.stat {{ background: var(--sheet); padding: 14px 16px; display: grid; gap: 2px; }}
.stat b {{ font-family: var(--font-mono); font-size: 1.5rem; font-weight: 500; font-variant-numeric: tabular-nums; }}
.stat span {{ font-size: .85rem; color: var(--ink-soft); }}
.cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }}
.panel {{ display: grid; gap: 10px; align-content: start; }}
.legend {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }}
.legend li {{ display: grid; grid-template-columns: 11em 1fr; gap: 12px; font-size: .92rem; }}
.legend span {{ color: var(--ink-soft); }}
.finds {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; font-size: .95rem; }}
.finds li::marker {{ color: var(--ink-soft); }}
.good {{ color: var(--ok); font-weight: 600; }}
.bad {{ color: var(--warn); font-weight: 600; }}
.filters {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; position: sticky; top: env(safe-area-inset-top, 0px); z-index: 2; background: var(--paper); padding-block: 10px; border-bottom: 1px solid var(--rule); }}
.filters button {{ font: inherit; font-size: .9rem; border: 1px solid var(--rule); background: var(--sheet); color: var(--ink); padding: 5px 14px; border-radius: 999px; cursor: pointer; }}
.filters button[aria-pressed="true"] {{ background: var(--accent); border-color: var(--accent); color: var(--sheet); }}
.filters button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.filters .count {{ margin-left: auto; font-family: var(--font-mono); font-size: .85rem; color: var(--ink-soft); }}
.cards {{ display: grid; gap: 20px; }}
.card {{ background: var(--sheet); border: 1px solid var(--rule); border-radius: 8px; padding: 16px; display: grid; gap: 12px; }}
.card-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }}
.card-head h3 {{ font-family: var(--font-mono); font-size: 1.05rem; font-weight: 500; }}
.state {{ font-size: .8rem; font-weight: 700; padding: 2px 10px; border-radius: 999px; }}
.state-yes {{ color: var(--mask); border: 1px solid var(--mask); }}
.state-no {{ color: var(--ink-soft); border: 1px solid var(--rule); }}
.ratio {{ font-family: var(--font-mono); font-size: .85rem; color: var(--ink-soft); font-variant-numeric: tabular-nums; }}
.tags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.tag {{ font-family: var(--font-mono); font-size: .78rem; background: var(--info-bg); padding: 1px 8px; border-radius: 4px; }}
.tag-muted {{ color: var(--ink-soft); }}
.note {{ margin: 0; font-size: .92rem; padding: 8px 12px; border-radius: 4px; max-width: 90ch; }}
.note-warn {{ background: var(--warn-bg); color: var(--warn); }}
.note-ok {{ background: var(--ok-bg); color: var(--ok); }}
.note-info {{ background: var(--info-bg); }}
.strip {{ display: grid; grid-template-columns: repeat(5, minmax(180px, 1fr)); gap: 8px; overflow-x: auto; padding-bottom: 4px; }}
figure {{ margin: 0; display: grid; gap: 4px; }}
figure img {{ width: 100%; max-width: 100%; display: block; border: 1px solid var(--rule); border-radius: 3px; cursor: zoom-in; background: #fff; }}
figcaption {{ font-size: .8rem; color: var(--ink-soft); }}
.credit {{ margin: 0; font-size: .78rem; color: var(--ink-soft); font-family: var(--font-mono); overflow-wrap: anywhere; }}
.zoom {{ position: fixed; inset: 0; background: rgb(0 0 0 / .85); display: grid; place-items: center; padding: 24px; z-index: 10; cursor: zoom-out; }}
.zoom img {{ max-width: 100%; max-height: 100%; border-radius: 4px; }}
@media (max-width: 640px) {{ .legend li {{ grid-template-columns: 1fr; gap: 0; }} }}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">RetroQuant lab · 2026-10-14 · subject-edge-check</div>
    <h1>主體線條檢查</h1>
    <p class="lede">確認「找出主體 → AI 找線條 → 只把主體範圍內的線條銳利化」這條流程的前兩步實際效果。這一頁只看偵測結果，還沒有做銳利化。全部在 CPU 上執行。</p>
  </header>

  <pre class="pipeline">影像 ─► DeepLabV3（主體範圍，S = 1 − P(背景) &gt; 0.5）─┐
     └► DexiNed（AI 線條機率 0～1）──────────────────┴─► 主體面積 ≥ 1%：只留主體內＋外擴 6px 的線條
                                                          主體面積 &lt; 1%：視為無主體，保留全圖線條</pre>

  <section class="stats" aria-label="摘要">
    <div class="stat"><b>{n}</b><span>測試圖片（Kodak 24 + Wikimedia 6）</span></div>
    <div class="stat"><b>{n_sub}</b><span>偵測到主體</span></div>
    <div class="stat"><b>{n_none}</b><span>無主體 → 全圖描邊</span></div>
    <div class="stat"><b>{kodak_none}/{kodak_n}</b><span>Kodak 中無主體的張數</span></div>
    <div class="stat"><b>{t_dl:.2f}s</b><span>DeepLabV3 平均每張</span></div>
    <div class="stat"><b>{t_dx:.2f}s</b><span>DexiNed 平均每張</span></div>
  </section>

  <div class="cols">
    <section class="panel">
      <h2>每一列的五張圖</h2>
      <ul class="legend">{legend}</ul>
      <p class="lede" style="margin:0">點圖片可以放大看。</p>
    </section>
    <section class="panel">
      <h2>初步觀察</h2>
      <ul class="finds">
        <li><span class="good">AI 線條很乾淨。</span>DexiNed 幾乎不理會草地、水面、毛髮這類紋理，只留下物體的輪廓和結構線；Sobel 則是整張圖都是雜點（看 voc_dog、kodim13）。</li>
        <li><span class="good">主體範圍準確。</span>VOC 類別（人、狗、貓、鳥、車、飛機、船）的遮罩貼合度都不錯，外輪廓也有被包進來。</li>
        <li><span class="bad">零星誤判會出問題。</span>kodim19 的誤判碎片加起來剛好超過 1%，被當成有主體，結果幾乎什麼都不會被銳利化。門檻規則要再設計（例如看最大連通區塊，而不是總面積）。</li>
        <li><span class="bad">「主體」的定義要討論。</span>kodim07 的花叢被當成盆栽、kodim04 主體佔 86% 等於全圖。</li>
        <li><span class="bad">DexiNed 的小毛病。</span>線條偏粗（2～4 像素），影像四周常出現假的邊框線（kodim01、kodim04），銳利化前需要處理。</li>
        <li>Kodak 有 {kodak_none} 張找不到主體，會走「全圖銳利化」的路線，和之前預期的風險一致。</li>
      </ul>
    </section>
  </div>

  <nav class="filters" aria-label="篩選">
    <button type="button" id="f-all" data-filter="all" aria-pressed="true">全部</button>
    <button type="button" id="f-subject" data-filter="subject" aria-pressed="false">有主體</button>
    <button type="button" id="f-nosubject" data-filter="nosubject" aria-pressed="false">無主體</button>
    <button type="button" id="f-warn" data-filter="warn" aria-pressed="false">需要注意（{n_warn}）</button>
    <span class="count" id="count">{n} 張</span>
  </nav>

  <section class="cards" id="cards">{cards}</section>
</div>

<script>
(() => {{
  const buttons = document.querySelectorAll(".filters button");
  const cards = document.querySelectorAll(".card");
  const count = document.getElementById("count");
  buttons.forEach(b => b.addEventListener("click", () => {{
    const f = b.dataset.filter;
    buttons.forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    let n = 0;
    cards.forEach(c => {{
      const show = f === "all" || c.dataset.flags.split(" ").includes(f);
      c.hidden = !show;
      if (show) n++;
    }});
    count.textContent = n + " 張";
  }}));
  document.getElementById("cards").addEventListener("click", e => {{
    if (e.target.tagName !== "IMG") return;
    const z = document.createElement("div");
    z.className = "zoom";
    z.innerHTML = `<img src="${{e.target.src}}" alt="${{e.target.alt}}">`;
    z.addEventListener("click", () => z.remove());
    document.body.appendChild(z);
  }});
  document.addEventListener("keydown", e => {{
    if (e.key === "Escape") document.querySelectorAll(".zoom").forEach(z => z.remove());
  }});
}})();
</script>
"""

if __name__ == "__main__":
    main()
