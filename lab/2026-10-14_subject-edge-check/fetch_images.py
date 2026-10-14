"""下載測試圖片到 images/。

- Kodak 24 張（768×512 無損 PNG）：https://r0k.us/graphics/kodak/
- Wikimedia Commons 上含 VOC 類別物件的照片（貓、狗、馬、車、鳥、牛），授權與作者記錄在 images/SOURCES.json

用法：python fetch_images.py
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "images"
UA = "RetroQuantLab/0.1 (student project; image fetch)"

KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"

# (存檔名, Commons 檔名, VOC 類別)
COMMONS = [
    ("voc_cat.jpg", "File:Cat03.jpg", "cat"),
    ("voc_dog.jpg", "File:YellowLabradorLooking_new.jpg", "dog"),
    ("voc_horse.jpg", "File:Nokota_Horses_cropped.jpg", "horse"),
    ("voc_car.jpg", "File:2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg", "car"),
    ("voc_bird.jpg", "File:Eopsaltria_australis_-_Mogo_Campground.jpg", "bird"),
    ("voc_cow.jpg", "File:Cow_female_black_white.jpg", "cow"),
]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def commons_info(titles: list[str]) -> dict:
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiextmetadatafilter": "LicenseShortName|Artist",
        "iiurlwidth": 1024, "titles": "|".join(titles),
    })
    pages = json.loads(get("https://commons.wikimedia.org/w/api.php?" + q))["query"]["pages"]
    info = {}
    for p in pages.values():
        ii = p["imageinfo"][0]
        meta = ii["extmetadata"]
        info[p["title"]] = {
            "url": ii.get("thumburl", ii["url"]),
            "page": ii["descriptionurl"],
            "license": meta.get("LicenseShortName", {}).get("value", "?"),
            "artist": re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "?")).strip(),
        }
    return info


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sources = {}

    for i in range(1, 25):
        name = f"kodim{i:02d}.png"
        if not (OUT / name).exists():
            (OUT / name).write_bytes(get(KODAK_URL.format(i)))
        sources[name] = {"set": "kodak", "page": "https://r0k.us/graphics/kodak/"}
    print("Kodak: 24 張")

    # Commons API 回傳的標題用空白而不是底線
    info = commons_info([t for _, t, _ in COMMONS])
    for name, title, voc in COMMONS:
        meta = info[title.replace("_", " ")]
        if not (OUT / name).exists():
            (OUT / name).write_bytes(get(meta["url"]))
            time.sleep(1)  # 對 Wikimedia 客氣一點
        sources[name] = {"set": "commons", "voc_class": voc, **meta}
        print(f"{name}: {meta['license']} / {meta['artist'][:40]}")

    (OUT / "SOURCES.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
