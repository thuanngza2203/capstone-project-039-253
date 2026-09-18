"""Migration một lần cho corpus hiện tại, KHÔNG được import bởi chunker.

Các số dòng đã được đọc/review trên snapshot trước migration. Không nhận diện
heading bằng isupper() ở runtime. Chỉ thêm marker, giữ nguyên câu và xuống dòng.
Chạy từ RAG-module: python scripts/migrate_headings.py [--apply]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/chunking/baseline/data"

# filename: (H2, H3); H1 luôn là dòng 1, trừ scab có title được thêm phía trên.
OUTLINES = {
    "apple_black_rot": (
        "10 50 69 88 150 165", "12 27 36 90 105 122 137"),
    "apple_cedar_rust": (
        "10 21 61 86 102 120 159 180 197 216", "23 48 122 137 146"),
    "apple_scab": (
        "1 4 7 10 15 20 27 34 40 45 57 69 75 82 99 119 134", "59 64 84 90 94 101 105 109 114"),
    "cherry_powdery_mildew": (
        "14 60 83 94 101 125 175 188 209 223 238 261", "16 45 127 144 162"),
    "corn_common_rust": (
        "14 25 44 70 130 153 211 226 297 315", "72 90 99 110 119 155 170 187 228 249 263 278"),
    "corn_gray_leaf_spot": (
        "16 34 83 98 138 155 177 199 215 234 250 294 316 331 348 362 381 398",
        "36 49 68 113 122 131 252 266 277"),
    "corn_northern_leaf_blight": (
        "16 81 99 155 191 206 231 249 265 301 319 330 343 361 391 406 424",
        "18 29 45 56 114 127 138 169 182 281 290 370"),
    "grape_esca": (
        "26 37 59 77 189 203 218 236 386 401 419 436",
        "88 114 127 152 171 243 262 273 288 299 308 323 339 366"),
    "peach_bacterial_spot": (
        "22 37 84 109 143 165 185 201 221 337 356", "39 69 116 134 174 230 243 260 276 289 311 324"),
    "pepper_bell_bacterial_spot": (
        "18 35 101 116 125 142 153 167 181 192 208 241 259 286 301 314 331 342 357 368 385 401 414 436 456 473",
        "37 58 69 84 264 273"),
    "potato_earrly_blight": (
        "16 32 89 104 147 182 207 218 229 249 266 477 495 511 526",
        "34 48 61 76 115 125 156 169 278 291 306 315 328 341 358 372 390 408 423 439 463"),
    "squash_powdery_mildew": (
        "16 33 47 62 69 85 98 111 127 166 182 191 206 221 270 281 296 308 324 337 435 455 476 494 508 532",
        "129 138 153 237 246 253 260 359 368 377 389 396 408 422"),
    "strawberry_leaf_scorch": (
        "14 25 81 95 107 129 161 174 187 201 213 362 380 397 412",
        "27 44 66 139 152 228 242 253 271 285 296 306 317 333 346"),
    "toamto_mosaic_virus": (
        "16 30 129 139 153 166 178 189 199 208 220 232 244 257 273 287 294 303 317 359 447 461 479 501 518 529 545 554",
        "44 61 74 94 109 118 324 336 347 369 380 391 404 413 424 435"),
    "tomato_bacterial_spot": (
        "21 39 58 99 135 150 159 174 183 192 204 218 231 244 256 269 287 300 314 330 343 356 365 372 392 412 427 442",
        "60 78 87 104 114"),
    "tomato_early_blight": (
        "14 27 74 91 105 118 132 160 221 339 357 371 386",
        "29 49 65 141 167 183 196 208 236 248 259 273 286 298 311 325"),
    "tomato_late_blight": (
        "24 40 119 139 148 165 178 188 200 210 221 237 251 266 343 411 429 447 461 482",
        "42 62 78 91 103 152 158 281 296 305 317 330 356 369 381 396"),
    "tomato_leaf_mold": (
        "23 34 46 64 83 183 195 206 221 233 246 260 274 293 330 344 359 373 384 393 407 421 434 493 511 524 539",
        "85 99 118 130 149 162 300 319 444 451 462 478"),
    "tomato_septoria_leaf_spot": (
        "16 28 43 118 140 160 175 187 197 212 230 466 484 503 517",
        "45 58 69 80 90 105 249 260 273 284 298 308 321 335 347 358 375 401 414 427 450"),
    "tomato_spider_mites": (
        "18 72 114 160 242 294 364 388 430 517 531 542 551 569 588 603 630 644",
        "30 46 58 79 86 95 104 121 128 137 149 167 185 198 211 224 253 267 280 301 321 332 339 351 377 398 409 416 445 458 469 480 493 503"),
    "tomato_target_spot": (
        "19 30 43 62 78 163 182 227 242 266 299 347 599 619 638 656",
        "80 94 109 124 133 147 200 214 277 290 310 320 334 363 379 392 405 418 427 440 451 461 475 484 496 514 530 547 557 571 584"),
    "tomato_yello_leaf_curl_virus": (
        "18 34 43 107 160 263 281 294 306 321 334 423 440 453 471 485 497 506 521 539",
        "55 64 77 86 95 121 133 142 149 172 186 199 210 222 234 251 346 362 372 379 393 405"),
}


def migrate(apply=False):
    if not BASELINE.is_dir():
        raise RuntimeError(f"Cần snapshot trước migration tại {BASELINE}")
    report = []
    changes = []
    for path in sorted(BASELINE.rglob("*.txt")):
        relative = path.relative_to(BASELINE)
        original = path.read_text(encoding="utf-8-sig")
        lines = original.splitlines(keepends=True)
        levels = {int(n): 2 for n in OUTLINES[path.stem][0].split()}
        levels.update({int(n): 3 for n in OUTLINES[path.stem][1].split()})
        if path.stem != "apple_scab":
            levels[1] = 1
        migrated = list(lines)
        outline = []
        for number, level in sorted(levels.items()):
            heading = lines[number - 1].strip()
            if not heading or heading.startswith(("-", "#")):
                raise ValueError(f"{relative}:{number}: snapshot khác outline đã review")
            migrated[number - 1] = "#" * level + " " + lines[number - 1]
            outline.append({"line": number, "level": level, "text": heading})
        # Scab có field TÊN TÀI LIỆU. Giữ field/body nguyên vẹn, thêm H1 từ value.
        prefix = f"# {lines[1].strip()}\n\n" if path.stem == "apple_scab" else ""
        text = prefix + "".join(migrated)
        restored = text[len(prefix):].splitlines(keepends=True)
        for number, level in levels.items():
            restored[number - 1] = restored[number - 1][level + 1:]
        assert "".join(restored) == original, relative
        target = ROOT / "data" / relative
        current = target.read_text(encoding="utf-8-sig")
        if current not in (original, text):
            raise RuntimeError(f"{relative} đã sửa sau snapshot; không ghi đè.")
        changes.append((target, text))
        report.append({"source": relative.as_posix(), "headings": outline,
                       "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                       "migrated_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "body_preserved": True, "added_title": bool(prefix)})
    if apply:
        for path, text in changes:
            path.write_text(text, encoding="utf-8", newline="\n")
        (BASELINE.parent.parent / "migration.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'Applied' if apply else 'Validated'} {len(changes)} files; inverse migration preserves every character.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    migrate(parser.parse_args().apply)
