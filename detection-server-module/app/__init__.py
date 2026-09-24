from pathlib import Path

from dotenv import load_dotenv

# Các biến đọc thẳng bằng os.getenv (YOLO_*, RAG_*, SAVE_SEGMENTATION_PREVIEW...)
# cũng lấy từ .env. Biến đã có trong môi trường tiến trình được ưu tiên.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
