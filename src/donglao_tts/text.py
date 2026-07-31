import re


def normalize_text(text: str) -> str:
    text = text.lower().strip()

    # Dấu ngắt câu -> dấu phẩy
    text = re.sub(r"\.{2,}", ",", text)      # ... -> ,
    text = re.sub(r"[;:?!]+", ",", text)
    text = re.sub(r"[–—]+", ",", text)

    # Loại bỏ dấu ngoặc, dấu nháy
    text = re.sub(r"[\"'“”‘’`´()\[\]{}<>]", "", text)

    # Chuẩn hóa dấu phẩy
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)

    # Xóa dấu phẩy ở đầu/cuối
    text = text.strip(" ,")

    return text
