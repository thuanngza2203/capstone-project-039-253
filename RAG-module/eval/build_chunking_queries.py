"""Gán nhãn từ dòng nguồn đã đọc, độc lập chunk ID/boundary.

Chỉ cần chạy lại khi chủ động sửa bộ nhãn. Snapshot raw không commit; JSONL kết
quả có trích đoạn và vị trí đầy đủ để dùng ngay với benchmark trong clone mới.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts/chunking/baseline/data"

# Query, file stem, evidence ranges (inclusive), required context, category.
# Các câu heldout được chốt trước khi chạy retrieval; không dùng để chỉnh chunker.
DEV = [
    ("Thối đen táo biểu hiện trên quả như thế nào?", "apple_black_rot", [(14, 18)], [], "exact_topic"),
    ("Đốm mắt ếch trên lá táo có viền và tâm màu gì?", "apple_black_rot", [(29, 31)], [], "symptom_only"),
    ("Ở Việt Nam có nên tỉa cành táo thối đen theo lịch tháng Hai Ba của Minnesota không?", "apple_black_rot", [(107, 107)], [(109, 109)], "qualifier"),
    ("Ghẻ táo bắt đầu trên lá non ra sao?", "apple_scab", [(28, 29)], [], "exact_topic"),
    ("sẹo sần sùi trên quả apple scab", "apple_scab", [(36, 38)], [], "mixed_language"),
    ("Có thể dùng danh sách Mancozeb Propineb Difenoconazole cho ghẻ táo ở mọi nước không?", "apple_scab", [(95, 95)], [(96, 97)], "qualifier"),
    ("Bào tử cedar apple rust có thể theo gió bay bao xa?", "apple_cedar_rust", [(65, 67)], [], "mixed_language"),
    ("Thuốc nêu cho táo cảnh bị gỉ sắt có dùng luôn cho táo ăn quả được không?", "apple_cedar_rust", [(184, 186)], [], "qualifier"),
    ("Phấn trắng trên lá anh đào non nhìn giống gì?", "cherry_powdery_mildew", [(20, 22)], [], "exact_topic"),
    ("Kiểm tra phấn trắng trên quả cherry non ở đâu?", "cherry_powdery_mildew", [(47, 51)], [], "mixed_language"),
    ("Gỉ sắt thông thường ở ngô nổi mụn trên mặt nào của lá?", "corn_common_rust", [(92, 96)], [], "exact_topic"),
    ("Vì sao vết đốm xám lá ngô có dạng chữ nhật?", "corn_gray_leaf_spot", [(51, 53)], [], "paraphrase"),
    ("chay la ngo phuong bac vet benh dai bao nhieu", "corn_northern_leaf_blight", [(31, 31)], [], "no_accent"),
    ("Esca làm lá nho trắng và nho đỏ đổi màu khác nhau thế nào?", "grape_esca", [(90, 94)], [], "exact_topic"),
    ("Cắt tỉa nho bị esca cần chú ý vết thương thế nào?", "grape_esca", [(275, 280)], [], "paraphrase"),
    ("Đào bị bacterial spot có thể nhầm với tổn thương do đồng không?", "peach_bacterial_spot", [(86, 90)], [], "confusable"),
    ("Bệnh ghẻ đào và đốm vi khuẩn có cùng tác nhân không?", "peach_bacterial_spot", [(145, 147)], [], "confusable"),
    ("DM = S²IR trong quản lý đốm vi khuẩn ớt là gì?", "pepper_bell_bacterial_spot", [(194, 203)], [], "formula"),
    ("Xử lý hạt ớt bằng chlorine có loại được vi khuẩn bên trong hạt không?", "pepper_bell_bacterial_spot", [(268, 270)], [], "qualifier"),
    ("Nước nóng xử lý hạt ớt có thể ảnh hưởng nảy mầm không?", "pepper_bell_bacterial_spot", [(277, 279)], [], "qualifier"),
    ("Mô bên trong củ khoai tây bị early blight có mềm ướt không?", "potato_earrly_blight", [(127, 131)], [], "negation"),
    ("Khoai tây early blight nguy cơ tăng ở bao nhiêu P-Day?", "potato_earrly_blight", [(410, 414)], [], "exact_topic"),
    ("Nấm phấn trắng trên bí có cần lá ướt liên tục không?", "squash_powdery_mildew", [(49, 51)], [], "negation"),
    ("Dùng dầu trị phấn trắng bí gần lúc dùng lưu huỳnh được không?", "squash_powdery_mildew", [(412, 415)], [], "qualifier"),
    ("Đốm ban đầu của leaf scorch trên dâu tây màu gì?", "strawberry_leaf_scorch", [(29, 34)], [], "mixed_language"),
    ("Lá dâu tây bị cháy lá nặng đổi màu giữa các đốm thế nào?", "strawberry_leaf_scorch", [(68, 73)], [], "exact_topic"),
    ("ToMV có truyền qua kéo cắt không?", "toamto_mosaic_virus", [(96, 101)], [], "exact_topic"),
    ("ELISA có thể hỗ trợ xác nhận virus khảm cà chua như thế nào?", "toamto_mosaic_virus", [(289, 291)], [], "acronym"),
    ("Dùng thuốc trừ nấm có chữa được cây đã nhiễm ToMV không?", "toamto_mosaic_virus", [(520, 524)], [], "negation"),
    ("Cà chua bị đốm vi khuẩn có nên tưới phun mưa?", "tomato_bacterial_spot", [(271, 276)], [], "exact_topic"),
    ("Quả cà chua xanh bị bacterial spot trông như thế nào?", "tomato_bacterial_spot", [(106, 111)], [], "exact_topic"),
    ("Cháy sớm cà chua xuất hiện ở lá nào trước?", "tomato_early_blight", [(31, 35)], [], "confusable"),
    ("Dưới lá cà chua bị mốc sương có lớp trắng lúc nào?", "tomato_late_blight", [(64, 66)], [], "symptom_only"),
    ("Dấu hiệu kết hợp hai mặt lá để nhận ra mốc lá cà chua?", "tomato_leaf_mold", [(185, 190)], [], "paraphrase"),
    ("Chấm đen giữa vết Septoria là gì?", "tomato_septoria_leaf_spot", [(71, 75)], [], "exact_topic"),
    ("Kiểm tra nhện đỏ bằng giấy trắng thế nào?", "tomato_spider_mites", [(269, 273)], [], "exact_topic"),
    ("Nhện đỏ hai chấm chỉ sống trên cà chua phải không?", "tomato_spider_mites", [(74, 76)], [], "multi_host"),
    ("Vòng đồng tâm của target spot có đủ phân biệt với early blight không?", "tomato_target_spot", [(111, 115)], [], "confusable"),
    ("TYLCV truyền qua loài côn trùng nào?", "tomato_yello_leaf_curl_virus", [(45, 45)], [], "acronym"),
    ("Protein V1 trong TYLCV làm gì?", "tomato_yello_leaf_curl_virus", [(123, 128)], [], "acronym"),
]

HELDOUT = [
    ("Canh tao bi thoi den loet vo mau gi?", "apple_black_rot", [(38, 40)], [], "no_accent"),
    ("Ghẻ táo cần thời gian ướt lá và nhiệt độ khoảng nào?", "apple_scab", [(71, 73)], [], "multi_fact"),
    ("Giống kháng cedar rust chịu lạnh được nêu cho vùng nào?", "apple_cedar_rust", [(163, 163)], [], "qualifier"),
    ("Lùi tưới hai tuần để chậm phấn trắng cherry có phải áp dụng ở mọi vườn?", "cherry_powdery_mildew", [(129, 133)], [], "qualifier"),
    ("Có thể dùng tháng Sáu Bảy Tám ở Mỹ làm lịch đốm xám lá ngô tại Việt Nam?", "corn_gray_leaf_spot", [(157, 159)], [(161, 161)], "qualifier"),
    ("Nhiệt độ và độ ẩm thuận lợi cho northern corn leaf blight?", "corn_northern_leaf_blight", [(157, 162)], [], "mixed_language"),
    ("Cắt ngang thân nho nghi Esca cần tìm dấu hiệu gì?", "grape_esca", [(129, 131)], [], "paraphrase"),
    ("Trên lá đào, đốm vi khuẩn mới xuất hiện có vẻ ướt hay khô?", "peach_bacterial_spot", [(43, 45)], [], "exact_topic"),
    ("Ngâm nóng hạt ớt giảm vi khuẩn bên trong được không và có rủi ro gì?", "pepper_bell_bacterial_spot", [(277, 277)], [(279, 279)], "multi_fact"),
    ("Củ khoai tây dễ nhiễm cháy sớm vào lúc nào và nấm xâm nhập qua đâu?", "potato_earrly_blight", [(106, 110)], [], "paraphrase"),
    ("Lá bí phủ trắng có chắc là phấn trắng không?", "squash_powdery_mildew", [(496, 501)], [], "negation"),
    ("Tại sao bệnh ở dâu tây được gọi là cháy lá?", "strawberry_leaf_scorch", [(68, 73)], [], "paraphrase"),
    ("Người hút thuốc chăm cà chua cần chú ý gì để hạn chế ToMV?", "toamto_mosaic_virus", [(111, 115)], [], "paraphrase"),
    ("Áp dụng ngưỡng Blitecast ở Wisconsin cho Việt Nam luôn được không?", "tomato_late_blight", [(398, 400)], [], "qualifier"),
    ("Vì sao tăng nhiệt độ đêm trong nhà kính có thể giảm mốc lá cà chua?", "tomato_leaf_mold", [(386, 390)], [], "paraphrase"),
    ("Vì sao có khi phun thuốc sâu xong nhện đỏ lại tăng?", "tomato_spider_mites", [(519, 524)], [], "paraphrase"),
    ("Bệnh xoăn vàng lá cà chua có thể xác nhận bằng xét nghiệm nào?", "tomato_yello_leaf_curl_virus", [(508, 513)], [], "paraphrase"),
    ("So sánh cách nhận ra lá táo bị thối đen và bị ghẻ.", "apple_black_rot", [(29, 31)], [], "multi_source"),
    ("Trong tài liệu có hướng dẫn chữa đạo ôn lúa không?", "", [], [], "no_answer"),
    ("Giá cà chua hôm nay tại Đà Lạt bao nhiêu?", "", [], [], "no_answer"),
]


def evidence(stem, ranges):
    if not ranges:
        return []
    path = next(RAW.rglob(stem + ".txt"))
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    result = []
    for first, last in ranges:
        quote = "".join(lines[first - 1:last]).strip()
        start = sum(map(len, lines[:first - 1]))
        assert quote and text[start:start + len(quote)] == quote
        assert text.count(quote) == 1, (stem, first, last)
        result.append({"source": path.relative_to(RAW).as_posix(), "quote": quote,
                       "raw_start_index": start, "raw_end_index": start + len(quote),
                       "raw_sha256": hashlib.sha256(text.encode()).hexdigest()})
    return result


if __name__ == "__main__":
    rows = []
    for split, queries in (("dev", DEV), ("heldout", HELDOUT)):
        for query, stem, spans, context, category in queries:
            row = {"id": f"q{len(rows) + 1:03}", "split": split, "query": query, "category": category,
                   "answerable": bool(spans), "expected_evidence": evidence(stem, spans),
                   "required_context": evidence(stem, context)}
            if category == "multi_source":
                row["expected_evidence"] += evidence("apple_scab", [(28, 29)])
            rows.append(row)
    target = ROOT / "eval/chunking_queries.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} queries ({len(DEV)} dev / {len(HELDOUT)} heldout).")
