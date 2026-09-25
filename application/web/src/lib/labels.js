// Tên tiếng Việt cho nhãn cây/bệnh. Nhận cả key chuẩn (`apple_scab`) lẫn nhãn detector
// (`Apple___Apple_scab`, `Corn_(maize)___Common_rust_`), giống app/chat/labels.py của detection.

const PLANT_VI = {
  apple: "Táo", cherry: "Anh đào", corn: "Ngô", grape: "Nho", peach: "Đào", pepper: "Ớt chuông",
  potato: "Khoai tây", strawberry: "Dâu tây", tomato: "Cà chua", squash: "Bí", orange: "Cam",
};

const DISEASE_VI = {
  apple_scab: "Ghẻ táo", black_rot: "Thối đen", cedar_apple_rust: "Gỉ sắt táo – tuyết tùng",
  powdery_mildew: "Phấn trắng", cercospora_leaf_spot: "Đốm xám lá", common_rust: "Gỉ sắt thông thường",
  northern_leaf_blight: "Cháy lá ngô phương Bắc", esca_black_measles: "Esca (sởi đen)",
  leaf_blight: "Cháy lá (Isariopsis)", bacterial_spot: "Đốm vi khuẩn", early_blight: "Cháy sớm",
  late_blight: "Mốc sương muộn", leaf_scorch: "Cháy lá", leaf_mold: "Mốc lá",
  septoria_leaf_spot: "Đốm lá Septoria", spider_mites: "Nhện đỏ hai chấm", target_spot: "Đốm mục tiêu",
  tomato_yellow_leaf_curl_virus: "Xoăn vàng lá (TYLCV)", tomato_mosaic_virus: "Khảm cà chua (ToMV)",
  healthy: "Khỏe mạnh",
};

const PLANT_KEYS = { corn_maize: "corn", maize: "corn", pepper_bell: "pepper", bell_pepper: "pepper",
  cherry_including_sour: "cherry" };
const PLANT_PREFIXES = ["cherry_including_sour", "pepper_bell", "corn_maize", "strawberry", "tomato", "potato",
  "pepper", "cherry", "squash", "apple", "grape", "peach", "corn"];
const DISEASE_KEYS = {
  cercospora_leaf_spot_gray_leaf_spot: "cercospora_leaf_spot", esca: "esca_black_measles",
  yellow_leaf_curl_virus: "tomato_yellow_leaf_curl_virus", mosaic_virus: "tomato_mosaic_virus",
  spider_mites_two_spotted_spider_mite: "spider_mites",
};

const slug = (value) => String(value).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");

export function plantKey(value) {
  if (!value) return null;
  const key = slug(value);
  return PLANT_KEYS[key] || key;
}

export function diseaseKey(value) {
  if (!value) return null;
  let key = slug(value);
  if (DISEASE_VI[key]) return key;
  const prefix = PLANT_PREFIXES.find((plant) => key.startsWith(`${plant}_`));
  if (prefix) key = key.slice(prefix.length + 1);
  return DISEASE_KEYS[key] || key;
}

const pretty = (value) => String(value).replace(/_+/g, " ").trim();

export function plantName(value) {
  if (!value) return null;
  const key = plantKey(value);
  return PLANT_VI[key] || pretty(value);
}

export function diseaseName(value) {
  if (!value) return null;
  const key = diseaseKey(value);
  return DISEASE_VI[key] || pretty(value);
}

export function subjectLabel(plant, disease) {
  return [plantName(plant), diseaseName(disease)].filter(Boolean).join(" · ");
}

export const PLANTS_VI = PLANT_VI;

// Tên tài liệu khi chưa lấy được từ RAG: "apple/apple_scab.txt" → "apple scab".
export function sourceFallback(source) {
  const file = String(source).split("/").pop().replace(/\.txt$/, "");
  return pretty(file);
}

export const ACTION_LABEL = {
  ACCEPT_QUERY: "Trả lời bằng tài liệu",
  REQUEST_IMAGE: "Yêu cầu gửi ảnh",
  ASK_CLARIFICATION: "Hỏi lại cho rõ",
  OUT_OF_SCOPE: "Ngoài phạm vi",
};

export const SCOPE_LABEL = {
  document: "Đúng tài liệu của bệnh",
  crop: "Các tài liệu của cây",
  healthy: "Cây khỏe (tài liệu của cây)",
  disease_multi_crop: "Bệnh có ở nhiều cây",
  none: "Toàn bộ kho",
  unknown_plant: "Cây lạ → toàn bộ kho",
  unsupported_disease: "Kho chưa có tài liệu",
  unknown_disease: "Không nhận ra bệnh",
  internal: "Backend groq (không dùng RAG server)",
};

export const INTENT_LABEL = {
  diagnosis: "Xác định bệnh", treatment: "Cách chữa", cause: "Nguyên nhân", prevention: "Phòng ngừa",
  general_info: "Thông tin chung", other: "Khác",
};
