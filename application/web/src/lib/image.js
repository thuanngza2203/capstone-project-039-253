// Thu nhỏ ảnh trước khi gửi: ảnh điện thoại 3–8 MB gửi qua 4G rất chậm, trong khi model chỉ
// dùng 640 px (tách lá) và 224 px (nhận diện). Đổi luôn HEIC của iPhone sang JPEG.
const MAX_EDGE = 1600;
const QUALITY = 0.88;

async function decode(file) {
  if ("createImageBitmap" in window) {
    try {
      return await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch {
      // Trình duyệt không đọc được định dạng này bằng createImageBitmap: thử <img>.
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function shrinkImage(file) {
  const name = file.name.replace(/\.[^.]+$/, "") || "leaf";
  try {
    const source = await decode(file);
    const width = source.width;
    const height = source.height;
    const scale = Math.min(1, MAX_EDGE / Math.max(width, height));
    if (scale === 1 && file.type === "image/jpeg") {
      return { blob: file, name: file.name, width, height, shrunk: false };
    }
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", QUALITY));
    if (!blob) throw new Error("toBlob failed");
    return { blob, name: `${name}.jpg`, width: canvas.width, height: canvas.height, shrunk: true };
  } catch {
    // Không giải mã được (ví dụ HEIC trên Chrome Android): gửi nguyên file, backend tự xử lý.
    return { blob: file, name: file.name, width: null, height: null, shrunk: false };
  }
}
