# Đồ án CS115 — B-cos Alignment for Inherently Interpretable CNNs and Vision Transformers

> Bài báo chính: M. Böhle, N. Singh, M. Fritz, B. Schiele, *"B-cos Alignment for Inherently Interpretable CNNs and Vision Transformers"*, IEEE TPAMI 2024 — [arXiv 2306.10898](https://arxiv.org/abs/2306.10898) — mã nguồn: <https://github.com/B-cos/B-cos-v2>

README này hướng dẫn **cách dùng tài nguyên đã tải**, **lộ trình đọc**, **gợi ý hướng làm slide** (chỉ là khung, không phải nội dung), **phân công cho nhóm 3 người** và **ý tưởng demo**.

> ⚠️ **Nhắc lại quy định:** Slide phải do nhóm **tự xây dựng hoàn toàn** — lời giải thích, câu chữ, hình vẽ, lưu đồ đều không được dùng AI tạo ra. README này chỉ chỉ đường đến tài liệu gốc và nêu câu hỏi để nhóm tự trả lời. **Demo** thì được phép dùng AI hỗ trợ lập trình.

---

## 1. Cấu trúc thư mục

```
CS115_PROJECT/
├── README.md                       ← file này
├── papers/
│   ├── 01_chinh/                   ← 2 bài báo cốt lõi (BẮT BUỘC đọc)
│   ├── 02_tien_than/               ← CoDA-Nets: ý tưởng "dynamic alignment" có trước B-cos
│   ├── 03_mo_rong/                 ← Các hướng phát triển sau B-cos
│   ├── 04_so_sanh_attribution/     ← Các phương pháp giải thích post-hoc để so sánh
│   └── 05_kien_truc_nen/           ← ResNet, ViT (kiến trúc mà B-cos thay thế lớp tuyến tính)
├── code/
│   ├── B-cos-v2/                   ← Repo chính thức (bản TPAMI) — dùng cho demo
│   └── B-cos-v1/                   ← Repo bản CVPR 2022 — có notebook toy dataset (Hình 2)
└── weights/                        ← Trọng số pretrained ImageNet (để chạy demo khi không có mạng)
```

## 2. Danh mục tài liệu

| Thư mục | File | Vai trò trong đồ án |
|---|---|---|
| `01_chinh` | `2306.10898_Bcos_Alignment_CNNs_ViTs_TPAMI2024.pdf` | **Bài báo đề tài.** Toàn bộ nền tảng toán nằm ở **Section 3** (trang 3–7) |
| `01_chinh` | `2205.10268_Bcos_Networks_CVPR2022.pdf` | Bản hội nghị đầu tiên, ngắn gọn hơn — đọc trước để nắm ý chính |
| `02_tien_than` | `2104.00032_CoDA_Nets_CVPR2021.pdf`, `2109.13004_CoDA_Nets_TPAMI.pdf` | Cùng nhóm tác giả; giới thiệu khái niệm *dynamic linear* & *alignment*; nguồn gốc của **grid pointing game** |
| `03_mo_rong` | `2411.00715_Bcosification_NeurIPS2024.pdf` | Chuyển một mạng pretrained thông thường thành B-cos (hướng phát triển) |
| `03_mo_rong` | `2301.08669_Holistically_Explainable_ViT.pdf` | B-cos cho Vision Transformer (cùng nhóm tác giả) |
| `04_so_sanh_attribution` | Saliency (Simonyan), Grad-CAM, Integrated Gradients, DeepLIFT | Các phương pháp **post-hoc** mà B-cos đem ra so sánh trong phần thực nghiệm |
| `04_so_sanh_attribution` | `1810.03292_Sanity_Checks_Saliency.pdf` | Chỉ ra lời giải thích post-hoc có thể "không trung thực" → động lực cho mô hình *inherently interpretable* |
| `05_kien_truc_nen` | ResNet, ViT | Kiến thức nền về kiến trúc |
| `code/B-cos-v1` | `CVPR2022_Poster.pdf` | Poster của tác giả — tham khảo cách tác giả tóm tắt (**không sao chép hình**) |

**Trọng số trong `weights/`** (lấy từ release chính thức của repo):

| File | Mô hình | Top-1 ImageNet |
|---|---|---|
| `resnet_18-68b4160fff.pth` | B-cos ResNet-18 | 68.7% |
| `densenet_121-b8daf96afb.pth` | B-cos DenseNet-121 | 73.6% |
| `bcos_simple_vit_ti_patch16_224-4b0824b1c1.pth` | B-cos ViT-Tiny | 60.0% |

Tài nguyên trực tuyến (không tải được về máy):
- Demo web của tác giả (Hugging Face): <https://huggingface.co/spaces/nps1ngh/B-cos>
- Notebook Colab của tác giả: <https://colab.research.google.com/drive/1bdc1zdIVvv7XUJj8B8Toe6VMPYAsIT9w>

---

## 3. Lộ trình đọc gợi ý

1. **Ngày 1–2 (cả nhóm):** Đọc lướt bài CVPR 2022 → nắm được "B-cos là gì, tại sao giải thích được".
2. **Ngày 3–6 (cả nhóm, kỹ):** Đọc **Section 3** của bài TPAMI, chép tay lại từng phương trình (1)→(30), tự kiểm chứng từng bước biến đổi.
3. **Song song (theo phân công):** Mỗi người đọc thêm tài liệu phụ cho phần mình phụ trách (xem mục 5).
4. **Section 4–5 (thực nghiệm):** Chỉ đọc đủ để hiểu *grid pointing game* và các bảng/hình chính — đây là phần **bổ trợ** theo quy định.
5. **Phụ lục (Supplement)** trong cùng file PDF: Sec. D có **chứng minh Eq. (3) ⇔ Eq. (4)** và lý do mã hóa ảnh 6 kênh — rất đáng đưa vào phần toán.

---

## 4. Bản đồ toán học của bài báo (để định hướng, không phải nội dung slide)

Bảng dưới chỉ ra **phương trình nào nằm ở đâu** và **câu hỏi nhóm cần tự trả lời** bằng lời của mình. Số phương trình theo bản arXiv `2306.10898v2`.

| # | Chủ đề | Phương trình | Câu hỏi để nhóm tự trả lời |
|---|---|---|---|
| A | Lớp tuyến tính viết theo cosine | (1), (2) | Vì sao tích vô hướng tách được thành độ dài × cosine? Đại lượng nào quyết định "độ khớp"? |
| B | Phép biến đổi B-cos | (3), (4) | Chứng minh (3) ⇔ (4) (gợi ý: Supplement Sec. D). Khi B = 1 thì sao? Vai trò của `sgn`? |
| C | Chặn trên của đầu ra | (5), (6) | Dấu "=" xảy ra khi nào? Khi B tăng, "cửa sổ phản hồi" thay đổi thế nào? |
| D | Mạng B-cos & dạng ma trận | (7)–(10) | Vì sao không cần hàm kích hoạt ReLU nữa? |
| E | **Tuyến tính động (dynamic linear)** | (11)–(15) | Tại sao tích các ma trận phụ thuộc đầu vào vẫn cho ra **một** ma trận W₁→L(x)? |
| F | Bản đồ đóng góp (contribution map) | (16) | Tổng đóng góp có bằng logit không (tính *đầy đủ — completeness*)? |
| G | Áp lực căn chỉnh (alignment pressure) | (17) + lập luận Sec. 3.2.2 | Vì sao chặn trên + BCE ⇒ trọng số buộc phải "căn chỉnh" với đầu vào? Vai trò của bias b = log(1/(C−1))? |
| H | MaxOut | (18) | MaxOut vẫn giữ tính tuyến tính động như thế nào? |
| I | Chuẩn hóa (Norm layers) | (19)–(27) | Vì sao phải bỏ β (bias) và bỏ trừ trung bình ở BatchNorm? Liên hệ với completeness |
| J | Attention là tuyến tính động | (28)–(30) | Viết lại softmax(·)·V·X thành W(X)·X |
| K | Mã hóa ảnh `[r,g,b,1−r,1−g,1−b]` | Sec. 4 + Supplement D.2 | Điều gì xảy ra với pixel đen (x = 0) nếu chỉ dùng 3 kênh? |

**Hình minh họa nên tự vẽ** (bằng matplotlib / GeoGebra / TikZ / draw.io — do nhóm tự thiết kế):
- Vector **w** và **x** trong 2D, góc giữa chúng, và đồ thị |cos θ|^B theo θ với nhiều giá trị B (cho mục B, C).
- Sơ đồ chuỗi các lớp → gộp thành một ma trận W(x) (mục E).
- Bài toán phân loại 2D tương tự **Hình 2** của bài báo — tự sinh dữ liệu và tự vẽ (xem Demo 1).
- So sánh bản đồ giải thích B-cos vs Grad-CAM vs Integrated Gradients trên cùng ảnh (xuất từ demo của nhóm).

---

## 5. Gợi ý hướng làm slide

> Đây là **khung mục lục** gợi ý. Nội dung, câu chữ, hình ảnh trong từng slide nhóm phải tự viết/vẽ.

| Phần | Gợi ý nội dung | Tỷ trọng |
|---|---|---|
| 1. Đặt vấn đề | Vì sao cần giải thích mô hình? Hạn chế của phương pháp post-hoc (dùng *Sanity Checks*) | ~10% |
| 2. Kiến thức nền | Tích vô hướng & cosine, ánh xạ tuyến tính / affine, gradient × input, tính *completeness* | ~15% |
| 3. **Phép biến đổi B-cos** | Mục A, B, C ở bảng trên — trọng tâm, có chứng minh | ~20% |
| 4. **Tuyến tính động & giải thích** | Mục D, E, F, K | ~20% |
| 5. **Vì sao trọng số căn chỉnh** | Mục G, H | ~10% |
| 6. Mở rộng sang kiến trúc hiện đại | Mục I, J (Norm, ViT) | ~10% |
| 7. Thực nghiệm (bổ trợ) | Grid pointing game, đánh đổi accuracy ↔ interpretability (tóm tắt ngắn) | ~5% |
| 8. Demo + Kết luận | Chạy demo trực tiếp; hạn chế; hướng phát triển (B-cosification) | ~10% |

Mẹo:
- Mỗi công thức quan trọng nên đi kèm **một hình** và **một ví dụ số nhỏ** (ví dụ x, w ∈ ℝ² cụ thể, tính tay B-cos với B = 1, 2, 3).
- Thống nhất ký hiệu toàn bộ slide (ŵ, ‖x‖, c(x, w), W₁→L(x)) theo đúng bài báo.
- Có một slide "bảng ký hiệu" ở đầu.

---

## 6. Phân công gợi ý cho nhóm 3 người

| Thành viên | Phần lý thuyết (slide) | Phần demo | Tài liệu đọc thêm |
|---|---|---|---|
| **Người 1** | Phần 1, 2, 3 (động lực, nền tảng, phép biến đổi B-cos, chứng minh (3)⇔(4), chặn trên) | **Demo 1** (neuron B-cos 2D tương tác) | CVPR 2022, Supplement Sec. D, *Sanity Checks* |
| **Người 2** | Phần 4, 5 (tuyến tính động, contribution map, alignment pressure, MaxOut, mã hóa 6 kênh) | **Demo 2** (kiểm chứng completeness) + **Demo 4** (train MLP nhỏ với nhiều B) | CoDA-Nets, Integrated Gradients, DeepLIFT |
| **Người 3** | Phần 6, 7, 8 (Norm, ViT, thực nghiệm, kết luận) | **Demo 3** (giải thích ảnh thật + so sánh Grad-CAM) — tích hợp tất cả demo vào một app | ViT, Holistically Explainable ViT, B-cosification, Grad-CAM |

Việc chung: cả nhóm đọc kỹ Section 3; mỗi tuần họp 1 lần để **mỗi người trình bày thử phần của mình cho hai người còn lại** (vừa kiểm tra hiểu bài, vừa thống nhất ký hiệu).

### Lịch trình mẫu (4 tuần — điều chỉnh theo deadline thật)

| Tuần | Mục tiêu |
|---|---|
| 1 | Đọc bài báo, chép lại & tự chứng minh các phương trình, chốt dàn ý slide |
| 2 | Mỗi người viết slide phần mình + bắt đầu demo |
| 3 | Hoàn thiện demo, xuất hình từ demo vào slide, ghép slide, thống nhất giao diện |
| 4 | Tập thuyết trình, chuẩn bị câu hỏi phản biện, rà soát checklist (mục 9) |

---

## 7. Ý tưởng demo (tập trung vào khía cạnh toán học)

| Demo | Minh họa điều gì | Gợi ý cách làm | Độ khó |
|---|---|---|---|
| **1. Neuron B-cos 2D** | Eq. (3)–(6): cửa sổ phản hồi hẹp dần khi B tăng; trọng số tối ưu thành "prototype góc" | Thanh trượt B và góc của w; tô màu mặt phẳng theo giá trị B-cos(x; w). Có thể tham khảo `code/B-cos-v1/Toy dataset (Figure 2).ipynb` | Dễ |
| **2. Kiểm chứng tuyến tính động** | Eq. (14), (16): `logit == sum(W(x) ⊙ x)` với B-cos; với ResNet thường thì **không bằng** (do bias) | Dùng `model.explanation_mode()` rồi lấy gradient theo đầu vào; in ra hai con số để so sánh | Trung bình |
| **3. Giải thích ảnh thật** | Hình 1 của bài báo; so sánh với Grad-CAM / IG | Dùng trọng số trong `weights/`, `model.explain(img)`; thư viện `captum` cho Grad-CAM/IG; giao diện Gradio hoặc Streamlit | Trung bình |
| **4. Áp lực căn chỉnh** | Sec. 3.2.2: B càng lớn → trọng số càng khớp với dữ liệu | Train MLP B-cos nhỏ trên MNIST với B ∈ {1, 1.5, 2, 2.5}; vẽ các hàng của W(x) dưới dạng ảnh 28×28 | Trung bình |
| 5. (Tùy chọn) Mini grid pointing game | Cách đánh giá định lượng | Ghép 2×2 ảnh khác lớp, đo % đóng góp dương rơi đúng ô | Khó |

### Cài đặt môi trường

Máy hiện đã có Python 3.13, PyTorch 2.10 (CUDA) và `einops`. Cần thêm:

```bash
pip install matplotlib pillow gradio captum
# (tùy chọn) cài gói bcos chính thức thay cho việc thêm đường dẫn repo:
pip install bcos
```

### Nạp mô hình B-cos từ trọng số cục bộ (không cần mạng)

```python
import sys, torch
from PIL import Image
sys.path.insert(0, "code/B-cos-v2")          # hoặc: pip install bcos
from bcos.models.pretrained import resnet18   # simple_vit_ti_patch16_224, densenet121 ...

model = resnet18(pretrained=False)
model.load_state_dict(torch.load("weights/resnet_18-68b4160fff.pth", map_location="cpu"))
model.eval()

img = model.transform(Image.open("cat.jpg").convert("RGB"))  # tự sinh 6 kênh [r,g,b,1-r,1-g,1-b]
img = img[None].requires_grad_()

out = model.explain(img)
print("Lớp dự đoán:", out["prediction"])
# out["explanation"]       -> ảnh RGBA của hàng W(x) tương ứng lớp dự đoán
# out["contribution_map"]  -> bản đồ đóng góp theo Eq. (16)
```

> Đoạn code trên được viết dựa trên mã nguồn (`bcos/models/pretrained.py`, `bcos/common.py`) nhưng **chưa được chạy thử** trên máy này — nhóm cần chạy kiểm tra trước khi dùng.

### Vị trí code ứng với công thức

| Công thức | File |
|---|---|
| B-cos (Eq. 4), chuẩn hóa ŵ, `|cos|^(B−1)`, MaxOut | `code/B-cos-v2/bcos/modules/bcosconv2d.py`, `bcoslinear.py` |
| Chuẩn hóa không bias / không trừ trung bình (Eq. 27) | `code/B-cos-v2/bcos/modules/norms/` |
| `explain()`, `explanation_mode`, contribution map | `code/B-cos-v2/bcos/common.py` |
| Bản cài đặt tối giản 1 file (dễ đọc nhất) | `code/B-cos-v2/extra/minimal_bcos_resnet.py` |
| ViT B-cos (Eq. 28–30) | `code/B-cos-v2/bcos/models/vit.py` |

Lưu ý: trong `explanation_mode`, hệ số động `|cos|^(B−1)` được **detach** (coi như hằng số khi lấy gradient), nhờ vậy gradient theo đầu vào chính là W(x). Đây là chi tiết rất đáng giải thích khi trình bày Demo 2.

---

## 8. Trích dẫn

```bibtex
@article{bohle2024bcos,
  title   = {B-cos Alignment for Inherently Interpretable CNNs and Vision Transformers},
  author  = {B{\"o}hle, Moritz and Singh, Navdeeppal and Fritz, Mario and Schiele, Bernt},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2024},
  doi     = {10.1109/TPAMI.2024.3355155}
}
@inproceedings{bohle2022bcos,
  title     = {B-cos Networks: Alignment is All We Need for Interpretability},
  author    = {B{\"o}hle, Moritz and Fritz, Mario and Schiele, Bernt},
  booktitle = {CVPR},
  year      = {2022}
}
```

Các bài trong `papers/` đều lấy từ arXiv; mã số arXiv nằm ở đầu tên file.

---

## 9. Checklist trước khi nộp

- [ ] Slide tập trung vào **cơ sở toán học** (Section 3); phần huấn luyện/siêu tham số chỉ là phụ
- [ ] Mỗi công thức/ý tưởng cốt lõi có **hình minh họa** do nhóm tự vẽ
- [ ] Có tham khảo thêm ít nhất 1–2 bài liên quan (CoDA-Nets, Grad-CAM, Sanity Checks, B-cosification…)
- [ ] Demo chạy được **offline** (đã có trọng số trong `weights/`) và minh họa được ít nhất một tính chất toán học
- [ ] **Không có nội dung slide nào do AI sinh ra** (câu chữ, hình, lưu đồ)
- [ ] Toàn bộ slide bằng **tiếng Việt** (thuật ngữ chuyên ngành có thể giữ tiếng Anh kèm giải thích)
- [ ] Trích dẫn nguồn cho mọi hình/bảng lấy từ bài báo
- [ ] Mỗi thành viên giải thích được phần của người khác (chuẩn bị cho câu hỏi)
