# Review và sửa bản thảo CVPR của CTA (27/09/2026)

Phạm vi: `paper_cvpr/` (abstract → conclusion, Fig. 1, Table 1–4, bib). Bản trước khi sửa được lưu ở
`paper_cvpr/draft_v3_before_review_20260927/`. Đã đối chiếu từng claim với `ti_wm/cta.py`,
`ti_wm/cta_parallel.py`, `scripts/cta_train.py`, `scripts/cta_round4.py`, `scripts/cta_collect_plus.py`,
config của 55018/55149 và `docs/CTA_INDEPENDENT_FEASIBILITY_REVIEW_20260927_VI.md`. Không chạy thí nghiệm mới;
chỉ biên dịch PDF qua sbatch (`build.sh`, job 55309 và các lần build sau đó).

## 1. Nhận định chung

Bản trước có khung phương pháp đúng với RESEARCH_DESIGN (target code có điều kiện, reader, feature decoder,
WM dự đoán code), nhưng:
- văn viết dạng câu cụt nối tiếp, thiếu câu chủ đề và liên kết giữa các đoạn;
- có claim không khớp với implementation (mục 2);
- một số lập luận reviewer sẽ bác ngay (baseline direct bị mô tả yếu hơn thực tế, "query many" chưa có bằng chứng
  trên PushT, số liệu motivation lấy từ hai tập episode khác nhau);
- chưa có kết quả nào: toàn bộ bảng vẫn là placeholder. Không sửa văn nào thay thế được phần này.

## 2. Lỗi nội dung đã sửa trong bản mới

| # | Bản trước | Thực tế / vấn đề | Sửa |
|---|---|---|---|
| 1 | "rate bounded by 128 bits", code 16 token là thứ reader nhận | WM chạy lúc deploy đưa **kỳ vọng codeword** (16×3 = 48 số thực) vào reader (`ParallelFSQWM.forward`) | Nói rõ bound chỉ áp dụng cho target code; thêm ablation "most likely code instead of Ŝ"; thêm vào Limitations |
| 2 | "2,000 rollouts, 87,860 decisions" | 87,860 là số **bank 8 candidate**; 32,590 bank nhiễu dùng chung context với 32,590 bank chuẩn → 55,270 decisions | Sửa số, mô tả nguồn dữ liệu (800 + 1,200 rollout, bank nhiễu, oracle-mixed) |
| 3 | Intro: feature distance +3 vs reader +8.5 "trên cùng true futures" | +3 (Gate C, n=100) và +8.5 (W2, n=400, reader S1-b nhãn coverage) là **hai tập root khác nhau** | Giữ ở dạng `\result{}` trỏ về hàng tương ứng của Tab. 2 (sealed test), ghi chú nguồn trong comment LaTeX |
| 4 | Direct scorer "phải học lại cho mọi goal", Table 1 "full model per goal" | Direct có thể tách f(C,A) → h(z,g) và cache z; đó là giả định về cách cài baseline, không phải giới hạn nguyên lý | Bỏ claim; thêm baseline **factorized direct scorer** (bottleneck cùng kích thước Ŝ, chỉ học ranking) — đối chứng quyết định cho việc giám sát bottleneck bằng tương lai |
| 5 | "Predict once, query many" như thể đã kiểm | PushT chỉ có một goal pose; 16 ảnh goal chỉ khác vị trí agent | Nói thẳng trong Setup và Limitations; tiêu đề vẫn giữ, nhưng cần task thứ hai/goal giữ lại (mục 4) |
| 6 | Eq. 1: `d_rank(D(C,S,g), ℓ)` như distortion từng mẫu | Rank là loss trên nhóm siblings | Viết lại Eq. 1 theo bank K siblings |
| 7 | Ranking loss có trọng số ở mọi nơi | Stage 1 dùng `rank_loss` không trọng số; chỉ WM dùng `weighted_rank` (σ = 0.01) | Sửa Eq. 2 và mô tả |
| 8 | Co-design penalty `λ‖S − E[S]‖²` | Round 2 (kết quả λ = .1 thật) dùng NLL của WM đóng băng, nội suy multilinear trên lưới FSQ (`fsq_predictability_nll`), đúng với L_source trong RESEARCH_DESIGN | Sửa |
| 9 | Ký hiệu φ cho cả encoder DINOv2 (`\varphi`) và target encoder (`q_\phi`); decoder dùng chung ψ với reader | Dễ nhầm | Encoder ảnh → `f`, decoder → `G_ξ`, `S = q_φ(C,τ)` tất định; cập nhật Fig. 1 |
| 10 | Related work: "target cố định trước" cho mọi WM | CompACT, GCQ học tokenizer | Sửa thành "biểu diễn quan sát, cố định hoặc học để tái tạo, độc lập với quyết định" |
| 11 | Thiếu TAP (Jiang et al., ICLR 2023) | Code rời rạc của đoạn trajectory có điều kiện theo state — prior art gần, reviewer sẽ nêu | Thêm vào related work + bib, nêu khác biệt (TAP code là latent action; code của CTA mô tả hệ quả quan sát được của một chunk cho trước) |
| 12 | "Decoder chống collapse theo nghĩa JEPA"; "WM là JEPA predictor" không nói target encoder không phải EMA | Ranking không làm code sụp về hằng; rủi ro là chuyên biệt hoá cho goal huấn luyện | Viết lại vai trò decoder; nói rõ target được học bởi reader + decoder và đóng băng trước Stage 2 |
| 13 | Reader "true future, uncompressed" ngầm là upper bound | FULL chỉ đọc frame cuối + proprio, còn code thấy cả frame trung gian | Caption Tab. 3: là tham chiếu, không phải upper bound |
| 14 | Chỉ báo success | Protocol Round 3/4 lấy normalized score làm primary | Tab. 2 thêm cột Score; TODO chốt primary metric trước sealed test |
| 15 | Một cột "ms/decision gồm cả proposal sampling" | Sampling DDPM chiếm phần lớn thời gian → che mất khác biệt giữa các WM | Tách Scoring (WM + reader cho K candidate) và Total |

## 3. Văn phong và cấu trúc

- Intro viết lại theo mạch: bối cảnh → hai vấn đề của target per-step khi chọn giữa candidate → cực còn lại
  (direct) → ý tưởng (đủ cho quyết định + chỉ phần lịch sử chưa có) → CTA → cách kiểm chứng → đóng góp.
- Problem formulation: hai trường hợp biên viết gọn là `S = τ` (per-step WM) và `S = A` (direct scorer).
- Experiments: mở đầu bằng ba câu hỏi, Setup ghi rõ split, nhãn, baseline cùng dữ liệu/loss/checkpoint selection;
  thêm Table 4 (ablation) và công thức retained gap.
- Thêm Conclusion không kèm kết quả và Limitations đầy đủ hơn.
- Hiện chiếm khoảng 6.3 trang nội dung, còn khoảng 1.7 trang cho kết quả và task thứ hai.

## 4. Còn thiếu để submit (không sửa được bằng văn)

1. Sealed test 400 episode, ba seed, công thức cuối **train lại từ đầu** (checkpoint hiện tại warm-start qua
   nhiều round; TODO trong §4.7).
2. Baseline chưa có: per-step latent WM, factorized direct scorer; endpoint WM phải retrain trên cùng dữ liệu.
3. Đo latency đúng: cùng GPU, precision, K, chia theo thành phần (encode, sampling, WM, reader), median/p95.
4. Chốt deployment bank (K = 8 hay 8 + 8 nhiễu; Round 6 đang chạy) và primary metric **trước** khi mở sealed.
5. Power: với tỷ lệ bất đồng như Round 4, n = 400 vẫn cho nửa độ rộng CI ≈ 5.7 điểm (review độc lập),
   nên hiệu ứng +4 có thể vẫn không sạch. Cần quyết định trước: hiệu ứng tối thiểu, hoặc claim
   non-inferiority với chi phí thấp hơn có biên xác định trước.
6. Task thứ hai (§5.5) hoặc goal giữ lại. Nếu không có, tiêu đề "Predict Once, Query Many" và claim về
   phần trung gian của segment cần thu hẹp.
7. Fig. 1 đọc được nhưng chữ nhỏ và toàn khối trừu tượng; nên vẽ lại với frame PushT thật, K chunk chồng lên
   ảnh và một ví dụ code/score (việc thiết kế, chưa làm).

## 5. Marker còn lại trong bản thảo

`\TODO{}` (đỏ đậm): task thứ hai; deployment bank; baseline chưa cài; lịch Stage 1; primary metric.
`\result{}` (đỏ): mọi số và đoạn diễn giải kết quả. Tắt cả hai trong `preamble.tex` trước khi nộp.

## 6. Vòng sửa thứ hai (27/09, theo phản hồi của user)

**Câu mở đầu sai.** Abstract và intro trước đây mở bằng "World models let a robot evaluate candidate actions".
World model chỉ *dự đoán* môi trường phản ứng thế nào với action; việc *đánh giá* cần cost/reward/goal (ở CTA là
reader). Tách dự đoán khỏi đánh giá lại chính là luận điểm của paper, nên câu đó mâu thuẫn với thesis. Dreamer (được
cite ở câu đó) cũng không chấm điểm candidate mà học policy. Đã viết lại: world model dự đoán, planner đánh giá dự
đoán so với goal. Cùng lượt sửa: bỏ "prevailing recipe" và "tractable" (overclaim), bỏ "the same true futures" (hai
số motivation đến từ hai tập episode), sửa câu MBPO/overoptimization, sửa "A^(1) là chunk policy gốc thực thi"
(thực ra baseline policy-only thực thi A^(1) có seed; arm OFFICIAL dùng RNG riêng).

**Fig. 1 vẽ lại bằng frame PushT thật.** Job 55314 (`scripts/cta_fig1_data.py`, ~1 phút MIG): closed loop
policy-only K = 8 trên dev roots 2100–2107 (≤ 20 decision/root). Chọn decision theo độ chênh nhãn geometry giữa
các proposal (không đọc score học được nào). Render gốc 512×512 (ảnh quan sát 96×96 được resize từ đây). Kiểm tra:
- clone tái tạo đúng observation mà model thấy (sai khác pixel 0);
- nhãn khớp log (0);
- score CTA tính lại khớp log trong nhiễu bf16.

Hình dùng root 2102, decision 16:
- 8 proposal đẩy block 41–54 px;
- sai số đỉnh sau 8 bước từ 15 đến 28 px;
- CTA4 chọn proposal 7, trùng oracle.

Ở 5/6 decision được render (đều là decision có độ chênh cao), lựa chọn của CTA trùng oracle. Đây **không phải** kết
quả vì mẫu nhỏ và đã chọn theo độ chênh.

Tái tạo hình với checkpoint cuối:
1. `sbatch scripts/slurm_cta_fig1.sh $PWD` (đổi checkpoint trong script).
2. `python3 scripts/cta_fig1_tex.py --decision <run>/r<root>_d<d> --out-tex paper_cvpr/fig/fig1_data.tex --img-dir paper_cvpr/fig/pusht`.

**Phát hiện cần đo lại: code 16 token gần như lặp một giá trị.** Trong 48 source code (6 decision × 8 proposal):
- trung bình chỉ có **1.8 giá trị khác nhau trên 16 token**;
- chỉ 35/256 index xuất hiện;
- ví dụ proposal 4 dùng index 92 cho cả 16 token.

Như vậy dung lượng thực của code thấp hơn nhiều so với 128 bit và các query của target encoder gần như trùng nhau.
Mẫu nhỏ và đã chọn lọc, cần đo trên toàn bộ dev decisions (số giá trị khác nhau trên mỗi code, entropy theo vị trí
token). Hệ quả nếu đúng:
- cách viết "16 tokens / 128 bits" chỉ là cận trên danh nghĩa;
- ablation M ∈ {8, 16, 32} gần như chắc không phân biệt;
- claim "mã hoá cả đoạn, kể cả frame trung gian" thiếu cơ sở cho tới khi token được dùng khác nhau.

Đã thêm "distinct token values per code" vào danh sách chẩn đoán ở §5.3 (placeholder), không đưa số vào paper.
