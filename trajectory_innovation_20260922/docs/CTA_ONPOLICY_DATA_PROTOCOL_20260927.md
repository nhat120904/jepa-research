# CTA: chẩn đoán on-policy (1) và dữ liệu Round 4 (2)

2026-09-27, trước khi submit. User yêu cầu làm cả (1) và (2) sau kết quả Round 3
(55077–55079): offline CTA3 ≈ CODE nhưng closed-loop không arm nào hơn P0; WM song song
overfit (dev NLL tăng 3,91 → 4,58 nats/token); thứ tự offline đảo ngược trong closed loop.

## Bộ chạy mới: closed loop theo lô, ghi đủ candidate

`ti_wm/cta_batch.py`, `scripts/cta_diag_onpolicy.py`. Nhiều root chạy song song như
`scripts/d_collect.py`. Ở **mỗi** quyết định: simulate đủ 8 candidate, ghi coverage native và
nhãn geometry, chấm bank bằng **mọi** scorer đã đăng ký (chấm chéo). Arm hành động chỉ chọn
nhánh nào thành trạng thái tiếp theo; arm học được không bao giờ đọc tương lai simulate.
Score chuẩn hóa như Round 3 (bỏ coverage lúc reset).

Kiểm tra correctness (không phải gate khoa học): trên trạng thái thật của 2 root đầu, điểm
theo lô phải khớp đường tuần tự đã preflight (`Planner` / `R3Planner`): median |Δ|/std ≤ .05,
Spearman trong bank ≥ .80 — cùng ngưỡng preflight hiện có.

Batch làm thay đổi số học nhỏ của DINO/UNet so với chạy tuần tự, nên kết quả chỉ ghép cặp
**trong** bộ chạy mới (P0 được chạy lại trong cùng bộ chạy). Không ghép với 55078.

## (1) Chẩn đoán trên model hiện có

Roots 2100–2199 (4 shard × 25). Arm hành động: P0, PHYS8 (oracle coverage), GEOM8 (oracle
geometry), FULL (reader geometry trên frame cuối thật), CTA3, NLL8, FRAME8, DIRECT3.
Scorer chấm chéo: FULL, CODE, CTA8E, DIRECT8, CTA3, NLL8, FRAME8, DIRECT3.

Trả lời ba câu:

1. **Target:** GEOM8 và PHYS8 so với P0 và so với nhau. Nếu GEOM8 không hơn PHYS8, việc chuyển
   sang geometry (quyết định từ 10 root) không có căn cứ closed-loop.
2. **Ranking on-policy:** retained gap của từng scorer trên trạng thái mà từng arm đi tới
   (ma trận arm × scorer), cho cả hai nhãn. So với hàng P0 (phân phối offline).
   Tụt mạnh trên trạng thái của chính arm → lệch phân phối / khai thác reader;
   giữ được mà success không tăng → target/horizon không dẫn tới success.
3. **Tầng reader:** FULL (tương lai thật) có tăng success không; nếu không, sửa WM chưa đủ.

## (2) Dữ liệu mới và Round 4

Thu thập `scripts/cta_collect_plus.py`: 1.200 episode mới, roots **31050–32249** (chưa dùng ở
đâu; tổng train 2.000 episode). Mỗi quyết định hai bank chung context:

- bank 0: 8 candidate seeded của policy (giống bank lúc deploy);
- bank 1: cùng 8 chunk, mỗi chunk dịch một offset hằng 2-D, candidate k có σ_k = 4k+4 đơn vị
  (4–32, tức ~0,75–6 px). Chỉ dùng cho training.

Lý do: sibling của policy gần như trùng nhau (97,8% cặp lệch block < 1 px), nên WM gần như không
thấy action thay đổi hậu quả thế nào. Bank 1 cho thấy tác động của action ở biên độ lớn hơn.

Thực thi: root chẵn theo P0; root lẻ thực thi candidate geometry-tốt-nhất trên một nửa số quyết
định (seeded), để dữ liệu có cả trạng thái mà planner đi tới. Trạng thái simulator chỉ dùng để
chọn nhánh thực thi và tính nhãn; deploy không đọc nó.

Encode `scripts/cta_encode_plus.py`: giữ **PCA gốc** (cta_feat_54490) để encoder/reader/FULL của
55018 vẫn hợp lệ; lưu token context + source code S (encoder đóng băng) cho dữ liệu cũ và mới.

Train `scripts/cta_round4.py`: công thức Round 3 giữ nguyên (warm start, loss, 32 bank = 4 × 8,
LR 1e-4), 8.000 update. Chọn checkpoint của từng mạng theo **chính objective của nó trên dev
offline** (roots 2000–2099; closed-loop 2100–2199 tách biệt). Mạng:

| Tên | Nội dung | Dữ liệu |
|---|---|---|
| NLL4 | WM song song, chỉ NLL | cũ + chuẩn mới + nhiễu mới |
| CTA4 | NLL + consistency + ranking qua reader đóng băng | cũ + chuẩn mới + nhiễu mới |
| CTA4S | như CTA4 | cũ + chuẩn mới (ablation: không có bank nhiễu) |
| DIRECT4 | direct scorer, weighted ranking | cũ + chuẩn mới + nhiễu mới |

Closed loop: P0, CTA4, NLL4, DIRECT4, CTA4S trên roots 2100–2199 với cùng bộ chạy; chấm chéo
thêm CTA3, DIRECT3, FULL. Tổng hợp gộp với (1) **chỉ khi** P0 ở hai lượt cho quỹ đạo giống hệt
(kiểm tra lựa chọn và nhãn từng quyết định); nếu khác, P0 thứ hai được đổi tên và báo ghép cặp
chéo không chính xác.

Cách đọc: CTA4 − CTA3 (dữ liệu), CTA4 − CTA4S (bank nhiễu), CTA4 − DIRECT4 (CTA so với direct
cùng dữ liệu), dev NLL theo thời gian (còn overfit không), và ma trận on-policy.

## Chuỗi job và ngân sách

Smoke GPU (test + mọi đường code mới trên input nhỏ) → song song: (1) diag 4 shard → tổng hợp;
(2) collect 24 shard → encode → train → closed 4 shard → tổng hợp. Dependency hủy chuỗi khi lỗi.

Ước lượng ~11–12 MIG GPU-giờ (diag ~2,7; collect ~3,6; encode ~1; train ~2; closed ~1,7;
smoke ~0,5). Đã dùng 51,4 MIG GPU-giờ từ 22/9 so với mức 60 đã duyệt ngày 24/9; sau chuỗi này
khoảng 63. User đã yêu cầu trực tiếp hai việc này; mọi vượt thêm phải hỏi trước.

Giới hạn: một seed train; roots dev đã dùng nhiều lần; không mở sealed roots 3000–3399.
Bank nhiễu và thực thi xen oracle là hai thay đổi dữ liệu cùng lúc; CTA4S tách được bank nhiễu,
chưa tách được thực thi xen oracle (có thể lọc theo root chẵn/lẻ trong lượt sau nếu cần).
