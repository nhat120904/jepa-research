# Geometry 55018: train xong, lỗi ghi báo cáo; tiếp tục từ checkpoint

2026-09-26. Xác minh bằng cả squeue và sacct: 55018 FAILED, exit 1:0,
40m27s; queue trống tại lần kiểm tra. Không có kết quả closed-loop mới.

## Kết quả đã lưu

Hai bộ unit test pass 30 + 24 tests. Native vertex transform khớp môi trường,
max error 0. Codec/reader/FULL/DIRECT hoàn tất 3000 updates; WM/prior hoàn tất
6000 updates. Checkpoint `train/cta.pt`, `train/dev_scores.npz` và dòng final
step 9000 của `train/metrics.jsonl` tồn tại trong:
`/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_geometry_e2e_55018/`.

Offline dev có 3011 bank, 100 roots (2000–2099). Target là âm khoảng cách
vertex tương ứng, không phải coverage. Retained gap đo phần lợi ích chọn
candidate giữ được so với oracle target trong cùng bank, lấy candidate 0 làm mốc;
không phải success rate và không so trực tiếp với retention coverage vòng trước.

| Tầng | Retained gap | Spearman trong bank |
|---|---:|---:|
| FULL: reader nhận endpoint thật | 0.693 | 0.437 |
| CODE: reader nhận code từ future thật | 0.549 | 0.350 |
| WM greedy | 0.294 | 0.147 |
| WM expected code | 0.467 | 0.295 |
| DIRECT | 0.462 | 0.289 |

Đây là point estimates đọc từ log cuối, chưa phải một thắng lợi có CI.
Expected code là kỳ vọng theo prefix greedy hiện tại, không phải marginal
chính xác trên mọi trajectory/code. Predicted code khác nhau ở 26.1% cặp sibling,
so với source code 80.3%; WM greedy vẫn gom nhiều candidate vào cùng code.
Điều này phù hợp với mất khả năng phân biệt khi decode, chưa chứng minh một
nguyên nhân duy nhất. FULL cũng chưa hoàn hảo; CODE còn mất thêm thông tin.

Nhãn geometry phân biệt candidate ở 157/1401 bank dev coverage-flat (khoảng 11%).
Train tương ứng 1295/10907. Flat/informative dùng margin 1e-3 của protocol;
không đồng nhất coverage-flat với toàn bộ candidate có coverage bằng zero.
Geometry bổ sung tín hiệu ở một phần plateau, không giải quyết mọi chuẩn bị
khi block chưa di chuyển và chưa chứng minh tăng native success.

## Lỗi và sửa

`ranking_metrics` trả về mảng NumPy `chosen`. Diagnostic native coverage mới
giữ nguyên mảng đó, trong khi `json.dumps(default=float)` không serialize được
mảng nhiều phần tử. Lỗi xảy ra sau khi lưu checkpoint, scores và final metrics;
không phải bằng chứng training thất bại hoặc NaN.

Đã bỏ `chosen` khỏi summary như ladder hiện có; thêm regression test và đường
recover đọc saved scores, kiểm tra identity với metadata và xác minh final update.
Recovery tái tính summary/CI trên compute node, giữ nguyên artifacts cũ, chạy
runtime consistency rồi full episode trên 10 roots 2100–2109 với 8 arms cũ.

Job **55052 submitted**, 1 MIG H100 3g.40gb, 8 CPU, 128 GB, cap 2 giờ. Không
train lại, không đổi target/model hoặc tăng tập eval. Không claim regression mới
pass trước khi batch chạy. Xem JOB_LEDGER.md cho source release và output paths.

## Việc tiếp theo

Hoàn tất closed loop trước khi mở thêm training. Nếu expected-code giữ được
lợi thế so với greedy trong control, thử adapt reader trên code dự đoán là bước
sửa cụ thể tiếp theo, với baseline và dữ liệu giữ cố định. Đây vẫn là giả thuyết:
reader không thể phân biệt hai candidate có cùng context và cùng predicted code.
Nếu GEOM8 không giúp native success, xem lại horizon/target hoặc candidate bank;
không mặc định lỗi còn lại chỉ thuộc WM. Mười root phục vụ debug, chưa đủ để
quyết định hiệu quả method hoặc claim paper.
