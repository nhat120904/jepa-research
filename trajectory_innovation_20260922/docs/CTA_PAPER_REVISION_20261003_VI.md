# CTA paper: sửa theo feedback về world model và CVPR (2026-10-03)

Bản rewrite trước đặt sai trọng tâm: đổi title thành action ranking, mở đầu áp đặt rằng lựa chọn action cần prediction, dùng quá nhiều kiểm tra có tương lai thật trong main, và thay figure giàu chi tiết bằng sơ đồ quá đơn giản. Đây là lỗi biên tập, không phải yêu cầu của phương pháp CTA.

Bản sửa khôi phục **Predict Once, Query Many: World Models that Forecast Conditional Trajectory Abstractions**. Manuscript dùng style CVPR sẵn có, bật `review`, anonymous authors, line numbers và confidential header. Kit là CVPR 2026; năm hiển thị 2027 và paper ID chưa được cấp. Trước submission thật cần đối chiếu kit/quy định của năm đích. Không có thay đổi margin hoặc giảm cỡ chữ của template.

## Đã sửa trong nội dung

- Abstract/Introduction đi từ prediction target của visual world models → spatial/per-frame targets và compact tokenizers → câu hỏi conditional segment target → phương pháp → bằng chứng. Direct scoring được trình bày như một phương án hợp lệ, không bị loại bằng premise.
- Method đặt world model ở trung tâm: q(C,tau) học source target từ future quan sát; p(S|C,A) forecast distribution; D(C,mean(S),g) đọc forecast. Reader không có A và deployment không dùng future thật. Categorical loss, centered reader consistency, weighted ranking, stage freezing, inputs và token dimensions được nêu rõ.
- Main chỉ có deployable methods: P0, CTA, END, DIR, DIR-large (offline), external DINO-WM. FULL/CODE/geometry selectors chỉ xuất hiện ở supplement như diagnostics, không phải performance claims hay upper bound của closed loop.
- Bốn figure gồm: ví dụ PushT thật với paths/frames/codes/scores; kiến trúc hai stage và gradient paths; learned ranking với paired-root uncertainty và parameter counts; quality/cost scaling theo K và G. Asset thực và số liệu giữ nguyên nguồn, không tạo simulator frame giả.
- Thêm candidate-expansion evidence đã lưu ở `cta_v2cl_kscale_55897/kscale.json`, không phải experiment mới. Khác cohort của bảng chính được ghi rõ. Component timing có cùng phạm vi cho CTA/END/DIR; không đưa DINO-WM scope khác vào các curves chính.
- Discussion nêu đúng giới hạn: single task/training seed, development roots đã dùng nhiều lần, batch sensitivity của historical control, ablations còn thiếu, one-target goal views và policy-dominated latency.

## Số liệu và mức độ chứng minh

| Bằng chứng | Kết quả | Điều có thể kết luận |
|---|---|---|
| L8 K8, 200 P0 dev roots, 5,558 banks | RG CTA .680, END .635, DIR .483, DIR-large .502 | CTA hơn trên shared-bank geometry ranking; paired CTA−END +.045 [.011,.079], CTA−DIR-large +.178 [.138,.220]. |
| K expansion, 100 dev roots, 2,810 P0 banks | CTA mean gain .608→.999 ×10^-3 (K8→64); paired increase .391 [.328,.458] ×10^-3 | Ranking benefit vẫn có trong logged-bank workload lớn hơn; không phải held-out test hoặc closed-loop claim. |
| Native control, 200 dev roots | L8 CTA 70%, P0 61%, END 69%; L15 CTA 60.5%, P0 51.5%, END 54% | Development win trước P0; lợi thế trước learned alternatives chưa chắc chắn, historical sampler cần thay bằng canonical qualification. |
| H100 3g.40gb MIG K8 | CTA scoring 4.11ms (G1),16.87ms (G16); END4.52/23.40; DIR2.54/26.81 | CTA có lợi về component cost khi repeated queries; DIR rẻ hơn ở G1. Policy765ms chi phối, nên chưa có total-planner speedup/Pareto. |

Hai bảng tổng hợp/curve có cohort và mục tiêu khác nhau. Không diễn giải RG thành success hay retained information; không gọi view của một goal là semantic-goal transfer. DIR-large chưa có complete closed-loop run. Source alphabet 128 bits và deployed mean48float32 values là hai đối tượng khác nhau.

## Kiểm tra artifact

Main build thành công bằng MacTeX/TeX Live 2026: 9 trang tổng, nội dung kết thúc ở trang8 rồi references; supplement4trang. Đã kiểm tra từng trang và hình/bảng; final log không có Overfull, undefined references/citations hay font-substitution warnings. Bibliography có21 nguồn được cite trong main (22 entries được lưu, một entry không sử dụng). Source/PDF hashes và provenance được lưu trong các manifest của paper.

`draft_v4_before_20261002_review/` bảo toàn bản gốc. `draft_v5_before_20261002_user_feedback/` bảo toàn bản rewrite bị phản hồi. Release mới không sửa các release cũ. Không thay code/experiment/checkpoint và không nộp Slurm job trong lượt sửa paper này.

## Phần nghiên cứu vẫn còn thiếu

Viết lại không bổ sung held-out multi-seed evidence. Cần chạy qualified canonical sampling/learned scoring và native closed-loop continuation đã chuẩn bị, giữ monthly usage cap và kiểm tra cả squeue/sacct. Bổ sung factorized direct, compact per-frame, conditional/intermediate controls khi chúng trả lời một quyết định cụ thể; ưu tiên reproducible control trước sweep. Endpoint và intermediate-event tasks đều phù hợp với CTA, không cần chuyển hẳn sang temporal tasks để tiếp tục pipeline.
