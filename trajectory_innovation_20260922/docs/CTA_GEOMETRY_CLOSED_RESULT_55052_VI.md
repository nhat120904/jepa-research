# Geometry closed loop 55052 và bước sửa tiếp

2026-09-26. Cả squeue và sacct đã kiểm tra: queue trống, **55052 COMPLETED,
exit 0:0, 36m30s**. Hai bộ test pass 30 + 25 tests. Preflight FULL/CODE/CTA/DIRECT
pass; within-bank correlation giữa runtime và cache lần lượt khoảng .996/.982/.958/.999.
Preflight hiện chưa có arm expected-code riêng; nên bổ sung vào consistency check
hiện có trong lượt adaptation, không tạo thêm một research gate.

Không train lại; dùng checkpoint 55018. Đã chạy đủ full native episode của 10
dev roots 2100–2109 cho 8 arms và aggregate. Không mở sealed roots.

## Kết quả

| Arm | Thành công | Mean max coverage |
|---|---:|---:|
| P0, policy gốc | 7/10 | .861 |
| PHYS8, oracle coverage sau chunk | 4/10 | .789 |
| GEOM8, oracle geometry sau chunk | 8/10 | .824 |
| FULL8, reader nhìn endpoint thật | 5/10 | .795 |
| CODE8, reader nhìn code future thật | 5/10 | .747 |
| CTA8, WM greedy | 6/10 | .825 |
| CTA8E, WM expected-code | 4/10 | .824 |
| DIRECT8 | 4/10 | .745 |

GEOM8 giữ cả 7 success của P0, thêm root 2109. CTA8 cứu root 2107 nhưng làm
mất success của 2102 và 2108. CTA8E mất 2101/2103/2106 và không thêm success.
CTA8 hơn DIRECT8 ở 3 roots, thua ở 1. Không contrast nào có McNemar p < .05;
GEOM8–P0 chỉ có một discordant pair, p=1. Bootstrap 10 roots rất rời rạc;
không dùng biên CI chạm zero để claim thắng. Mean max coverage của GEOM8 vẫn
thấp hơn P0 nên ngay cả hai metric native cũng chưa cho ưu thế nhất quán.

## Diễn giải đúng phạm vi

- End-to-end chạy được sau sửa reporting; không còn dừng ở offline/preflight.
- Geometry oracle 8/10 là tín hiệu có thể tiếp tục PushT, chưa xác lập headroom
  toàn task. PHYS8 4/10 không phủ định các kết quả coverage oracle lớn trước đó.
- Offline expected-code .467 vượt greedy .294 về geometry retained gap nhưng
  closed-loop success đảo thứ tự 4/10 vs 6/10. Không thể chọn decoder dựa riêng
  vào offline ranking; cũng chưa đủ mẫu để kết luận expected-code tệ hơn thật.
- FULL8/CODE8 cũng dưới P0 dù không phải dự đoán future bằng WM. Vì vậy sửa WM
  không được đảm bảo sẽ giải mọi lỗi control. Đây không phải ladder nhân quả trong
  closed loop: mỗi arm đi đến trạng thái/bank khác nhau sau khi chọn action khác.
- Chưa tách được lỗi reader, thiếu thông tin S, hoặc target/horizon. Phân bố trạng
  thái khi triển khai có thể khác dữ liệu train; chưa đo ở run này để khẳng định.
- Oracle là oracle của target trong chunk 8 bước, không phải oracle thành công
  episode hay một trần tuyệt đối cho mọi learned policy.

## Một bước sửa ưu tiên

**Adapt reader trên code thực sự dùng lúc triển khai**, giữ source encoder,
codebook và WM frozen. Hiện reader được train trên source code rời rạc; lúc
deploy lại nhận code dự đoán sai hoặc vector expected-code nằm giữa các mã.
Đây là một mismatch cụ thể có thể sửa, không phải nguyên nhân đã được chứng minh.

Một lượt implementation/training/evaluation có:

1. Trộn source code, greedy prediction và expected-code khi train reader, với
   label geometry hiện có. Giữ source code để tránh phá khả năng đọc mã thật.
2. Control cùng khởi tạo, data/batch và số updates, nhưng chỉ tiếp tục học source
   code. Giữ frozen checkpoint trước adaptation làm reference. So sánh như vậy
   phân biệt lợi ích học predicted code với việc đơn giản train reader lâu hơn.
3. Nối thẳng sang closed loop trên cùng 100 dev roots đã dùng 2100–2199; công bố
   cả greedy và expected-code, không chọn cấu hình thắng trên 10 roots vừa xem.
   Matched policy/K/horizon/target, P0 và DIRECT làm reference. Đây vẫn là dev,
   không phải sealed test hoặc đảm bảo power cho improvement nhỏ.
4. Ghi số lần đổi lựa chọn khỏi candidate 0, score ties và gain/loss theo root
   trong evaluation để debug trực tiếp. Không mở chuỗi thí nghiệm điều kiện riêng.

Giới hạn: reader không phân biệt được hai action nếu context và predicted code
giống hệt nhau. Adaptation không tự sửa collision của greedy. Nếu adaptation
không cải thiện control, sửa WM theo khả năng phân biệt candidate là nhánh tiếp
theo; không tăng lambda co-design đã gây collapse theo quán tính. Chưa đổi task,
chưa tăng code capacity, chưa thay target/horizon cùng lượt để giữ kết quả đọc được.

Đây là hướng đề xuất, chưa submit job mới trong lượt báo cáo này.

## Artifacts

Run: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_geometry_e2e_55052/`.
`aggregate/summary.json`: success, paired differences, McNemar, timing.
`e2e_report.json`: completed stages và mean max coverage.
`closed/roots_2100_2109.jsonl`: root/arm outcomes và decision records.
`train/train_report.json`: recovered offline ladder và native-coverage diagnostic.
