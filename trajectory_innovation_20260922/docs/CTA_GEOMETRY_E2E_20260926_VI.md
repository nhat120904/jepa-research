# PushT geometry: làm end-to-end rồi debug

2026-09-26. User nhắc lại lựa chọn đã có trong CTA_E2E_PROTOCOL.md: triển khai
method đầy đủ, chạy closed loop, dùng ladder để tìm chỗ tụt và sửa. Không tiếp tục
dùng chuỗi research gates làm điều kiện trước mỗi bước implementation. Root
AGENTS.md yêu cầu compute qua Slurm, giới hạn thời gian, kiểm tra duplicate và
giữ artifacts; không bắt buộc dừng sau mỗi diagnostic. Đây là điều chỉnh cách làm
của assistant, không phải user cần xin exception từ protocol.

## Một lượt development tự chạy đến cuối

Script `scripts/cta_geometry_e2e.py`, launcher `scripts/slurm_cta_geometry_e2e.sh`.
Một GPU MIG 3g.40gb, 8 CPU, 128 GB, **hard cap 3 giờ**, không có array hoặc
training sweep. Tất cả source được snapshot. Không submit bước tiếp chỉ vì điểm
offline đẹp. Không chờ user phê duyệt giữa relabel, train và closed loop.

1. Unit tests kỹ thuật và kiểm tra vertex transform với gym-pusht native.
2. Tính nhãn registration từ `phys8` của collection cũ; kiểm tra root/decision/t/
   action/coverage khớp feature cache. Feature files được symlink, metadata mới
   ở run directory riêng. Cache gốc không đổi.
3. Warm-start tất cả model từ Round 1 `cta_train_54717`. Codec/reader/FULL/DIRECT
   cùng nhận 3000 updates, LR 1e-4, cùng target. WM/prior nhận 6000 updates sau khi
   freeze source và tạo lại code. Giữ reconstruction, saturation và action contrast;
   lambda=0, không co-design, không reader adaptation.
4. Lấy batch source trên toàn bộ decisions, reconstruction học cả phase nhãn hòa;
   pairwise loss không ép preference khi label chênh dưới margin. Đây là thay đổi
   sampling cùng với target, nên không claim một causal ablation chỉ của reward.
5. Lưu checkpoint rồi tính full offline ladder trên dev. Greedy và expected-code
   cùng được báo cáo. Không yêu cầu tier nào vượt threshold mới được chạy control.
6. Preflight inference/cached features, giữ nguyên panel 32 banks và thresholds.
   Sửa việc lặp giải nén NPZ, không hạ tiêu chuẩn để chạy qua.
7. Closed loop **10 dev roots 2100–2109**, full native episode, cùng policy/K=8/
   execute8: P0, PHYS8, GEOM8, FULL8, CODE8, CTA8, CTA8E, DIRECT8.
8. Aggregate native success, max episode coverage, paired differences và logging
   các quyết định. Mười root chỉ phục vụ phát hiện lỗi/triển vọng, không đủ làm
   bằng chứng improvement 4–5 pp hoặc paper confirmation.

Lỗi dữ liệu, NaN, model compatibility hoặc preflight thất bại vẫn phải dừng và
ghi lỗi; một metric khoa học thấp không chặn bước kế tiếp. Nếu hết hard cap, giữ
checkpoint và từng root đã ghi, báo cáo partial thay vì gọi hoàn tất.

## Target và giới hạn của claim

Target = âm mean Euclidean distance giữa 8 vertex tương ứng của vật và goal,
chia 512. Dùng hình học native gym-pusht, không sao chép tọa độ local của GPC.
Margin 1e-3 tương ứng 0.512 pixel mean distance. Goal cố định của task hiện tại.
Đây là ứng dụng registration objective của GPC vào runtime đang có, không phải
reproduction GPC (khác policy, data, K, horizon và môi trường).

Để reuse code, trường metadata `cov8` trong **view mới** mang registration target;
`native_cov8` giữ coverage gốc. Config/report ghi rõ điều này. `phys8` không được
đưa vào input của learned reader, source hoặc WM. Chỉ GEOM8 là oracle dùng pose
tương lai thật, tương tự PHYS8 dùng coverage thật. FULL8 là reader đọc **endpoint**
thật; CODE8 đọc code từ các frame thật. Hai arm này cũng là diagnostic privileged.
CTA8/CTA8E/DIRECT8 chấm candidate trước khi candidate được mô phỏng để logging.

Sửa target không tự chứng minh giải được bước chuẩn bị khi vật đứng yên, sửa
free-running code collision, hay có whole-path advantage. Các diagnostics nằm
trong cùng run để biết vấn đề còn ở đâu. Không mở sealed roots hoặc đổi sang
RinseBowls ở lượt này. Không claim whole method fully self-supervised: reader
nhận nhãn task và gradient ranking ảnh hưởng source code.

## Kết quả CPU 55016 đã có trước lượt này

55016 COMPLETED, exit 0:0, 1 giây; đã xác minh bằng squeue và sacct.
Hai panel Scrub cũ có 8 và 7 bank hoàn chỉnh. Contact count phân biệt candidate
ở 5/8 và 2/7 bank; candidate tốt nhất vượt candidate0 ở 3/8 và 1/7 bank.
Released-success trong 8 bước hòa ở toàn bộ 15 bank. Đây là các partial-contact
prefix dev được chọn trước đây, không phải task-wide success/headroom estimate.
RinseBowls chỉ pass ví dụ predicate bằng evaluator native; chưa có robot rollout,
visual probe hoặc CTA training trên task đó.
