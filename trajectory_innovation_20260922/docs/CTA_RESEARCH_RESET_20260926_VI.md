# CTA: chốt lại câu hỏi research và dừng compute lệch trọng tâm

2026-09-26. User yêu cầu tránh experiment tốn tài nguyên nhưng không chứng
minh CTA. Đã dừng scope array 54989 và aggregation 54990, giữ nguyên artifacts.
Đây là quyết định về relevance/compute, không phải kết quả âm của hai codec.

## Trạng thái triển khai

Squeue trống; sacct xác nhận 54989_0 và _1 CANCELLED sau 9m50s mỗi task;
_2/_3 và 54990 bị hủy trước khi chạy. 54988 CPU tests/smokes COMPLETED 37s.
Không có final trained-codec comparison. Khoảng 19m40s MIG allocation của hai
task đang chạy đã sử dụng, không phải đủ một experiment. Không submit job mới.

## Chỗ lệch giữa proposal và experiment

Proposal gốc muốn conditional segment code giữ thông tin cho một query family,
dự đoán được từ action, tái sử dụng cho nhiều query, và tốt hơn compact-frame
codes/direct queries ở trade-off quality/compute. Proposal không đòi mọi task
phải có hidden history; efficiency cũng có thể là contribution nếu được đo
với các baseline mạnh và accounting đầy đủ.

PushT ban đầu là qualification của proposal/control interface. Sau khi visual
queries và hindsight readers không đạt, reader chuyển sang coverage cuối 8
bước. Điều này tạo được signal điều khiển hữu ích nhưng thu hẹp câu hỏi sang
endpoint scoring. Cải thiện trên target đó không tự chứng minh whole-path
abstraction, conditional coding advantage, hay query reuse advantage.

48.1% train banks không có ranking spread >1e-3 (11.3% toàn zero, 33.7%
nonzero exact-flat, 3.0% near-tie). Global oracle improvement không giải quyết
blind spot của các bank này. Cũng không được suy ra mọi bank flat đều có
terminal-value differences; điều đó cần evidence riêng.

## Bản đồ claim/evidence

| Claim | Đã có gì | Còn thiếu gì |
|---|---|---|
| Pipeline CTA chạy được | Source, FSQ, WM, reader và Round-1 closed loop | Round-2 runtime preflight chưa hoàn tất sau sửa precision |
| Có support để rerank trên PushT | W2 PHYS8 .765 vs P0 .625, n=400 | Không phải evidence cho path information |
| Code giữ endpoint-ranking information | lambda=0 CODE .706 vs endpoint FULL .785 | Không phải kiểm chứng temporal sufficiency |
| Conditioning giúp nén | Encoder/reader có C | Chưa có matched conditional/unconditional frontier |
| Whole-trajectory tốt hơn frame/endpoint code | Source nhận frames 2/4/6/end | Chưa có matched evidence; scope experiment đã dừng chưa ra kết quả |
| Action-conditioned prediction hữu ích | WM vượt prior; soft code vượt DIRECT offline | Greedy collisions cao; chưa có native closed-loop win rõ |
| Predictability co-design có lợi | Đã có matched lambda=.1 experiment | Cấu hình đó làm code nghèo đi, retention giảm rõ |
| Predict-once/query-many có lợi | Có giao diện reader query-conditioned | Nhiều ảnh của cùng goal không thay cho held-out query family/latency advantage |
| Kết quả method đủ cho paper | Infrastructure và diagnostic evidence | Chưa có contribution lõi được chứng minh và confirmation cần thiết |

Không nói 'CTA chỉ còn thiếu WM tốt hơn'. Chưa có evidence đủ để kết luận
toàn bộ idea sai, nhưng validation của central contribution vẫn còn thiếu.

## Công việc tiếp theo trước khi cấp thêm training compute

Chốt một query/task contract thể hiện đúng claim. Nếu ưu tiên whole-trajectory,
cần kiểm tra bằng dữ liệu và evaluator sẵn có rằng cùng context/query, các
candidate có endpoint observation gần nhau vẫn khác câu trả lời cần thiết
vì diễn biến giữa đoạn; endpoint+context baseline thực sự thiếu thông tin.
Không dùng riêng sự giống nhau của arm pose thay cho kiểm tra endpoint image/state.

Ba điều kiện trước model training:

1. Query về diễn biến cần thiết cho task, được xác định độc lập với outcome
   method; không thêm query trang trí rồi gọi là control benefit.
2. Thông tin cần thiết có thể đọc từ permitted trajectory observations;
   nếu chỉ simulator contact flag thấy được, abstraction vision cũng không
   tự giải được. Shared history phải khớp giữa endpoint và path controls.
3. Candidate bank tạo khác biệt có ích tại horizon thực thi, và evaluator
   phân biệt được chúng; tránh long frozen-policy tail làm signal bị xóa.

Rà soát local history đã cho hai cảnh báo cụ thể:

- ScrubCuttingBoard từng nhắm accumulated contact thay vì endpoint. Đây là
  nguồn dữ liệu cũ có thể audit lại về task properties, không phải arena đã
  qualify cho CTA. GR00T one-chunk continuation chưa confirm headroom;
  learned-composer tests thất bại và spatial-map/union là control mạnh.
  Không mở lại training nhánh đã đóng chỉ vì đổi tên method.
- Scene từng được cho là cần hidden memory, nhưng single frozen latent
  probe đạt 98–99% các biến liên quan. Không chọn lại arena này bằng lập luận
  'task nhiều bước nên endpoint không đủ'.

Chưa có arena mới được qualify. Bước tiếp theo là xác định premise từ data/
task semantics và thiết kế một audit nhỏ có tiêu chí dừng, không mặc định
submit thêm WM, continuation rollouts, hay endpoint/trajectory training.
PushT giữ vai trò diagnostic phụ nếu sau này cần; không còn là đường training
chính để hy vọng tự xuất hiện evidence cho whole-trajectory abstraction.

Follow-up cùng ngày: đã sàng lọc source native và xác minh metadata RinseBowls;
ưu tiên task này cho premise checks, chưa qualify để train. Xem
`CTA_TASK_QUALIFICATION_20260926_VI.md`. Audit CPU mới chỉ kiểm tra ngữ nghĩa
native và đọc lại short-chunk labels của bank đã có; không nối training GPU.
