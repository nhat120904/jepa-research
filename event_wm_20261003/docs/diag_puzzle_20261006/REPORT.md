# Nguyên nhân unified STATE thất bại trên puzzle — 2026-10-06

## Kết luận

Nút thắt chính đã xác định ở mức thành phần là **cost-to-go học được đánh giá sai khoảng cách tới goal trên các bài puzzle dài**. Lỗi phụ là **search coi nhiều biểu diễn số của cùng bảng đèn là state khác nhau**, làm lãng phí ngân sách tìm kiếm. Các log đã kiểm tra không cho thấy skill là nguyên nhân chính.

Đây là chẩn đoán checkpoint puzzle `wm_57636`, dùng state observation của benchmark, không dùng SAM2. Không suy rộng sang pixel reader hoặc mọi checkpoint cũ.

## Đối chứng giữ cùng candidate và ngân sách search

Các root lấy từ episode 0 của từng task trong dev seed 0 đã lưu, làm tròn 3 chữ số trong JSON. Giới hạn 20,000 expansions; xử lý theo batch khiến lần chạy hết ngân sách báo 20,053. Oracle dynamics và oracle h dùng quy luật puzzle của simulator và solver GF(2), chỉ phục vụ quy nguyên nhân.

| Dynamics dùng trong search | Cost-to-go | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---|---|---|---|---|
| WM học được | h học được | Không tìm được plan | Không tìm được plan | Không tìm được plan | Không tìm được plan |
| Dynamics chính xác | h học được | Plan 10 lần bấm; 2,005 expansions | Hết ngân sách | Hết ngân sách | Hết ngân sách |
| WM học được | Khoảng cách chính xác | Plan 10; 533 expansions | Plan 14; 789 | Plan 16; 917 | Plan 20; 1,173 |

Các plan ở dòng cuối đều đạt goal khi kiểm tra bằng dynamics chính xác. Điều này xác định h học được là nút thắt của các root khó trong ngân sách hiện tại; không chứng minh WM hoàn hảo trên toàn bộ state space.

## 1. Cost-to-go bị quá lạc quan và xếp hạng sai

- Trong search task 2, một state thực sự còn **9 lần bấm** được h chấm **0.699**. Sau khi đưa state về prototype rest gần nhất để loại nhiễu, h vẫn chỉ **0.867**. Vì vậy lỗi không chỉ do input có nhiễu nhỏ.
- Root task 5 cần **20 lần bấm**, h chấm **8.064**. Trên chuỗi đúng của task 2, khoảng cách thật giảm 9 → 8 → 7 nhưng h tăng 5.81 → 6.05 → 6.47.
- Thay chỉ h bằng khoảng cách đúng, giữ nguyên WM học được, tìm được cả bốn plan khó. Thay WM bằng dynamics đúng mà giữ h học được vẫn thất bại ở task 3–5.
- Cách train hiện tại dùng Bellman bootstrap với successor do WM dự đoán. Loss nhỏ không xác nhận h là số lần bấm tối thiểu thực tế. Các đối chứng xác định lỗi chức năng của h; chưa tách được nguyên nhân huấn luyện sâu hơn giữa phân phối cặp S/G, mục tiêu bootstrap và khả năng biểu diễn của mạng.

Chẩn đoán cũ “h gần như luôn bằng 30” không còn mô tả checkpoint này: hiện h có thể rất nhỏ dù còn xa goal.

## 2. Search nhận quá nhiều phiên bản của cùng bảng

Trong task 2, các candidate được tạo ra có **229,186 search key khác nhau nhưng chỉ 3,513 bảng bật/tắt khác nhau**. Một bảng có tới **3,735 key**. Đây là số biểu diễn được tạo, không phải số node đã mở rộng.

WM dự đoán continuous state, còn key lượng tử hóa từng thành phần vị trí/appearance. Sai số nhỏ có thể đổi key dù trạng thái bật/tắt không đổi.

Chỉ đổi key sang key của prototype rest gần nhất, giữ nguyên WM và h: task 2 tìm được plan 10 event sau **1,941 expansions**, thay vì hết ngân sách. Đối chứng này chưa kiểm tra thực thi closed-loop. Projection output WM cũng giải được task 2, nhưng vẫn thất bại task 3–5; projection riêng input h không giải được task 2–5. Vì vậy chuẩn hóa state là sửa lỗi phụ, chưa đủ giải toàn bộ puzzle.

Prototype được lấy từ train rest states; phép projection không chứa quy tắc bấm puzzle. Tuy nhiên không nên áp dụng mù quáng phép snap này lên tọa độ liên tục của cube vì có thể hạn chế các vị trí mới.

## 3. Skill và WM đang làm được phần nào?

Trong các task thất bại 2–5, **746/746 event đã kết thúc và được ghi log** đạt target của nút được chọn và đổi đúng toàn bộ nhóm đèn tương ứng với một lần bấm. Không tính các event chưa kết thúc. Log cho thấy planner cuối cùng lặp cùng nút, làm bảng quay đi quay lại, trong khi skill vẫn thực thi lệnh.

Với chuỗi bấm đúng dài 4/10/14/16/20 của năm task, raw learned WM dự đoán đúng các bit bật/tắt suốt chuỗi và trạng thái cuối qua goal test. Có drift continuous nhưng các thử nghiệm này không cho thấy thiếu candidate hoặc goal test ngăn mọi lời giải dài.

## Cube tốt nhưng puzzle kém có ý nghĩa gì?

Cube có ít object và chuỗi di chuyển ngắn. Puzzle có 20 nút, một lần bấm đổi nhiều đèn và các bài khó cần 10–20 lần bấm; tiến gần goal không tương đương làm từng đèn giống goal hơn. Một heuristic đủ dùng cho cube vẫn có thể dẫn search sai trên puzzle.

Bằng chứng này cho thấy unified hiện chưa giải tốt planning tổ hợp dài. Nó chưa chứng minh mọi thành công trên cube chỉ là hand-engineering, cũng chưa chứng minh generalization giữa các family.

## Kết quả method và phạm vi

- Puzzle checkpoint 57636: dev learned closed-loop vẫn **6/30**. Seed 1 bị timeout sau 96 episode hoàn tất, 20 success; không có score hoàn chỉnh 100 episode. Seed 2 chưa chạy.
- Cube checkpoint 57637: seed 1 **99/100**, seed 2 **97/100**, cùng cấu hình unified state. Một training seed; các eval seed đã được xem qua các vòng phát triển.
- Các đối chứng oracle ở đây là planning offline, **không phải kết quả method mới hoặc baseline fair**. Chưa sửa core source, chưa train lại, chưa có closed-loop sau sửa.
- Hướng sửa có căn cứ: xử lý các state tương đương trong search bằng đặc tính học từ rest states, đồng thời sửa cách học/đánh giá h ở khoảng cách dài. Chỉ tăng thời gian train hoặc ngân sách search chưa được chứng minh giải quyết nguyên nhân.

## Nguồn và trạng thái job

Checkpoint: `/mnt/data/nhatnc129/jepa/event_wm/state_puzzle-4x5-play-v0_57614/wm_57636/u_model.pt`.

Checkpoint SHA256: `6268297187c50fbf65cdf6169a5c3964b3011ad6d00b177d03a26b143ebc67c6`.

Frozen source remote: `/mnt/data/nhatnc129/jepa/event_wm/diag_puzzle_cause_20261006/source/`.

- `u_wm.py`: `fcb59df4a34aea4e4a68f3046f7aaff1673ce6d87804ae6dfbb5970850653215`
- `lightsout.py`: `5546d4b1cb63f2f3b38efe830b810c93f365ebc40bb69598dade44ca922e65b0`

Jobs 57669/57670/57671 đều COMPLETED, exit 0, lần lượt 22 giây / 1 phút 59 giây / 1 phút 13 giây. Mỗi job 1 GPU, 2 CPU, 12 GB; time limit 15/5/5 phút. Đã kiểm tra cả squeue và sacct sau hoàn tất; account không còn job chạy tại lần kiểm tra. Quota tháng đã kiểm tra trước mỗi submission, nằm dưới trần 90% người thứ năm.

Dữ liệu chi tiết: [57669](report_57669.json), [57670](report_57670.json), [57671](report_57671.json), [skill trace audit](skill_trace_check.json). Script và sbatch nằm cùng thư mục. Output remote: `/mnt/data/nhatnc129/jepa/event_wm/diag_puzzle_cause_20261006/job_<id>/report.json`.
