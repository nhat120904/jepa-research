# Demo cho giáo sư — đã tạo, không phải kết quả learned WM

Job CPU 53629, hoàn tất 42 giây, exit 0. Artifacts:

- HTML tương tác, tự chứa toàn bộ ảnh, mở offline:
  `/mnt/data/nhatnc129/jepa/predictive_abstraction/prepare_53629/demo/index.html`
- GIF dùng trong slide:
  `/mnt/data/nhatnc129/jepa/predictive_abstraction/prepare_53629/demo/oracle_demo.gif`
- Ảnh cuối của ví dụ được cứu:
  `/mnt/data/nhatnc129/jepa/predictive_abstraction/prepare_53629/demo/preview.png`

Tải HTML về laptop rồi mở bằng trình duyệt, không cần Python/server. GIF có ba ca
liên tiếp; HTML cho chọn từng ca, play/pause và kéo thanh thời gian.

## Lời trình bày khoảng một phút

“Em đang thử world model dự đoán một bản tóm tắt quỹ đạo thay vì toàn bộ chuỗi frame.
Từ cùng bản tóm tắt, mình hỏi được: có đến A không, có đến B không, và có đến A
TRƯỚC B không. World model nhận quan sát hiện tại/lịch sử và một chuỗi action đề xuất;
nó phải dự đoán trước khi thực thi, rồi giúp chọn action.

Demo hiện tại dùng simulator cung cấp tương lai thật để minh họa thứ model cần phân
biệt. Cùng proposal bank, đường mặc định thành công 11/48 trường hợp; nếu chọn đúng
candidate thì được 19/48. Đây là headroom +16.7 điểm phần trăm của một chunk, KHÔNG
phải cải thiện đã đạt được của method, cũng không phải episode success của MPC.

Em đã chuẩn bị dữ liệu RGB/action phân nhánh với cùng query cho nhiều candidate.
Bước đang làm là so sánh summary với dự đoán frame đầy đủ và dự đoán score trực tiếp.
Chỉ nếu summary giữ được thông tin và chọn action tốt hơn thì mới đầu tư sang control.”

## Các ca trong demo

Lấy ca đầu tiên theo mỗi nhóm, không tìm video đẹp nhất: prefix 12 (default thua,
oracle thắng), prefix 3 (default đã thắng), prefix 0 (cả bank không có lời giải).
Replay và kết quả đối chiếu với artifacts jobs 53529/53568. Goal A/B là overlay để
người xem hiểu, không phải thêm vào ảnh training. Không gọi cột oracle là learned WM.

## Demo tiếp theo cần có gì

Sau checkpoint hợp lệ, thêm cột learned-selection và đường score predicted/actual,
dùng đúng cùng candidate bank. Giữ ca thất bại và báo toàn bộ mẫu. Không dùng profile
20 bước làm kết quả method. Wall chỉ là mechanism sandbox; demo này không chứng minh
transfer sang robot/manipulation hoặc đủ đóng góp cho một method paper.
