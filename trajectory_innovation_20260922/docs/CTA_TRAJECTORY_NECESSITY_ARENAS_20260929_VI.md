# Arena chứng minh thông tin trajectory cần thiết — 2026-09-29

Review tiếp theo câu hỏi người dùng “chả lẽ không có arena nào chứng minh được trajectory là cần thiết?”. Chỉ đọc source/literature; không submit job hay đổi experiment đang có.

## Sửa cách đặt tiêu chuẩn

Có benchmark phù hợp. Cần phân biệt ba mệnh đề:

1. Endpoint observation/physical configuration không đủ để đánh giá outcome của đoạn đã đi qua.
2. Learned temporal summary giữ thông tin đó đủ tốt cho planning.
3. CTA có quality/compute tốt hơn những cách giữ/predict thông tin đó khác.

(1) có thể xác lập từ semantics và kiểm chứng bằng paired physical rollouts. (2–3) là câu hỏi thực nghiệm. Không đòi arena mà mọi compact event summary đều thất bại: bản thân CTA cũng là compact summary. Có counter/DFA giải được semantics không bác bỏ trajectory necessity; nó cho một baseline.

Một direct Q(C,A,q) vẫn có thể học outcome mà không xuất trajectory tường minh. Vì vậy claim là temporal information cần thiết ngoài physical endpoint, không phải mọi solver bắt buộc dựng raw video hay chỉ CTA giải được.

## 1. RoboCasa RinseBowls — native task progress

Pinned native source: [rinse_bowls.py](https://raw.githubusercontent.com/robocasa/robocasa/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/washing_dishes/rinse_bowls.py).

`update_state` tăng timer khi bowl dưới nước; đạt 25 update thì latch rinsed; rời nước trước threshold reset timer. Native success cần tất cả bowls rinsed và gripper xa hai bowls. Do đó cùng endpoint predicate và tổng thời gian dưới nước có thể khác success: 25 liên tục khác 13 + ngắt + 12. Đây là counterexample về semantics; chưa phải cặp physical rollouts endpoint-matched đã chạy.

Task hợp nếu user cần trajectory cho task progress thay vì thêm safety metric. Dùng context đủ lịch sử chung cho mọi arm; thông tin đã rửa trước chunk không được ngầm cấp riêng cho CTA. Event predictor + resettable counter là baseline hợp lệ. Không cấp simulator rinsed/timer vào visual endpoint baseline rồi gọi đó là endpoint thuần.

Local protocol `docs/CTA_TASK_QUALIFICATION_20260926_VI.md` đã ghi metadata dữ liệu và những phần chưa xác nhận: policy competence, camera visibility và sibling diversity. Những phần đó chưa có kết quả không đồng nghĩa semantics đã thất bại. Không mở lại Scrub cũ; không bắt buộc một chuỗi gate trước end-to-end run.

## 2. SafeManip trên RoboCasa — nhiều temporal properties

[Paper](https://arxiv.org/html/2605.12386v1), [official repository](https://github.com/chengyuehuang511/SafeManip), [specs source](https://github.com/chengyuehuang511/SafeManip/blob/main/SafeManip/monitor/specs.py).

Paper định nghĩa 10 templates/8 categories trên 50 RoboCasa tasks; có release-after-inside, reach-only-when-open, cross-contamination ordering và recovery. Đây là họ query có sẵn, không cần tự phát minh reward để CTA thắng. Ví dụ release-before-inside và release-after-inside có thể cùng kết thúc với vật ở trong ngăn nhưng khác compliance; cần xác minh cặp thực trong bank, không coi ví dụ là measurement.

Chọn trước các property có sự kiện đọc được từ camera/history, ví dụ enclosure/release; force/contact cực ngắn và contamination predicate không hiển thị có rủi ro observability. Evaluator dùng privileged state: hợp lệ làm training supervision/evaluation nếu khai báo; không đưa ground-truth predicates/monitor state tương lai vào deployment.

Repo có specs và monitor code, nhưng README hiện nói working simulator/policy forks chuyển sang các thư mục `_safemanip` bị gitignore, và installation paths chưa cập nhật. Không gọi whole pipeline clone-and-run. Cần pin commit và tích hợp exporter đúng với simulator local nếu triển khai. Đây là ứng viên giàu semantics cho CTA, chưa phải lựa chọn được chứng minh rẻ nhất hoặc CTA thắng.

Các template có obligation vượt chunk cần continuation semantics và common past context; không đánh fail mỗi chunk vì eventually chưa hoàn thành. Giới hạn ban đầu ở sự kiện trong execution horizon hoặc có value của monitor continuation.

## 3. Safety-Gymnasium Goal/Push với hazard — mechanism experiment đơn giản

[Hazard docs](https://safety-gymnasium.readthedocs.io/en/stable/components_of_environments/objects/geom.html), [Vision observation docs](https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_vision.html), [hazards code](https://github.com/PKU-Alignment/safety-gymnasium/blob/main/safety_gymnasium/assets/geoms/hazards.py).

Hazards là vùng không collision vật lý, tạo cost khi đi vào. Vì vậy physical endpoint không ghi lại cumulative hazard exposure. Hai đường từ chung start tới chung end có thể khác native accumulated cost. Suffix `Vision` bật visual observations; tên Safe Vision một mình không đảm bảo pixel observation.

Hợp làm controlled mechanism figure: trade-off progress và native safety cost, không chỉ reward đứng yên. Không biến thành main claim query generalization chỉ với một binary risk label. Chưa xác minh released competent chunk proposer phù hợp local; không quảng cáo là chạy ngay miễn phí.

## Quyết định nghiên cứu

Nếu ưu tiên native task-progress semantics: RinseBowls rõ nhất trong shortlist đã kiểm tra. Nếu ưu tiên nhiều query/order cho CTA: SafeManip enclosure/release là ứng viên mới mạnh về semantics, có integration debt cần ghi rõ. Nếu ưu tiên thí nghiệm cơ chế sạch: Safety-Gymnasium hazard là lựa chọn tốt.

LIBERO-Safety đang có hướng tích hợp local vẫn là lựa chọn thực dụng cho manipulation, không tự ý bỏ/chuyển jobs. Phải giữ caveat early termination/padding đã ghi trong review 09-28 khi diễn giải endpoint-versus-trajectory.

Bằng chứng paper nên kết hợp endpoint-near pairs có khác native temporal outcome (diagnostic), full-distribution closed-loop control và matched-budget comparisons. Không chọn main evaluation chỉ gồm những pair thuận lợi; không ép endpoint nhận context yếu hơn. So với direct critic, event predictor+monitor, compact frame WM; counter giải được semantic không phải lý do loại arena.
