# CTA: chọn bài kiểm tra trước khi train thêm

2026-09-26. Tiếp theo CTA_RESEARCH_RESET_20260926_VI.md. Đây là quyết định
qualification, không phải tuyên bố đã tìm được arena mà CTA sẽ thắng.

## Quyết định cụ thể

**Ưu tiên kiểm tra RinseBowls của RoboCasa. Chưa chuyển training sang task đó.**
Giữ PushT làm diagnostic phụ, dừng mở rộng train để tìm lợi thế whole-trajectory
trên coverage cuối chunk. Không mở lại composer Scrub hoặc MIKASA.

Lý do chọn RinseBowls từ source native đang cài (RoboCasa commit
`4f8a2980def75a55dff96b990745b83540425f09`):

- Có hai bát. Mỗi bát phải ở dưới dòng nước trong **25 lần update liên tục**.
- Nếu rời nước trước ngưỡng, timer của bát đó reset về 0. Đã đủ ngưỡng thì cờ
  rinsed được giữ lại. Success còn đòi gripper ở xa cả hai bát.
- Vì thế endpoint predicate hoặc tổng thời gian dưới nước không mô tả đầy đủ
  điều kiện native. Ví dụ 25 bước liên tục và 13 + ngắt + 12 bước không tương đương.
- `check_obj_under_water` dùng vị trí bát so với water site và trạng thái vòi.
  Có cơ sở để kiểm tra từ hình ảnh, nhưng **chưa chứng minh đọc được ở camera thực tế**.
- Query theo danh tính bát gắn trực tiếp với mục tiêu có sẵn; không cần tự thêm
  reward “trajectory đẹp”. Hai query vẫn quá ít để tự tuyên bố query amortization.

Registry native có dữ liệu `v1.0/pretrain/composite/RinseBowls/20250718`,
horizon episode 2850. Không có entry target cho RinseBowls trong registry đã đọc.
Hai vị trí dữ liệu local chuẩn đã kiểm tra chưa có metadata task này. Sau khi web
viewer không đọc được URL, đã xác minh trực tiếp metadata HF bằng HTTP tại revision
`522e4ffa3bb9d9729f79b241c65fc9e589f2659c`: **100 episodes, 140551 frames,
20 fps, 3 camera 256×256, state 16D, action 12D**. Chỉ đọc metadata vài KB; chưa
tải parquet/video, chưa xác minh replay alignment, visibility hay độ đa dạng nhãn.
Metadata chỉ có split train 0:100; cần tự khóa episode split trước mọi fitting.
Checkpoint GR00T đã chạy trên Scrub/RinseSinkBasin không đồng nghĩa đã qualify trên
RinseBowls. Simulator-family hoạt động không thay cho task-specific smoke test.

## Tại sao không chọn các hướng khác ngay

| Task/hướng | Điểm thuận lợi | Điều cản việc train ngay |
|---|---|---|
| PushT hiện tại | Có oracle coverage gain và pipeline chạy được | Nhãn cuối đoạn; nhiều bank flat; chưa test trajectory advantage |
| ScrubCuttingBoard | Có data và bank đã lưu; native history thật | Label playback lệch 62/504 episode; contact khó đọc trong probe cũ; confirmation tail không chắc; composer đã đóng |
| RinseSinkBasin | Native ba vùng đã rửa; baseline local 6/10 | Rất dễ được giải bằng ba bit + OR; pilot gần hoàn thành bị bão hòa; chưa qualify bank ở phase khác |
| RinseCuttingBoard | Native dwell + tắt vòi sau khi rửa | Tổng thời gian đơn giản, nước nóng thêm yêu cầu quan sát; `_check_success` tự tăng counter nên extra calls làm đổi nhãn |
| RinseBowls | Native continuity/reset, per-object queries, update khác success check | Chưa có data audit, pixel readability hoặc policy headroom trên task này |
| TurnOffSimmeredSauceHeat | Có lỗi tắt sớm bị latch | Thời gian và màu sốt ở endpoint có thể là control rất mạnh; không tự chứng minh cần trajectory code |
| MIKASA-Robo | Task temporal/procedural có sẵn | Local SAPIEN/Vulkan đã fail; nhiều task nhớ cue quá khứ khác với dự đoán future segment |
| Scene | Pipeline local, milestone native | Probe single-frame trước đây đã đọc được 98–99% biến liên quan; không được tái dùng premise thiếu lịch sử |

Danh sách này là sàng lọc source và evidence có sẵn, không phải exhaustive benchmark
search hoặc ranking thực nghiệm giữa các arena.

## Audit CPU được triển khai ở vòng này

Script `scripts/cta_task_premise_audit.py`, config `configs/cta_task_premise_audit.json`.
Một job CPU, 1 core, 2 GB, hard cap 5 phút. Không tải data, chạy physics, render,
load model, encode, train, hoặc tạo continuation mới. Không có job nối tự động.

1. Trích nguyên ba method native của RinseBowls từ source được snapshot, cấp chuỗi
   predicate giả lập để kiểm tra tính liên tục/reset/latch. Hai cặp được cố định
   trước khi chạy: H32 từ timer=0, H8 từ timer=23; trong mỗi cặp, cùng context,
   cùng tổng số bước dưới nước và cùng endpoint predicate nhưng native success khác.
   Đây **chỉ là kiểm tra ngữ nghĩa evaluator**. Không gọi đó là hai trajectory robot
   khả thi, cùng ảnh cuối, hoặc evidence trong candidate bank.
2. Đọc lại đúng hai artifact Scrub `qualification_52419` và `qualification_52524`,
   là protocol có restore compiled model. Dùng mọi prefix hoàn chỉnh, giữ danh sách
   prefix thiếu và kiểm tra fidelity. Lấy đúng 8 native steps sau anchor, không lấy
   nhãn sau policy tail thay cho nhãn của chunk.
3. Báo cáo spread và oracle-over-candidate0 cho từng native component riêng:
   contact count, count cap, sweep range, sweep milestone, released success.
   Không ghép thành scalar mới để chọn ra một metric thuận lợi.
4. Lưu tail length và terminal outcomes riêng để thấy phạm vi extrapolation;
   không suy diễn nguyên nhân terminal success, không fit ranker từ những nhãn đó.

Audit bank này trả lời một khoảng trống cụ thể trong evidence cũ: **candidate có
khác nhau ngay trong chunk không**, trước khi policy tiếp tục. Nó không mở lại
nhánh Scrub, không chứng minh ảnh chứa event, không xác nhận native increment là
value dài hạn. Chỉ dùng các prefix dev được chọn ở partial-contact, không đại diện
cho toàn bộ phase của task. Không đọc lại test set của codec/composer cũ.

## Điều kiện tiếp tục với RinseBowls — chưa thực hiện ở vòng này

Thứ tự được cố định để tránh train rồi mới phát hiện premise sai:

1. **Data và evaluator.** Xác minh metadata/download nhỏ, frame cadence, camera,
   action alignment, số lần gọi update mỗi native step và replay labels. Không
   quy đổi 25 updates ra giây trước khi kiểm tra cadence. Success checker RinseBowls
   đọc cờ; vẫn phải audit wrapper không gọi update thừa. Đọc full prefix để lập
   shared context; không cấp timer oracle riêng cho CTA.
2. **Observability và endpoint control.** Dùng đoạn quan sát thật, chia episode,
   so sánh C+endpoint với C+toàn đoạn tại cùng thời điểm/horizon/query. Xét cả phase
   tiếp cận, đang rửa và rời nước, giữ cả label-flat windows. Đối chiếu endpoint
   images/proprio, không chỉ arm pose. Nếu endpoint đã đủ trên data hoặc full
   observed sequence không đọc được nhãn, dừng claim trajectory-information ở đó.
3. **Bank/headroom.** Một pilot có cap riêng, policy frozen và K=8 từ cùng snapshot,
   chọn anchor bằng thông tin quá khứ. Trước hết dùng native H8 hiện có; chỉ đổi H
   nếu data cho thấy H8 không bao phủ sự kiện cần thiết, và ghi đó là đổi action
   commitment. Không dùng chuỗi boolean giả lập làm bằng chứng candidate thực tế.
   Candidate labels chỉ tính đến đoạn thực thi; flat bank để policy chọn candidate0.
   Không tự thêm distance shaping hoặc nối policy tail dài để làm nhãn có spread.
4. **Chỉ sau khi ba bước trên đạt:** huấn luyện một ladder nhỏ. Full observed path
   → true code → predicted code; so cùng context với endpoint, conditional frame
   codes, DIRECT(C,A,q), và per-frame event predictor + bộ đếm/reset đơn giản.
   Control automaton rất mạnh; learned temporal code phải có lợi về chất lượng
   hoặc cost đo được, không chỉ thắng baseline bỏ quên continuity.

Chưa có quantitative acceptance threshold vì chưa xác minh data/cadence và panel.
Phải khóa panel, metrics, mức cải thiện tối thiểu và cap **trước** pilot/probe tiếp
theo; không gọi một vài ví dụ source là gate pass cho training. Nếu chỉ phase gần
threshold có headroom thì báo cáo đúng phase đó, không khẳng định giải được ranking
trên đường đi. Success native đầy đủ vẫn là metric closed-loop cuối cùng.

## CTA được sửa điều gì và không tự giải quyết điều gì

Sửa target từ một số coverage cuối đoạn sang thông tin native có cấu trúc: bát nào
đã rửa đủ, continuity có bị ngắt, trạng thái cần cho bước tiếp. Target cung cấp bởi
evaluator sẵn có để supervision/evaluation; policy deployment không được query
simulator future. Đây vẫn là task-supervised abstraction, không phải tự khám phá
progress từ video hoặc task-agnostic reward learning.

Việc này kiểm tra đúng giá trị thông tin giữa đoạn và giảm nhầm lẫn với endpoint
scoring. Nó **không** đảm bảo có nhãn tốt ở mọi phase, không tạo candidate tốt nếu
policy bank không có, và không tự chữa WM/code collapse. Co-design lambda=.1 vẫn
là cấu hình đã thất bại; không bật lại khi premise mới chưa pass.

## Nguồn

- Local pinned RoboCasa: `.../latent_scope_stage_a/src/robocasa365`, commit ở trên.
- [Native RinseBowls](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/washing_dishes/rinse_bowls.py).
- [Native sink predicates](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/models/fixtures/sink.py).
- [Dataset registry](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/utils/dataset_registry.py).
- [Pinned dataset metadata](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos/blob/522e4ffa3bb9d9729f79b241c65fc9e589f2659c/pretrain/composite/RinseBowls/20250718/lerobot/meta/info.json).
- `latent_scope_20260909/docs/SCRUB_COMPOSITION_AUDIT_52617.md`,
  `COMP_PILOT_REEVAL_RESULT_52642.md`, `COMP_GROUNDED_RESULT_52655.md`,
  `MIKASA_GATE0_RESULT_53286_53288.md`, `STAGE_B0_2_RESULT_52399.md`.
