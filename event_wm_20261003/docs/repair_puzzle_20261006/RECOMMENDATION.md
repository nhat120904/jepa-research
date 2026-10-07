# Sửa unified puzzle: hướng có căn cứ, 2026-10-06

## Vấn đề và mục tiêu

Checkpoint state puzzle 57636 thất bại ở planning dài. Đối chứng trước cho thấy h đánh giá một bảng còn 9 lần bấm là 0.70; giữ learned WM nhưng thay h bằng khoảng cách đúng tìm được cả bốn plan dài. Search còn tạo nhiều key cho cùng bảng. Mục tiêu là sửa hai phần này mà không đưa quy luật bấm puzzle, solver GF(2), goal evaluation hoặc executor scripted vào method.

**Khuyến nghị sau pilot hoàn tất:** giữ sửa biểu diễn train/WM/h/search nhất quán và continued h training trên biểu diễn đó: cả 5 task đã thành công, mỗi task 1 episode phát triển. Task3–5 chưa tìm được full first plan nhưng closed loop với replanning vẫn giải được. Ưu tiên xác nhận checkpoint này trên đủ episode trước khi đổi tiếp cách train. LHBL là đối chứng cải tiến hợp lý cho full-plan reliability và tốc độ search; chưa triển khai/chưa xác nhận trong lượt này.

## Sửa biểu diễn trước

Một thuộc tính vốn rời rạc trong play data cần có cùng biểu diễn ở dữ liệu thật, rollout WM, input h và search key. Suy ra các giá trị hợp lệ từ train rest states, đưa dự đoán gần chúng về cùng biểu diễn và khử candidate trùng sau chuẩn hóa. Vị trí và chiều cao liên tục không nên bị ép về một ít prototype vị trí.

Pilot 57680 nhận diện các thuộc tính có <=8 giá trị sau làm tròn 1e-5 từ train before/after; đây là nhận diện support hữu hạn bảo thủ, không phải giải pháp hoàn chỉnh cho pixel noisy. Các thuộc tính liên tục được để nguyên. Pilot cache chỉ chạy khi toàn bộ support hữu hạn, và kiểm tra candidate goal-dependent tương đương candidate cache; điều kiện này là điều kiện cache, không phải điều kiện áp dụng toàn bộ phương pháp.

Code cũ train h trên S thật nhưng successor và một nửa G là continuous imagined state; đây là lệch biểu diễn đã thấy trong code. Việc lệch này có phải nguyên nhân đủ để giải thích h sai hay không vẫn là giả thuyết. Pilot chuẩn hóa cả hai phía và train tiếp để kiểm tra bằng planning và closed loop.

Search còn nên bỏ các heap entry cũ khi state đó đã có đường ngắn hơn, và chỉ mở rộng lại một state nếu path cost tốt hơn lần mở rộng trước. Đây là sửa quản lý duplicate chung cho các domain; chưa triển khai trong pilot.

## Sửa cách học h bằng search trong WM

Hướng tiếp theo để thử có kiểm soát: thay một phần single-step backup bằng limited-horizon graph backup, kèm replay các state đã mở rộng. Pilot hiện tại đã giải được closed loop; mục tiêu của hướng này là cải thiện độ tin cậy và tốc độ planning, không mặc định rằng phải thay h mới giải được task.

1. Lấy S từ train data và tạo G từ một rollout WM. Không dùng official evaluation goals để train.
2. Chạy một search ngắn, chẳng hạn 32–64 expansions, trong learned WM với cùng candidate/goal test/key như inference.
3. Lưu các node và cạnh của graph vừa mở rộng. Cho node goal target 0; frontier dùng h_target. Tính target của các node bên trong bằng shortest path tới frontier cộng giá trị frontier, xử lý cả cycle.
4. Train h trên các node này và giữ một phần cặp train ngẫu nhiên/near-goal để phủ phân phối rộng.

Tức là h học từ nơi planner thực sự mắc kẹt và từ nhiều bước chuyển đã được mô hình dự đoán, thay vì chỉ học `1 + min h_target(next_state)`. Target từ graph cục bộ vẫn là xấp xỉ, không phải khoảng cách thật toàn cục. Số expansions không đồng nghĩa số event theo chiều sâu.

Đây là một hướng dùng thuật toán chung với learned transitions, không cần quy luật bấm đèn. [Beyond Single-Step Updates / LHBL](https://arxiv.org/abs/2511.10264) đề xuất lấy state từ limited search và backup theo graph, nhằm giảm lệch phân phối train/search và các vùng heuristic bị đánh giá thấp. [DeepXube](https://arxiv.org/abs/2603.23873) cũng mô tả việc dùng các node đã mở rộng làm dữ liệu và hỗ trợ LHBL. Sự phù hợp với lỗi hiện tại là suy luận từ chẩn đoán; các bài báo không xác nhận nó sẽ sửa checkpoint này.

## Những lựa chọn chưa có lý do làm trước

- Tăng width/steps thêm: đã thử width2048 và150k steps, chưa giải các task dài. Cần thay nguồn target/coverage trước khi mở rộng thêm compute.
- Tăng search budget: có thể giải thêm nhưng chưa sửa h hoặc duplicate; phải so sánh cùng budget và báo runtime.
- Dùng độ dài random walk làm khoảng cách tối thiểu: random walk có thể bấm lặp hoặc đi vòng. Độ dài đó chỉ là path cost/upper bound khi đường đạt goal, không phải shortest distance.
- Dùng oracle h hoặc quy tắc puzzle trong planner: đối chứng chẩn đoán đã đủ; không đưa chúng vào learned method.
- Train lại skill hoặc SAM2: chưa có bằng chứng chúng là nút thắt của checkpoint state này.

## Pilot đã triển khai và bằng chứng hiện có

Thí nghiệm giữ WM và skill, tiếp tục h15000 bước từ checkpoint cũ, same architecture/lr1e-4, batch512, half real pairs/half60000 new canonical imagined-walk pairs. Source/cache được cô lập. Candidate cache được kiểm tra trên1024 cặp train để giữ cùng operator. Planning trên cả5saved dev roots với20k expansions, sau đó learned closed loop1episode/task seed0. Đây là thử phát triển, không phải benchmark cuối hoặc kết quả generalization.

Checkpoint train hoàn tất nằm ở remote `repair_puzzle_20261006/job_57684/u_model.pt`, training log local `training_57684.json`. Job57685 kiểm tra phép canonicalize vectorized/scalar cho kết quả giống hệt nhau, rồi hoàn tất planning:

| Task | Plan đầy đủ | Expansions | h ban đầu |
|---|---|---:|---:|
| 1 | 4 event | 149 | 3.92 |
| 2 | 10 event | 1,749 | 6.45 |
| 3 | Không | 20,053 | 8.08 |
| 4 | Không | 20,053 | 8.31 |
| 5 | Không | 20,053 | 8.69 |

Task3–5 vẫn có best_h chỉ1.20/1.16/1.19 khi hết ngân sách. Loss training cuối0.011 không đồng nghĩa tìm được toàn bộ plan dài từ root. Tuy nhiên điều này không quyết định kết quả closed-loop: replanning có thể giải được bài dù lần search đầu không tìm được plan hoàn chỉnh. Giữ sửa biểu diễn nhất quán là hướng đã có bằng chứng; search-derived state replay và limited-horizon backup là hướng tiếp theo cho h và planning dài, chưa được thử trong pilot này.

Job57686 đã COMPLETED00:03:57, exit0:0, được xác nhận bằng cả squeue và sacct ngày2026-10-06 lúc08:54:33UTC/15:54:33ICT. Final JSON đã lấy về `loop_57686_final.json`:

| Task | Thành công | Simulator steps | Replans đã ghi | Full first plan | Thời gian search đầu |
|---|---|---:|---:|---|---:|
| 1 | Có | 119 | 3 | 4 event | 0.76s |
| 2 | Có | 306 | 9 | 10 event | 2.88s |
| 3 | Có | 463 | 13 | Không | 24.60s |
| 4 | Có | 566 | 17 | Không | 24.51s |
| 5 | Có | 640 | 19 | Không | 22.28s |

Cả 5 episode không có event timeout. First-plan-found0.4; toàn bộ loop5workers mất3.9phút. Task3–5 không có full first plan nhưng partial plan và replanning đưa robot tới goal. Mỗi task1episode seed0; các task dài trước đó thất bại trong dev cũ. Không ngoại suy pilot5/5 thành success rate100-episode benchmark; không so trực tiếp với6/30 cũ như hai protocol bằng nhau. Số replans/logged events không phải phép đếm đầy đủ các lần bấm: đoạn cuối có thể đạt goal trước khi event được ghi.

Job57680 bị hủy trong goal preparation chậm, chưa train;57684 lưu model rồi bị hủy để tối ưu canonicalization;57685 planning xong nhưng phần simulator lỗi thiếu dependency `sfa_code`;57686 bổ sung dependency snapshot và chỉ chạy loop, không train lại. Các manifest/source gốc và log được giữ. Đây là một cấu hình khoa học và một checkpoint mới, các job tiếp theo là tiếp tục/khắc phục runtime.

Thay đổi batch, mixed precision và cached successors làm khác compute/training throughput so với57636, nên nếu pilot cải thiện thì không quy toàn bộ cải thiện riêng cho một thao tác snap. Không dùng exact-distance labels hoặc eval goals cho sampling/training/early stopping. Support hữu hạn từ training cũng chưa cho phép khẳng định transfer sang giá trị/scenario mới.

Lịch sử: lúc02:14:43UTC/09:14:43ICT, gateway SSH báo connection refused và direct target timeout; khi đó chỉ xác nhận được task1–3 và trạng thái RUNNING00:02:48. Snapshot đó được giữ ở `loop_57686_partial_verified.json`. Kết nối đã hoạt động lại khi kiểm tra15:54ICT; kết quả cuối ở trên thay thế trạng thái chưa rõ của task4–5. Hiện squeue của account rỗng.

Bước xác nhận tiếp theo là đánh giá checkpoint sửa hiện tại trên protocol100episodes và các seed chưa dùng để chỉnh sửa. Sau đó so single-step với LHBL trên cube và puzzle, giữ WM/skill/candidates/data/horizon và ngân sách compute phù hợp; đo closed-loop success, expansions, latency, và lỗi rollout WM. LHBL có thể dùng learned transitions và goal test chung, không cần solver puzzle. Nhưng đây là phần thích nghi đề xuất: bài báo dùng transition của environment, chưa chứng minh target nhiều bước từ learned WM của mình sẽ tốt hơn. WM sai có thể tạo đường tưởng tượng rẻ, làm h học sai; skill/reader yếu không được sửa chỉ bằng cách đổi target h.

LHBL đã được tác giả đánh giá trên Rubik,35-sliding-tile và7x7LightsOut. Rubik khác task cube xếp khối của mình. Khả năng giúp task robot nhiều event là suy luận cần thử, không phải kết quả bài báo hay pilot này. Cùng thuật toán train trên nhiều domain cũng không đồng nghĩa một h checkpoint dùng zero-shot cho mọi task; hiện model vẫn train riêng theo environment. Cube hiện có ít headroom success hơn puzzle, nên cải thiện search có thể thể hiện ở latency hoặc task dài hơn.

Sau một sửa đổi hiệu quả, cần cùng cấu hình trên cube và puzzle, closed-loop đủ episode và training seeds, cùng candidates/horizon/data/budget để so sánh baseline; dev pilot không thay thế bước đó. Các model/core files ở project không bị thay đổi bởi thí nghiệm này.
