# Global latent và event discovery: cân nhắc lại trong phạm vi hướng hiện tại

Ngày 07/10/2026. Phân tích tài liệu/source, chưa triển khai frontend mới, chưa có kết quả so sánh global/object trên cùng protocol. Không sửa source/config frozen hoặc submit job trong phần nghiên cứu này.

## Kết luận và phạm vi giữ nguyên

Global latent có thể giảm phần proposal, association và identity bookkeeping, nhưng chưa chắc giảm độ khó của toàn method. Event discovery, latent state completeness, termination, executable candidates và độ chính xác rollout dài vẫn phải giải quyết. Không có đủ bằng chứng để thay đường chính object-centric bằng một pooled vector chỉ vì ít module hơn.

Giữ câu hỏi gốc: từ offline pixel play và continuous actions, discover interaction events có thể thực thi, học event-conditioned transitions chính xác trên chuỗi dài, học cost-to-go và search trên candidates hữu hạn, rồi execute/replan bằng learned skill. Không chuyển mặc định sang online option RL, fixed-horizon action-chunk MPC, hoặc chỉ đi theo latent distance tới goal.

Object-centric là inductive bias để giữ state và locality của effects; không phải bản thân novelty và không cần mặc định gắn tên semantic. Global latent + learned event là đối chứng có ý nghĩa. Đường ảnh dùng object hiện có tiếp tục là lựa chọn triển khai chính trong đề xuất; claim cuối cùng phải dựa vào matched learned closed-loop.

## Global latent có nhiều nghĩa

1. Một vector pooled của toàn ảnh: bỏ explicit identity nhưng dễ mất các thay đổi nhỏ; cần kiểm tra capacity, mục tiêu training và readout. Không có định lý rằng vector global luôn mất thông tin.
2. Một tập patch/spatial features không có object IDs: vẫn là scene representation chung, giữ vị trí và chi tiết, có thể ground event bằng attention/vùng tương tác mà không giữ persistent object identity.
3. Một codebook với mỗi code là toàn bộ scene configuration: khác distributed latent. Với 20 biến nhị phân có tối đa `2^20` cấu hình; một code cho mỗi cấu hình gây vấn đề combinatorial coverage. Đây không phải lập luận chống mọi VQ hoặc global representation: nhiều categorical factors có thể giữ compositionality.

Flatten một tập object tokens thành vector cũng không xóa cấu trúc object đã học. Vì vậy so sánh phải nêu rõ model/encoder/object supervision, không chỉ đặt nhãn global/object cho tensor shapes.

## Có thể extract event mà không tracking object như thế nào?

Đề xuất global variant sau là thiết kế để thử, chưa phải thuật toán đã chứng minh trên data hiện tại:

1. Video/actions -> encoder có history -> planning context và các feature transient cần cho control. Context phải giữ mọi thay đổi bền liên quan tới nhiệm vụ; không yêu cầu bỏ toàn bộ robot information nếu nó ảnh hưởng feasibility. Skill vẫn nhận full observation/history.
2. Boundary proposals dựa trên context change, sự ổn định sau thay đổi và action/history. Không dùng mỗi `norm(z[t]-z[t-1])` của pretrained vector để định nghĩa event: tay, bóng, occlusion và small-object changes có thể tạo signal rất khác nhau. Confidence/rest là prior mềm, không phải nhãn simulator.
3. Học/refine segmentation với command compact, policy reconstruction và chi phí complexity; một segment cần giữ được interaction hoàn chỉnh. Chỉ sparsity/slowness có thể bỏ small objects; chỉ compression có thể giữ wander; chỉ endpoint prediction có thể tạo shortcut hoặc cắt từng frame.
4. Event encoder gán code `c` cho đoạn tương tác dựa vào state/action trajectory dưới ràng buộc policy. Command có thể là `c`, hoặc thêm parameter cục bộ/visual anchor khi code đơn thuần không đủ. Không cấp sẵn full-scene after-state như lệnh đầu vào WM.
5. Train macro WM `F(z_start, e) -> z_end` trên events đã được suy ra; train policy `pi(action | current history, e)` và causal termination. Full-segment posterior có thể nhìn future trong offline training, nhưng candidate generation và stop head ở runtime chỉ dùng thông tin đã quan sát.
6. Candidates là code/local parameter có support từ TRAIN và được đề xuất từ current state; học h và search như hướng đang theo. Không tìm tùy ý trên latent continuous toàn không gian rồi coi điểm WM dự đoán là executable.

Boundary extraction vẫn cần objective và prior. Không cần lookup task hoặc named object labels; cũng không tự trở thành fully learned chỉ vì input là một latent vector. Các khoảng object change dùng cho WM và đoạn approach–interaction–release dùng cho BC cần alignment nhưng không nhất thiết có cùng start time.

## Điểm cần sửa trong đề xuất objective trước

[OPOSM, ICML 2023](https://proceedings.mlr.press/v202/freed23a/freed23a.pdf), Sec. 2.1, chỉ ra latent skill được tối ưu joint bằng action likelihood và endpoint dynamics likelihood có thể mã hóa các biến đổi mà agent không điều khiển được. Paper dùng cập nhật EM để posterior khớp cấu trúc policy/prior; experiments dùng fixed skill horizon. Đây là lý do không coi tổng BC+WM+compression là bảo đảm causal hoặc executable.

Trong đề xuất của mình, phải phân biệt event inference với dynamics fitting. Một lựa chọn là suy ra command chủ yếu theo policy likelihood/compression và prior, rồi fit WM trên command đó, thay vì để WM gradient tùy ý ghi tương lai vào code. Dùng stop-gradient đơn thuần là guard về implementation, không thay thế lập luận của paper hoặc chứng minh causal identification từ pixels/partial observability. Cần kiểm tra predicted successor khớp actual successor khi learned skill được thực thi, không chỉ matched offline segments.

Hindsight local target vẫn có thể là một command hợp lệ: phải sinh được target ấy lúc inference và policy phải thực hiện được. Encoded uncontrolled side effects hoặc full after-scene không được gọi là desired local target để tránh vấn đề này.

## Các nguồn sát câu hỏi

- [LOVE, NeurIPS 2022](https://arxiv.org/html/2212.04590): learned boundaries, skill policy và compression; vẫn cần minimum-duration training constraint, và noisy offline data được nêu là open challenge. Không chứng minh mỗi learned option tương ứng một atomic robot/object event.
- [THICK, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/13b45b44e26c353c64cba9529bf4724f-Paper-Conference.pdf): dùng sparsely changing context và fast latent dynamics để học temporal abstraction. Chứng minh object tracking không phải điều kiện cần cho mọi event-like hierarchy; chưa phải nghiệm đầy đủ cho offline executable interaction vocabulary của mình.
- [Hi-LeWM](https://arxiv.org/html/2607.12547): latent macro-actions/subgoals tốt theo model có thể khó execute do support mismatch. Động cơ giữ finite/data-supported candidates; không khẳng định global WM luôn yếu.

EAWM vẫn là related work về event-aware representation loss. Nó không thay thế phần boundary/skill/candidate discovery cần thiết ở đây.

## Bằng chứng local và những gì chưa được kết luận

- README ghi pipeline puzzle ban đầu dùng vector code bits với detected changes, không cần object-set transformer. Nhưng XOR event identity dựa vào tính chất state-independent effects của Lights Out; không bê sang cube/scene.
- `JOB_LEDGER.md` mục unified representation ghi slow-map attempts đã mất small cubes, jitter hoặc giữ arm/shadow. Đây là thất bại của các recipe cụ thể; không phải thử nghiệm đầy đủ cho mọi global learned latent.
- Compact slot attempt cũng thất bại, và SAM2 tracking/identity còn lỗi. Không được dùng các probe đó để khẳng định object branch đã thắng global branch.
- Các run STATE mới có semantic adapters và separate checkpoints; không đánh giá image frontend hoặc chứng minh representation chung trên pixels.

## Khuyến nghị quyết định hiện tại

Không pivot toàn method sang unstructured global latent. Tiếp tục đường ảnh với local/object features được học, bỏ phụ thuộc vào RGB/cột simulator/semantic names. Giữ event record, command và controller endpoints tách biệt; làm boundary và termination học được với prior chung, đánh giá trong end-to-end run.

Global/spatial latent variant nên kiểm tra cùng câu hỏi và backend, không thành agenda mới. Giữ shared image backbone/pretraining budget, TRAIN episodes, native tasks/horizons, training/eval seeds và planning resources; disclose phần segmentation pretraining khác nhau nếu không match được. Bắt đầu với cùng event supervision để cô lập representation, rồi chỉ thêm learned-discovery comparison nếu cần trả lời sự khác biệt về extraction. Đây là cách phân tích attribution, không phải chuỗi go/no-go gates trước implementation.

Đánh giá cần gồm boundary split/merge, identity/state quality khi applicable, skill success ở held-out starts, WM actual successor sau skill, multi-event closed-loop, và rollout/search errors. Boundary đẹp hoặc thấp latent MSE không đủ. Nếu global variant tốt ngang hoặc hơn trên held-out learned closed-loop và đơn giản hơn ở runtime/training, chọn nó là hợp lý; hiện chưa có số liệu đó.
