# Chọn hướng tiếp theo từ một arena đã có

Ngày rà soát: 2026-09-08. Trạng thái: **ứng viên cho phép thử rẻ, chưa phải programme đã được qualify**. Không có thí nghiệm mới trong lần research này.

## Quyết định

Tôi đề xuất kiểm tra **học dynamics ổn định trước sai lệch ngoài tập trạng thái hợp lệ, trong khi giữ độ nhạy với chuyển động thật**. Điểm xuất phát là **checkpoint LeWM–Reacher và reset/replay của ORDER**, không phải một task occlusion hoặc contact mới. Nguyên lý mượn từ học hệ động lực trên manifold và ổn định ngang manifold (*transverse stability*).

Đây là hướng tôi ưu tiên kiểm tra trong các ứng viên vừa rà, **không phải kết luận đã tìm thấy novelty đủ mạnh cho top conference**. Bản thân Jacobian regularization, manifold projection và ổn định dynamics đều có prior art. Chỉ nên phát triển thành method nếu có bằng chứng rằng phân biệt sai lệch biểu diễn với biến thiên vật lý mang lại lợi ích planning mà các cách sửa thông thường không có.

Đề xuất belief/channel trước giữ nguyên `STOP_BEFORE_ARENA`. Không xây arena cho nó. Không hồi sinh ORDER, không gộp các null cũ thành bằng chứng rằng predictor ở mọi môi trường đều hết headroom.

## 1. Có một con số thật trước khi xây method

[RUN_STATUS.md của ORDER](/Users/nhatcuong/code_project/vin-research/order-jepa/RUN_STATUS.md) ghi nhận trên LeWM–Reacher:

| Phép chọn trên cùng panel candidate | Thành công |
|---|---:|
| Mã hóa endpoint thật rồi chọn bằng latent cost | 100% |
| Chọn bằng endpoint do predictor dự đoán | 41,5%, CI 95% [35,0%; 48,5%] |

Dynamics excess physical regret là 1,3007, CI [1,1287; 1,4859]. Panel có 200 anchors thuộc 25 trajectories. Witness và các cặp diagnostic không tham gia lựa chọn ordinary candidate.

**Giới hạn quan trọng:** đây là khả năng chọn trên một panel cục bộ được tạo quanh một action sequence khả thi. Goal được tạo từ witness. Con số 100% không phải oracle closed-loop success trên toàn benchmark, và 58,5 điểm phần trăm không phải mức method có thể thu hồi. Candidate coverage thuận lợi là một phần thiết kế. Khoảng tin cậy đã cluster theo episode; không được coi 200 anchors là 200 episode độc lập.

Tuy vậy, nó chứng minh được một điều hẹp và hữu ích: **trong setup này, có lỗi quyết định do rollout, trong khi cost của endpoint thật hoạt động trên chính panel ấy**. Đây là lý do chọn arena, không phải chứng minh cơ chế mới.

ORDER vẫn bị loại: predicted order-vector cosine 0,8936, amplitude ratio 1,0115; khoảng cách order-ranking chỉ 1,11 điểm phần trăm với CI chứa 0. Không có cơ sở sửa “thiếu thứ tự action”.

Original DINO-WM PushT là điểm bắt đầu kém hơn cho một predictor method: sau loại witness, true-latent và predicted-latent selection đều 36,5%; CI của dynamics excess regret chứa 0. Tăng chất lượng predictor ở đó có thể lại cải thiện metric mà không cải thiện quyết định.

## 2. Câu hỏi và nguyên lý

**Câu hỏi:** khi rollout tự dùng latent nó vừa dự đoán làm input, sai số có bị khuếch đại theo những hướng không tương ứng với một lịch sử quan sát vật lý hợp lệ không? Nếu có, có thể dập những sai lệch ấy mà vẫn giữ các khác biệt do trạng thái và action thật gây ra không?

Một ví dụ trực quan: hai cấu hình cánh tay thật gần nhau tạo ra hai latent khác nhau. Model cần giữ khác biệt đó để phân biệt action. Nhưng một latent trung gian do model tưởng tượng có thể lệch theo hướng không giống bất kỳ cấu hình/lịch sử thật nào. Predictor không nhất thiết đã được huấn luyện cách phản ứng với sai lệch này.

Làm toàn bộ predictor “mượt hơn” có thể giảm cả hai loại khác biệt. Đề xuất chỉ có giá trị nếu **cấu trúc hướng sai lệch** giúp tránh đánh đổi đó. Cần kiểm tra trên lịch sử ba frame của checkpoint, không áp một giả định Markov lên một frame.

Trong phương án method, giữ encoder cố định; ước lượng không gian biến thiên cục bộ từ những lịch sử thật của tập train; fine-tune predictor để giảm việc sai lệch ngoài không gian đó lan thành sai số rollout. Loss dự đoán và đối chứng biến thiên thật giữ lại action/state sensitivity. Công thức, estimator và giới hạn nằm trong [METHOD_SPEC_EN.md](METHOD_SPEC_EN.md).

**Không đồng nhất ba thứ:** ổn định khỏi manifold dữ liệu, ổn định vật lý của robot và độ thẳng của quỹ đạo theo thời gian. Robot có thể có dynamics vật lý giãn khoảng cách, và contact có thể không trơn. Không được ép contraction toàn hệ rồi gọi đó là physical consistency.

## 3. Prior art giới hạn claim như thế nào?

| Nguồn primary | Đã làm gì liên quan | Hệ quả cho hướng này |
|---|---|---|
| [Learning vector fields… geometrically constrained operator-valued kernels, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/318f3ae8be3c97cb7555e1c932f472a1-Abstract-Conference.html) | Học vector field với ràng buộc tangent và tích phân giữ geometry | Không claim phát minh học dynamics trên manifold hoặc normal correction |
| [Towards Unraveling and Improving Generalization in World Models](https://arxiv.org/html/2501.00195v1) | Đề xuất input–output Jacobian regularization cho rollout | Global Jacobian regularization là baseline bắt buộc; acceptance chưa xác minh |
| [GRASP, bài tác giả giải thích cơ chế](https://bairblog.github.io/2026/04/20/grasp/) và [paper](https://arxiv.org/abs/2602.00475) | Nêu độ nhạy ở normal directions của state input; dùng action gradients trong lifted planning | Không claim phát hiện failure này đầu tiên. Cần phân biệt sửa learned dynamics với thay planner |
| [A Control Theory of Predictability in Latent World Models](https://arxiv.org/html/2607.10362v1) | Phân tích off-manifold error, query distribution và plan-cost discrepancy | “Prediction error thấp không đủ cho control” không còn là novelty; đây là preprint |
| [Sub-JEPA](https://arxiv.org/html/2605.09241v1) | Điều chỉnh Gaussian regularization của representation bằng random subspaces | Không đánh tráo subspace anti-collapse thành dynamics stability; preprint trong ledger này |

Có xác nhận proceedings trực tiếp cho paper manifold ở ICLR 2025 và [DINO-WM ở ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html). Một số công trình 2026 ở trên được dùng làm **novelty threats dù chưa xác minh acceptance**. Bài GRASP có venue link trong ledger cũ, nhưng lần này trang conference không truy cập được; không nâng trạng thái chỉ từ ledger. Danh sách này là rà soát hướng cụ thể, không phải danh mục đầy đủ mọi paper accept 2025–2026.

Các hướng khác bị hạ ưu tiên trong lượt này:

- Parallel/multi-horizon prediction và consistency: [Fast LeWorldModel](https://arxiv.org/abs/2606.26217) đã tiến vào đúng không gian này; repo còn có null rollout-matched training. Chưa có khác biệt method đủ cụ thể.
- Online sửa residual từ quan sát: [Feedback World Model](https://arxiv.org/abs/2605.15705) đã dùng feedback state/latent observer cho robot control. Không đề xuất lại như idea mới.
- Belief/channel matching: bị reduction audit của chính repo loại trước khi xây arena.
- Generic ensemble, rerank, acquisition, temporal/action straightening: các đối chứng trong repo đã bác bỏ những phiên bản cụ thể. Không dùng tên mới để bỏ qua kết quả cũ.

## 4. Arena này thực sự thừa hưởng được gì?

| Thành phần | Chọn cụ thể | Mức xác minh |
|---|---|---|
| Simulator/task | `swm/ReacherDMControl-v0`, `qpos_match`, RGB 224 | Collector có source; log cũ ghi exact reset/pixel/repeat error bằng 0 |
| Checkpoint | `quentinll/lewm-reacher`, revision `62adae4b71dc474ddf8f794c476ebfe737a743ca` | HF metadata và config được lấy trực tiếp ngày 08/09; log cũ ghi đã score weights |
| Model | ViT-tiny patch14; embedding192; predictor6 layers, history3, hidden192, MLP2048 | Đọc config pinned, đã lưu bản sao |
| Action/controller | Hai action liên tục của môi trường; block5; horizon5 blocks | Không cần skill library hoặc low-level controller mới |
| Data policy | Random action Uniform[-1,1], collector cũ có deterministic seeds | Có code thu, chưa tạo dataset mới trong lượt này |
| Goal/cost | Goal image; mean squared latent distance | Giữ nguyên để cô lập dynamics |
| Success | Mọi joint có wrapped angular error <0,05 rad | Có hàm chính xác trong `order_jepa/core.py` |
| Oracle/replay | Restore physics state và replay từng candidate | Có implementation và log qualification |
| Baseline đầu | Released model, cùng candidate và cost | Có kết quả cũ để đối chiếu |

LeWM là encoder huấn luyện end-to-end từ pixels; **không phải checkpoint trên frozen DINOv2**. Chúng ta freeze encoder ở phép thử này để kiểm soát biến số. [Paper LeWM](https://arxiv.org/html/2603.19312v2) mô tả họ model nhỏ khoảng 15M tham số; thông số chính xác cho checkpoint lấy từ config, không suy ra số tham số bằng tên model.

**Không được nói “mọi thứ đã có cache”.** Collector Reacher lưu các scalar/candidate rows; nó không lưu đầy đủ latent history và từng intermediate endpoint cần cho phép thử mới. Cần mở rộng phần ghi dữ liệu và chạy lại replay. Đây là instrumentation của arena có sẵn, vẫn tốn công, nhưng không phải thiết kế task/controller/signal từ đầu. Source `stable-worldmodel` và weights trên server chưa được kiểm tra live trong lượt này; bản pull local không đủ để tự nhận runnable end-to-end trên Mac.

## 5. Phép thử đầu tiên, trước training

Thiết kế cụ thể ở [METHOD_SPEC_EN.md](METHOD_SPEC_EN.md). Trình tự bắt buộc:

1. **Replay 32 anchors dev**, lưu latent sau mỗi block, so autoregressive rollout với dự đoán bước cuối được cấp history thật. Oracle thứ hai chỉ để định vị error accumulation, không phải baseline deployable. Nếu ngay cả teacher forcing cũng không cải thiện quyết định thì premise về lan truyền sai lệch yếu đi rõ rệt.
2. Dùng một bank lịch sử train, thử phép chiếu cục bộ không cần fine-tune. So với không sửa, global PCA, nearest-neighbor, shrinkage và random subspace có correction norm được ghép tương đương. Nếu random/shrinkage làm tốt như nhau, hướng geometry không được qualify.
3. Kiểm tra lại trên candidate do CEM tạo ra, không chỉ candidate quanh witness. Tách panel để biết gap có thật ở distribution planner truy vấn hay chỉ do thiết kế diagnostic.

Các phép thử phải báo **physical selected regret và success**, cùng latent error/action sensitivity. Latent error, normal gain hoặc “ra ít trạng thái lạ hơn” không tự mở training.

Ngân sách đề xuất trước số liệu đầu tiên: tối đa **một ngày công instrumentation**, **8 GPU-hours và 64 CPU core-hours** cho preflight/smoke; dừng nếu vượt, không tự mở rộng thành dự án hạ tầng. Đây là **trần chi**, không phải benchmark runtime đã đo. Không có giờ GPU nào đã được sử dụng hoặc job nào đã được submit trong lượt research.

## 6. Khi nào xứng đáng thành method paper?

Chỉ khi cùng lúc có ba kết quả:

- Cấu trúc tangent/normal đáng tin trên held-out histories và hữu ích hơn random subspace/equal-energy damping; không xóa các phản ứng vật lý hợp lệ.
- Learned method cải thiện physical decisions hơn **multi-step fine-tuning, isotropic noise/Jacobian regularization và projection không train**, trên cùng data, checkpoint và compute.
- Lợi ích sống sót khi CEM tối ưu lại và khi chạy closed-loop; được lặp trên một manipulation arena đã có, chẳng hạn LeWM–OGBench Cube, sau một gate độc lập tại đó.

Reacher chỉ là control pilot. Thắng Reacher chưa đủ cho claim robot manipulation. Cube là nơi để cố bác bỏ khả năng transfer, không được mặc định sẽ thắng: [rollout_repair_gate](../../../vin-research/rollout_repair_gate/outputs/stage1/analysis/DECISION.md) đã có `STOP_DYNAMICS_REPAIR`. Không train Cube trước khi xác minh lại đúng mechanism/headroom trên replay sẵn có.

Nếu kết quả cuối chỉ là “thêm một anisotropic penalty thắng vanilla một ít”, tôi cũng chưa khuyến nghị viết method paper top conference. Nếu projection đơn giản đã giải quyết toàn bộ lợi ích, không thêm mạng để tạo cảm giác contribution lớn. Đóng góp đáng theo đuổi phải là chứng minh được **phần dynamics nào cần ổn định, phần nào phải giữ nhạy, và phân biệt ấy tạo lợi ích control mà baseline mạnh bỏ lỡ**.

## 7. Khuyến nghị đầu tư

Cho ứng viên này **một phép thử có giới hạn**, chưa dành phần lớn ngân sách mới cho nó. Nó đứng trước proposal occlusion/belief vì có checkpoint, arena, reset và một decision gap đã đo; nó chưa có bằng chứng hơn baseline để đứng vào vị trí programme chính. Kết quả research hiện tại là lựa chọn một câu hỏi có thể bị bác bỏ rẻ hơn và có spec rõ hơn, không phải một lời bảo đảm rằng lần này sẽ có paper.
