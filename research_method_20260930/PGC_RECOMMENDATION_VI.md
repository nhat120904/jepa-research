# Đề xuất thay thế: Predictor-Space Goal Calibration (PGC)

Ngày: 30/09/2026. **Trạng thái: thiết kế nghiên cứu chưa triển khai, chưa train, chưa có kết quả PGC.** Thay thế [OTD](RECOMMENDATION_VI.md) sau [phản biện](OTD_REVIEW_VI.md). Không thay đổi agenda hoặc job của các phiên khác.

## 1. Lựa chọn và phạm vi

Tôi chọn thử **học một hiệu chỉnh goal phụ thuộc context để chấm các future của world model đã freeze**, rồi đánh giá bằng planner closed-loop hiện có. Tên làm việc: Predictor-Space Goal Calibration (PGC).

Ý tưởng: goal được encode từ ảnh thật, còn candidate endpoint do predictor sinh ra. Chúng dùng **cùng hệ tọa độ latent**, nhưng prediction error có thể tạo một sai lệch theo context/horizon khi so khoảng cách. Thay vì fine-tune toàn WM hoặc học scalar reward không ràng buộc, học một dịch chuyển goal dùng chung cho mọi candidate trong lần replan. Goal vật lý và điều kiện success giữ nguyên.

**Arena chính: OGBench-Cube.** Secondary có thể dùng TwoRoom vì runtime/checkpoint đã được hỗ trợ, nhưng chưa có số đo chứng minh đủ headroom. Reacher/PushT gần bão hòa sau sửa baseline là reference/negative controls, không phải arena chính để tìm gain. DINO-WM là lựa chọn kiểm tra chuyển sang kiến trúc WM khác nếu tích hợp kịp; không giả định mọi checkpoint và task đã sẵn sàng.

Đây là lựa chọn có chi phí tích hợp thấp hơn OTD: dùng frozen embeddings, nhánh CEM đã có, objective hook và evaluator hiện có. **Chưa đủ bằng chứng để gọi đây là hướng chắc chắn thành paper.** Novelty hẹp và phải kiếm được bằng hiệu quả, khả năng generalize và đối chứng đơn giản.

## 2. Điều repo đã đo

| Evidence đã ghi | Cách dùng đúng |
|---|---|
| [Ladder Cube](../bottleneck_ladder_20260930/JOB_LEDGER.md): trên 20 candidates/root, native predicted endpoint top-1 success 0,76; true rendered endpoint 0,81; any candidate 0,82. Informative subset n=33: 0,85 → 0,97. | Có khoảng hụt quyết định do dùng future dự đoán. Không phải gain 12 điểm closed-loop và chưa biết phần nào là bias chung. |
| Cùng ladder: history h1/h3 closed-loop 0,675/0,669, 10 paired CEM seeds. | Bổ sung history không tự giải quyết Cube. Baseline mới vẫn phải có context/prefill đúng. |
| [Audit endpoint đã sửa restore](../diagnosis/results/ogb_true_endpoint_corrected/TRUE_ENDPOINT_DECISION.md): predicted latent-L2 16/32; true endpoint 21/32; physical best 25/32. | GoalMSE trên future thật cũng chưa là physical metric hoàn hảo. PGC không được hứa sửa tất cả lỗi này. Đây là cohort khác với ladder mới. |
| [Action-response Cube](../action_response_cube_20260928/README.md): response 35/50 vs prediction-only 36/50; mixed 34/50 vs 37/50. | Response auxiliary chưa thêm gain. Loss đầy đủ vẫn có MSE nên không được nói nó hoàn toàn bỏ qua bias chung. |
| Learned physical scorer 28/50, prediction-only/native cost 36/50; scorer yếu cả trên true held-out endpoints. | Học một cost head mới đã có kết quả yếu; phải đối chiếu, không được chỉ đổi tên. |
| [Fixed CEM refinement](../cem_stopping_20260929/docs/REFINEMENT_RESULT_20260929_VI.md): fixed pair (1,10) khoảng 71% dev; adaptive không thêm giá trị. | Comparison phải gồm lựa chọn fixed mạnh, không chỉ released CEM-30. Con số này chưa là baseline mới chạy cùng PGC. |

## 3. Cơ chế khác với action-response

Gọi z_i là latent của endpoint thật, x_i là latent WM dự đoán, g là goal encoder. Native cost là J_i=||x_i−g||²; cách chia cho số chiều chỉ nhân mọi cost cùng một hằng số.

Xét mô hình sai số lý tưởng hóa **x_i=z_i+b(C)** cho mọi action trong cùng context. Khi đó:

1. x_i−x_j=z_i−z_j: action-response hoàn toàn đúng.
2. Nhưng chênh lệch cost bị sai: (J_i−J_j)−(||z_i−g||²−||z_j−g||²)=2 b(C)^T(z_i−z_j).
3. Dùng goal hiệu chỉnh g̃=g+b(C) khôi phục đúng cost của endpoint thật trong giả định này.

Ví dụ toán một chiều, **không phải experiment**: endpoints thật 0 và 2, goal 0, bias −2. WM dự đoán −2 và 0; native costs 4 và 0 nên chọn sai. Action-response vẫn đúng 2. Dịch goal thành −2 cho costs 0 và 4, chọn đúng.

Điều này chỉ chứng minh một cơ chế có thể tồn tại. **Chưa đo rằng common bias là bottleneck chính của Cube.** Hãy đo nó trong run tích hợp bằng oracle bank-mean bias correction trên dev cache, cùng training và closed-loop; không mở thêm một chuỗi qualification gates.

Giới hạn cũng rõ: nếu x_i=x_j, dịch goal không phân biệt được hai action. PGC không tái tạo dynamics bị mất, không bảo đảm sửa distortion phi tuyến hay metric encoder không tương ứng vật lý.

## 4. Method tối thiểu

### Deployment

- Freeze encoder, WM, preprocessing và normalization của baseline.
- C là lịch sử quan sát và action đã thực thi hợp lệ; g là embedding goal image; H là horizon cố định hoặc được đưa vào input nếu dùng nhiều horizon.
- Một MLP nhỏ uψ(C,g,H), output zero-initialized, trả vector cùng kích thước latent; có norm bound và identity/ridge regularization chọn bằng train/dev.
- Tính g̃=g+uψ một lần mỗi replan, giữ nguyên cho toàn bộ CEM candidates/iterations của lần đó.
- Chấm Jψ(A)=||F(C,A)−g̃||² bằng native CEM. Adapter không đọc candidate A hoặc privileged future; A chỉ đi qua WM như baseline.
- WM rollout không cần gradient theo action để train adapter. Với CTA extension, reader chỉ nhận C, code S và calibrated goal; vẫn không đọc A. Extension đó chưa chứng minh đóng góp riêng của trajectory code.

### Supervision cùng context

Tái sử dụng hoặc thu thập bank M candidate chunks ở cùng context C: action, future ảnh thật, goal ảnh và metadata reset. Freeze encoder để thu z_i thật; frozen WM tạo x_i với **đúng context history**. Primary objective chỉ dùng các image embeddings, không dùng cube distance hoặc success label.

Simulator restore cần privileged state khi thu nhánh, nên mô tả rõ **simulator-assisted branching**; điều này không biến thành privileged deployment. Không gọi việc thu counterfactual branches là miễn phí hoặc giống hệt dữ liệu passive của WM gốc.

Đặt c_i=||x_i−g||², t_i=||z_i−g||², r_i=(c_i−mean(c))−(t_i−mean(t)), X_i=x_i−mean(x). Train:

**Lψ = mean_i[(2 X_i^T uψ−r_i)²] / s² + λ||uψ||².**

s là một scale toàn cục fit trên training data, không phải std riêng của mỗi bank. Centering loại bỏ offset cost chung và tương đương khớp tất cả pairwise cost differences. Bank không có action spread tự cho ít tín hiệu, không được chia cho một std gần 0 để khuếch đại nhiễu.

Các x_i, z_i và g dùng làm target được detach. Gradient chỉ cập nhật adapter. Dùng bank initial/final CEM populations để tránh chỉ học vùng near-elite hẹp; các control phải dùng cùng bank. Nếu planner mới đi ra ngoài proposal coverage, chỉ làm một vòng bổ sung bank có mục đích, áp dụng cùng budget/data cho các arm liên quan.

Với một bank cố định, nghiệm ridge của objective chưa chia scale là u*=2 X^T(4XX^T+λI)⁻¹r. Đây là oracle diagnostic về khả năng biểu diễn của translation, **không phải deployment estimator** vì cần future thật. Không tính oracle trên test rồi feed nó cho planner.

### Không che giấu tương đương toán học

||x−g−u||²=||x−g||²−2x^T u+constant(C,g).

Vì vậy PGC là **một lớp residual cost tuyến tính theo predicted future**, với coefficients phụ thuộc context/goal. Dịch goal cũng tương đương trừ cùng vector khỏi mọi future dự đoán. Không thể nhận novelty cho “bias compensation” hoặc “một cost family hoàn toàn mới”.

Research hypothesis thực sự là: **ràng buộc cùng một hiệu chỉnh query cho toàn candidate set, học bằng relative native costs, có thể cần ít counterfactual data và generalize tốt hơn endpoint-MSE correction hoặc scalar cost head linh hoạt.** Điều này chưa được chứng minh; generic head có expressivity lớn hơn và có thể thắng.

## 5. Các paper top-tier định vị câu hỏi

| Công trình được nhận | Cách đặt vấn đề/design | Phần PGC phải khác và chứng minh |
|---|---|---|
| [UPN, ICML 2018](https://proceedings.mlr.press/v80/srinivas18b.html) | Học representation và differentiable planning end-to-end bằng imitation; latent image-goal metric có ích cho transfer. | Học goal metric đã có. PGC sửa interface của WM đã freeze từ cùng-context futures, không học lại representation qua imitation. |
| [Value Equivalence, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html) | Model có thể tương đương cho planning mà không khớp mọi chi tiết trạng thái. | “Decision-useful hơn reconstruction” không mới. PGC cần một correction cụ thể và evidence control. |
| [CVAML, ICML 2025](https://proceedings.mlr.press/v267/voelcker25a.html), [full paper tác giả](https://cvoelcker.de/assets/pdf/paper_cvaml.pdf) | Chỉ ra calibration failure của value-aware losses với probabilistic models rồi sửa objective. | Đây là prior art về calibration/decision-aware learning. Không lấy lý thuyết stochastic loss của họ làm chứng minh lỗi goal-space của PGC. |
| [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | Predict pretrained spatial visual features rồi planning bằng goal-distance. | Là interface phù hợp để thử adapter; không có bảo đảm derivative, cost geometry hoặc mọi checkpoint cần dùng đều sẵn. |

Chỉ dùng accepted work làm nền thiết kế ở bảng này. Đã tìm thêm các cụm goal calibration/correction, bias và latent planning trên proceedings/CVF; không thấy một bản trùng chính xác qua lượt tìm đó. **Không thấy qua tìm kiếm không phải chứng minh novelty.** Khi viết related work vẫn cần kiểm tra cả các preprint gần nhất, dù không dùng chúng như paper top-tier được nhận.

Điều học từ paper đơn giản là độ khớp giữa câu hỏi, can thiệp và evidence. Không thể suy ra lý do accept chỉ từ số layer hay số loss. Một adapter nhỏ đủ cho paper nếu làm rõ failure mode và tạo kết quả tổng quát; chỉ thêm vài điểm trên một benchmark thì chưa đủ.

## 6. Đối chứng bắt buộc, không dựng baseline yếu

| Arm | Câu hỏi phân biệt |
|---|---|
| Native WM + GoalMSE, corrected history/prefill | PGC có hơn baseline đúng không? |
| Static/global translation học từ train | Chỉ cần offset cố định hay cần context? |
| Context-only bias predictor b(C), train endpoint MSE, score x−b(C) | Có gì hơn sửa bias thông thường? |
| Context+goal translation nhưng train endpoint-bias MSE | Relative-cost supervision có ích riêng hay chỉ architecture? |
| PGC relative-cost loss | Full method. |
| Native cost + scalar residual h(C,g,x), cùng target centered costs, data và gần bằng params | Ràng buộc translation có lợi hơn cost head thông thường không? |
| Prediction-only WM fine-tune; native-cost rank fine-tune nếu budget đủ | Có cần goal adapter hay fine-tune objective trực tiếp đã đủ? |
| Best fixed CEM pair (1,10), và released CEM-30 khi so compute | Gain có vượt việc chọn search budget tĩnh đúng không? |

Native-cost rank follow-up trong [source](../action_response_cube_20260928/native_followup.py) **chưa chạy** và label hiện tại là physical argmin; không được gọi là null result của native-image objective. Nếu dùng làm control, phải đổi/disclose targets. Learned physical scorer lịch sử cũng không phải một đối chứng đã khớp objective và context với PGC.

Có thể staged controls theo câu hỏi, tránh array speculative. Run đầu tiên tích hợp native, context-bias MSE và PGC vào training + native closed-loop. Sau khi xử lý correctness, thêm scalar head và objective ablations; không dùng một offline win để tuyên bố method.

## 7. Tài sản có thật và việc vẫn phải sửa

- [Branch collection](../action_response_cube_20260928/pipeline.py) đã có same-state restore, candidate actions, encoded actual endpoints, goal và episode/start metadata. Cache cũ **chỉ lưu context một frame**; không giả định predicted endpoints của cache đã là history-3 corrected baseline. Tái dùng actual targets/actions khi xác nhận hash/stop semantics; reconstruct context và recompute predictions trên compute node.
- [Task wrappers](../state_estimation_20260930/se/tasks.py) có Cube, Reacher, PushT và TwoRoom, cùng checkpoint mappings và solver/execute interface. [PrefillPolicy](../state_estimation_20260930/se/policy.py) có causal context/past actions cho first plan. Source đã đọc; chưa runtime-test PGC hook.
- [Ladder source](../bottleneck_ladder_20260930/scripts/ladder.py) cung cấp correct history và true-endpoint collection/scoring. Dùng lại, không tự viết simulator reset khác.
- Align **5 action blocks × 5 simulator steps**, normalization, goal offset và stopping. Branch thật có thể terminate sớm; WM dự đoán đủ horizon. Phải thống nhất masks/padding/terminal targets hoặc thu nonterminal endpoint đúng H, không trộn nhãn silently. Giữ evaluator native để comparison có nghĩa.
- Goal adapter output dimension tùy backbone. Với patch-latent DINO-WM, cần shared patch mapping hoặc low-rank output; đây là tích hợp bổ sung, chưa là tài sản đã chạy.

## 8. Evidence cần để thành một bài

Main claim dự kiến, **chưa được phép viết như kết quả**: query calibration sửa một phần decision error của frozen visual WM với ít adaptation data và chi phí deployment nhỏ.

Evidence nên gồm: native closed-loop success và paired intervals trên held-out episodes; biến thiên training seeds; khả năng chuyển sang goal/context chưa dùng để fit; đường data-efficiency của vài mức data có chủ đích; ít nhất arena thứ hai và ưu tiên WM thứ hai; total latency mean/p95 và chi phí thu branches/train. Offline cost differences và oracle là giải thích cơ chế, không thay thế control.

Test episodes tách theo episode, không chỉ candidate. Dev không dùng làm test. Existing “sealed” manifests của một programme không tự bảo đảm chưa bị programme khác dùng; kiểm tra overlap từ metadata trước khi khóa manifest mới. Baseline và PGC dùng cùng roots, solver seeds, candidates trong offline comparisons, horizon, action execution, reset, history và rollout count; khác compute phải công khai.

Không đòi 25 training seeds theo SD của một metric khác. Với head nhỏ có thể bắt đầu 3–5 training seeds; đủ hay không phụ thuộc paired uncertainty thực đo. Closed-loop roots lặp qua CEM seeds không được coi là training seeds độc lập.

Nếu chỉ global/context bias MSE đã bằng PGC, contribution goal-conditioned relative-cost không được chứng minh. Nếu scalar head ngang hoặc hơn trên cùng data/compute và generalization, không gọi translation tốt hơn. Nếu oracle translation cũng không giúp trong bank đủ thông tin, chẩn đoán action-dependent/metric error và sửa đúng quan sát; không thêm vài loss để cứu tên method. Đây là diễn giải kết quả trong workflow tích hợp, không phải gate bắt người dùng duyệt thêm trước implementation.

## 9. Lịch và ngân sách

[CVPR 2027](https://cvpr.thecvf.com/Conferences/2027/Dates): registration **10/11/2026 AoE**, paper **16/11/2026 AoE**, supplement **23/11/2026 AoE**. Từ 30/09 còn khoảng 47 ngày tới paper deadline.

- 01–04/10: Cube integrated implementation, cache/history alignment, train small heads, closed-loop native comparison và bias diagnostic trong cùng run.
- 05–11/10: sửa correctness/bottleneck đã quan sát; scalar-head, MSE và objective controls; đo total latency/data collection cost.
- 12–24/10: secondary arena hiện có, một WM khác nếu tích hợp hợp lý; khóa independent evaluation và training seeds. Không xây RoboMimic arena mới làm dependency bắt buộc.
- 25/10–03/11: held-out confirmations, uncertainty và figures; viết song song từ đầu.
- 04–16/11: hoàn thiện evidence/limitations, paper và reproducibility. Không dùng deadline để nới quota.

Đây là lịch dự kiến, không phải cam kết hoàn thành. Training head nhẹ nhưng simulator/encoding và closed-loop vẫn tốn compute. Chỉ ước lượng tổng budget sau khi đo bounded pilot; không biến số dự phóng tháng 10 thành quyền submit.

Theo [AGENTS.md](../AGENTS.md), mọi model loading/training/physics/encoding/bulk analysis chạy sbatch, explicit resources/time, peer checks; trước GPU submission/CPU array kiểm tra tháng hiện tại và giữ mức sử dụng kể cả planned time-limit dưới 50% người thứ 5 về GPU/CPU/memory. squeue+sacct bắt buộc khi báo live job state. **Lượt nghiên cứu này không load model, không chạy physics, không submit job và không xác minh live quota.**
