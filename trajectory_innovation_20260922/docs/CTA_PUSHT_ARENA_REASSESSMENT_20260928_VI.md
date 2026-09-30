# CTA: độ mạnh của PushT và lựa chọn arena — 28/09/2026

Đây là review theo yêu cầu nghiên cứu, không phải kết quả thực nghiệm mới. Đã đọc README, ledger mới nhất,
RESEARCH_DESIGN (kể cả §5a), protocol/code local, summary 55351 và nguồn benchmark/paper gốc trên web.
Không submit job, không thay runner hay cấu hình của các phiên đang chạy. Đánh giá khả thi dưới đây là nhận định,
không phải bằng chứng CTA đã thắng arena mới.

**Kết luận:** PushT đủ làm kết quả phát triển đáng báo cáo, chưa đủ làm bằng chứng chính cho một paper CTA mạnh.
Ưu tiên thực nghiệm trajectory trên **LIBERO-Safety TSA-L1 + FSHOA-L1**; phương án thứ hai là
**VLA-Arena safety_dynamic_obstacles + safety_hazard_avoidance**. Giữ PushT làm endpoint/control arena.
RinseBowls là ứng viên tốt về native temporal progress nhưng chưa có mức sẵn sàng tương đương.

**PushT: số liệu đã kiểm tra trực tiếp.**

Nguồn: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_onpolicy8_aggregate_55351/summary.json`.
200 paired development roots 2200–2399; một training seed. Cả 55349_0–7 và 55351 COMPLETED,
đối chiếu squeue và sacct trong lượt review này.

| Arm | Success | Native normalized score |
|---|---:|---:|
| P0 | 122/200 = 61% | .96087 |
| CTA4 | 142/200 = 71% | .96141 |
| CTA8O | 138/200 = 69% | .95597 |
| ENDPOINT8O | 138/200 = 69% | .96456 |
| DIRECT8O | 124/200 = 62% | .95790 |
| GEOM8, privileged selector | 160/200 = 80% | .97050 |

CTA4−P0 success +10 pp, paired bootstrap CI [2,18], McNemar p=.02446; 46 roots chỉ CTA4 thành công,
26 roots chỉ P0 thành công. Native-score difference +.00053, CI [−.01716,.01783].
CTA8O−ENDPOINT8O 0 pp [−8,8]; CTA8O−DIRECT8O +7 pp [−2.5,16].

Diễn giải:

- Có tín hiệu learned control thực sự, không chỉ offline ranking. Không nên gọi kết quả vô giá trị.
- CI và p-value là mô tả exploratory trên development. Nhiều round/chọn cấu hình và một seed không cho phép
  đọc p=.024 như một xác nhận độc lập sau khi freeze phương pháp.
- Primary metric của Round 3/4 là normalized native score; success tăng trong khi score gần như không đổi.
  Một giả thuyết là thay đổi quanh ngưỡng success, nhưng summary chưa chứng minh cơ chế này.
- CTA8O và ENDPOINT8O bằng success không chứng minh tương đương; khoảng bất định vẫn ±8 pp.
  Nó cũng chưa chứng minh lợi ích riêng của trajectory. Hai arm có lịch warm-start khác nhau (ledger).
- GEOM8 là một heuristic dùng tương lai thật để chọn chunk, không phải trần tối ưu mọi planner.
  80% là headroom của selector cụ thể, không phải mục tiêu mà CTA chắc đạt được.
- Không so 71% trực tiếp với số PushT của DINO-WM/LeWM/D-JEPA khi khác roots, policy bank, horizon,
  termination, mục tiêu và ngân sách. Phải có comparison trong cùng protocol hoặc bảng tách biệt.

Để thành bảng chính đáng tin: freeze công thức sau lượt development hiện có; train lại từ đầu nhiều seed;
test trên roots chưa dùng chọn mô hình; chốt metric trước khi mở test; so ENDPOINT, cached/factorized DIRECT,
compact per-step/parallel WM trên cùng dữ liệu và candidate bank. Báo uncertainty theo root và biến thiên training
seed, không coi mọi seed×root là môi trường độc lập. Đo scorer và total latency riêng trên cùng GPU, K và horizon.
400 roots và ba seed là một thiết kế dự kiến trong review trước, không phải bảo đảm significance.
Nếu claim ngang chất lượng nhưng rẻ hơn, định nghĩa non-inferiority margin trước test.

**Arena nào thực sự kiểm tra trajectory?**

Điều cần là thông tin sự kiện trong đoạn thực thi ảnh hưởng mục tiêu, nhưng không được đại diện tốt bởi
endpoint observation với cùng context. Long-horizon, nhiều subtask hoặc có vật cản tự chúng chưa đủ.
Ví dụ phù hợp: hai chunk đưa cốc tới gần cùng một vị trí, một chunk chạm vật cản giữa đường rồi tách ra,
chunk còn lại tránh được. Đây là ví dụ cơ chế cần tìm trong dữ liệu thực, chưa phải counterexample đã đo.

Phân biệt endpoint ảnh/proprio với full simulator state có các cờ lịch sử. Nếu cấp cho endpoint baseline
bit 'đã va chạm' hay timer 'đã rửa', nó đã có một temporal summary; nếu vô tình cấp chúng cho CTA thì leakage.
Một endpoint predictor nhận C,A cũng có thể gián tiếp mã hóa thông tin đường đi nếu representation/loss cho phép.
Vì thế cần operational definition rõ, không tuyên bố bất khả thi tuyệt đối cho mọi endpoint model.

CTA hợp nhất khi nhiều sự kiện quan sát được xảy ra trong horizon, code giữ chúng rẻ hơn dự đoán cả chuỗi,
và nhiều query có ý nghĩa dùng lại code. Với duy nhất một cost nhị phân, predictor trực tiếp một scalar là
baseline rất mạnh. Chỉ thắng endpoint ở collision không đủ chứng minh learned abstraction hơn direct risk head.

**So sánh các ứng viên.**

| Arena | Bằng chứng phù hợp | Rủi ro/quyết định |
|---|---|---|
| LIBERO-Safety TSA-L1, FSHOA-L1 | Native contact constraints, moving obstacles; released π0.5; adapter local đang dựng | Ưu tiên thực dụng. Chưa có learned CTA result; cần đủ safe/unsafe alternatives trong bank |
| VLA-Arena DynamicObstacles, HazardAvoidance | Native costs theo quá trình, task assets và π0 checkpoint public | Dự phòng tốt; cần đo lại policy competence và restore, không suy headroom từ leaderboard |
| RoboCasa RinseBowls | Native continuous dwell/reset; progress phụ thuộc lịch sử | Tốt nhất trong shortlist nếu yêu cầu tiến độ task thay vì safety; policy/camera/horizon còn chưa qualify |
| Safety-Gymnasium Goal/Push với hazard | Cumulative cost phân biệt đi qua hazard rồi rời khỏi nó | Mechanism experiment gọn; cần xác minh pixel inputs và policy phù hợp, không giả định vision chỉ vì tên suite |
| SafeVLA-Bench trên LIBERO/RoboCasa | Nhiều query safety/temporal, tận dụng benchmark hiện có | Rất đáng theo dõi cho query reuse; chưa xác minh official evaluator release trong lượt này |
| SafeStage; ForesightSafety-VLA | Có execution-time/process risk | Mở rộng liên quan; tích hợp, observability và checkpoint chưa được audit đủ để xếp trước lựa chọn đang có |
| OGBench cube, LIBERO-Goal thường | Endpoint progress/control có ý nghĩa | Hợp cho generalization; không tự chứng minh whole-path necessity |

VLA-Arena có native cost predicates như contact, force, distance và các suite được đặt tên ở bảng trên:
[repository](https://github.com/PKU-Alignment/VLA-Arena),
[cost specification](https://github.com/PKU-Alignment/VLA-Arena/blob/main/docs/scene_construction.md).
Đã xác minh có [π0 LoRA checkpoint chính thức](https://huggingface.co/VLA-Arena/pi0-vla-arena-fintuned-LoRA),
không cần mặc định train VLA mới. Chưa chạy checkpoint này local.

RinseBowls tại pinned commit yêu cầu 25 update liên tục dưới nước; rời nước trước ngưỡng làm reset timer;
cờ rinsed đã đạt được giữ lại. [Native source](https://raw.githubusercontent.com/robocasa/robocasa/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/washing_dishes/rinse_bowls.py).
Nhận định: liên tục 25 và 13+ngắt+12 có cùng tổng thời gian nhưng khác kết quả; đây là probe continuity hợp lý.
Tuy vậy, cần history đủ dài dùng chung cho mọi arm và một event predictor + counter/reset baseline.
Xem audit local `docs/CTA_TASK_QUALIFICATION_20260926_VI.md` về dữ liệu, horizon và chưa có policy qualification.
Không phục hồi chuỗi gate cũ: khi triển khai, tích hợp các diagnostic này vào một lượt end-to-end có giới hạn.

Safety-Gymnasium có hazard costs; BuildingGoal1 ghi risk-area cost nhưng tài liệu cũng liệt kê lidar observations.
Do đó phải khóa wrapper pixel/proprio thực tế trước khi gọi đây là visual CTA benchmark.
[BuildingGoal documentation](https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_vision/building_goal.html).

[SafeVLA-Bench](https://safevla.org/) thêm đánh giá STL sau rollout trên LIBERO/RoboCasa; trang cập nhật 26/09
ghi π0.5 LIBERO SR 96.6%, safety 77.6%, SBU 21.4%. Đó là số bên ngoài, không phải matched-bank oracle headroom.
Nhận định: hấp dẫn hơn một cost duy nhất cho query reuse, nhưng force spikes có thể khó suy từ ảnh.
Không nhầm với repository Jatshi/SafeVLA-Bench về speech/clarification, một dự án khác tên giống nhau.

[SafeStage](https://arxiv.org/abs/2609.21223) tách rủi ro trước/trong/sau thao tác;
[ForesightSafety-VLA](https://arxiv.org/abs/2606.27079) dùng process costs/risk exposure trên RoboTwin.
Nhận định: cả hai đáng có trong related benchmark discussion; hiện chưa đủ xác minh triển khai để chuyển sang.

**Vì sao ưu tiên LIBERO-Safety, và giới hạn của khuyến nghị.**

[Code chính thức](https://github.com/LIBERO-SAFETY/LIBERO-Safety) và
[π0.5 checkpoint](https://huggingface.co/LIBERO-Safety/pi05_libero_safety) đã phát hành.
Ưu thế thực dụng là proposal policy có sẵn và workspace đã có runtime, snapshot/restore, branch và evaluator.
Chọn toàn bộ 5 task TSA-L1 và 5 task FSHOA-L1 theo semantics trước khi xem CTA thắng task nào.
Không chọn riêng root có collision để báo main success.

Audit BDDL local trong `docs/CTA_LIBSAFE_PROTOCOL.md` đính chính: chỉ 13/45 task contact có obstacle thực sự
di chuyển, không phải 35/45. L1 hai suite trên chiếm 10 task; ba task còn lại là cùng HRI banana ở L0–L2.
Mocap obstacle không giữ dấu dịch chuyển sau contact như vật tự do, nhưng điều đó không chứng minh endpoint
không thể suy collision từ các cue khác. Cần đo endpoint-matched diagnostics và kết quả trên toàn phân phối.

Protocol local ghi π0.5 collision/violation khoảng 6–16% episode tùy suite/level.
Trang tác giả mô tả nhiều thất bại là collision-free incompletion:
[failure cases](https://libero-safety.github.io/#evaluation).
Vì vậy không thể diễn giải toàn bộ 35–45% failure là cơ hội trajectory-safety. Chặn một va chạm còn có thể
dẫn tới timeout; tỷ lệ va chạm không chuyển 1:1 thành safe-success gain.

Khác biệt evaluator mới phát hiện: [paper §4.1](https://arxiv.org/html/2606.23686v1#S4.SS1)
dừng ngay khi vi phạm, nên SR đã là safe success. Trong local `scripts/libsafe/headroom.py:66–90`, cost được
ghi nhưng loop chỉ dừng tại success/horizon; `ti_wm/libsafe_runtime.py:190–199` cũng vậy khi branch.
Không dùng local raw success so với published SR. Với cùng prefix, horizon và reset, local safe_success có thể
cho cùng outcome nhị phân như early-stop, nhưng bước chạy, chi phí và dữ liệu sau violation khác nhau.
Nên dùng native termination cho main evaluation; nếu giữ continuation để phân tích, ghi rõ diagnostic variant.
Đây là chênh lệch protocol có thể quản lý, không phải lý do kết luận toàn bộ headroom job vô ích.

Không để terminal flag, độ dài branch hoặc padding tiết lộ nhãn cho endpoint/source oracle diagnostic.
Collision ngắn có thể xảy ra giữa hai frame sampled: phải kiểm tra cadence và full-path reader, không mặc định
mọi native contact/force label đều quan sát được từ RGB. History giúp ước lượng chuyển động obstacle phải giống
nhau ở tất cả các arm. Safety label supervision được mô tả rõ ở reader; không đưa simulator state/event labels
vào deployment C hoặc S và không gọi toàn hệ fully self-supervised.

**Lượt thực nghiệm có sức phân biệt, không mở một sweep arena.**

Tận dụng setup/headroom đang có; nối một collection → training → closed-loop run trên hai L1 suite.
K=8 với frozen π0.5 là điểm bắt đầu theo protocol local. Tất cả arm dùng cùng cách sinh bank, context, horizon,
training split và labels; sau khi hành động khác nhau, context tự nhiên khác nhau, nên không nói bank vẫn giống
nhau giữa những trạng thái khác nhau. Ghép episode theo root và seed.

| Control | Câu hỏi nó giải quyết |
|---|---|
| P0 | Reranking có cải thiện chính policy nguồn? |
| ENDPOINT | Intermediate information có thêm ích lợi ngoài frame cuối? |
| Factorized DIRECT: z=f(C,A), score=h(z,q) | Target học từ future trajectory có ích hơn bottleneck task-only? |
| Direct multihead progress/risk | Một vài sufficient statistics định sẵn có giải task rẻ hơn? |
| Compact FRAME, có parallel multi-step prediction | CTA có trade-off tốt hơn mô hình giữ chuỗi? |
| Per-step event predictor + aggregation/monitor | Có cần learned segment abstraction cho họ query này? |
| Actual-path / source-code / predicted-code diagnostic | Lỗi từ khả năng quan sát, nén, hay dự đoán? |
| Simulator oracle, báo riêng | Trong bank có action hữu ích mà selector học được bỏ lỡ? |

Metric chính nên là native safe success. Báo thêm collision/violation, incompletion, execution time, scorer
median/p95 và total median/p95; không thưởng policy đứng yên chỉ vì cost thấp. Mọi compute oracle/branch
chỉ phục vụ collection/evaluation, không tính như deployable CTA.

Primary task labels lấy trực tiếp từ benchmark. Dùng query theo goal và constraint identity có sẵn; nếu
thêm thresholds/windows hoặc hold-out query compositions, gọi rõ đó là diagnostic extension. Đổi vài trọng số
của cùng hai scalar không đủ chứng minh query-many. Cache z của DIRECT và predicted sequence của FRAME công bằng.

Ở đoạn thực thi h=5 trong một proposal H=10, safety metric trực tiếp gắn với prefix đã commit. Không ghi vi phạm
trên suffix chưa execute như observed outcome của planner. Nếu cần H dài hơn để thấy lợi thế, công khai thay đổi
commitment và áp dụng cho mọi arm; không kéo dài chunk riêng cho endpoint để làm nó yếu.

Một kết quả mạnh có thể là CTA tăng safe success so với ENDPOINT và DIRECT, hoặc giữ chất lượng của FRAME
trong một margin xác định trước với latency/memory thấp hơn rõ. Nếu chỉ giảm risk prediction error offline,
hoặc chỉ thắng P0, claim representation vẫn còn thiếu. Không phải cả hai dạng thắng đều bắt buộc; claim phải
theo đúng bằng chứng đạt được.

**Literature cập nhật và ranh giới đóng góp.**

| Nguồn gốc | Điều CTA không thể coi là mới riêng |
|---|---|
| [Fast-LeWM, 24/06/2026](https://arxiv.org/abs/2606.26217) | Action-prefix prediction, nhiều future latents song song |
| [Hi-LeWM, 14/07/2026](https://arxiv.org/abs/2607.12547) | Temporal hierarchy trên LeWM |
| [DA-LeWM, 19/08/2026](https://arxiv.org/abs/2608.18746) | Căn chỉnh biểu diễn với thứ hạng planning |
| [D-JEPA, 21/09/2026](https://arxiv.org/abs/2609.24749) | Học quan hệ decision-relevant giữa candidate futures |
| [AD-WM, 24/09/2026](https://arxiv.org/abs/2609.30264) | Giữ khác biệt action cho counterfactual MPC |
| [AdaJEPA](https://github.com/agentic-learning-ai-lab/adajepa), [Sandwich-Residuals, 18/09](https://arxiv.org/abs/2609.21740) | Thích ứng world model lúc test |
| [RaMP, NeurIPS 2023](https://boyuan.space/ramp-rl/) | Dự đoán cumulative features theo action chunk để đổi reward |

Các nguồn trên xác lập overlap; không dùng raw leaderboard khác protocol để phán CTA thua/thắng.
Giá trị có thể bảo vệ của CTA là **conditional segment representation học từ tương lai, dự đoán được từ action,
giữ thông tin cần cho một họ truy vấn đường đi, với trade-off quality/compute đo được**.
Nếu chỉ còn hai head progress/collision thì phải giải thích bằng thực nghiệm vì sao cần codec.
Nominal 128-bit source indices cũng chưa bằng deploy code: các review local đã ghi expected FSQ code là
48 continuous scalars. Báo actual deployment representation; hard-code compression là ablation riêng.

**Trạng thái quan sát được trong lượt review.**

Squeue+sacct ngày 28/09: 55601 (OGBench closed-loop) và 55614_0 (OGBench v2 training) RUNNING;
55616 (PushT v2 train) PENDING; 55603 setup FAILED; 55634 LIBERO-Safety headroom PENDING.
Đây là snapshot trạng thái, không suy setup chưa được sửa vì có thể có job thay thế từ phiên khác.
Không lấy submission hoặc training đang chạy làm kết quả. Không submit duplicate và không đụng dirty files
của các phiên khác. Lượt triển khai tiếp theo phải dùng ledger/state mới nhất và kiểm tra quota tháng theo AGENTS.md.
