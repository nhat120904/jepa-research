# Chọn lại câu hỏi–task–baseline cho nghiên cứu planning/control

Ngày rà soát: 07/09/2026. Mục tiêu theo lựa chọn của tác giả: **method mới cải thiện planning/control**. Đây là đề xuất nghiên cứu và protocol, không phải kết quả thực nghiệm mới. Không chạy training hoặc simulator trong lần rà soát này.

## 1. Quyết định mình khuyến nghị

**Chuyển trọng tâm khỏi sửa latent terminal cost trên stack cũ; ưu tiên kiểm tra planning ở cấp kỹ năng trên state quan sát đầy đủ.** Câu hỏi cụ thể là: *model có cần dự đoán cả trạng thái bàn giao và thời điểm bàn giao giữa hai kỹ năng để chọn được chuỗi hành động tốt hơn không?*

Đây là lựa chọn cho **pilot đầu tiên**, chưa phải kết luận rằng đã có method đủ mới để viết paper. Options/SMDP đã tồn tại lâu; “thêm duration”, “thêm termination” hoặc “thêm hierarchy vào JEPA” riêng lẻ không đủ mới. Đóng góp còn phải tìm và kiểm tra là một cách học model có thể dùng lại khi thay đổi điều kiện dừng, giữ được sự phụ thuộc giữa thời gian và trạng thái bàn giao, và thực sự cải thiện kế hoạch với cùng bộ kỹ năng.

**Hướng dự phòng:** tính nhất quán của action-conditioned dynamics dưới các cách chia cùng một khoảng thời gian vật lý. Hướng này dễ dựng thí nghiệm hơn, nhưng nguy cơ incremental so với TAWM và continuous-time models cao hơn. Không nên đặt tên một JEPA mới cho cả hai hướng lúc này.

Phân bổ đề nghị cho vòng nghiên cứu tiếp theo: khoảng 60% công sức vào kiểm tra task–baseline và headroom của hướng kỹ năng; 25% vào pilot thời gian vật lý; 15% vào novelty audit và chốt protocol. Đây là phân bổ công sức kiểm chứng, không phải bật đồng thời nhiều training campaign. Nếu baseline chính không tái lập được trong ngân sách setup đã định, chuyển thứ tự hai hướng.

| Hướng | Câu hỏi có thể kiểm chứng | Task mở đầu | Baseline quyết định | Đánh giá hiện tại |
|---|---|---|---|---|
| A — trạng thái và thời điểm chuyển kỹ năng | Dự đoán đúng *khi nào và trong trạng thái nào* chuyển skill có giúp composition không? | OGBench antmaze-medium/large, state | CompPlan, policy gốc, fixed-length skill model, option model có termination, HIQL/HSVL | Ưu tiên pilot; novelty và tái lập là hai rủi ro lớn |
| B — nhất quán theo thời gian vật lý | Cùng action signal nhưng cách chia rollout khác nhau có làm planner chọn sai không? | DMC pendulum/reacher, state; sau đó một task động lực học khác | TAWM, TD-MPC2 với Δt và mixed-rate data, continuous-time model | Dự phòng; dễ kiểm tra, novelty chưa vững |
| C — sửa search bằng uncertainty/value pessimism | Có khác hẳn các cost repair cũ và MRS.Q không? | Nếu làm: DMC với reward và online RL | MRS.Q, TD-MPC2, MR.Q | Không ưu tiên ở vòng này |

## 2. Cần sửa cách diễn giải mạch nghiên cứu cũ

Các null result là lý do tốt để đổi phân bổ công sức. Nhưng chúng không tạo thành một chứng minh loại trừ toàn bộ encoder → predictor → geometry → memory.

- **Oracle dynamics:** cùng encoder/planner mà latent cost thất bại, physical reference cost thành công, cho thấy predictor error không cần thiết để tạo ra failure trong cấu hình được audit. Chưa suy ra predictor hoặc planner vô can trên mọi task. Xem [CURRENT_STATUS.md](/Users/nhatcuong/code_project/vin-research/diagnosis/docs/CURRENT_STATUS.md).
- **Action geometry:** note tháng 8 ghi nhận continuation giảm false valleys nhưng success 62.6% → 62.4%, chênh lệch −0.2 điểm phần trăm, CI [−2.47, 1.80]. Đây là bằng chứng chống intervention đó và mức cải thiện lớn trong protocol đó. Không phải chứng minh mọi geometry method đều vô ích. Đáng chú ý, explicit cosine-curvature loss và continuation là hai can thiệp khác nhau. Xem [technical note](/Users/nhatcuong/code_project/vin-research/AUGUST_2026_TECHNICAL_NOTE.md:38).
- **PS-JEPA:** kết quả ngày 05/09 cho thấy frame latent đọc được bốn biến kiểm tra với balanced accuracy khoảng 98.5–99.8%, recurrence không thêm lợi ích. Các nhãn đó là trạng thái hiện tại; không được biến kết quả thành định lý về toàn bộ lịch sử hoặc predictive sufficiency. Xem [protocol mới](/Users/nhatcuong/code_project/vin-research/scene_progress_wm/docs/SCENE_PS_JEPA_PROTOCOL.md:85) và [label code](/Users/nhatcuong/code_project/vin-research/scene_progress_wm/scripts/history_swap_audit.py:76).
- **Progress cost:** confirm gần nhất pooled 32/100 → 33/100, CI chênh lệch [−9, +10] điểm phần trăm. Chưa có bằng chứng tăng planning; CI này cũng chưa đủ hẹp để bác bỏ mọi gain nhỏ. Xem [confirm](/Users/nhatcuong/code_project/vin-research/scene_progress_wm/docs/SCENE_PROGRESS_WM_PROTOCOL.md:459).

Vì vậy không nên lấy câu “PS-JEPA là giả thuyết duy nhất chưa bị loại” trong bản kể cũ làm trạng thái hiện tại. Cũng không nên dùng replay thành công của dataset như một upper bound toán học của planner. Nó là một reference chứng minh có trajectory giải được task.

Điều nên mang sang hướng mới là **cách cô lập lỗi bằng can thiệp**, không phải một giả thuyết chung rằng mọi world model đều hỏng vì cùng một nguyên nhân.

## 3. Literature thực sự cho phép kết luận gì?

[PAPER_LEDGER.md](/Users/nhatcuong/code_project/vin-research/research_direction_audit_20260907/PAPER_LEDGER.md) ghi 20 công trình có acceptance main conference đã đối chiếu, cùng các preprint/đối thủ liên quan. Đây là tập rà soát có chủ đích quanh quyết định nghiên cứu; không phải census toàn bộ 2025–2026, cũng không phải ranking “paper mạnh nhất”.

Nhận xét của bạn **đúng như một heuristic thiết kế nghiên cứu**: một nguyên lý bên ngoài có ích khi nó dẫn đến một thay đổi cụ thể ở đại lượng học hoặc thuật toán quyết định. TD-JEPA dùng dự đoán dài hạn theo policy; Temporal Straightening nhắm vào điều kiện thuận lợi cho gradient planning; SVL dùng survival learning; các planner dùng ý tưởng từ optimal control. Nhưng acceptance không chứng minh nguyên lý đó áp dụng được sang task khác.

Ba điều chỉnh đáng kể:

1. **DINO-WM là ICML 2025.** Không gán ICLR chỉ vì phiên bản submission trước đó. [Proceedings](https://proceedings.mlr.press/v267/zhou25t.html).
2. **Causal-JEPA có main ICML 2026**, ngoài bản workshop được dẫn trong tài liệu ban đầu. Không dùng riêng workshop PDF để kết luận không có main acceptance. [Poster chính thức](https://icml.cc/virtual/2026/poster/63623).
3. **GeoWorld không phải bằng chứng robot MPC.** Evaluation chính là procedural planning trên CrossTask/COIN. Bài có thật và accepted CVPR 2026, nhưng regime khác câu hỏi manipulation của bạn. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_GeoWorld_Geometric_World_Models_CVPR_2026_paper.html).

Quan trọng hơn danh sách JEPA là những đối thủ làm thay đổi baseline:

- [CompPlan, ICML 2026](https://icml.cc/virtual/2026/poster/62821) đã lập kế hoạch bằng cách ghép pretrained policies với world model qua nhiều timescale. “Đổi primitive actions thành skills” không còn là novelty.
- [TAWM, ICML 2025](https://proceedings.mlr.press/v267/nhu25a.html) đã đưa Δt vào dynamics và các thành phần control. “Cho model biết action duration” không còn là novelty.
- [SVL, ICML 2026](https://icml.cc/virtual/2026/poster/61425) đã học phân phối thời gian đến goal. “Survival loss cho goal-reaching” không còn là novelty.
- [MRS.Q, ICML 2026](https://icml.cc/virtual/2026/poster/62452) cho thấy search có thể hại dù model chính xác, và xử lý overestimation bằng value ensemble. Kết quả này gần về hiện tượng với nghiên cứu của bạn, nhưng khác frozen image-cost setting; không được coi hai cơ chế đã chứng minh đồng nhất.

Bài học thực dụng là: **chọn nguyên lý cùng với một đối thủ mạnh đã sử dụng nguyên lý gần đó**, rồi xác định chính xác thứ còn thiếu. Câu “vanilla JEPA chưa có X” yếu hơn nhiều câu “baseline mạnh nhất đã có X nhưng vẫn không giải được Y vì Z”.

## 4. Hướng A: học model của việc bàn giao giữa các kỹ năng

### Câu hỏi và ví dụ

Giả sử policy đi tới một vị trí trung gian. Chạm được vị trí đó chưa đảm bảo bước tiếp theo dễ: robot có thể tới nhanh nhưng đang lao sai hướng, hoặc tới chậm với tư thế phù hợp để rẽ. Trong manipulation, đưa vật vào một vùng có thể thành công trong khi tư thế tay khiến hành động tiếp theo khó thực hiện.

Câu hỏi: **một model học đồng thời trạng thái kết thúc skill và thời gian thực thi, có điều kiện theo cách dừng skill, có chọn được composition tốt hơn model chỉ dự đoán endpoint hoặc thời gian riêng rẽ không?**

Đây là giả thuyết của báo cáo, chưa có kết quả hỗ trợ từ repo hiện tại.

### Đại lượng học và cơ chế dự kiến

Giữ cố định một family policy π(a|s,u), với u là subgoal/skill parameter. Đặt b là điều kiện dừng được cung cấp trong protocol; τ là thời điểm policy dừng hoặc bị timeout. Model dự đoán:

`pθ(s_exit, τ, outcome | s_start, u, b)`

`outcome` phân biệt đạt điều kiện, failure thực sự, và timeout. Timeout không tự động là thất bại vĩnh viễn. Với rollout kết thúc trước khi skill hoàn thành, chỉ biết τ vượt quá thời gian quan sát; không được gán bừa một exit state.

Planner ghép các phân phối này theo state bàn giao và ngân sách thời gian còn lại. Chỉ trong task terminate-on-success/deadline rõ ràng mới dùng xác suất đạt goal trước H. Với benchmark có reward khác, cần model cả reward tích lũy trong skill hoặc dùng đúng functional của benchmark; endpoint và duration không đủ cho arbitrary rewards.

**Phần có thể tạo đóng góp:** một estimator dùng lại trajectory để học nhiều điều kiện dừng hợp lệ, query được điều kiện mới, và giữ được coupling của τ với residual exit state. Có thể dùng censored likelihood cùng conditional density model; chưa có lý do bắt buộc dùng JEPA hay flow matching ngay từ đầu. Thử mô hình đơn giản trước.

### Novelty còn lại hẹp ở đâu?

| Đối thủ | Thứ đã có | Điều phải chứng minh thêm |
|---|---|---|
| Options/SMDP, Universal Option Models | Mô hình option, termination, abstract planning | Không gọi lại đối tượng toán học cũ thành đóng góp mới |
| CompPlan | Policy composition và nhiều horizon | Điều kiện dừng theo state và cơ chế ước lượng mới có lợi hơn lựa chọn horizon tốt nhất không? |
| SVL/HSVL | Phân phối thời gian tới goal, xử lý censoring | Joint exit-state/time có ích hơn duration/value và hierarchical actor không? |
| Abstract world models với options | Abstraction phục vụ planning | Có cách học/transfer khác thực chất, ngoài thay neural architecture không? |
| AgentOWL | Joint option/world-model learning | Không nhận “học skill cùng model” làm novelty |
| Cost of Commitment | Tradeoff giữa duration, deliberation và model error | Không nhận “duration nên thích nghi với uncertainty” làm novelty |

Nguồn: [UOM, NIPS 2014](https://papers.nips.cc/paper/5590-universal-option-models), [CompPlan full paper](https://arxiv.org/html/2602.19634v1), [SVL full paper](https://arxiv.org/html/2604.17551v1), [abstract models/options](https://arxiv.org/abs/2406.15850), [AgentOWL](https://arxiv.org/abs/2602.02799), [Cost of Commitment, ICML 2026](https://icml.cc/virtual/2026/poster/66480).

**Mức tin cậy novelty: thấp–trung bình.** Mình chưa tìm được bằng chứng đủ để tuyên bố estimator dự kiến là mới. Cần viết được một khác biệt thuật toán so với neural option model có termination; nếu không viết được, dừng trước training lớn. “Chưa thấy paper đúng tên” không phải chứng nhận novelty.

### Chọn task và data cùng baseline

1. **Development:** `antmaze-medium-navigate-v0` và `antmaze-large-navigate-v0` của OGBench, state quan sát đầy đủ. Medium để kiểm tra implementation, large để đo nhu cầu composition. Không lấy giant/humanoid làm cửa đầu vì dễ mắc trần low-level policy.
2. **Transfer sau khi qua gate:** cube manipulation, bắt đầu một cấu hình ít vật như cube-double; freeze trước danh sách task/goal test. Không mặc định task cube còn headroom: phải đối chiếu CompPlan và baseline mạnh trên đúng cấu hình.
3. **Pixels là một nhánh mở rộng**, sau khi state-based method thắng. Nếu pixels thất bại, đo representation separately; không dùng failure đó để giết kết luận state-based, cũng không gọi kết quả state-based là visual world modeling.

[OGBench chính thức](https://github.com/seohongpark/ogbench) cung cấp môi trường/dataset và reference implementations. Cần pin version/config trước chạy; tên và khả năng hỗ trợ visual phải kiểm tra theo checkout sử dụng.

**Regime đề nghị chốt cho pilot A:** pretrain policy từ dataset OGBench, freeze policy, rồi cho tất cả model/baseline một tập rollout simulator bổ sung chung, với trần sàng lọc ban đầu 100.000 primitive transitions cho mỗi task. Đây là đề xuất giới hạn thu dữ liệu, chưa phải số đo sample complexity. Báo riêng kết quả trước và sau dữ liệu bổ sung; giữ test starts/goals tách khỏi collection. Khi bắt đầu thực nghiệm phải tính cả reset/branching simulator và mọi dữ liệu baseline dùng vào ledger.

Hai regime dữ liệu phải tách rõ. Nếu học từ dataset offline, đoạn tương lai dưới behavior policy **không phải** một sample kết quả dưới target skill khác. Relabel điều kiện dừng chỉ hợp lệ khi action/policy thực thi tương thích; không được relabel skill tùy ý. Pilot trên có tương tác bổ sung, nên không báo như strict offline OGBench. Một nhánh strict offline chỉ đáng mở khi có estimator off-policy hợp lệ; đó có thể là phần nghiên cứu khó, không phải một chi tiết implementation được mặc định đã giải xong.

### Baselines tối thiểu

- Frozen GC policy trực tiếp; cùng skill bank dùng trong method.
- Fixed-length skill transition model, có sweep duration trên validation.
- **CompPlan** với policy family và dữ liệu tương ứng; không chỉ lấy CEM primitive-action làm đối thủ.
- Neural option model có state-dependent termination; thêm heuristic dừng khi đạt subgoal với cùng thông tin.
- Model dự đoán `p(s_exit)` và `p(τ)` riêng, cùng capacity/data, để kiểm tra coupling.
- HIQL hoặc HSVL làm đối thủ end-to-end GCRL. Không ép chúng dùng kiến trúc yếu hơn; báo riêng so sánh cô lập cơ chế và so sánh hệ thống tốt nhất.

Code OGBench và [HSVL chính thức](https://github.com/Simple-Robotics/hierarchical-survival-value-learning) đã tìm thấy. **Chưa xác minh được code chính thức của CompPlan trong lần rà soát này**; paper có pseudocode, nhưng như vậy chưa đủ gọi baseline sẵn sàng. Đây là một điều kiện chặn mở campaign, không phải lý do bỏ baseline.

### Gate rẻ trước khi train method

**A0 — implementation và năng lực skill.** Reproduce short-goal success của GC policy. Nếu chính skill không đi được các chặng ngắn trong vùng dữ liệu, chưa có cơ sở thử model composition. Không sửa bằng controller đọc privileged state rồi dùng controller đó làm evidence cho method học.

**A1 — headroom do bàn giao.** Trên một tập development cố định, giữ skill sequence/proposal budget và primitive-action budget. So fixed/geometric switching với state-conditioned switching, dùng simulator rollout chỉ như oracle diagnostic. Ghi rõ độ mạnh oracle và chi phí tìm kiếm; không gọi nó là controller triển khai. Mục tiêu sàng lọc đề nghị: reference có thêm ít nhất 10 điểm phần trăm success hoặc lợi ích thực chất đã định trước. Nếu không có, giả thuyết switching không đáng ưu tiên trên task đó.

**A2 — simple baseline kill.** Cho heuristic dừng tại subgoal và learned termination chuẩn cơ hội công bằng. Nếu chúng thu được toàn bộ gain, chưa cần model mới.

**A3 — học có thực hiện được không?** Trên held-out trajectories và start states, so joint model với factorized model rồi đưa cả hai vào cùng planner. Model phải thắng control, không chỉ likelihood hoặc timing RMSE. Trong môi trường deterministic với policy deterministic và state đủ, phân phối có thể gần Dirac: nếu coupling distribution không có vai trò, bỏ câu chuyện probabilistic ngay.

**A4 — confirm.** Khóa hyperparameters rồi chạy nhiều training seeds trên tập goal/episode chưa dùng chọn task. Đề nghị target gain tối thiểu +5 điểm phần trăm so baseline mạnh nhất, paired CI loại 0, giữ hoặc cải thiện task gốc. Đây là ngưỡng quyết định đề xuất, không phải universal standard. Nếu CI còn rộng, ghi INCONCLUSIVE thay vì “null đủ mạnh”.

### Khác gì với `event_smdp_h0` và PS-JEPA?

Hướng cũ hỏi observer/history/progress và từng dùng simulator, automaton, skill thủ công. Hướng này phải học dynamics của kỹ năng và chứng minh composition trên task có state đủ; không dựa vào milestone latching hoặc frame aliasing. Nếu implementation lại cần automaton gắn tay và oracle skills để có gain, nó chưa tạo được bước tiến cần thiết so với repo cũ.

## 5. Hướng B: nhất quán theo thời gian vật lý

### Câu hỏi và loss tối thiểu

Một action giữ nguyên trong 40 ms có thể được dự đoán bằng một bước 40 ms hoặc hai bước 20 ms. Nếu hai cách dự đoán mâu thuẫn, planner thay discretization có thể đổi quyết định vì model, dù action vật lý không đổi.

Với state đủ và action u giữ cố định, flow đúng thỏa:

`F(F(s,u,Δ1),u,Δ2) = F(s,u,Δ1+Δ2)`

Ứng viên đơn giản: thêm loss giữa hai vế, **cùng supervised transition loss**, rồi kiểm tra lợi ích control khi control interval biến thiên. Một model collapse cũng thỏa consistency; loss này riêng lẻ không đủ. Với dynamics stochastic, đẳng thức đúng cho composition của transition kernels, không nhất thiết cho conditional means. Với partial observability, latent hiện tại có thể không đóng dưới dynamics; đó là lý do state-first.

Không áp đẳng thức trên cho hai action khác nhau hoặc đổi thứ tự action. Với action signal piecewise-constant, phải so hai cách phân hoạch **cùng signal** và giữ đúng mọi thời điểm đổi action. Đây không phải quay lại ORDER-JEPA.

### Task và baseline

Development trên DMC pendulum/reacher với state, giữ physics integration step và thay số bước physics giữa hai control commands. Dùng ba loại test tách biệt: interval cố định đã thấy; interval chưa thấy trong khoảng hỗ trợ; interval thay đổi trong episode. Dropped observations là một bài toán khác, không trộn nhãn với action bị giữ lâu hơn.

Tất cả đối thủ cùng dữ liệu, cùng phân phối Δt, cùng thời lượng episode tính bằng giây, cùng horizon vật lý. Reward cần tích lũy nhất quán theo thời gian; discount có thể đặt `γ(Δt)=exp(−ρΔt)` khi chọn continuous-time objective. Nếu benchmark gốc dùng discount theo step, báo rõ đây là task variant và vẫn báo task gốc.

**TAWM là baseline bắt buộc**, cùng TD-MPC2 được thêm Δt/mixed-rate training, multi-step overshooting, và action-conditioned continuous-time model. [TAWM paper](https://proceedings.mlr.press/v267/nhu25a.html), [code](https://github.com/anh-nn01/Time-Aware-World-Model). Không so một method được biết Δt với baseline bị giấu Δt rồi gọi đó là tác dụng consistency.

[ODEWorld](https://arxiv.org/html/2607.27924v1) là novelty threat về physical-time latent dynamics, dù phiên bản kiểm tra chưa đưa action conditioning vào predictor. Việc đó không làm “thêm action vào ODE” tự động mới. CompPlan cũng đã có cross-horizon consistency, dù đối tượng toán học khác. Cần đối chiếu continuous-time control/system-identification ngoài JEPA trước claim.

### Gate và lý do chỉ xếp thứ hai

Trước hết đo TAWM dưới protocol đúng. Nếu Δt conditioning cộng mixed-rate data đã giữ control tốt, không có vấn đề đáng giải. Tiếp theo cần cho thấy partition inconsistency làm đổi lựa chọn có hậu quả vật lý; chỉ vẽ error theo Δt chưa đủ.

So sánh consistency loss với tăng supervised data, tăng rollout depth và baseline continuous-time cùng compute. Nếu metric consistency giảm mà return không tăng, dừng như bài học action-curvature. Nếu chỉ thắng khi cố tình lấy test interval vượt xa dữ liệu, đó có thể là extrapolation benchmark nhân tạo, chưa phải lý do làm flagship paper.

Ưu điểm là code nền công khai và task đầu nhỏ; nhược điểm là ý tưởng nền rất chín. **Mức tin cậy novelty: thấp.** Hướng này là fallback có thể falsify nhanh, không phải đề xuất “chắc ra paper”.

## 6. Cách tránh trần model/encoder/task từ đầu

Không có một “oracle ceiling” duy nhất. Cần phân biệt năm khoảng chênh:

| Tầng | Kiểm tra cần có | Kết luận khi không qua |
|---|---|---|
| Task có giải được trong budget? | Reference controller/trajectory với cùng horizon; ghi rõ thông tin ưu tiên | Đổi horizon/task, không tăng model mù quáng |
| Dữ liệu có hỗ trợ quyết định cần thiết? | Coverage của start, action/skill, duration và subgoal; tách train/test theo trajectory | Thu dữ liệu công bằng hoặc chọn task khác |
| Low-level policy có thực thi được? | Short-horizon skill success dưới cùng observation | Sửa/reproduce skill; chưa kiểm tra được high-level model |
| Representation có đủ cho mechanism? | State-first, rồi probe/decoder và matched observational intervention | Tách perception problem khỏi planning method |
| Baseline mạnh còn thiếu gì? | Oracle thay đúng một thành phần với cùng budget; đo success/return | Không có headroom thì không phát minh loss cho thành phần đó |

Probe tốt không bảo đảm planner dùng tốt; probe kém cũng không đủ chứng minh information-theoretic impossibility. Oracle có nhiều thông tin hơn chỉ là reference. Đặc biệt, không chọn baseline yếu để tạo ra headroom: đối thủ end-to-end mạnh phải luôn hiện diện trong báo cáo.

Task selection được phép dựa trên **baseline và mechanism gate trên development split**. Không chọn task sau khi nhìn method thắng ở đâu rồi dùng chính các episode đó làm confirm. Giữ lại benchmark chuẩn và báo toàn bộ tập test đã đăng ký, kể cả task không thắng.

## 7. Ngân sách và kế hoạch quyết định

Chưa benchmark throughput hoặc xác minh GPU khả dụng, nên không đưa con số GPU-hour như một phép đo. Đặt **trần chi phí**, profile rồi quyết định số runs.

| Mốc | Công việc | Trần công sức đề nghị | Điều kiện ra quyết định |
|---|---|---|---|
| 0 | Pin code/data/config; kiểm tra CompPlan implementation; viết novelty delta | 1–2 ngày làm việc | Không có baseline hoặc khác biệt thuật toán cụ thể → chưa mở A |
| 1 | Reproduce baseline trên một task nhỏ, profile train/eval | Tối đa 2 ngày setup bổ sung | Sai reproduction → FIX, không tính là method failure |
| 2 | Gate headroom và simple-baseline kill | Một batch diagnostic hữu hạn | Không có gap hoặc heuristic giải hết → STOP hướng/task |
| 3 | Method nhỏ nhất, 2–3 training seeds, development only | Trần đề nghị 48 GPU-hour tổng sau profile | Không có gain control đáng kể → STOP/INCONCLUSIVE |
| 4 | Confirm trên ≥5 training seeds, goals held out, task thứ hai | Chỉ lập ngân sách sau mốc 3 | Ước lượng effect, CI và chi phí; không mở sweep vô hạn |

48 GPU-hour là **giới hạn đầu tư đề nghị**, không phải dự đoán rằng tất cả baseline sẽ fit. Nếu reproduction vượt trần, giảm scope trước hoặc chuyển hướng B. Simulator CPU-hour, số environment transitions, storage và planning latency phải ghi riêng; đừng gói vào một GPU-hour.

Số episode cần dựa trên biến thiên đo được. 3 seeds × 100 episodes chưa tự động đủ lực bác bỏ gain 5 điểm phần trăm. Dùng paired start/goal sets, uncertainty có cluster theo training seed, và xác định trước maximum sample size. Nếu sequential screening nhiều lần, tách screen khỏi confirm độc lập hoặc dùng phương pháp inference phù hợp; không liên tục nhìn CI rồi dừng khi vừa đạt ý muốn.

## 8. Những hướng không nên chi phần lớn công sức mới

- Một cost/probe/ensemble khác trên frozen DINO stack, nếu không có lý do cụ thể khiến nó thoát shared representation error và thắng các control rẻ đã có.
- Memory/belief JEPA dựa vào premise OGBench Scene thiếu lịch sử, khi audit mới không hỗ trợ premise được dùng.
- Straightening/curvature loss khác mà chưa chứng minh component headroom trên downstream control.
- Object-centric JEPA tổng quát: Causal-JEPA và LPWM đã chiếm phần lớn lời hứa dễ nói; phải tính chi phí segmentation, tracking và policy integration.
- Uncertainty “tăng theo horizon” như quy luật phổ quát. Dynamics ổn định/absorbing có thể làm uncertainty giảm; probabilistic prediction cũng không đồng nghĩa calibrated epistemic uncertainty.
- Thêm flow, hyperbolic geometry, survival loss hoặc TD target chỉ vì chúng xuất hiện ở paper accepted. Mỗi thứ cần task và đối thủ phù hợp.

**Quyết định cụ thể sau lần rà soát:** chưa tiếp tục train một biến thể JEPA lớn. Việc đầu tiên đáng làm là khóa bộ ba **state-based OGBench + policy-composition baseline + câu hỏi về trạng thái/thời điểm bàn giao**, rồi chạy gate rẻ khi bắt đầu giai đoạn experiment. Nếu không qua novelty/reproduction/headroom, chuyển sang kiểm tra TAWM dưới thay đổi control interval. Không có bằng chứng hiện tại để hứa rằng một trong hai sẽ thắng; có đủ cơ sở để tránh lặp lại các vòng train không xác định được nguyên nhân thất bại.
