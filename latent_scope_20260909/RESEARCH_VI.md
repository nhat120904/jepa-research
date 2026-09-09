# Chọn lại hướng trong JEPA-style latent world models

> Update, 2026-09-09: The user prefers existing released tasks/datasets and English research documents. The [English native-arena assessment](ARENA_FIT_EN.md) supersedes the custom PushCube pilot below. Native RoboCasa tasks support a broader path-dependent prediction question, but do not by themselves justify the signature/order-specific method claim.

Ngày khảo sát: **09/09/2026**. Phạm vi: simulator/GPU, method cho robot planning, hướng tới ICML/NeurIPS 2027 theo các cửa sổ dự kiến của người dùng. Đây là kết quả khảo sát và một ứng viên nghiên cứu có cơ chế cụ thể; **chưa phải method đã qua novelty audit đầy đủ hay có kết quả thực nghiệm**. Không chạy simulator, tải checkpoint hoặc train model trong lần này.

## Kết luận lựa chọn

**Nên tiếp tục JEPA/latent world models.** Những null trong repo là bằng chứng chống lại các giả thuyết và pipeline đã thử. Chúng không chứng minh latent prediction là ngõ cụt. Sai lầm trong đề xuất gần nhất là đổi luôn đối tượng nghiên cứu thành dynamics của trạng thái vật lý tường minh.

Tôi ưu tiên câu hỏi sau để kiểm chứng tiếp:

> **Có thể học một JEPA dự đoán đồng thời latent cuối đoạn và một biểu diễn có thể ghép nối của diễn biến trong đoạn, để lập kế hoạch thao tác có ràng buộc dọc đường và chuyển sang yêu cầu mới mà không phải rollout từng frame hay train lại world model không?**

Tên mô tả: **composable trajectory-target JEPA**. Không dùng tên Path-JEPA: đã tìm thấy một công trình mang tên đó. Đóng góp mong muốn nằm ở **đối tượng mà latent predictor học**, cách bảo toàn sự phụ thuộc giữa đường đi và trạng thái bàn giao, và tác dụng của chúng trong visual planning.

Đây là lựa chọn cho một vòng qualification ngắn. Tôi **chưa khuyến nghị cam kết toàn bộ bốn tháng**: signature prediction, temporal-logic planning và stochastic JEPA đều đã có prior art đáng kể.

## Vì sao không cần rời JEPA

Các ví dụ được xác minh cho thấy cùng một họ có nhiều trục phát triển:

| Công trình | Vai trò trong lập luận |
|---|---|
| [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | Mốc latent prediction từ visual features phục vụ planning. |
| [TD-JEPA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html) | Dùng TD để học dự đoán latent dài hạn phụ thuộc policy; thay đổi cấu trúc dự đoán, không cần tái tạo ảnh. |
| [Temporal Straightening, ICML 2026, trang tác giả](https://agenticlearning.ai/temporal-straightening/) | Thay đổi cách học representation phục vụ planning bằng một nguyên lý từ perceptual learning. |
| [Causal-JEPA, paper](https://arxiv.org/abs/2602.11389), có trong [danh mục ICML 2026](https://icml.cc/Downloads/2026) | Object-level masked prediction là một trục khác về cấu trúc biểu diễn và tương tác. |

TD-JEPA của **Bagatella và cộng sự** là latent-predictive zero-shot RL, không phải chỉ DINO-WM được đổi một loss rồi chạy CEM. Không nhập nhằng nó với các paper khác cũng viết tắt TD-JEPA. [Code chính thức](https://github.com/facebookresearch/td_jepa) hỗ trợ state/RGB và các baseline FB, HILP, ICVF, BYOL cùng một số biến thể. Khả năng chạy code trên cluster chưa được thử ở đây.

Không suy từ acceptance sang “mọi claim đều đã được tái lập”, hoặc gọi đây là danh sách đầy đủ/những paper mạnh nhất của cả năm.

## Ba hướng đã so sánh

| Hướng | Nguyên lý mượn | Đánh giá hiện tại |
|---|---|---|
| **Trajectory-target JEPA** | Path signatures/rough paths: mô tả thứ tự, moment dọc đường và quy tắc ghép đoạn | Ưu tiên qualification. Câu hỏi khác terminal metric và predictor patch cũ; có falsifier mạnh. |
| Latent model thích nghi với cơ chế mới | Bayesian filtering, system identification, adaptation | Giữ như trục dài hạn, chưa chọn method. [AdaJEPA](https://agenticlearning.ai/adajepa/) đã thích nghi ngay trong MPC; [VJEPA](https://arxiv.org/abs/2601.14354) đã đặt stochastic JEPA/belief vào khung xác suất. “Thêm context/belief” không đủ. |
| JEPA có cơ chế tương tác ghép được giữa vật thể | Modular causal mechanisms, relational models | Có tiềm năng dài hạn nhưng khoảng cách novelty hiện chưa rõ: Causal-JEPA đã nhắm interaction reasoning; [WM3C](https://www.charonwangg.com/projects/wm3c/) đã dùng causal components cho composition. Chưa chọn một method cụ thể ở nhánh này. |

Các lựa chọn tưởng còn trống cũng đã có đối thủ gần: [PSG-JEPA](https://arxiv.org/abs/2608.06799) về physical grounding; [ProWorld](https://arxiv.org/abs/2608.01926) về progress-aware hyperbolic representation; [Bis-JEPA](https://github.com/genglongling/Bis-JEPA) về invariant/bisimulation representation. Đây là các nguồn preprint/code trong khảo sát này, không được gom tất cả vào nhóm main-conference accepted.

## Câu hỏi khoa học của ứng viên chính

Một robot chuyển vật tới cùng một khay đích có thể làm đúng hoặc sai quy trình: đi qua vùng kiểm tra trước khi đặt xuống, tránh vùng cấm trong suốt đường đi, hoặc giữ vật ở một vùng đủ lâu. Ảnh cuối giống nhau không quyết định được các điều kiện này.

**Không được suy rằng mọi JEPA đều không biết thứ tự.** DINO-WM dự đoán một chuỗi latent, nên một rollout đủ chính xác cùng evaluator phù hợp hoàn toàn có thể giải bài toán. TD-JEPA/SF có temporal discount, nên cũng không phải “bag of states không có thông tin thời gian”. Câu hỏi thực tế là liệu một target cô đọng của *cả đoạn* có giữ được thông tin cần cho planning tốt hơn trong ngân sách dữ liệu/tính toán hữu hạn.

Đặc biệt, future occupancy ở từng thời điểm chưa xác định joint path distribution. Ví dụ giải tích, không phải kết quả robot:

| Phân phối đường đi | Hai đoạn đầu | P(A rồi B) |
|---|---|---:|
| P | AA hoặc BB, mỗi loại 1/2 | 0 |
| Q | AB hoặc BA, mỗi loại 1/2 | 1/2 |

Hai phân phối có cùng marginal A/B ở từng thời điểm, do đó cùng mọi discounted occupancy trên bảng này. Thêm một trạng thái cuối C chung không đổi điều đó. Nhưng xác suất thỏa thứ tự khác nhau. Đây chỉ là giới hạn của **bản tóm tắt marginal**; một transition kernel đầy đủ, mô hình trajectory đầy đủ hoặc augmented state có thể phân biệt chúng.

[CompPlan](https://arxiv.org/html/2602.19634v1), §2, cũng phân biệt occupancy/cumulant kỳ vọng với trajectory-level statistics. Không được diễn giải đó thành bằng chứng CompPlan thất bại trên task chưa chạy, hay rằng nó không thể mở rộng với automaton.

## Nguyên lý và bản dịch sang JEPA

Path signature là một họ đặc trưng tuần tự với tích không giao hoán. Đặc trưng bậc cao chứa tương tác có thứ tự; signature của hai đoạn nối nhau được ghép bằng Chen's identity. Expected-signature prediction đã có từ [Levin–Lyons–Ni, 2013](https://arxiv.org/abs/1309.0260). Đây là nguyên lý mượn, không phải theorem mới.

Lõi observation model vẫn là:

\[
z_t=E_\theta(o_{t-L:t},a_{t-L:t-1}),
\qquad u=a_{t:t+h-1}.
\]

Thay target chỉ gồm endpoint bằng target chung:

\[
Y_{t,h}=\left(z_{t+h},\;\operatorname{LogSig}_{\le m}(X_{t:t+h})\right),
\quad
X(s)=\left(s,\int_t^s \rho(z_v)\,dv\right).
\]

Trong đó \(\rho\) là projection latent nhỏ. Time channel và tích phân dùng đơn vị cố định cho toàn bộ trajectory; không normalize thời gian riêng từng đoạn rồi coi quy tắc composition còn nguyên. Endpoint cung cấp trạng thái để dự đoán đoạn kế; signature cung cấp thông tin diễn biến trong đoạn.

Predictor học **joint distribution** \(p_\phi(Y_{t,h}\mid z_t,u,h)\). Một hiện thực thử được là conditional flow matching trên target latent từ EMA encoder. Target tương lai chỉ xuất hiện ở nhánh training; rollout không được đọc ảnh tương lai. Giữ loss latent prediction và anti-collapse như baseline, thêm joint target; không có pixel decoder và không dùng learned physical-state simulator làm lõi.

Khi planning, lấy mẫu cặp endpoint–summary từ *cùng một sample*, dùng endpoint đó sinh đoạn tiếp theo, rồi ghép signatures của từng particle. Chấm query trên các particle. **Không** nhân hai expected signatures độc lập để giả vờ có kỳ vọng của đường ghép:

\[
\mathbb E[S_1\otimes S_2]\ne
\mathbb E[S_1]\otimes\mathbb E[S_2]
\quad\text{nói chung}.
\]

Đây là lý do cần joint endpoint–summary. Sự phụ thuộc được giữ bằng conditional rollout theo từng particle; Chen's identity không tự giải quyết sai số model hay thiếu Markov sufficiency của latent.

Signature bậc hữu hạn **không** phân biệt mọi trajectory, không bảo đảm mọi temporal-logic query đọc được tuyến tính, và không cung cấp safety certificate. Các định lý về full signature không tự áp dụng cho một projection 8 chiều, degree 2/3 được học từ ảnh. Readout ceiling phải được đo trước.

## Novelty boundary: phần nào thực sự chưa giải quyết?

| Prior art / baseline | Phần đã có | Điều ứng viên phải chứng minh thêm |
|---|---|---|
| TD-JEPA | Policy-conditioned long-term latent prediction, successor structure | Lợi ích của path-dependent joint target ngoài reward additive trong feature span; không được làm yếu TD-JEPA bằng cách cấm task-state augmentation. |
| CompPlan | Multi-horizon jumpy occupancy models và policy composition | Bảo toàn thông tin trong đoạn có ích cho path constraints; so cả phiên bản có temporal monitor, không chỉ bản gốc. |
| VJEPA; RSSM-style stochastic latent rollout | Latent probability, filtering và uncertainty | Lợi ích đến từ structured segment target, không chỉ thêm stochasticity hoặc capacity. |
| [hint², preprint 08/2026](https://arxiv.org/html/2608.13678v1) | Dự đoán abstract proposition sequence, temporal-logic guidance qua hai cấp world model | Một latent target học từ quan sát phải có lợi ích về reuse/generalization/accuracy-cost ngoài fixed proposition model. Đây là đối thủ trực tiếp, không được bỏ qua. |
| [Anticipatory RL, preprint 04/2026](https://arxiv.org/html/2604.04662v1) | Đã đề xuất path-law/signature prediction, TD và composition | Không nhận “signature + future prediction + RL” là mới. Tính đúng và lợi ích của conditional joint latent implementation phải được đánh giá độc lập; không mượn các theorem của paper làm bảo chứng. |
| Path-JEPA, được liệt kê trong [CV của tác giả](https://ashutosh17dec.github.io/assets/files/Ashutosh_Singh_Resume.pdf) | Signature + JEPA cho skeleton action recognition được khai là đang review | Không claim lần đầu ghép hai công cụ. Chưa tìm được full paper/acceptance để đối chiếu chi tiết, nên novelty vẫn mở. |
| Transformer trajectory encoder + direct prediction | Có thể học summary tự do của cả clip | Cấu trúc signature phải hơn một target có cùng dimension/compute mà không có algebra. |
| JEPA rollout + exact task automaton | Đã có thể dùng dự đoán nhiều bước để kiểm tra thứ tự | Nếu baseline này thu toàn bộ gain, target mới chưa có lý do tồn tại. |

**Một paper chỉ thêm signature auxiliary loss rồi báo PushT tăng vài điểm chưa đáp ứng mục tiêu.** Đóng góp có cơ hội thành method paper nếu chứng minh được target chung và quy tắc ghép là một cách học visual latent segment models hữu ích cho nhiều query mới, với advantage có thể quy về cơ chế. Hiện đây vẫn là giả thuyết.

## Chọn task, model, data và baseline cùng lúc

Các con số sau là **cấu hình khởi đầu đề xuất**, không phải cấu hình đã chạy hay tính toán power từ variance chưa có.

| Thành phần | Lựa chọn ban đầu |
|---|---|
| Arena cơ chế | Variant của ManiSkill `PushCube-v1`: thêm vùng A/B, vùng cấm và temporal success monitor; giữ physics/robot/controller stack gốc. |
| Ý nghĩa task | Thao tác đưa vật qua khu xử lý/kiểm tra rồi tới đích, không đi vào vùng cấm. Variant mang tính kiểm soát cơ chế, không đủ làm toàn bộ evidence của final paper. |
| Quan sát | RGB top-down 128×128, 3 frame causal + proprio; không cung cấp object pose/contact state ở deployment. Cùng mọi arm. |
| Action | Pilot giới hạn chuyển động end-effector trên mặt bàn, 2D delta commands; giữ gripper/height cố định qua controller. Đây là adapter phải triển khai và kiểm tra, không có sẵn dưới đúng cấu hình đề xuất. |
| Horizon | Dự đoán segment 8/16 control steps; MPC nhìn 64 steps, thử extrapolation 128 sau đó. Planning qua piecewise-constant action knots, mỗi knot 8 steps. |
| Encoder | End-to-end small ViT: patch 8, width 192, depth 6; causal pooling ra latent 256. Thêm baseline frozen DINOv2 ViT-S/14, nhưng không để checkpoint cũ quyết định target ceiling. |
| Predictor | Conditional transformer/flow network: width 384, depth 6; action/time conditioning. Đếm parameters và profile thật trước khi khóa capacity match. |
| Segment target | Endpoint latent 256 + logsignature của 8 projected channels và 1 time channel, degree 2; degree 3 là một ablation được khai trước. |
| Planner | CEM: 256 candidates, 5 iterations, 4 joint particles/candidate; execute 4 steps rồi replan. Sweeps 1/4/16 particles trên validation và báo wall time. Những số này chưa là tối ưu. |
| Query | Giới hạn ba template: ordered regions, forbidden region, dwell duration. Query được mã hóa bằng region IDs/thời lượng và trạng thái monitor hiện tại; readout nhận query cùng signature của **observed** train trajectories. Không claim arbitrary LTL. Cấp cùng thông tin query, nhãn và history monitor cho đối thủ. |
| Evaluation | Thành công phải thỏa mọi quy tắc toàn episode và endpoint; báo riêng SR, từng loại violation, control steps, inference wall time. Cấm coi endpoint success là full success. |

[ManiSkill task cards](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html) xác nhận PushCube và các task/tabletop demos. **Variant, action adapter và data collector nêu trên chưa được xây hay smoke-test.** Không có checkpoint công khai cho variant này được xác minh.

Data pilot: 2.000 episodes × tối đa 128 control steps, trần 256.000 transitions. Tập thu gồm 40% randomized waypoint pushing, 40% perturbations quanh controller, 20% action chunks ngẫu nhiên có giới hạn; giữ failure và forbidden-region violations. Controller dùng privileged state để thu dữ liệu được phép nếu mọi arm dùng cùng tập và không đọc state đó tại deployment. Tổng interaction kể cả unsuccessful resets/collection phải được ghi riêng.

Split theo episode, layout và collection seed, không split các window lân cận. Hold out một nhóm vị trí vùng và một số query compositions; cần giữ các atomic events tương ứng trong train. Không cố tình lấy toàn train quanh một lời giải oracle rồi dùng panel đó làm headroom của CEM.

World-model pretraining không dùng task reward. Query decoder pilot nhận cùng ngân sách 2.000 labeled train clips cho mọi arm; báo nhánh unsupervised representation + supervised task readout đúng tên, không gọi cả hệ là reward-free. Probe/readout training dùng train clips, validation riêng, không fit trên candidate do test planner chọn. Phân phối decoder ngoài train là một rủi ro chính vì repo đã thấy planner khai thác surrogate.

Baselines tối thiểu phải bao gồm: (1) autoregressive JEPA + temporal monitor/readout; (2) cùng encoder stochastic one-step latent model + particles; (3) direct segment endpoint model; (4) endpoint và summary được học/lấy mẫu độc lập; (5) direct segment model với learned Transformer trajectory target cùng dimension; (6) temporal-proposition model theo ý tưởng hint²; (7) shared waypoint/goal policy + task automaton. TD-JEPA và CompPlan là so sánh bắt buộc khi claim mở rộng sang policy-level zero-shot planning; không giả định đưa chúng vào continuous-action CEM là tái lập đúng thuật toán gốc.

## Thí nghiệm đầu tiên phải loại được hướng này

**G0 — solvability và perception.** Controller tham chiếu phải làm được variant với cùng action bounds; visual baseline phải thực hiện được các đoạn push đơn giản. Nếu không, sửa interface/perception hoặc bỏ arena; không dùng failure tầng dưới làm bằng chứng target tốt hơn.

**G1 — target ceiling, trước khi train dynamics mới.** Dùng frozen DINO và encoder từ một baseline JEPA nhỏ đã warm-up trên train split; không coi random encoder là phép thử ceiling. Từ observed train/test clips, so query readout trên: endpoint; mean/multi-time features; exact task monitor; Transformer clip latent; signature degree 2/3 của cùng visual features. Đánh giá unseen layouts/queries. Nếu signature giữ thông tin kém hoặc không hơn baseline clip target dưới cùng budget, không train campaign mới. Đây là phép đo target adequacy, không phải proof of planning gain.

**G2 — phép bác bỏ rẻ nhất.** Cho JEPA autoregressive được đọc toàn predicted trajectory cùng task monitor; cho policy được dùng explicit monitor và subgoals. Nếu cả hai đã giải task ổn định ở chi phí chấp nhận được, đề xuất chưa có headroom. Không chỉ so với endpoint-L2 yếu theo construction.

**G3 — cơ chế joint target.** Train các arm cùng data, rồi so ở matched wall time và matched transition/parameter budgets. Joint target phải hơn factorized target và learned clip target trên planning thực tế, không chỉ target MSE. Nếu chỉ cải thiện nhờ thêm labels/capacity/particles, bỏ claim method.

**G4 — confirm trước mở rộng.** Sau screening 3 train seeds, khóa task/config, dùng 5 seeds và các episode đánh giá ghép theo initial conditions. Đề xuất decision threshold: +8 percentage points full success, CI paired phân cấp loại 0, hoặc cùng success trong tolerance 2 points với ít nhất 2× giảm latency. Đây là tiêu chí thiết kế, không là chuẩn thống kê phổ quát. Khoảng tin cậy còn rộng thì kết luận inconclusive; không tự gọi là null. Claim efficiency chỉ phụ khi không có generalization/control advantage, theo ưu tiên của người dùng.

Controls: endpoint-only task không yêu cầu thứ tự; deterministic dynamics; shuffled temporal order trong target; cùng tổng nhãn; cùng query decoder. Nếu chỉ có gain trên một task A→B được tạo riêng để ưu ái signature, chưa có final paper.

## Đường tới kết quả có sức nặng và giới hạn hiện tại

Arena thứ hai nên là thao tác vận chuyển/pick-place nhiều giai đoạn. [robomimic](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) có Transport và Can; các demonstration thành công không tự cung cấp đủ counterfactual/failure coverage, nên cần collection bổ sung chung cho các model. Không mặc định dataset có sẵn đã phù hợp với temporal constraint mới. Giữ một task chuẩn không sửa để đo mức đánh đổi; variant mới bổ sung câu hỏi khoa học, không thay toàn bộ external validity.

Khác ORDER-JEPA: ORDER học chênh endpoint giữa hai thứ tự action, giữ target/cost cũ. Ứng viên này học thông tin *nội bộ* của đoạn dù endpoint có thể giống nhau. Khác PS-JEPA: không dựa vào premise ảnh hiện tại mất progress; cung cấp cùng history/task monitor cho baseline. Khác skill-exit proposal: thời lượng segment ban đầu cố định; đối tượng mới là latent path summary gắn với endpoint, không phải học stopping time.

Ước lượng công việc: 1–2 tuần G0–G1, 2–3 tuần G2–G3 nếu simulator/controller chạy được. Đề xuất trần qualification 200 GPU-hours, không phải dự báo throughput và chưa phải ngân sách được cấp. Có thể cần 1 GPU 24–48GB cho các model nhỏ, nhưng **chưa profile memory/runtime**, chưa xác minh tài nguyên thực tế. 10.000 training updates đầu dùng để đo wall time/VRAM trước quyết định tiếp; không extrapolate số giờ từ số parameters. Nếu hết trần mà chưa qua gate, ghi engineering/compute inconclusive và đánh giá lại.

Chưa có: số đo headroom; target/readout ceiling; implementation hay kiểm chứng của Path-JEPA; một novelty claim vượt được toàn bộ prior; bằng chứng empirical về joint composition. Vì vậy quyết định hiện tại là **ưu tiên câu hỏi này để qualification**, không tuyên bố đã tìm ra một method chắc đủ ICML. Nếu nó giảm về stochastic latent rollout + known signature pooling mà không có gain tái lập, cần bỏ method claim, trong khi vẫn giữ định hướng dài hạn JEPA.
