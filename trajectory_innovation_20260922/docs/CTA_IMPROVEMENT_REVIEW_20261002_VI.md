# CTA: sửa gì để cải thiện control và làm paper mạnh hơn?

Ngày 02/10/2026. Theo yêu cầu tìm hiểu kỹ của user. Đọc code hiện tại, artifacts qua L=15, các review cũ và nguồn paper chính thức; chạy hai loại audit có giới hạn qua Slurm. Không mở sealed roots, không huấn luyện model mới. Các đề xuất dưới đây là giả thuyết cần kiểm nghiệm, không phải kết quả cải thiện đã đạt được.

**Khuyến nghị:** tiếp tục CTA, sửa tính tái lập của proposal trước; sau đó ưu tiên WM và loss chọn action ở L=15 trên dữ liệu hiện có. Đặt contribution quanh lợi ích của target học từ tương lai so với bottleneck task-only ở cùng quality/compute. Goal transfer trên OGBench là phần mở rộng phù hợp nếu giữ claim reuse; temporal arena mới không phải điều kiện để tiến bộ.

## 1. Phát hiện mới làm thay đổi thứ tự ưu tiên

CPU audit **56686 COMPLETED 0:0**, squeue/sacct xác nhận. Artifact:
[audit.json](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_improvement_audit_56686/audit.json).

### 1.1 Reproduction lệch lớn ở cấp episode

Hai lần chạy L=15 có cùng 200 roots, cùng checkpoint policy và các source files runtime/batched loop/evaluation đã so byte-identical:

| Arm | 56374 | 56597 | Roots đổi outcome success | Roots đổi số bước |
|---|---:|---:|---:|---:|
| P0 | 100 | 103 | **43/200** | 76 |
| GEOM8 | 152 | 149 | **29/200** | 93 |

Chênh lệch tổng ±3 che đi nhiều episode thay đổi. Hai run khác batch size và các scorer được load. Không gộp chúng như các trajectory tái tạo chính xác. Các CI nội bộ của development không thay được kiểm tra này.

GPU audit **56690 COMPLETED 0:0, 2 phút**, giữ observation và candidate seeds của root 2205 cố định. Khi đổi số root trong batch từ 1 sang 4/25/100, initial actions khác tối đa **.0563/.0371/.0356 world units**. P0 root 2205, chạy riêng hay ghép với root 2206, có max coverage **.5075/.9382** dù cùng seed. Chế độ prototype cố định shape theo root cho cùng success ở bước 233, coverage .952236 trong cả hai trường hợp.

Đây là bằng chứng cho một nguồn sai lệch là proposal batch-size dependence, không phải kết luận đã giải thích mọi flip của hai run lớn. CuDNN deterministic=True vẫn không làm phép tính với các shape khác nhau bitwise identical. Không gọi đây là simulator-restore bug: fidelity của restore là một phép kiểm tra khác.

Đã thêm [CanonicalPolicyRunner](/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922/ti_wm/pusht_canonical.py): policy conditioning tính một lần cho mỗi history; denoising dùng microbatch cố định 8 candidate và padding khi cần. Collection và diagnostic closed loop có `--proposal-mode canonical`; mode và microbatch được ghi vào report. Mặc định batched cũ được giữ cho lịch sử; checkpoint và dirty edits trước đó được bảo toàn.

Job **56693 FAILED 1:0** sau 2:05 vì assertion đòi bitwise equality cho cả native coverage. Tuy vậy production sampler qua kiểm tra initial bank/prefix K=1/8/16 bitwise. CPU job **56705 COMPLETED 0:0** phân tích log lưu lại: root/decision/t/chosen/geometry đều identical; 4/128 coverage labels lệch tối đa **3.33e-16**, kết quả episode identical. Audit được sửa để đòi exact proposal prefixes và discrete/geometry logs, coverage atol=1e-12, rtol=0; lưu report trước assertion. Không đổi trạng thái lịch sử của GPU job thất bại. Kết quả và source snapshot:
[reproduction.json](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_repro_review_56693/results/reproduction.json).
[logcheck.json](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_repro_logcheck_56705/logcheck.json).
CPU job **56709 COMPLETED 0:0** kết hợp các prefix checks và log-field checks trên artifacts 56693 thành report với `bounded_contract_pass=true`, không rerun model hay simulator:
[contract report](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_repro_logcontract_56709/logcheck.json).
Một root được chọn để debug chưa chứng minh invariance trên toàn phân phối, learned scorer hoặc phần cứng khác; log này cũng không lưu toàn bộ simulator state để claim bitwise full-state trajectory equality.

Chế độ per-root hiện tại có thể làm batch evaluation chậm hơn. Tối ưu tiếp bằng fixed larger microbatch/padding chỉ sau khi kiểm tra invariance theo grouping, ordering, K và active-root count; không đổi sang dynamic shapes rồi bỏ numerical contract. K=1 được padding lên 8 nên timing P0 canonical không thay cho native K=1 policy latency: báo riêng native policy, overhead numerical contract và incremental cost K=8. Kiểm tra scorer/visual-encoder batch dependence cùng trong closed-loop experiment kế tiếp. Đo chi phí thật, không credit việc chuẩn hóa sampler thành novelty CTA.

### 1.2 Training sampler thiên về bank nhiễu, event thành công ít được ưu tiên

L=15 có 18,606 standard banks và 18,606 perturbed banks. Pool lấy 75% từ geometry-informative banks, 25% từ mọi bank. Vì perturbed có nhiều spread hơn, tỷ lệ exposure thực là **57.23% perturbed / 42.77% standard**; deployment dùng standard policy proposals.

| Split | Mixed native-success banks | Bị bỏ qua vì cả bank dưới geometry margin | Geometry oracle bỏ lỡ success có sẵn |
|---|---:|---:|---:|
| Standard train, n=18,606 | 513 | 45 | 6 |
| Perturbed train, n=18,606 | 1,233 | 4 | 10 |
| Selection, n=1,476 | 49 | 6 | 1 |

"Mixed success" nghĩa là cùng bank có candidate đạt native success và candidate không đạt. Dưới sampler hiện tại, chỉ **5.56%** exposure thuộc loại này.

Hai kết luận cần cùng được giữ:

- Có lý do ưu tiên native boundary/event trong sampling và decision loss, thay vì chỉ các bank có spread hình học lớn.
- True geometry oracle vẫn thường chọn được candidate thành công khi nó tồn tại. Dữ liệu không hỗ trợ việc quy toàn bộ thất bại cho target geometry sai, hoặc thay nó hoàn toàn bằng một target mới chưa kiểm chứng.

### 1.3 Không có bằng chứng rằng chỉ kéo dài training sẽ sửa được

CTA stage-2 selection gap L=15 tăng .085 → .208 → .260 → .325 tại steps 2k/4k/6k/8k, rồi .324 ở 10k. Source codec được chọn tại 8k/16k, direct tại 12k/16k. Đây là plateau ở criterion hiện tại, không phải proof converged globally; không đủ lý do chạy dài hơn như can thiệp duy nhất.

### 1.4 Training và evaluation cần khớp hơn

Stage 1 cộng losses của CODE/FULL/DIRECT rồi clip gradient norm toàn bộ parameters bằng một ngưỡng 1.0. Các nhóm không chia sẻ weights nhưng một nhóm gradient lớn có thể giảm update của cả các nhóm còn lại. Khi retrain recipe cuối, dùng optimizer/gradient clipping riêng cho từng nhóm và ghi gradient norm; đây là confound cần loại bỏ, chưa có bằng chứng nó gây toàn bộ CTA–DIRECT gap. DIRECT parameter-matched đã được train riêng là bằng chứng bổ sung có giá trị.

Stage 2 giữ ba losses NLL/score-consistency/ranking ở hệ số unit. Scalar NLL lớn hơn không chứng minh gradient của nó chi phối. Đo gradient norms ở vài checkpoint để quyết định trọng số, tránh sweep tùy ý. Success-bonus và hindsight hooks đã có nhưng **run L=15 hiện tại để cả hai bằng 0**; không mô tả chúng như can thiệp đã thử thành công.

So sánh L=8 và L=15 còn khác tập training, ngoài execution interval. Chọn L=15 làm operating point có lý do là WM gap và headroom hiện tại, không claim đã chứng minh CTA càng tốt khi L tăng. Muốn claim horizon robustness cần matched data construction, token/visual budgets và retraining recipe.

## 2. Sửa method nào có cơ sở nhất?

### Ưu tiên 1: giữ source tốt, sửa prediction và selection trên đúng bank

L=15 source CODE giữ geometry gap **.667**, predicted CTA **.374** trên selection. Trên các crossing của P0, CODE chọn được success khoảng **91%**, CTA khoảng **37%**. Chỗ tụt này đặt WM/readout of predicted code thành ưu tiên cụ thể.

Lượt end-to-end tiếp theo nên:

1. Dùng proposal contract canonical cho mọi arm, cùng roots, K=8 và L=15; ghi phiên bản numerical policy. Giữ checkpoint source/reader hiện tại cho pilot này để tránh đồng thời đổi target và predictor.
2. Có continuation control của CTA cũ nhận đúng cùng updates, và áp dụng cùng loss/data treatment cho DIRECT/ENDPOINT.
3. Lấy phần lớn batch từ standard banks; dành một phần xác định trước, chẳng hạn 1/4, cho mixed-success banks; phần còn lại giữ geometry-informative và flat banks. Đây là cấu hình pilot được đề xuất, không phải tỷ lệ tối ưu đã biết. Chọn trên train/selection, không chọn trên closed-loop roots.
4. Thêm loss ưu tiên chọn được một candidate thành công trong mixed bank, đi qua reader:

   `L_hit = -log(sum_{k: hit_k=1} softmax(score_k / T))`.

   Giữ dense geometry ranking và code NLL. Loss này không bắt một thứ tự giả giữa các candidate đều đạt success, và cung cấp gradient ở native boundary dù geometry difference dưới margin. WM nhận C,A; reader vẫn chỉ nhận C,S,goal. Native hit labels dùng để train decision loss, không vào input deployment.
5. Tính metric selection trên toàn split và đúng cách average 16 goal images như deployment; hiện model selection dùng 4 goal images và cap 1,200 banks. Định nghĩa criterion trước run, giữ dense quality bên cạnh hit capture; không chọn checkpoint theo một panel hiếm hoặc xem closed-loop rồi đổi primary.
6. Nối training → closed loop → aggregate trong cùng protocol; đọc source/predicted retention, capture, false selections, success và native score. Không dùng offline win làm điểm kết thúc.

Giả thuyết: training hiện phân bổ capacity theo các khác biệt lớn/easy, trong khi các khác biệt quyết định success cần được ưu tiên. Chưa biết gain cuối cùng; recovery và long-horizon failures vẫn có thể hạn chế lợi ích.

Nếu cần đổi predictor, control liên tục **cùng 48 chiều**, giữ source/reader và dữ liệu, là phép thử có sức phân biệt hơn việc tăng nhiều layers. Current categorical predictor factorizes FSQ coordinates và reader đọc mean code. So với predictor continuous được supervise cùng target sẽ cho biết NLL/quantization structure giúp hay làm khó learning. Có thể kiểm tra joint 256-way token head sau đó nếu coordinate factorization được chẩn đoán là vấn đề; các hidden features hiện vẫn được tính jointly, nên không nói predictor hoàn toàn bỏ tương quan token.

Không bật lại co-design λ=.1 như một giải pháp mặc định: cấu hình đó đã làm source code collapse. Co-design tương lai cần bảo vệ retained query information bằng teacher/constraint rõ, và so matched continuation. Anti-collapse hoặc tăng token diversity tự nó không chứng minh giữ đúng thông tin.

### Ưu tiên 2: làm lợi ích của future target và compute có thể được quy kết

Current Scorer self-attends trên C≈321 tokens, goal=256 và code=16. Như vậy code ít token nhưng C/goal vẫn được xử lý lại cho mỗi candidate/query. Một reader query cross-attention nhỏ với context/goal caches có lý do compute rõ hơn. Cache phải áp dụng công bằng cho các baseline; không mặc định code ít token sẽ nhanh.

Control quan trọng nhất là **factorized DIRECT**:

`z = f(C,A); score = h(C,z,g)`.

Đặt z cùng 48 chiều/token layout, backbone/readout/caching tương ứng với CTA; học task losses nhưng không future-code target. Direct 14-layer hiện có chỉ khớp tham số, chưa thay control này. Thêm continuous future-supervised bottleneck để tách lợi ích future supervision khỏi FSQ nếu giữ claim về discrete coding.

Đo quality/compute frontier thực tế với một anchor configuration và vài mức budget có mục đích, cùng GPU/precision/K/data/horizon. Tách shared visual encode, policy sampling, predictor, reader, total, p95, memory, training/branch-data cost. Runner seconds có simulator/cross-scoring không phải deployment time. Sampler ~770 ms chiếm phần lớn total hiện tại; speedup riêng 4.1 vs29 ms không thành speedup toàn planner cùng tỷ lệ.

Phải có compact per-frame predictor mạnh, có thể parallel; chỉ thắng DINO-WM 256-token hoặc autoregressive không đủ bảo vệ contribution trước CompACT. Official DINO adapter hiện dùng current agent velocity và render current observation 224 trực tiếp; CTA dùng agent-position history và 96-px observation. Đây là khác input operating point, dù không dùng future observations. Báo external official control riêng và dùng matched input/control cho causal comparison; không gọi tất cả là input/data-matched.

### Ưu tiên 3: nối representation với value dài hạn khi objective ngắn hạn thật sự giới hạn

Nếu sửa WM vẫn không chuyển ranking thành control, một chunk-return target có bootstrap là hướng hợp lý:

`Y_g = sum_{j=0}^{h-1} gamma^j r_j(g) + gamma^h (1-done) V_bar(x_h,g)`.

Ở đây h là số bước thực sự executed trước termination; goal timing, episode truncation, reward và continuation operator phải được định nghĩa. Teacher V chỉ train từ train data/observations được phép; label cho reader tạo offline, deployment vẫn chỉ đọc predicted S. Không giả định simple squared TD cho shortest path, hay hồi quy elapsed time trên behavior demos là optimal value.

Baseline bắt buộc: chunk Q(C,A,g) dùng cùng return/teacher và dữ liệu, endpoint+V cùng history, và matched CTA. Cùng target tốt hơn cho mọi method không tự chứng minh abstraction. Project đã có GCIVL checkpoints và TD-reader review; đây không phải lần đầu TD được cân nhắc. Không dùng continuation rollout dài mặc định: local continuation pilots từng ít headroom và tốn compute.

Hướng này có chi phí/rủi ro teacher error cao hơn native-hit intervention, nên xếp sau cho PushT. OGBench phù hợp hơn để tận dụng play trajectories và goal-conditioned value.

## 3. Arena và câu hỏi nào có thể làm paper mạnh hơn?

Giữ PushT cho matched closed-loop/control-cost. Dùng **OGBench hiện có** nếu cần goal reuse/transfer; task labels là endpoint vẫn hợp lệ.

Một phép thử reader-goal transfer cần giữ lại semantic goals khỏi task losses và model selection **của toàn pipeline**, kể cả Stage 1 codec. Existing encoder đã học tất cả goals không được dùng để gọi leave-goals-out. Có thể giữ dynamics observations không nhãn của các vùng đó nếu công khai đây là query-supervision transfer, không phải unseen-dynamics generalization. Hindsight relabeling không được đưa held-out goal hoặc vùng success tương đương vào training labels.

Mọi arm dùng cùng goal-conditioned proposal policy và same candidate construction. Tách CTA's gain khỏi gain do nâng policy. Official per-step HIQL không thể biến thành fixed open-loop chunk bằng cách đọc simulated futures để sinh suffix. Muốn policy chunk mạnh hơn cần một policy chunk thực sự hoặc thay predictive object sang feedback option và mô tả đó là hướng mới.

Q-chunking/FQL là lựa chọn có cơ sở để nâng một BC chunk proposer, nhưng đồng thời tạo baseline chunk-Q mạnh. Đây là một nhánh có mục tiêu nếu support hiện tại không đủ; không cần chạy policy training lớn trước khi sửa vấn đề PushT đã biết.

Không ưu tiên LIBERO-Safety frozen pi0.5/K8/H5 hiện tại: pilot không có violating candidate ở 2,038 oracle decisions. Không rerun missing shard chỉ để hoàn thành số tập khi nó chưa tạo safety choice. RinseBowls có native temporal semantics hấp dẫn nhưng chưa có qualified policy/control; không có bằng chứng nó là đường rẻ nhất tới paper.

Nếu muốn giữ headline intermediate events, chọn đúng một objective native trong đó event ảnh hưởng completion và có proposal diversity quan sát được. So endpoint/history, event predictor+monitor, direct statistic heads và CTA trên cùng task distribution. Thêm decorative query hoặc task nhiều bước không đủ. Đây là mở rộng claim, không phải tiền điều kiện để tiếp tục CTA endpoint.

## 4. Literature giới hạn novelty thế nào?

| Nguồn chính thức | Điều đã có | Hệ quả cho CTA |
|---|---|---|
| [CompACT, CVPR 2026](https://arxiv.org/abs/2603.05438) | Tokenizer observation 8–16 tokens, frozen visual features, compact WM planning; có parallel prediction trong manipulation | Ít token/FSQ/one-pass không đủ làm novelty. Cần đối chứng compact frame và lợi ích target conditional segment/query. |
| [Value Equivalence](https://arxiv.org/abs/2011.03506) | Model learning dựa trên những phép đánh giá value cần cho planning | Giữ decision-relevant information là nguyên lý có trước; không đổi cách diễn đạt thành một theorem novelty. |
| [RaMP](https://www.boyuan.space/ramp-rl/) | Dự đoán tích lũy random features của action sequence để tái sử dụng cho downstream rewards | Dự đoán một lần, chấm nhiều reward và bỏ rollout từng bước đã có tiền lệ gần. |
| [V-GPS](https://github.com/nakamotoo/V-GPS) | Value guidance để steering proposal của generalist policy | Reranking frozen policy bằng learned score không phải contribution độc lập mới. |
| [Value-guided JEPA](https://arxiv.org/abs/2601.00844) | IQL/value-shaped representation cho planning | Thay latent L2 bằng value/TD chưa đủ mới. |
| [Q-chunking](https://arxiv.org/abs/2507.07969) | Critic trong chunk-action space, chunk TD backups và behavior constraints | Chunk-Q là direct baseline mạnh khi CTA thêm continuation value. Paper này chủ yếu offline-to-online; không suy kết quả benchmark sang fully offline CTA. |
| [LeWorldModel](https://arxiv.org/abs/2603.19312) | JEPA gọn train ổn định end-to-end từ pixels | Đối thủ latent WM không nhất thiết là model to, pretrained, rollout chậm. |
| [Temporal-Distance JEPA](https://arxiv.org/abs/2607.25337) | Representation/goal-cost học để phù hợp planning | Phân biệt temporal-distance regression với Bellman TD; không claim mọi value approach thất bại từ C2. |
| [Adaptive Q-Chunking](https://arxiv.org/abs/2605.05544) | Chunk-length adaptation trong offline-to-online RL | Adaptive execution interval cần contrast riêng; không đủ novelty chỉ vì chọn L động. |

Contribution có thể bảo vệ: **future-supervised conditional segment target giữ thông tin cho những truy vấn quyết định, dự đoán được từ proposed actions, và tạo lợi ích quality/compute hay query transfer so với task-only bottleneck và compact frame target**. Kiến trúc ghép module không thay evidence. Source 128-bit bound và continuous predictive mean phải được tách rõ; khi chưa có hard-code deployment, viết compact continuous predictor supervised by FSQ targets.

## 5. Một lượt phát triển có giới hạn và gói paper cần đạt

Không đề xuất sweep rộng hoặc chuỗi novelty/headroom gates. Lượt tiếp theo có thể là một end-to-end experiment gồm:

- Runtime canonical đã kiểm chứng; P0/GEOM8 anchors và checkpoint CTA hiện tại.
- CTA matched continuation và CTA native-hit-aware/data-rebalanced; DIRECT và ENDPOINT cùng intervention, update budget và candidate bank.
- Một source/reader cố định cho pilot, native labels có sẵn; không recollect toàn dataset để thử loss trước.
- Training + closed loop + aggregate, benchmark deployment timing riêng. Phần training trên existing features và frame reader không load privileged future lúc deploy.
- Đọc method-minus-control và method-minus-baseline; metric thấp dùng để định vị rồi sửa, không tự động abandon CTA. Nếu đã sửa target cho mọi arm mà direct bắt kịp thì kết luận đúng, không credit objective gain cho abstraction.

Sau development: retrain công thức cuối từ đầu nhiều seeds; freeze metric, candidate/numerical policy, data, horizon, training/selection recipe trước test. Số roots phải dựa trên effect/discordance và ngân sách; 400 không tự bảo đảm thấy +4 pp, và p>0.05 không chứng minh non-inferiority.

Paper mạnh cần đồng thời: closed-loop hữu ích; một comparison giải thích vì sao học target từ future có ích; compute thật; phạm vi query/temporal khớp claim; uncertainty và reproducibility. Không yêu cầu thắng mọi baseline ở mọi arena, nhưng chỉ số offline cao trên một dev set không đủ.

Deadline [CVPR 2027 chính thức](https://cvpr.thecvf.com/Conferences/2027/CallForPapers): đăng ký 10/11, paper 16/11, supplement 23/11/2026 AoE. Từ 2/10 đến paper deadline khoảng 45 ngày. Scope thực dụng là một PushT comparison có đóng góp rõ và một OGBench query/control result nếu có điều kiện, thay vì dựng đồng thời nhiều simulator.

Review đã kiểm tra monthly rankings và cả peer jobs trước các submission. GPU audits có cap 5 phút/job, CPU audit cap 5 phút. Không có training campaign queued; mọi lượt tiếp theo vẫn phải tuân cap 50% của user thứ 5 tính planned time limits trên GPU/CPU/memory. Trạng thái, source snapshots và artifacts được ghi tại [JOB_LEDGER.md](/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922/JOB_LEDGER.md).
