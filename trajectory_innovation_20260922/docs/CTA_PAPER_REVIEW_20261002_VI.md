# Phản biện bản thảo CTA theo rubric nghiên cứu — 02/10/2026

## 1. Nhận định chính và phạm vi đánh giá

CTA đã có một phương pháp được cài đặt và bằng chứng phát triển có giá trị: học target từ tương lai, dự đoán target từ context/action chunk và chấm bằng reader không đọc action. Kết quả lịch sử cho thấy ranking tốt và một số cải thiện success so với policy mặc định. Tuy nhiên, bằng chứng hiện có chưa xác nhận lợi thế closed-loop so với các baseline mạnh trên dữ liệu độc lập, chưa xác nhận khả năng chuyển sang goal mới, và chưa đủ để quy toàn bộ lợi ích cho conditional segment target. Vấn đề tái lập do hình dạng batch của proposal còn làm giới hạn cách sử dụng các kết quả lịch sử.

Bản viết lại nên là **bản thảo phát triển có kết quả sơ bộ thật và ranh giới kết luận rõ**, thay vì bản đề xuất chỉ có placeholder hoặc bản kết quả cuối giả định rằng thí nghiệm chưa chạy đã thành công. Rubric yêu cầu mạch **Problem → Importance → SoTA → Gap → Novelty → Method → Results → Validation → Discussion → Contribution → Impact**; rubric cũng cho phép pilot và benchmark sơ bộ khi nghiên cứu đang tiếp diễn. Không thể suy ra điểm acceptability hoặc xác suất được CVPR chấp nhận từ checklist này.

Đối chiếu dựa trên [rubric người dùng](</Users/nhatcuong/.codex/attachments/3b90d3ff-589a-467e-8e80-9976e4d34a40/Pasted text.txt>), ba audit độc lập về [phương pháp](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/docs/paper_review_20261002/method_audit.md), [bằng chứng](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/docs/paper_review_20261002/evidence_audit.md), [literature](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/docs/paper_review_20261002/literature_audit.md), source H100 và các summary/config đã lưu. Báo cáo này không chạy model, physics, thống kê mới, test tích hợp hay job. Nó đánh giá luận điểm và bằng chứng, không xác nhận chất lượng dàn trang của PDF cuối.

Nguồn bản thảo: [main.tex](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/paper_cvpr/main.tex), [problem](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/paper_cvpr/sec/3_problem.tex), [method](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/paper_cvpr/sec/4_method.tex), [experiments](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/paper_cvpr/sec/5_experiments.tex), [conclusion](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/paper_cvpr/sec/6_conclusion.tex). Đầu ra đã biên dịch của bản viết lại là [main.pdf](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/paper_cvpr/main.pdf) và [supplement.pdf](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/paper_cvpr/supplement.pdf); trạng thái cuối: main 8 trang gồm references, supplement 3 trang; build local thành công và các trang đã được kiểm tra trực quan.

## 2. Mạch khoa học cần giữ theo rubric

| Thành phần rubric | Luận điểm phù hợp với CTA | Điểm cần kiểm soát |
|---|---|---|
| Problem | Trong candidate-based visual planning, cần dự đoán hệ quả của action chunk đủ tốt để chọn proposal trước khi thực thi. | World model dự đoán; reader/cost đánh giá. Không đồng nhất hai chức năng. |
| Importance | Số token dự đoán và chi phí đọc nhiều candidate/goal có thể ảnh hưởng ngân sách suy luận. | Token ít chưa chứng minh whole-planner nhanh hơn; sampling có thể chiếm phần lớn latency. |
| SoTA | Visual latent WM, compact tokenizer, policy guidance và value-aware modeling đã giải quyết các phần của bài toán. | Không dựng một baseline yếu giả định mọi phương pháp khác đều dự đoán từng pixel hoặc không dùng thông tin quyết định. |
| Gap | Cần kiểm tra liệu future-supervised conditional segment target tạo lợi ích quality/compute hơn endpoint target hoặc action bottleneck trên cùng planner. | Đây là câu hỏi nghiên cứu cần chứng minh; không tuyên bố không có prior art chỉ vì tên CTA mới. |
| Novelty | Một target/interface và recipe cụ thể: source nhìn `C,tau`, predictor nhìn `C,A`, reader nhìn `C,S,g`; target tổng hợp các quan sát trong segment. | FSQ, nén token, reranking và query-independent prediction riêng lẻ không đủ novelty. |
| Method | Nêu chính xác input, horizon, sampling, loss, frozen modules, privileged training labels và checkpoint selection. | Tách cấu hình L8 lịch sử, L15 v2 và native-hit continuation chưa chạy. |
| Results | Dùng số từ artifact hoàn tất, chỉ chọn những kết quả trả lời câu hỏi chính. | Offline RG, oracle, selection và learned closed-loop success là các loại bằng chứng khác nhau. |
| Validation | Matched candidates/protocol, numerical invariance, uncertainty theo root, held-out evaluation và component timing. | Đối chiếu DINO-WM hiện có khác input/data/precision; ghi rõ thay vì gọi hoàn toàn matched. |
| Discussion | Giải thích vì sao ranking cao chưa chuyển đều thành success; vì sao prediction/readout có thể là bottleneck cụ thể ở L15. | Không kết luận WM luôn là bottleneck hoặc co-design luôn thất bại. |
| Contribution/Impact | Đóng góp có thể là kiến thức về target/interface giúp planning dưới ngân sách cụ thể, nếu được xác nhận. | Không thay bằng “đã xây hệ thống”; không mở rộng sang arbitrary goals, path events hoặc robotics nói chung khi chưa có kiểm chứng. |

Ba câu hỏi nên hiện rõ ở đầu phần experiments: **CTA giữ được ranking nào sau dự đoán? Ranking đó cải thiện learned closed-loop selection đến đâu so với matched baseline/control? Chi phí thành phần và tổng planner thay đổi bao nhiêu trên cùng điều kiện đo?**

## 3. Định vị SoTA và đóng góp có thể bảo vệ

Literature nên tổng hợp theo đối tượng dự đoán và giao diện đánh giá, không liệt kê từng paper rời rạc.

| Nhóm prior art | Khả năng đã có | Khác biệt CTA cần chứng minh |
|---|---|---|
| [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) | Dự đoán patch features để planning đến image goal. | Một target segment được học bằng reader/reconstruction, thay cho giữ spatial feature map của từng bước. |
| [CompACT](https://arxiv.org/abs/2603.05438) | Tokenizer observation chỉ 8–16 token, gắn với latent WM và compute tradeoff. | Ít token và one-pass không đủ. Cần giá trị của conditional segment target so với compact observation/endpoint target. |
| [GPC](https://computationalrobotics.seas.harvard.edu/GPC/) | Forecast hệ quả proposal của frozen policy rồi rank/refine tại inference. | CTA thay representation dự đoán bằng target compact được học; reranking policy proposal không phải phát minh độc lập. |
| [TAP](https://arxiv.org/abs/2208.10291) | State-conditioned discrete latent action codes và trajectory reconstruction để tìm kiếm trajectory. | CTA forecast outcome code từ một action chunk đã cho và không cho reader đọc chunk; conditional compact trajectory code đã có tiền lệ. |
| [Value Equivalence](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html) | Dùng tài nguyên biểu diễn cho mô hình hữu ích với value-based planning. | Decision-aware compression là động cơ đã có; CTA chưa chứng minh định lý value equivalence. |

Đề xuất giữ **hai đóng góp phương pháp và một đóng góp thực nghiệm có điều kiện**:

1. **Conditional segment target cho proposal scoring:** học source FSQ code từ context và future segment, forecast từ proposed chunk, đọc qua giao diện action-blind. Điểm khoa học là cách chọn target và đường truyền thông tin, không phải tên mới cho hệ thống.
2. **Recipe để giữ thông tin hữu ích sau forecast:** reader ranking và feature reconstruction cho source; categorical prediction, centered reader-score consistency và weighted ranking cho predictor. Native-hit continuation là thử nghiệm phát triển thêm về task alignment, chưa phải kết quả xác nhận.
3. **Đánh giá có đối chứng để phân biệt representation, prediction và control:** matched learned baselines, actual/source/predicted reading, closed-loop outcomes và cost. Đóng góp này chỉ trở thành kiến thức thực nghiệm mạnh khi các contrast và uncertainty cần thiết hoàn tất.

Factorized direct `h(C,A) → D(h,g)` là đối chứng quan trọng nếu claim nhấn vào future supervision: nó cũng có thể cache một vector và tái sử dụng cho nhiều goal. Reader action-blind là constraint hữu ích, nhưng các outcome-evaluation interface khác cũng không nhất thiết đọc action trực tiếp. Endpoint task vẫn hợp lệ cho CTA; không cần pivot sang temporal task trước khi cải thiện pipeline hiện tại.

## 4. Ma trận claim–evidence

| Claim | Bằng chứng hiện có | Cách viết được phép / validation còn thiếu |
|---|---|---|
| Source target có capacity tối đa 128 bit | FSQ levels `(8,8,4)`, M16; source alphabet hữu hạn. | Đúng cho source code cố định/evaluation mode. Không gọi đây là measured entropy hay deployment bitrate. |
| Reader deployment nhận code nhỏ | Predictor trả M×3 kỳ vọng coordinate; M16 là 48 số liên tục. | Mô tả đúng tensor/interface. Bound 128 bit không áp dụng cho predictive mean. |
| Deployment không đọc future/action qua reader | Source review: source nhìn `C,tau`, predictor `C,A`, reader `C,S,g`; score cache được tính trước simulation. | Boundary đúng theo code; GPU smoke/integration của native-hit bản mới vẫn chưa chạy. Privileged poses/hits là training/diagnostic labels. |
| CTA ranking tốt hơn END/DIRECT ở L8 | Cùng P0 banks, CTA RG .680; END .635; DIRECT .483; parameter-matched DIRECT .502, có paired CI. | Kết quả development ranking thật. Chưa chứng minh nguyên nhân là conditional coding hoặc tương lai supervision. |
| CTA tăng closed-loop success so với P0 | L8 140/200 vs 122/200; L15 121/200 vs 103/200, historical mode. | Báo rõ development roots dùng lại, một seed, historical proposal mode và paired uncertainty. Không chuyển thành canonical result. |
| CTA thắng baseline learned mạnh | Contrast CTA–END/DINO closed-loop có CI bao gồm 0; CTA–DIRECT L15 còn không nhất quán giữa bootstrap và McNemar quanh .05. | Chưa có cơ sở cho dominance hoặc non-inferiority. Báo các thống kê cùng nhau, không chọn phép kiểm thuận lợi. |
| Source→predicted gap chỉ ra bottleneck L15 | Selection RG CODE .667 vs CTA .374; native crossing capture CTA .3714 vs CODE .9143 trên 35 cơ hội. | Chẩn đoán cụ thể cho cấu hình/phân phối này. Selection không phải independent test; cơ hội crossing hiếm. |
| CTA có lợi ích compute | Timing lịch sử CTA G16 16.87 ms, END 23.40 ms, DIRECT 26.81 ms; common sampling khoảng 765.33 ms. | Component evidence; không suy whole-planner speedup tương ứng. Cần benchmark deployment riêng dưới canonical mode và cùng precision/accounting. |
| Conditional encoder chỉ dành bit cho điều context chưa biết | Encoder được phép đọc context; decoder cũng đọc context. | Là cơ chế cho phép conditional encoding, chưa là tính chất đã đo. Cần conditioning control nếu giữ claim causal. |
| Reconstruction bảo đảm chống collapse/goal transfer | Có reconstruction loss; co-design lịch sử còn có code collapse. | Chỉ nói khuyến khích giữ future features. Không có guarantee hoặc goal-transfer result. |
| Predictive mean thể hiện đầy đủ uncertainty | Có distributions theo coordinate và mean của chúng. | Sai nếu diễn giải mạnh: distributions khác nhau có thể cùng mean; nonlinear reader không đọc kỳ vọng score. |
| Query reuse/path retention đã được xác nhận | Predictor không nhận goal; source nhận intermediate observations. | Là architecture facts. 16 PushT goal images cùng một target pose; distinct-goal transfer và intermediate-event benefit chưa được cô lập. |
| Native-hit intervention cải thiện CTA | Source, pure helper tests và launcher đã chuẩn bị trong DNS failure. | Chỉ mô tả hypothesis/design. Chưa có checkpoint treatment, GPU forward/backward hay closed-loop improvement. |

Không dùng ladder như một chuỗi chất lượng bắt buộc giảm dần: ở L8 predicted CTA RG cao hơn source CODE. FULL chỉ đọc endpoint/proprioception qua reader khác, còn source CODE nhận thêm intermediate observations; FULL→CODE không cô lập riêng tổn thất compression. Source→predicted dùng cùng frozen code reader là forecast diagnostic sạch hơn, nhưng vẫn không phải decomposition của episode success.

## 5. Kết quả sơ bộ nào đáng đưa vào main paper?

Các số sau lấy từ summary đã lưu, không tái tính trong lượt review. L8 và L15 cùng 200 development roots 2200–2399 và một training seed, nhưng training data khác nhau; đều dùng historical batched proposals. Bảng đủ cho thông điệp phát triển, không thay held-out test.

| Horizon | P0 | CTA v2 | ENDPOINT | DIRECT | DINO-WM adapter |
|---|---:|---:|---:|---:|---:|
| L8 success /200 | 122 | 140 | 138 | 135 | 136 |
| L15 success /200 | 103 | 121 | 108 | 103 | 112 |

Ở L15, CTA−P0 là **+9 điểm phần trăm [1,17]**, exact McNemar p=.044371. CTA−END là **+6.5 [−2.5,15.5]**, p=.182077; CTA−DINO là **+4.5 [−4.5,13.5]**, p=.385669. CTA−DIRECT là **+9 [.5,17.5]**, nhưng McNemar p=.050452; cần giữ cả hai thông tin. Các contrast mean normalized score được kiểm tra đều có CI bao gồm 0. Vì vậy, câu “CTA improves development success over P0” có ranh giới cụ thể; câu “CTA outperforms all learned baselines” chưa được hỗ trợ.

Ranking L8 trên cùng P0-state banks là evidence tốt hơn cho một câu hỏi khác: CTA−END RG **+.045 [.011,.079]**, CTA−parameter-matched DIRECT **+.178 [.138,.220]**. Capacity đơn thuần chưa đóng ranking gap, nhưng factorized direct chưa có nên không thể gán causal effect cho future supervision.

Timing lịch sử cần tách shared context, predictor/reader và sampling. CTA nhanh hơn END/DIRECT trong phép đo G16, nhưng DIRECT nhanh hơn CTA ở G1 (2.54 vs 4.11 ms). G16 ở đây là 16 view của một target pose, đo computation scaling chứ không đo semantic query transfer. DINO timing đã gồm own encoding, còn CTA/END/DIRECT chưa gồm shared context; input và equal precision cũng chưa hoàn toàn matched. Giữ bảng latency ở dạng component benchmark với disclosure, không viết một hệ số whole-planner speedup từ các cột không cùng định nghĩa.

K64 và co-design nên đưa vào supplement nếu giúp giải thích failure. K64 closed-loop chỉ hoàn tất 50 roots ở subset được lưu; không gọi đó là completed 100-root population result. Co-design λ=.1 từng làm source RG .706→.458, perplexity 131.197→1.371 trong khi WM CE giảm .5577→.0860 nats. Nó cho thấy objective dễ dự đoán có thể làm mất thông tin quyết định trong cấu hình đã thử, không chứng minh co-design luôn sai.

## 6. Method và theory: những điều phải sửa cho chính xác

Source code phải được viết như grid-valued `S ∈ (Q8 × Q8 × Q4)^M`; integer token index chỉ là biểu diễn tương đương. Nếu lấy kỳ vọng trực tiếp trên index tùy ý thì không đúng với code. Với frozen deterministic encoder, `I(tau;S|C)=H(S|C)≤128 bit`; implementation dùng finite-alphabet constraint, không tối ưu/đo conditional entropy hoặc entropy-coded rate.

Decoder chỉ tái tạo **endpoint/intermediate image features**, không tái tạo future proprioception dù source encoder đọc nó. Normalization bằng copy-baseline error không có nghĩa mọi code hằng đều có loss bằng 1: decoder vẫn có thể đoán từ context. Predictor được supervision bởi labels thông qua ranking/consistency/native-hit losses; đúng là labels không phải input, nhưng không đúng nếu nói training “không dùng labels”.

v2 không train context-only prior, nên bỏ câu “ước lượng mutual information từ excess NLL”. Ngay cả khi có hai predictor, cross-entropy gap còn chứa chênh lệch model approximation error; không tự là unbiased MI estimate. `S=A` và continuous `S=tau` nên là architectural comparators, không phải hai trường hợp đặc biệt đúng toán học của cùng finite-alphabet encoder `q(C,tau)`.

Có thể giữ một lemma ngắn: nếu discrepancy tối đa giữa **centered source-reader và predicted-reader scores** là ε, reader-score regret của lựa chọn predicted không vượt `2ε`; source margin lớn hơn `2ε` bảo toàn winner. Đây là analysis của ranking interface, không phải theorem mới về coverage/success. Logistic ranking không calibrate reader scores sang physical utility; thêm một định lý task regret mà không nêu calibration/control assumptions sẽ làm bản thảo yếu hơn.

Cấu hình L15 v2 56504 thực tế: M16, L15, lr `3e-4`, dropout .1, weight decay .05, stage1 16,000/stage2 10,000 updates, 16 decisions/update, warmup500. Selected codec/CTA là step8000, FULL16000, DIRECT12000, ENDPOINT8000. Selection dùng retained geometry gap trên bốn goal views cố định; final ladder/deployment dùng đủ 16. v2 được train từ đầu, nên không gộp nó vào câu “mọi checkpoint đều warm-started”. Native-hit continuation mới dùng full 16-view selection và criterion khác; phải ghi thành recipe riêng.

## 7. Reproducibility và native-hit: đã làm gì, chưa làm gì?

Hai historical L15 runs trên cùng roots/weights từng cho P0 100 rồi 103 successes, với **43/200 success flips**; GEOM8 có **29/200 flips**. Proposal batch shape là một nguyên nhân đã được chứng minh; chưa có accounting đầy đủ cho mọi flip và không được mặc định gọi đó là simulator restore failure.

Canonical policy adapter cố định denoising microbatch K8; xử lý scorer theo từng root đã được chuẩn bị trong source continuation nhưng chưa được qualification đầy đủ. Kiểm tra đã lưu cho root 2205 khi chạy riêng so với gom cùng root 2206 cho thấy proposal prefixes K1/8/16 và các discrete/geometry fields trùng; coverage khác tối đa `3.33e-16`, đạt tolerance `1e-12`. Job 56693 vẫn là **FAILED 1:0** do assertion coverage bitwise; CPU log analysis sau đó xác nhận bounded contract chứ không đổi trạng thái job. Kiểm tra này chưa chứng nhận toàn bộ roots, learned scorers, backend hay hardware. Canonical còn đổi proposal so với historical mode, nên lịch sử không thể được đổi nhãn thành canonical results.

Theo [báo cáo nối tiếp H100](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/docs/CTA_H100_RESUME_20261002_VI.md), tại thời điểm đối chiếu ngày 02/10, DNS/controller/accounting access đang lỗi và chưa nộp job native-hit mới. Source đã có CTA/ENDPOINT/DIRECT × BASE/CTRL/HIT cùng P0/GEOM8, full metadata identity, các kiểm tra numerical đã chuẩn bị trong source và strict aggregate validation. Chín helper tests thuần Python được ghi nhận pass local; integration tests, GPU smoke, backward/frozen invariance và learned closed loop vẫn chưa chạy. Pure helper test pass không chứng minh planner pass.

Native-hit plan kết hợp **standard-heavy sampling, mixed-hit replay và hit loss**. Contrast treatment–continuation-control đo can thiệp kết hợp này; không được gọi là isolated hit-loss ablation. Training caches còn từ historical sampler, evaluation dùng canonical; khác biệt đó phải nằm trong experiment description.

## 8. Validation tiếp theo: có thứ tự, có giới hạn

Không cần thêm một chuỗi novelty/headroom gate trước implementation. Thứ tự hợp lý là sửa correctness, chạy end-to-end bounded pilot, đọc bottleneck quan sát được rồi quyết định validation cần thiết cho claim cuối.

| Ưu tiên | Công việc | Kết luận mà nó có thể cung cấp |
|---|---|---|
| 1 — correctness | Khi scheduler truy cập lại: CPU integration/regression tests; GPU smoke có forward/backward, frozen-reader checks và qualification tích hợp cho proposal/scorer grouping/native-hit outcomes. | Xác nhận code mới chạy đúng contract. Không phải success experiment. |
| 2 — bounded canonical controls | Continuation hữu hạn; 11 acting arms trên 20 development roots, K8/L15, canonical mode, cùng checkpoint lineage/update/selection budget; aggregate full episodes. | Pilot phân biệt BASE/CTRL/HIT và native-objective alignment của CTA so với END/DIRECT. Dùng để debug/chọn bước tiếp; chưa xác nhận headline. |
| 3 — measurement và attribution cần cho claim | Đo predictor, reader, shared encoding, policy và total deployment latency cùng GPU/precision/K/L; nếu giữ claim future supervision, bổ sung factorized direct đủ khớp. | Phân biệt representation benefit với target/loss/compute confounds. Chỉ chạy control giải quyết quyết định hoặc claim cụ thể. |
| 4 — độc lập và uncertainty | Sau khi recipe được chốt, freeze metric, candidate/numerical policy, data split, precision và selection; đánh giá roots độc lập, training seeds phù hợp ngân sách. | Xác nhận generalization của quality/compute result với uncertainty. Không dùng lại development win làm final evidence. |
| 5 — claim mở rộng có chọn lọc | Conditionality/end-only/equal-budget controls hoặc held-out semantic goals/event task khi quyết định giữ các claim tương ứng. | Chỉ cần nếu paper muốn claim những tính chất này. Endpoint task vẫn đủ hợp lệ để phát triển CTA trước đó. |

Số roots/seeds cuối phải dựa trên effect, discordance và compute budget; 400 roots không tự bảo đảm power, còn p>.05 không chứng minh non-inferiority. Phải phân biệt confidence interval của development estimate với xác nhận độc lập. Cần giữ source/config/checkpoint hashes, job IDs, shard/root identities, completion state, version/clone identity và partial artifacts trong [ledger](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/JOB_LEDGER.md).

Mọi compute tiếp theo phải dùng `sbatch`, tài nguyên/time limit rõ và cả `squeue`/`sacct` để xác minh trạng thái. Trước GPU submission hoặc CPU array, kiểm tra ranking tháng hiện tại cho GPU/CPU/memory và tính planned full time limits; giữ usage của `nhatnc129` không vượt 50% user xếp thứ 5. DNS chưa cho kiểm tra thì không dùng quota/states cũ để hợp thức hóa submission mới.

## 9. Hình, discussion và conclusion cần đóng vai trò gì?

Theo rubric, hình phải phục vụ lập luận khoa học. Một bộ tối thiểu có ích gồm: overview tách training/deployment và privileged boundary; paired closed-loop comparison với uncertainty; component cost theo số goal views; source/forecast failure diagnostic; numerical grouping illustration. Hình chọn theo large geometry gap chỉ là minh họa đã được lựa chọn, không phải representative success evidence. Axis/units, caption, precision/horizon/candidate count và nguồn artifact phải đọc được ở kích thước paper; trạng thái dàn trang cần được kiểm tra trên PDF thực sau build.

Discussion nên dẫn từ **quan sát → giả thuyết có căn cứ → hệ quả cho thiết kế**. Ví dụ: L8 ranking mạnh nhưng success contrast với END/DINO yếu; L15 source đọc native-crossing tốt hơn predictor trên ít cơ hội; sampling chiếm phần lớn latency; co-design giảm CE nhưng làm source mất ranking. Các quan sát này giải thích bước native-hit/continuation tiếp theo, đồng thời ngăn việc suy rằng token ít, CE thấp hay oracle headroom cao tự động đem lại control tốt.

Conclusion nên giữ thành tựu cụ thể của kiến trúc và preliminary results, cùng giới hạn: privileged training labels/branching simulator, pretrained expert proposal policy, finite source alphabet so với continuous deployment mean, reuse of development roots, backend-sensitive historical proposals, chưa có held-out multi-seed hoặc distinct-goal test. Một greedy immediate oracle không phải upper bound của long-horizon episode success. Giới hạn lợi ích trung gian/query transfer cần nêu thẳng, nhưng không biến thành yêu cầu bỏ endpoint direction.

## 10. Các chỉnh sửa biên tập và reference cần khóa

Abstract nên dành phần lớn nội dung cho câu hỏi target, phương pháp, một hoặc hai preliminary findings có scope, và phần chưa xác nhận. Không mở bằng world-model background dài hoặc đưa native-hit expected improvement thành result. Main paper nên bỏ bảng trống giả sealed400/three seeds; supplement giữ provenance, schedules, numerical repair, partial K64 và co-design failure.

Literature audit đã kiểm tra nguồn sơ cấp; cần sửa citation metadata và cách phân nhóm: GPC là **Generative Predictive Control**; DINO-WM dùng DINOv2 còn V-JEPA 2-AC dùng V-JEPA 2 encoder; TAP/CompACT/Delta-IRIS là các prior gần; V-GPS dùng offline-RL value guidance; DynaGuide là gradient guidance của denoising, không chỉ ranking. DreamSteer/Robot Critics là preprint khi chưa có venue được xác minh; không gán venue tùy ý. Xem các [correction và primary URLs đã kiểm tra](/Users/nhatcuong/code_project/vin-research/trajectory_innovation_20260922/docs/paper_review_20261002/literature_audit.md).

Mục tiêu của lượt viết lại là làm người đọc thấy rõ **CTA đã làm được gì, phát hiện nào có nguồn thật, khác biệt nào còn là hypothesis, và thí nghiệm hữu hạn nào sẽ phân xử claim đó**. Văn phong và rubric có thể làm luận điểm rõ hơn; chúng không thay thế kiểm chứng control và held-out evidence còn thiếu.

## 11. Bản sửa đã thực hiện

Đã viết lại toàn bộ abstract, introduction, related work, problem, method, experiments và discussion/conclusion bằng tiếng Anh; thay bảng projected bằng archived development evidence; bổ sung so sánh literature, paired uncertainty và nguồn provenance. Có ba hình vector dùng source/observations thật: training/deployment boundary, L8 ranking và L15 prediction diagnostic. Phụ lục tách rõ cấu hình, checkpoint selection, kiểm định, công thức và phạm vi qualification.

Main PDF có 8 trang gồm references (bắt đầu trang 7); supplement có 3 trang. Build local với MacTeX thành công; không còn cảnh báo overfull/underfull hoặc citation/reference chưa resolve, mọi trang đã được kiểm tra hình thức. Nội dung đã được review độc lập về evidence và literature. Bản cũ nằm trong `paper_cvpr/draft_v4_before_20261002_review/`; manifest ghi hash source/PDF và 13 artifact metadata nguồn. Việc hoàn thiện bản thảo không đồng nghĩa đã hoàn tất validation thực nghiệm còn thiếu ở các phần trên.
