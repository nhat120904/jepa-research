# Review đề xuất TD / hindsight reader cho CTA — 2026-09-29

## Kết luận

Đáng triển khai một thử nghiệm end-to-end có giới hạn để kiểm tra **giá trị dài hạn học từ dữ liệu có cải thiện chọn action chunk không**. Chưa có cơ sở gọi đây là phương án tốt nhất, thay toàn bộ reader có nhãn, hay lời giải cho arena chứng minh trajectory. Ba câu hỏi cần tách:

1. Value dài hạn có tốt hơn progress hình học ngắn hạn?
2. Với cùng value objective, CTA có tốt hơn DIRECT và ENDPOINT ở quality/compute?
3. Trong trường hợp nào thông tin bên trong segment cần thiết ngoài endpoint/history và cumulative reward/termination?

Đổi reader giải quyết trực tiếp câu 1. Câu 2 và 3 vẫn cần đối chứng. Không cần đòi chứng minh câu 3 trước khi tiếp tục cải thiện pipeline hiện tại.

## Bằng chứng mới trong workspace

Đã kiểm tra `squeue` và `sacct`: PushT 55666_0–7 và aggregate 55676 COMPLETED; OGB v2 55675 vẫn RUNNING lúc review (elapsed 1:48). Không coi số partial của 55675 là kết quả cuối. Không submit job trong review này.

PushT aggregate: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_v2cl_aggregate_55676/summary.json`, 200 development roots 2200–2399, một training seed:

| Arm | Success | Mean native score |
|---|---:|---:|
| P0 | 122/200 | .960874 |
| GEOM8 privileged diagnostic | 160/200 | .970499 |
| CTA4 | 142/200 | .961408 |
| CTAV2 | 140/200 | .959566 |
| ENDV2 | 138/200 | .956092 |
| DIRV2 | 135/200 | .972704 |
| DINOWM | 136/200 | .974226 |

CTAV2−P0: +9 pp, paired CI [+1,+17.5], p=.0444; native-score contrast −.001308, CI [−.01617,+.01410]. CTAV2−CTA4: −1 pp [−8.5,+6.5]; CTAV2−DIRV2: +2.5 pp [−5.5,+10.5]; CTAV2−ENDV2: +1 pp [−7.5,+9.5]. V2 chưa cho gain rõ so với CTA4 hoặc matched learned baselines. Đây vẫn là development evidence; không suy diễn superiority hay equivalence từ các khoảng này.

OGB v1 final 55601 theo ledger: P0 4, CTA 14, FRAME 18, ENDPOINT 21, DIRECT 23 /100. Cải thiện objective chung có lý do thực nghiệm, nhưng CTA còn phải chứng minh lợi ích riêng so với hai baseline mạnh hơn.

Đọc FULL trên future thật ở các state CTAV2 đã thăm đạt geometry-retention .74, CODE .60, predicted CTA .64. Các số này không cùng nghĩa với closed-loop success và không định vị duy nhất reader/codec/WM. Thử decode-to-FULL không cải thiện cũng chỉ loại trừ một cách readout, không loại trừ mọi objective reader.

## Các chỗ cần sửa trong chẩn đoán và literature

- **C2:** `docs/GATE_C2_PROGRESS_READER_PROTOCOL.md:15–36` dùng hai future frames cuối, proprio và goal; target `log(1+g-u)` là temporal gap từ endpoint đến future goal trong cùng trajectory, không phải elapsed time từ đầu demo. Đúng là regression theo behavior không đảm bảo shortest distance. Nhưng thiếu context ban đầu C không tự nó là lỗi: value từ một Markov state không cần C. Chưa có ablation xác nhận nguyên nhân thất bại đó. Confirm r0 +4.5 pp [−4,+13], r1 +2.5 [−5.5,+10.5] không phải bằng chứng mọi reader không nhãn đều thất bại.
- **Đã có TD trong project:** job 55589 train GCIVL rồi bị cancel trong final eval; còn checkpoints 100k–400k tại `/mnt/data/nhatnc129/jepa/ogbench/exp/OGBench/cta_gcivl/sd000_s_55589.0.20260928_104703/`. Chưa xác nhận một CTA TD-reader end-to-end hoàn chỉnh; nhưng nói chưa lần nào thử TD ở cấp project là sai.
- **Temporal-Distance JEPA** dùng temporal-gap regression và heuristic cross-trajectory negatives; không phải Temporal-Difference bootstrap như đề xuất. Paper triển khai temporal cost ở topology tasks và latent L2 của representation đã train temporal objective ở contact tasks. Đây là cảnh báo về transfer của cost, không phải kết luận TD Bellman sẽ thất bại trên PushT. [Paper](https://arxiv.org/html/2607.25337v1).
- **Value-Guided Action Planning with JEPA World Models** đã đưa IQL/value shaping vào JEPA representation và planning geometry. Không claim novelty ở việc dùng value không nhãn cho WM. [Paper](https://arxiv.org/html/2601.00844v1).
- GCIQL/GCIVL, CRL, QRL, TMD không phải một thuật toán TD duy nhất. QRL dùng cấu trúc/quy tắc quasimetric; CRL dùng contrastive learning. Chọn một operator cụ thể, tránh ghép tên để ngầm có bảo đảm shortest path. [QRL](https://proceedings.mlr.press/v202/wang23al.html), [TMD](https://arxiv.org/html/2509.20478v2), [GCIVL implementation](https://github.com/seohongpark/ogbench/blob/master/impls/agents/gcivl.py).

## Target phải được định nghĩa đầy đủ

Định nghĩa hai đại lượng khác nhau:

- `d(x,g)`: continuation cost từ trạng thái/observation history x đến goal, dưới policy/operator đã xác định.
- `D(C,S,g)`: cost từ đầu chunk, khi cố định chunk đầu tiên được S mô tả rồi tiếp tục theo operator trên.

Target `k` hoặc `H+d(endpoint,g)` thuộc đại lượng thứ hai; không phải chỉ remaining distance *sau* segment. Chọn positive cost và minimize, hoặc negative value và maximize nhất quán.

Với deterministic undiscounted first-hitting cost, dạng khái niệm là:

```
y = tau_g                  nếu đạt goal lần đầu trong segment ở tau_g
y = h + d_bar(x_h, g)      nếu chưa đạt goal
```

`h` là số bước thực sự execute, không phải luôn H. Với discounted implementation:

```
y = sum_{j=0}^{h-1} gamma^j c_j(g) + gamma^h m_h(g) d_bar(x_h, g)
```

Phải xác định timing của cost/goal, terminal mask, và value teacher. Không bootstrap một segment-reader bằng cách đưa endpoint vào nó khi kiến trúc đó cần cả S. Plain squared TD dưới behavior sampling không tự cho shortest distance; dùng đúng expectile/Bellman operator của GCIVL/GCIQL nếu muốn implicit policy improvement. Discounted hitting cost không còn đúng đơn vị số bước và không luôn tương đương maximize success dưới finite episode budget.

**Goal achievement không tự có từ hindsight.** Official OGBench GCDataset sinh synthetic success bằng index equality giữa current sample và goal sample; không nhận biết mọi state có cube nằm trong native success radius. Goal relabeling cung cấp positive pairs không cần pose, nhưng goal ảnh bất kỳ không đồng nghĩa task goal arbitrary có semantics đúng. Random goals không chắc unreachable; nhiều frame khác nhau có thể cùng task success. Cần current-goal/self positives hoặc một boundary condition tương đương để anchor value. [Official dataset code](https://github.com/seohongpark/ogbench/blob/master/impls/utils/datasets.py).

Bank sibling cho transition khác nhau từ cùng root, hữu ích để so sánh action consequences. Nhưng continuation target vẫn là ước lượng của teacher, không trở thành ground-truth counterfactual long-horizon return chỉ vì có future thật trong chunk.

Khi relabel goal, termination do original task success không tự động là terminal cho goal mới. Time limit cũng không tự động là goal success. Không biến padded duplicate frames thành genuine transitions, không giả định có suffix sau khi simulator đã dừng. Đặc biệt cần kiểm tra dữ liệu có lưu đủ frames/terminal timing để nhận ra hit ở bước k; bank subsampling không quan sát tất cả bước.

## Vì sao lập luận trajectory hiện chưa đứng vững

Ở fixed h, không có hit và mỗi bước cost bằng nhau, target chỉ khác nhau qua `d_bar(x_h,g)`. **Objective này trở thành endpoint-value ranking.** Nó có thể cải thiện planning rất nhiều mà không cần thông tin nội bộ segment.

Protocol hiện tại còn làm lập luận hit-and-leave yếu hơn:

- `ti_wm/cta_runtime.py:27–43`: PushT dừng sau success và lặp terminal frame cho các sample còn thiếu.
- `scripts/ogbench/ogb_collect.py:36–58`: visual OGB dùng pre-step success timing, thực hiện thêm một action rồi terminate/pad. Có một bước trễ, nhưng không có việc tùy ý chạy hết chunk dài sau success như lập luận đề xuất.

Do đó “native success ở bất kỳ bước nào” không đủ để chứng minh full segment cần thiết. Nếu có early-hit advantage, baseline mạnh phải gồm endpoint + predicted cumulative reward/hit/termination; đây là cấu trúc macro-action value quen thuộc. Nếu lợi ích do velocity không có trong một frame, cần endpoint hai-frame/history control. Một-frame image không Markov là vấn đề partial observability, chưa phải bằng chứng cho toàn trajectory.

Điều này không cấm dùng CTA trên endpoint tasks: ở đó claim hợp lý là compact prediction có quality/compute tốt, robustness hay cross-goal reuse tốt hơn. Nó chỉ ngăn suy diễn trajectory necessity từ một objective endpoint-factorized.

## “Không nhãn” và fairness

`scripts/ogbench/ogb_cta_train_play.py:47–91` dùng cube positions cho moving-frame sampling, hindsight labels và progress; selection cũng đã dùng geometry diagnostics. Chỉ thay target reader mà giữ pose-trained codec/sampling/selection không tạo ra pipeline hoàn toàn không state supervision. Có thể claim hẹp “reader objective không dùng dense task labels” và disclose phần khác; muốn claim toàn pipeline phải kiểm tra/retrain các thành phần liên quan.

Giữ reader registration/cube-distance làm **privileged-supervision baseline**, không gọi là upper bound: nó có thông tin đặc quyền khi train, nhưng không bảo đảm optimal control. Simulator ground truth chỉ ở evaluation/oracle diagnostic là setting khác với dùng nó tạo training targets.

DIRECT học cùng targets là direct chunk critic; chỉ gọi GCIQL khi thật sự có đúng objective/operator của GCIQL. FULL trên future thật là diagnostic; nếu dùng trong deployment nó là privileged, không phải learned planner. FULL-predicted là một baseline riêng.

Teacher GCIVL hiện có dùng IMPALA pixels, CTA dùng DINO/PCA features. Không thể đưa predicted DINO latent trực tiếp vào IMPALA teacher. Một cách hợp lệ là teacher tạo target trên real observations, rồi train compatible value head; tất cả representation arms dùng cùng targets và quy tắc fitting. Cần đánh giá trên predicted inputs của mỗi WM vì value tốt trên real frames chưa bảo đảm tốt trên imagined futures.

## Visual-antmaze và bước tiếp theo

Visual-antmaze là arena tốt để kiểm tra **topology-aware value/navigation transfer**; chưa phải lựa chọn đã được chứng minh tốt nhất cho trajectory abstraction. Giá trị của endpoint có thể mã hóa đường vòng quanh tường. Velocity ambiguity cần history control. Native environment mặc định terminate tại goal, visual variant dùng pre-step success; cùng caveat hit-and-leave. [Official environment](https://github.com/seohongpark/ogbench/blob/master/ogbench/locomaze/maze.py).

Adapter hiện tại là manipulation-specific; action dimension, restoration và policy proposals của ant cần sửa, không chỉ đổi tên env. Shortest maze geodesic bỏ qua gait/stability nên là diagnostic có state đặc quyền, không phải true optimal control oracle. Không nên mở thêm arena chỉ vì TD có vẻ hợp navigation, cũng không biến oracle pilot thành chuỗi gate bắt buộc.

Khuyến nghị iteration kế tiếp trên cube, dùng play đã có:

1. Reuse/check GCIVL checkpoint 400k như initialization/reference; xác định goal convention, discount, terminal treatment và compatible teacher/readout. Không mặc định teacher đã tốt chỉ vì train xong.
2. Một end-to-end run có giới hạn, cùng bank/proposals và data: CTA-TD, ENDPOINT-TD, DIRECT-TD; FULL trên real futures chỉ là diagnostic. Giữ các supervised arms làm control phù hợp, tránh confound đổi data+model+objective cùng lúc.
3. Tích hợp diagnostic ở cùng root: đọc real full future → source code → predicted code, cùng target value, để quyết định sửa teacher/reader, codec, hay WM. Báo closed-loop success, native metric, latency, paired uncertainty; offline TD loss không thay thế control results.
4. Nếu các arms cùng tăng, đó là gain từ value objective. Nếu CTA tăng hơn ENDPOINT/DIRECT ở matched budget, mới có evidence riêng cho abstraction. Sau một implementation chạy đúng, port sang PushT để kiểm tra transfer; không cần sweep nhiều loại reader trước.

Không có cơ sở xác nhận estimate “4 GPU giờ” đã gồm teacher fitting, adaptation, train các arms và closed-loop evaluation. Trước actual submission vẫn phải kiểm tra peer work, account quota và resources theo AGENTS.md.

Research question nên giữ ở mức: **Trong ngân sách dự đoán/planning cố định, code tương lai điều kiện hóa có giữ được thông tin cần để đánh giá nhiều goal tốt hơn direct value và endpoint/full-future prediction không?** TD là công cụ kiểm tra và cải thiện câu hỏi này. “Reader không nhãn” và “nhìn thấy event giữa chunk” là các claim riêng, chỉ thêm khi đã có protocol và bằng chứng tương ứng.
