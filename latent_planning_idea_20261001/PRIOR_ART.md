# Prior art và claim boundaries — 01/10/2026

Accepted main-track papers dùng để học cách đặt vấn đề và thiết kế phép so; preprints/workshops chỉ dùng kiểm tra collision và baseline gần nhất. Bounded search không chứng nhận không còn prior art.

## Main-track primary sources

| Paper | Điều đã có | Hệ quả cho proposal |
|---|---|---|
| [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | Frozen pretrained visual features làm target cho learned dynamics và zero-shot goal-image planning. | Giữ functioning native WM/task interface; không claim latent planning hoặc frozen visual target là mới. |
| [SVL, ICML 2026](https://proceedings.mlr.press/v306/tiofack26a.html); [full text](https://arxiv.org/html/2604.17551) | Goal-conditioned first-hit distributions, hazard likelihood, censored traces và value estimators; visual và state-based offline GCRL. | Survival formulation và first-hit head đã có. Proposed distinction là candidate-conditioned, native-rate event evaluation của imagined JEPA rollouts; còn phải chứng minh hơn direct hazard critic. |
| [DWSL, ICML 2023](https://proceedings.mlr.press/v202/hejna23a.html) | Supervised distributions của time distance và policy learning từ offline interaction, kể cả image inputs. | Không claim distributional temporal targets hoặc không dùng Bellman backup là mới. |
| [THICK, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/13b45b44e26c353c64cba9529bf4724f-Abstract-Conference.html) | Hierarchical WM dùng sparse latent context changes để học temporal abstraction. | Không claim event abstraction/variable-time WM nói chung là mới. Proposed head không phải một hierarchy mới. |
| [Puppeteer, ICLR 2025](https://www.nicklashansen.com/rlpuppeteer/); [method](https://arxiv.org/html/2405.18418v3) | Frozen low-level WM/MPC tracker; high-level WM học actual command outcomes, có termination prediction. | Loại claim “dự đoán achieved state của frozen MPC” và generic termination head. |
| [Jumpy/CompPlan, ICML 2026](https://proceedings.mlr.press/v306/farebrother26a.html); [method](https://arxiv.org/html/2602.19634) | Policy/time-scale-conditioned occupancy models và composition của continuous goal-conditioned policies. | Coarse controller-outcome modeling không mới chỉ vì q liên tục hay executor frozen. |
| [Average-Reward Learning and Planning with Options, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/c058f544c737782deacefa532d9add4c-Paper.pdf) | Option termination-state model, duration và cumulative reward. | Thêm stopped-latent/duration heads không tự tạo method contribution. |

Bài học cụ thể từ SVL: một cách học value tương đối đơn giản vẫn cần phép so giữ actor cố định để cô lập estimator. Áp dụng ở đây bằng same-candidate/same-execution planner và direct survival critic. Bài học từ DINO-WM: representation phải được dùng trong một planning interface hoạt động; không lấy offline feature loss làm đích kết luận.

## Direct recent collisions, không gán acceptance main track

| Source | Collision |
|---|---|
| [Fast-LeWM, preprint, v2 28/09/2026](https://arxiv.org/html/2606.26217); [official code](https://github.com/Yuntian-Gao/Fast-LeWorldModel) | Parallel action-prefix latent prediction đã có. Native-rate event proposal không được lấy prefix encoder/parallel prediction làm claim riêng. Cần kiểm tra checkpoint và strong planning baseline thực tế. |
| [D-JEPA, September preprint](https://arxiv.org/abs/2609.24749) | Executed-outcome supervision để sửa goal-relative predictive features/ranking. Generic learned latent correction không còn là claim sạch. |
| [Hi-LeWM, workshop/preprint](https://arxiv.org/html/2607.12547) | Macro action model, latent waypoint handoff, support constraints và execution timing. | 
| [ACID, July preprint](https://arxiv.org/abs/2607.02403) | Inverse action reconstruction residual làm planning cost; không đề xuất inverse consistency như novelty. |
| [Closing the Train-Test Gap, preprint](https://arxiv.org/abs/2512.09929) | Planner-generated correction/adversarial training; không claim dùng candidate rollout data nói chung là mới. |
| [LeWMRO, workshop implementation](https://github.com/24GUNV/LeWMRO) | Terminal/prefix/running objectives và waypoint controls; simple scoring controls cần có. |

## Native-task sources

- [Stable-worldmodel PushT specification](https://github.com/galilai-group/stable-worldmodel/blob/main/docs/envs/pusht.md): success gồm pusher position, block position và orientation. Không loại pusher pose như nuisance mà giữ nguyên tên benchmark.
- [Native objective source](https://raw.githubusercontent.com/galilai-group/stable-worldmodel/main/stable_worldmodel/planning/objective.py): đọc cùng version local trước implementation.
- Local corrected execution: `cem_stopping_20260929/cemstop/tasks.py` dừng theo native termination; `bottleneck_ladder_20260930/scripts/ladder.py` encode stopped endpoint.
- Local temporal objective: `bottleneck_ladder_20260930/scripts/feedback_objectives.py` có `rh1_min`, thiếu matched `rh5_min`.

## Vì sao chưa chọn các hướng khác

Generic reachability/value heads, inverse cycles, coarse controller models, online residual correction và action-prefix acceleration đều là vùng hợp lệ nhưng claim rộng đã có. Chúng chỉ nên được chọn khi một mechanism hẹp và phép so cụ thể tạo thêm giá trị. Trong lượt này, nguồn repo cho native-rate event hypothesis rõ hơn các alternative trên; chưa có evidence rằng nó lớn hơn về actual success.

Không dùng tìm kiếm prior art như lý do loại cả world-model planning hoặc survival learning. Cũng không dùng việc chưa tìm thấy exact combination để khẳng định combination đủ mới.
