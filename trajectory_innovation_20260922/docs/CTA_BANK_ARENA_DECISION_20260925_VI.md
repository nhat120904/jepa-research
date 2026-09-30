# CTA: candidate bank, LIBERO, arena thay thế và đường tới CVPR

Ngày 2026-09-25. Đây là phân tích và đề xuất pilot, không phải kết quả thực
nghiệm mới hay protocol đã được chạy. Không thay đổi Round 2 đang chạy.

## Quyết định đề xuất

1. Giữ PushT làm primary: bank K=8 đã có headroom dương với CI rõ trên
   400 root. Đợi hai nhánh Round 2, rồi đánh giá closed-loop matched.
2. Sửa cách diễn giải LIBERO: L3 phủ định hiệu quả của selector dùng dense
   progress sau 10 bước, không phủ định toàn bộ khả năng chọn trong bank.
   Làm một pilot hữu hạn về continuation value trước khi bỏ runtime hiện có.
3. Arena mới ưu tiên khảo sát: RoboMimic Transport-MH với image Diffusion
   Policy CNN đã phát hành. Square-MH là phương án kiểm tra runtime/efficiency,
   chưa phải lựa chọn tốt để tìm success headroom lớn.
4. Không train lại VLA, không chuyển sang raw-action CEM, không mở nhiều
   arena cùng lúc. Mục tiêu paper cần bằng chứng riêng cho trajectory code,
   thay vì chỉ một cải thiện success trên một goal cố định.

## 1. Bất đẳng thức oracle cần đúng đối tượng

Tại cùng state và cùng bank B, với true value Q cố định:

    Q(s, A_selected) <= max_{A in B} Q(s, A).

Nhưng rollout success của CTA không nhất thiết <= success của controller
greedy theo progress sau H bước. Hai controller đi vào state khác nhau;
progress_H không bằng xác suất thành công cuối episode. Optimal controller
bị giới hạn trong proposal support mới là trần về expected return; controller
ORACLE8 hiện tại không tính được controller tối ưu đó.

Do đó phải tách:

- **Support:** trong bank có action tốt theo value đủ dài hay không?
- **Scoring horizon:** metric sau một chunk có ưu tiên đúng các action đó?
- **Retention/prediction:** code và WM có giữ được thông tin để chọn?

Bank giữ proposal gần phân phối policy nhưng không loại bỏ exploitation:
chọn max trên K cũng có thể khuếch đại sai số ranking. K lớn cần kiểm tra lại
generalization và có control cùng bank/cùng compute.

## 2. PushT chưa cho thấy proposal là bottleneck chính

Nguồn nội bộ: docs/W2_CLOSED_LOOP_RESULT_54455_54474.md.
Trên 400 root: P0 .625, PHYS8 .765, actual-future reader .710.
PHYS8−P0 = +14 pp [8.75,19.25]; reader−P0 = +8.5 pp [2.75,14.25].
Đây là bằng chứng có thể đạt lợi ích bằng cách chọn trong bank hiện tại,
dù không chứng minh mọi root thất bại đều có candidate tốt.

Round 1 source-code bank distinctness .79 so với predicted .29 cũng chống
lại suy luận rằng toàn bộ code trùng chỉ vì action bank trùng: ở nhiều cặp,
encoder có thể phân biệt tương lai thật nhưng WM đã xóa khác biệt đó.
Distinctness không tự bảo đảm khác biệt quan trọng cho task; cần collision
metric trên các cặp có label khác nhau (Round 2 đã thêm).

Gate B từng đo K=8 vs K=32: mean one-decision coverage gain .0074 vs .0105,
nhưng chưa chứng minh tăng K làm closed-loop success tốt hơn. Tăng bốn lần
số candidate chỉ giải quyết được thiếu support khi candidate mới thực sự
chứa tương lai tốt; nó còn làm bài toán ranking khó hơn.

## 3. LIBERO: điều biết và điều chưa biết

Implementation `scripts/libero/l3_oracle.py`, dòng chọn progress sau run_chunk,
và `ti_wm/goal_progress.py` cho thấy:

- Hành động thực thi 10 bước.
- On/In dùng khoảng cách object tới target; joint goals dùng progress của joint.
- Metric không tính xác suất thành công sau khi policy tiếp tục từ state mới.

L3 có P0 .75, ORACLE8 .74, paired −1 pp [−10,+7]. Kết luận hợp lệ:
**scorer privileged này chưa cải thiện control với operating point đã đo.**
Không được đổi câu đó thành “LIBERO-Goal không có candidate headroom”.
Task cần grasp, lift, mở fixture hoặc đi vòng có thể cần action không cải thiện
ngay khoảng cách cuối. Đây là giải thích có cơ sở từ code, chưa được kiểm chứng
trên trace của các failure cụ thể.

L1 có 50/100 initial roots vừa thành công vừa thất bại qua ba policy seeds.
Điều này cho thấy rollout outcome thay đổi theo noise. Nó không chứng minh
một lần chọn chunk sửa được thất bại; cũng không chứng minh năm root thất bại
ở cả ba seeds là bất khả cứu với mọi sample khác. L1 và L3 còn có device/seed
operating points khác nhau, nên không gộp trực tiếp các success rates.

### Pilot đề xuất trên LIBERO hiện có

Không train WM trong pilot. Giữ checkpoint, action cadence, K=8 và task suite.
Dùng development init 0–9 đã được mở; giữ init 10–49 nguyên trạng.

Trước hết đọc lại trace L3 trên CPU compute node: báo cáo theo task, tỷ lệ
bank progress hòa, số lần selector đổi candidate, và khác biệt thắng/thua.
Trace chỉ có progress không đủ chứng minh mọi future state giống nhau.

Sau đó lấy tối đa 20 root (init 0,1 của cả 10 task), tối đa hai anchor định
trước theo thời gian episode; không chọn anchor dựa trên nhánh nào thành công.
Nếu episode đã kết thúc thì ghi rõ anchor không tồn tại, không thay bằng
anchor thuận lợi khác. Từ mỗi anchor:

1. Clone faithfully, chạy 8 candidate chunks.
2. Tiếp tục mỗi branch bằng cùng frozen policy đến terminal, với 4 noise
   streams độc lập. Cùng stream id được dùng giữa siblings để giảm variance.
3. Dùng streams 0–1 chọn candidate; streams 2–3 ước lượng gain so với
   candidate 0. So thêm lựa chọn theo progress 10 bước trên cùng outcomes.
4. Bootstrap theo root, báo cáo uncertainty và win/loss, không chỉ naive
   max-success trên chính samples đã dùng chọn winner.

Giới hạn là 20 × 2 × 8 × 4 = 1,280 continuations trước early termination.
Đo cost trên 2 root trước; chỉ chốt giới hạn CPU/GPU giờ từ timing thật.
Đây không phải “cheap” vô điều kiện: CPU SmolVLA và rendering có thể tốn nhiều.

Pilot chỉ giúp chọn phép thử kế tiếp:

| Quan sát | Hành động có cơ sở |
|---|---|
| Continuation ranking có gain, myopic ranking không có | Ưu tiên value/horizon/temporal reader, giữ policy bank |
| K=8 không có tín hiệu nhưng nested bank lớn hơn có tín hiệu | Proposal coverage đáng thử; mọi scorer phải dùng cùng bank mới |
| Cả hai còn CI rộng | Chưa xác định được nguyên nhân; không gọi arena bị refute |
| Không thấy tín hiệu trong ngân sách pilot và pipeline chậm | Tạm ngừng LIBERO vì chi phí/cơ hội, không claim không có headroom |

Một single-anchor null không phủ định lợi ích của repeated closed-loop selection:
PushT đã có đúng hiện tượng đó. Nếu continuation pilot dương, vẫn cần kiểm tra
selector ở nhiều quyết định trước khi thu thập CTA data lớn.

Continuation evaluation không được lén đưa future actions vào test-time WM.
Nếu reader được train với continuation value thay cov10, phải ghi rõ target
Q^pi của continuation policy cố định. Nếu đổi code để biểu diễn cả continuation
ngẫu nhiên, đó là thay đổi đối tượng dự đoán, cần protocol riêng.

## 4. Arena và checkpoint đã kiểm tra từ nguồn chính thức

Tất cả điểm dưới đây là **điểm ghi trong tên checkpoint của tác giả**, chưa
được reproduce và không phải số đo headroom. Không so chúng trực tiếp với P0
trong repo. Chọn checkpoint theo quy tắc upstream trước khi đọc CTA outcomes.

| Lựa chọn | Artifact công khai kiểm tra được | Nhận định |
|---|---|---|
| RoboMimic Square-MH, image DP CNN | train_0 checkpoint `epoch=0050-test_mean_score=1.000.ckpt`; config và ba training seeds | Dễ hơn Transport về interface, nhưng nguy cơ ceiling rất lớn |
| RoboMimic Transport-MH, image DP CNN | train_0 `epoch=2850-test_mean_score=0.909.ckpt`; config và ba seeds | Arena mới ưu tiên pilot: multi-stage, nhưng vẫn chưa có oracle headroom |
| RoboMimic Tool Hang-PH, image DP CNN | train_0 `epoch=2150-test_mean_score=0.955.ckpt` | Không ưu tiên nếu mục tiêu cần gain success lớn; checkpoint đã gần ceiling |
| LIBERO-Goal, SmolVLA hiện tại | Đã reproduce operating point, clone fidelity đã kiểm tra | Rủi ro setup thấp nhất; sửa phép đo headroom trước |
| RoboCasa365 | Official DP/Openpi/GR00T support; LeRobot có tài liệu SmolVLA RoboCasa | Nhiều goal/stage, nhưng thêm mobile-base interface, assets, rendering, clone và reproduction |
| CALVIN | CALVIN và DiWA có code/data/model assets | Hợp temporal evaluation, nhưng chưa xác minh ở đây một base-policy checkpoint đầy đủ cho bank; không coi là plug-and-play |

Nguồn checkpoint:

- [Square-MH](https://diffusion-policy.cs.columbia.edu/data/experiments/image/square_mh/diffusion_policy_cnn/train_0/checkpoints/)
- [Transport-MH](https://diffusion-policy.cs.columbia.edu/data/experiments/image/transport_mh/diffusion_policy_cnn/train_0/checkpoints/)
- [Tool Hang-PH](https://diffusion-policy.cs.columbia.edu/data/experiments/image/tool_hang_ph/diffusion_policy_cnn/train_0/checkpoints/)

RoboMimic MH có 300 successful demonstrations từ 6 demonstrator có mức thành
thạo khác nhau. Đây là motivation hợp lý cho variation, không bảo đảm policy
sau training giữ multimodality hữu ích. Dataset dùng khởi tạo/reproduction;
CTA vẫn cần sibling branches từ cùng state, không tự có counterfactual tuples
chỉ bằng tải demonstrations. [Dataset chính thức](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html).

Transport config sử dụng 4 camera streams, action biểu diễn 20 chiều (hai
robot, rotation 6D), episode limit 700; Square có hai cameras và 10 chiều.
Cả hai dùng history 2, prediction horizon 16, action horizon 8.
Do đó Transport hợp path hơn nhưng tốn hơn về encoding/rendering/collection.
[Transport config](https://diffusion-policy.cs.columbia.edu/data/experiments/image/transport_mh/diffusion_policy_cnn/config.yaml),
[Square config](https://diffusion-policy.cs.columbia.edu/data/experiments/image/square_mh/diffusion_policy_cnn/config.yaml).

Rủi ro setup cụ thể: original DP pin robomimic .2 và một robosuite commit dựa
trên mujoco-py, trong khi docs dataset mới hỗ trợ robosuite 1.5.1. Không ghép
checkpoint cũ vào physics/action interface mới rồi diễn giải reproduction kém
là headroom. Legacy runtime cần isolated env và reset/clone checks. Upstream
runner còn ghi chú sửa bug aggregation số eval seeds; dùng code đã sửa và
fresh seeds, không cố khớp một filename score bằng cách khôi phục bug.
[DP environment](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/conda_environment.yaml),
[DP runner](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/env_runner/robomimic_image_runner.py).

Các hướng dự phòng:
[RoboCasa policy support](https://robocasa.ai/docs/build/html/benchmarking/policy_learning_algorithms.html),
[LeRobot RoboCasa](https://github.com/huggingface/lerobot/blob/main/docs/source/robocasa.mdx),
[DiWA assets](https://diwa.cs.uni-freiburg.de/).
Không tự launch chúng chỉ vì benchmark rộng hoặc policy success thấp; policy
quá yếu có thể không chứa hành vi thành công trong proposal support.

## 5. Khi nào thay bank

Thứ tự hợp lý: giữ native noise sampling, đo nested K={1,8,16/32}, rồi mới
thử mixture từ các policy training seeds tốt và độc lập nếu đã có checkpoints.
K tăng phải báo cáo sampling + encoding + scoring + total latency. Thay bank
cần dữ liệu branch tương ứng; không chỉ đổi bank ở test làm WM bị distribution shift.

Nếu mixture bank giúp, so CTA với DIRECT, medoid/diversity control và một
unranked mixture baseline trên đúng cùng bank, cũng như policy đơn mạnh nhất.
Không credit toàn bộ mixture gain cho trajectory abstraction.
Không chọn checkpoint yếu hơn, giảm denoising steps hay tăng temperature chỉ
để tạo P0 thấp mà không báo cáo control tương ứng.

## 6. Triển vọng Round 2

Snapshot squeue/sacct lúc 18:27 UTC: checks 54932 COMPLETED 0:0;
54933_0 (lambda=0) RUNNING; 54933_1 pending; comparison 54934 pending.
Log kiểm tra xác nhận 30 + 16 unit tests passed và tiny training smoke.
Không có outcome của nhánh co-design tại thời điểm snapshot.

Không có cơ sở cho xác suất kiểu “70% thành công”. Phân biệt ba mức:

- **Cải thiện offline:** hợp lý để thử, vì tier-2 giữ nhiều tín hiệu hơn tier-3.
- **Vượt lambda=0 và DIRECT với CI rõ ở closed loop:** chưa có bằng chứng mới;
  Round 1 chỉ +4 pp vs P0 và CI chứa zero.
- **Round 2 tự tạo ra một paper CVPR đủ mạnh:** không nên đặt kế hoạch vào đó.
  Nó chưa thay đổi proposal support, horizon, task diversity hay các ablation
  cần để chứng minh abstraction.

Co-design còn có thể làm source target đổi nhanh, mất thông tin để dễ dự đoán,
hoặc chỉ giúp expected-code mà greedy vẫn collision. Các diagnostics đã khóa
phải được đọc cùng matched control; không rút kết luận từ minibatch train loss
hay ladder trên 600 decision trước khi final alignment/eval hoàn tất.

## 7. Phần research cần làm để CTA có lý do tồn tại

Hiện source đọc cả path nhưng reader loss là cov8 của một task. Điều đó chưa
buộc method phải giữ thông tin thứ tự hay event trong trajectory. Có thể một
endpoint encoder hoặc direct scalar scorer làm tốt tương đương.

Phép thử quan trọng nhất cho ý tưởng: một code dùng lại cho query về endpoint,
max/progress trong đoạn, thời điểm event, hoặc thứ tự event hợp task, với
held-out query/goal được tách trước training. Chỉ thêm query khi nhãn/định nghĩa
được xác định rõ và query có variation thật ở horizon đang dùng.

Không ép nâng horizon bằng cách thực thi open-loop lâu hơn làm policy yếu đi.
Prediction horizon và execution cadence là hai biến khác nhau. Longer-horizon
code cần action inputs sẵn có trước execution hoặc cần định nghĩa distribution
dưới continuation policy; không dùng future observations để sinh test actions.

Các đối chứng tối thiểu liên quan tới claim:

- cached DIRECT cùng input access và budget;
- endpoint-only source code;
- conditional per-frame code cùng tổng bit budget;
- codec có/không context ở reader;
- co-design vs equal-update continuation;
- latency toàn stack, đặc biệt khi tăng số query Q hoặc số candidate K.

Nếu không thắng success nhưng có efficiency frontier tốt, đó là một claim
khác phải khóa trước fresh confirmation. Không chuyển headline sau khi đã đọc
test. CTA phải cho lợi ích đo được; tên kiến trúc và positive dev number không
thay được các control này.

## 8. Lịch đề xuất nếu nhắm CVPR 2027

Website chính thức: registration 10/11/2026, submission 16/11/2026,
supplement 23/11/2026, AoE. Từ 25/9 còn khoảng 52 ngày tới submission.
[CVPR dates](https://cvpr.thecvf.com/Conferences/2027/Dates).

Mốc dưới đây là đề xuất phụ thuộc budget người dùng, không phải cam kết runtime:

- 25/9–2/10: kết thúc Round 2; đánh giá matched closed loop PushT; pilot
  LIBERO đúng loại headroom; Transport feasibility chỉ setup/reproduction,
  không launch training sweep.
- 3–12/10: chốt arena thứ hai từ qualification, chốt temporal query và
  Round 3 nếu còn cần. Nếu chưa có predicted-code signal vượt control,
  giảm scope thay vì tiếp tục mở thêm benchmark.
- 13–26/10: chạy method và decisive ablations với ba seeds; khóa claim,
  hyperparameters và power/budget của confirmation trước khi mở test.
- 27/10–5/11: confirmation, latency, failure analysis, figures và draft.
- 6–16/11: ổn định kết quả và hoàn thiện paper, giữ buffer cho failures.

Đầu ra của lượt nghiên cứu này là quyết định và nguồn kiểm chứng. Không có
job arena mới, dataset download lớn, rollout hay policy training được submit.
