# Hướng thử learned đầu tiên: dự đoán sự kiện đạt goal ở bước điều khiển

Ngày: 01/10/2026. Source được đọc tại commit `2159e6734ad33930e7d0386d4f6eb5628f432a1b`.

Trạng thái: **hypothesis và thiết kế pilot; chưa có kết quả mới, chưa xác nhận novelty đủ cho CVPR**. Không submit job trong lượt tìm idea này. Tài liệu này không đổi agenda của các job đang chạy.

## Lựa chọn

Nghiên cứu một adapter nhỏ cho latent world-model planner: **giữ dự đoán dynamics ở thời gian thưa, nhưng dự đoán sự kiện đạt image goal ở thời gian của bộ điều khiển**. Tên mô tả tạm thời: *Native-rate Event Prediction for Latent World-Model Planning*.

Điểm cần kiểm chứng là liệu một planner chấm tương lai ở mốc 5/10/15/20/25 bước có bỏ qua hoặc xếp sai các action chunk đã chạm goal tại một bước nằm giữa các mốc. Adapter học từ rollout thực của candidate, dùng latent/context và proposed actions lúc deployment, không dùng future observations hay simulator state lúc deployment.

Đây là một hướng nghiên cứu quanh LeWM/DINO-WM/JEPA, không phải pivot sang video perception. Nó cũng không yêu cầu frozen adapter phải thắng SOTA trước khi được train.

## Bằng chứng thực có và phần chưa biết

1. Current Cube ladder cho thấy successful candidates đã có trong population; trên 33 informative roots, predicted endpoint ranking chọn thành công khoảng .85, true stopped-endpoint ranking khoảng .97. **Đây là oracle selection trong candidate bank, không phải dự báo gain closed-loop 12 điểm %.** Nguồn: [bottleneck ledger](../bottleneck_ladder_20260930/JOB_LEDGER.md).
2. Corrected historical audit 32 roots: predicted latent-L2 chọn 16/32, true same-renderer endpoint chọn 21/32, physical best chọn 25/32. Thay prediction bằng true endpoint loại 4.51 cm regret; encoder cùng terminal-L2 vẫn còn 3.29 cm regret. Không kết luận encoder một mình là lỗi. Nguồn: [corrected audit](../diagnosis/results/ogb_true_endpoint_corrected/TRUE_ENDPOINT_DECISION.md).
3. `Task.execute` dừng ngay khi environment terminate. Ladder sau đó render và encode **trạng thái cuối thực thi**; model score vẫn dùng endpoint của năm action blocks. Các target có thể khác thời điểm. Nguồn: [execution](../cem_stopping_20260929/cemstop/tasks.py), [ladder](../bottleneck_ladder_20260930/scripts/ladder.py).
4. `MinOverTime` hiện lấy minimum trên **block endpoints**, không phải mọi primitive step. Các arm hiện có so `rh1_min` với `rh5_term`, đồng thời đổi execution cadence. **Chưa có đối chứng `rh5_min` cùng lịch thực thi.** Nguồn: [feedback objectives](../bottleneck_ladder_20260930/scripts/feedback_objectives.py).
5. Action-response fine-tuning đã cải thiện loss mà không cải thiện Cube success. Reacher gần trần sau history prefill. Vì vậy không dùng generic response loss hoặc Reacher làm câu chuyện chính.

Điều chưa biết: bao nhiêu ranking errors thực sự do hits nằm giữa các mốc; bao nhiêu do dynamics/cost sai ở mọi mốc; một ordinary goal-success classifier có giải quyết đủ hay không. Terminal-horizon MPC là một lựa chọn thông thường, không tự nó là lỗi correctness.

## Research hypothesis

**Dự đoán endpoint feature ở thời gian thưa có thể đủ cho progress, nhưng không đủ để xếp hạng action chunks theo sự kiện kết thúc task ở thời gian điều khiển. Một event predictor nhẹ, huấn luyện trên imagined contexts và outcome thực, có thể khôi phục phần thông tin này mà không phải chạy một visual dynamics model dày hơn.**

Ví dụ minh họa, chưa phải case đã đo: plan A chạm vùng goal ở bước 3 rồi sẽ đi ra ngoài ở bước 5 nếu cứ tiếp tục; plan B không chạm goal nhưng có latent bước 5 gần ảnh goal hơn. Benchmark terminate-on-success ghi A là thành công. Chấm endpoint hoặc minimum ở bước 5/10/... có thể không phân biệt đúng.

Nếu đa số lỗi là ở các trạng thái chưa bao giờ chạm goal, giả thuyết time-resolution không giải thích được chúng. Không đổi tên generic learned cost thành event method sau khi thấy kết quả.

## Can thiệp tối thiểu

Giữ frozen LeWM encoder/predictor và corrected context. Với một candidate action chunk, native WM sinh coarse imagined history. Một module nhỏ xử lý từng primitive-action prefix trong block, tạo event token; reader nhận token và goal embedding để dự đoán hazard đạt goal tại bước ấy. Reader không cần đọc raw action trực tiếp.

Với bước primitive j:

\[
h_j = P(\tau=j\mid \tau\ge j,C,A_{0:j-1},g),\qquad
P(\tau\le H)=1-\prod_{j=1}^{H}(1-h_j).
\]

Công thức hazard/survival là **kế thừa prior art**, không phải novelty. Module tại bước j chỉ nhận action prefix đã thực thi tới j và coarse predictions sinh từ các block trước j; không nhận coarse endpoint của chính block nếu endpoint ấy đã đọc action suffix sau j. Nếu không giữ điều này, model có thể dự đoán một early hit bằng action chưa xảy ra.

Huấn luyện trên **coarse contexts được native WM dự đoán**, với labels từ rollout thực của cùng actions. Không chỉ train reader trên true future features rồi deploy trên predicted features. Successful traces cho first-hit time; traces chưa hit cho no-hit tới thời điểm quan sát cuối. Failure termination và timeout cần phân biệt, không mặc định đều là thông tin giống nhau.

Pilot đầu dùng event probability để rerank **cùng final candidate pool và final mean** của native CEM. GoalMSE dùng cho progress khi event estimate không phân biệt được candidates; quy tắc fallback phải chốt bằng validation data, không chọn lại trên test. Cả baseline lẫn method đều chọn từ cùng pool, cùng execution cadence. Chỉ đưa event score vào toàn bộ CEM search nếu kết quả rerank cho thấy nó đổi quyết định hữu ích; scoring under optimizer shift phải được kiểm tra khi thay interface này.

Không thêm stopped-latent head lúc đầu: với một terminal goal, state sau hit không còn continuation cần chấm, nên thêm head này chưa có lý do thực dụng.

## Đối chứng quyết định câu chuyện

| Đối chứng | Phân biệt điều gì |
|---|---|
| Corrected native GoalMSE; mean và best candidate trong cùng pool | Gain có chỉ do đổi cách lấy action cuối CEM không? |
| `rh5_min`, running-distance và native schedule | Sửa scoring đơn giản có giải quyết đủ không? |
| Goal-success classifier trên coarse predicted states, cùng data/capacity | Gain có chỉ do học predicate của task thay latent-L2 không? |
| Coarse hazard head và direct context/actions/goal hazard critic | Native-rate prediction và imagined WM context có mang thông tin riêng không? |
| Dense/native-rate future predictor + cùng success reader | Event-only micro prediction có trade-off quality/compute hữu ích không? |

Các head learned phải nhận cùng event labels và branch budget. Dense predictor có supervision thêm feature targets; khác biệt cần ghi rõ, không giả vờ hoàn toàn cùng supervision. Đối chứng quan trọng này không nhất thiết cần train full WM từ đầu: có thể bắt đầu bằng capacity-matched local prefix predictor trên cùng frozen features.

Khi tiến tới paper cần so với baseline mạnh/released gần nhất dưới protocol tương thích, không chỉ LeWM original. `Fast-LeWM` là một preprint collision và baseline về action-prefix prediction cần kiểm tra, không phải một bài top-tier đã xác minh acceptance.

## Pilot một tuần, tích hợp chẩn đoán vào learned run

- Ngày 1–2: replay branch inputs trên **training episodes độc lập với dev/test roots**; lưu first-hit step, executed length, termination reason, hit predicate tại từng primitive step. Trong chính lượt replay đo missed hits tại coarse boundaries và thêm các simple temporal-scoring controls. Không dùng existing dev bank để train rồi báo held-out trên cùng episodes.
- Ngày 3–5: train event adapter, coarse success classifier và direct survival critic; chạy closed-loop pilot cùng candidates/horizon/execution. Một cấu hình nhỏ cho mỗi arm; không làm speculative sweep. Model loading, physics, rendering, encoding và analysis lớn chạy Slurm compute nodes.
- Ngày 6–7: confirmation trên starts/episodes chưa dùng chọn method, ít nhất ba training seeds nếu budget cho phép; báo paired uncertainty theo root/episode, native success, time-to-success và toàn bộ planning latency. Seed lặp ở cùng root không phải independent task sample.

Cube là pilot chính vì có checkpoint và restore đúng. PushT là replication của interface, giữ native success gồm cả pusher pose; không tự đổi thành block-IoU. Reacher là sanity check trên baseline corrected, không dùng baseline h1 yếu để tạo gain. Mở rộng sang một bộ robot-manipulation thứ hai chỉ sau khi xác minh released checkpoint/data và chi phí, không hứa có sẵn trong một tuần.

Đây là thứ tự chạy dự kiến, không phải job đã được submit. Trước mọi GPU submission phải kiểm tra ranking **tháng 10**, tính cả time limit theo rule account; không tái dùng mức quota tháng 9.

## Novelty cần kiếm được bằng chứng

Claim có thể bảo vệ nếu chạy tốt: **dự đoán task events ở native control rate trên coarse JEPA rollouts cải thiện learned closed-loop planning, và đạt chất lượng tương đương hoặc tốt hơn dense rollout với compute thấp hơn.**

Không claim first survival critic, first temporal abstraction, first controller-conditioned WM, hay first action-prefix predictor. Không claim thêm vài heads tạo novelty.

Nếu `rh5_min` đạt cùng gain, kết quả là sửa baseline/scoring. Nếu coarse success classifier đạt cùng gain, câu chuyện là learned goal predicate, chưa có bằng chứng cho native-rate mechanism. Nếu direct critic tốt ngang/better, vai trò WM chưa kiếm được. Nếu chỉ giảm event NLL/offline regret mà success không đổi, đây lại là proxy-only improvement. Nếu event method có gain held-out nhưng không hơn dense predictor về trade-off, vẫn có control result, nhưng claim quality/compute phải thu hẹp.

**Độ tin cậy hiện tại:** scope/implementation fit tốt; có source-level mismatch cụ thể; effect size chưa biết; first-hit novelty đã có prior art mạnh. Đây là hướng đáng thực hiện một learned pilot, chưa phải một hướng đủ chứng cứ để dành toàn bộ budget CVPR.

Xem [prior-art ledger](PRIOR_ART.md) cho claim boundaries và các primary sources.
