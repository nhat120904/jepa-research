**Review độc lập về hiệu quả và tính khả thi của CTA — 27/09/2026**

**Nhận định:** CTA có cơ sở kỹ thuật và đáng tiếp tục trên PushT. Kết quả hiện tại chứng minh một pipeline học được đã hoạt động và có tín hiệu ranking hữu ích. Chưa chứng minh được cải thiện closed-loop đáng tin cậy, lợi thế chất lượng/compute, hay tính mới đủ mạnh cho CVPR. Bản tóm tắt được cung cấp đánh giá quá mạnh ở các câu “kỹ thuật đã ổn”, “128 bit ngang tương lai thật”, và “chỉ còn thiếu arena/headroom”.

Review này đọc thiết kế, protocol, mã nguồn, snapshot Round 4, các JSON tổng hợp đã có và tài liệu nghiên cứu gốc. Không chạy lại simulator/model, không mở sealed roots, không submit thí nghiệm mới. Các CI bên dưới là CI trong artifact gốc, không phải phân tích lại dữ liệu thô. Các đề xuất sau đây là khuyến nghị nghiên cứu, không phải protocol đã được phê duyệt hay sửa đổi.

**1. Bằng chứng hiện có đáng tin đến đâu?**

Nguồn chính là [summary Round 4](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_plus_r4_agg_55170/summary.json), phạm vi 100 development roots, một seed huấn luyện, runner ghép cặp nội bộ.

| Arm | Success | Native normalized score trung bình |
|---|---:|---:|
| P0 | 64/100 | 0.9196 |
| PHYS8: chọn theo coverage thật cuối chunk | 64/100 | 0.9052 |
| GEOM8: chọn theo geometry thật cuối chunk | 73/100 | 0.9352 |
| CTA4 | 68/100 | 0.9377 |
| NLL4 | 64/100 | 0.9128 |
| DIRECT4 | 60/100 | 0.9338 |
| CTA4S: không thêm bank nhiễu trong training | 68/100 | 0.9343 |

- CTA4−P0: success +4 điểm phần trăm, CI [−7,+15], McNemar p=0.608. Native score +0.0180, CI [−0.0097,+0.0505].
- CTA4−DIRECT4: success +8 điểm, CI [−3,+20], p=0.243. Native score chỉ +0.00385, CI [−0.0202,+0.0330].
- CTA4−NLL4: success +4 điểm, CI [−7,+15].
- CTA4−CTA4S: success 0 điểm, CI [−10,+10]. Chưa chứng minh bank nhiễu tạo thêm lợi ích closed-loop.

Không có bằng chứng đủ chắc cho superiority của CTA4 trong các đối chiếu chính. Cũng không có bằng chứng rằng CTA vô hiệu. Điểm ước lượng tích cực và độ bất định lớn cùng tồn tại. Protocol Round 3 chuyển primary sang normalized native score; Round 4 tiếp tục công thức đó. Vì vậy không nên chỉ chọn success, nơi chênh lệch với DIRECT trông lớn hơn, làm câu chuyện duy nhất sau khi thấy số.

Điểm tích cực có căn cứ là ranking geometry vẫn hữu ích trên trạng thái planner tự đi tới, thay vì chỉ trên dữ liệu P0. Training report cũng cho thấy dev NLL của nhánh NLL4 giảm từ khoảng 3.44 ở update 1000 xuống 3.20 nats/token ở update 8000; hiện tượng dev NLL tăng của Round 3 không lặp lại theo cùng chiều ở nhánh này. Đó là tiến bộ kỹ thuật đáng giữ, dù chưa đủ xác định nguyên nhân cải thiện hoặc bảo đảm success ngoài tập development.

Các root này đã phục vụ phát triển qua nhiều round. Chúng chưa phải xác nhận độc lập sau lựa chọn mô hình. Ba training seeds giúp đo biến thiên do huấn luyện; không biến cùng một tập root thành ba tập môi trường độc lập.

GEOM8 73% là kết quả của một selector tham lam dùng geometry cuối chunk. Nó **không phải trần tối ưu** của planner hay toàn bộ candidate bank trong nhiều bước. Khoảng +9 điểm là headroom đo được của selector cụ thể, và chính khoảng đó còn có CI [−1,+19]. PHYS8 chỉ bằng P0 cho thấy tối ưu nhãn ngắn hạn hoàn hảo vẫn có thể không tăng success cả episode.

**2. “128 bit” chưa mô tả CTA4 đang triển khai.**

Source codec lượng tử hóa 16 token, mỗi token có 8×8×4=256 giá trị, nên chỉ số source code có thể được đóng gói trong 128 bit. Tuy nhiên [ParallelFSQWM.forward](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_plus_r4closed_55169/shard_0/code/ti_wm/cta_parallel.py:44) trả kỳ vọng của từng tọa độ. [Runtime](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_plus_r4closed_55169/shard_0/code/ti_wm/cta_batch.py:94) đưa trực tiếp các kỳ vọng này vào reader.

Đó là 16×3=48 scalar liên tục: tensor hiện tại FP32 chiếm 192 byte, tức 1536 bit; FP16 giả định là 96 byte, tức 768 bit. Các số này chỉ tính đầu vào biểu diễn của reader, chưa tính logits, embedding, context, trọng số và workspace. Giới hạn thông tin của source code rời rạc không tự áp dụng cho tensor kỳ vọng liên tục.

Hơn nữa, loss consistency/ranking truyền gradient qua reader vào WM. Các tọa độ liên tục có thể được điều chỉnh để chấm điểm tốt, kể cả ở giữa những điểm của lưới FSQ. Đây có thể là thiết kế hiệu quả; nhưng thành công của nó không chứng minh planner triển khai qua một kênh rời rạc 128 bit. Với reader phi tuyến, đọc kỳ vọng code cũng không bằng kỳ vọng của kết quả đọc code.

Cách diễn đạt đúng hiện nay: “predictor gọn 48 chiều, học với mục tiêu source code FSQ và loss quyết định”. Muốn giữ claim 128 bit phải đánh giá deployment với hard indices/quantization thật và báo kết quả riêng. Nên có một predictor liên tục cùng kích thước làm control để biết lượng tử hóa đóng góp gì.

**3. “Ngang đọc tương lai thật” chỉ đúng với một proxy và một reader cụ thể.**

Trên các trạng thái do CTA4 đi tới, artifact Round 4 cho biết:

| Nhãn dùng đánh giá retention | CTA4 | FULL | DIRECT4 |
|---|---:|---:|---:|
| Geometry | 0.6211 | 0.5811 | 0.4372 |
| Native coverage | 0.5500 | 0.7310 | 0.3023 |

Trên trạng thái P0, tương ứng geometry là 0.6720/0.6395/0.5196 và coverage là 0.6272/0.7423/0.4242.

Retention là tỷ số tổng gain nhãn ngắn hạn của candidate được chọn so với candidate 0, chia tổng gain của candidate tốt nhất theo nhãn đó. Nó không đo tỷ lệ thông tin vật lý giữ lại và không phải tỷ lệ episode thành công.

FULL ở đây đọc **endpoint thật và future proprio**, không đọc toàn bộ trajectory. Đây là một reader học được, có sai số và kiến trúc riêng; không phải oracle Bayes hay upper bound thông tin. CTA4 ngang FULL về geometry cho thấy một bottleneck dự đoán được có thể đủ cho reader/proxy này. Nó không chứng minh WM đã hết dư địa, bảo toàn toàn bộ tương lai, hoặc ngang một WM mạnh trên nhiều task.

Khoảng cách coverage là dấu hiệu còn cần giải thích giữa surrogate geometry, reader, biểu diễn dự đoán và long-horizon control. Không nên quy toàn bộ vấn đề còn lại cho thiếu root hoặc thiếu headroom. Tách lỗi predictor khỏi lỗi source/readout cần so sánh source code thật và predicted code qua cùng reader trên cùng bank; ngay cả phép so sánh đó vẫn chỉ xác định lỗi đối với họ truy vấn được đo.

**4. Trajectory đã có trong kiến trúc; đóng góp của nó chưa được chứng minh.**

[FutureTokens/SourceEncoder](/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922/ti_wm/cta.py:97) đã đọc frame ở bước 2,4,6,8, có vị trí thời gian; decoder cũng reconstruct các frame trung gian. Do đó nói “design chưa bao giờ bật phần trajectory” là không chính xác về implementation.

Điều đúng là nhãn quyết định hiện tại dùng endpoint geometry, nên chưa buộc reader tận dụng sự kiện trung gian và chưa tách được lợi ích đó khỏi endpoint compression. Endpoint task vẫn hợp lệ để kiểm tra CTA như một planner gọn. Không cần bỏ PushT hoặc bắt buộc pivot temporal trước khi cải thiện end-to-end.

Chỉ đổi nhãn thành tổng reward/cost của chunk chưa đủ làm bằng chứng trajectory abstraction. Với một objective cố định, DIRECT(C,A,q) có thể học ngay tổng cost. Nếu chỉ có hai số “progress” và “hazard cost”, một predictor hai đầu ra là baseline mạnh, rẻ và cần thiết.

Một phép thử trajectory rõ hơn có các đoạn tương lai với endpoint tương tự nhưng sự kiện trung gian khác nhau, sự khác biệt ảnh hưởng metric task thật và CTA cải thiện quyết định so với endpoint control. Nếu claim thêm thứ tự thời gian thì cần objective thật sự phụ thuộc thứ tự; tổng cost không chứng minh khả năng này. Phép kiểm tra cặp tương lai chỉ nên là diagnostic hỗ trợ kết quả trên phân phối task đầy đủ.

**5. Tính mới có cửa, nhưng không nằm ở từng thành phần riêng.**

| Công trình gốc | Điều đã có và hệ quả cho CTA |
|---|---|
| [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | Dự đoán latent thị giác có điều kiện action để planning, không cần pixel reconstruction. CTA phải hơn một baseline latent phù hợp, không chỉ hơn sinh video nặng. |
| [CompACT, CVPR 2026](https://arxiv.org/abs/2603.05438) | Nén mỗi observation xuống rất ít token cho world-model planning. “Ít token, planning rẻ” đã là prior art. |
| [Δ-IRIS, ICML 2024](https://arxiv.org/abs/2406.19320) | Tokenization theo context và dự đoán delta. “Chỉ mã hóa phần mới so với quá khứ” không phải đóng góp độc lập mới. |
| [GCQ, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/f18d5002b0c1e18d371805ea363d430b-Abstract-Conference.html) | Nén observation–action sequences trong không gian và thời gian. Nén segment tự nó chưa đủ phân biệt. |
| [RaMP, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/b048dd19ba6d85b9066aa93b4de9ad4a-Paper-Conference.pdf) | Dự đoán cumulative features theo action sequence để chấm điểm nhiều reward và planning. Đổi sang nhãn tổng cost đặt CTA gần đối thủ này hơn. |
| [Jumpy World Models, 2026](https://arxiv.org/abs/2602.19634) | Dự đoán multi-step occupancy ở nhiều timescale cho policy composition. “Bỏ rollout từng bước” cũng không phải mới riêng. |

Theo đánh giá của tôi, đóng góp có thể bảo vệ là: **học một biểu diễn segment có điều kiện theo quan sát, có thể dự đoán từ action, giữ thông tin cần cho một họ truy vấn quyết định, và đạt trade-off chất lượng/chi phí tốt hơn các baseline phù hợp**. Cần bằng chứng cho tổ hợp đó, không chỉ một sơ đồ ghép các module có sẵn.

Predict-once/query-many không độc quyền của CTA. Một direct model có thể tách thành z=f(C,A) và score=h(z,q), cache z và chỉ chạy head cho từng query. Paper draft hiện mô tả direct luôn phải chạy toàn mạng lại cho mỗi goal là giả định về baseline implementation, không phải hạn chế nguyên lý. Cần cached/factorized DIRECT làm đối thủ.

Nhiều ảnh của cùng goal cố định PushT chưa chứng minh chuyển sang goal/task mới. Nếu giữ claim query reuse/generalization, cần truy vấn khác nhau về ngữ nghĩa hoặc objective và tập giữ lại phù hợp. Reader hiện dùng Transformer trên cả C, code và goal cho mỗi query; vì thế ít code token không bảo đảm chi phí reader nhỏ.

Co-design codec cho predictability cũng chưa là kết quả thành công: thử nghiệm lambda=.1 trước đó giảm source retention. Round 4 giữ source cố định và thay predictor/loss/data. Không nên trình bày thành công hiện nay như bằng chứng cho co-design source codec.

**6. Hiệu quả tính toán chưa được đo đúng để kết luận.**

Runner Round 4 mô phỏng mọi candidate và chấm chéo nhiều scorer cho từng arm. Trường `seconds`/`score` trong closed_report là chi phí của diagnostic runner, không phải latency riêng của CTA4 hay DIRECT4. Nó hữu ích cho ngân sách chạy thí nghiệm; không dùng trực tiếp để claim speedup deployment.

Cần đo tổng thời gian thực sự dùng để chọn action: encode history/goal, sinh proposal, WM, reader và selection; cache được tính đúng, cùng phần cứng, precision, K, horizon và query count. Báo thêm component timing, median/p95 và memory. Nếu proposal policy chiếm phần lớn thời gian thì giảm mạnh WM có thể chỉ giảm ít tổng latency.

Đối chứng per-frame không nhất thiết phải autoregressive chậm; compact per-frame và parallel multi-step predictor cũng hợp lệ. So CTA với DINO-WM chỉ khác mục tiêu scoring hoặc supervision sẽ khó quy kết speed/quality cho abstraction.

**7. Tăng lên 400 root là hữu ích, nhưng không bảo đảm thấy +4 điểm.**

CTA4 thắng riêng P0 ở 19 root và thua riêng ở 15 root: tỷ lệ bất đồng q=0.34, chênh d=0.04. Xấp xỉ plug-in cho sai số chuẩn chênh lệch ghép cặp là sqrt((q−d²)/n). Nếu tỷ lệ này giữ nguyên, với n=400, nửa độ rộng CI 95% vẫn khoảng 5.7 điểm phần trăm. Chênh +4 điểm vẫn có thể có CI qua 0.

Xấp xỉ power 80%, kiểm định hai phía mức 5%, cho hiệu ứng thật +4 điểm dẫn đến cỡ khoảng 1,660 root độc lập. Đây là tính toán minh họa dựa trên pilot nhiễu, không phải đề nghị submit 1,660 root hay khẳng định số chính xác. Nên xác định hiệu ứng tối thiểu có ý nghĩa, discordance, độ chính xác và ngân sách trước confirm. Nếu claim “ngang chất lượng nhưng rẻ hơn”, cần biên non-inferiority xác định trước; p>0.05 không chứng minh tương đương.

**8. C2 nào hợp lý hơn?**

Safety-Gymnasium có lý do task rõ hơn hai ví dụ ManiSkill được đề xuất: cost tích lũy theo bước có thể ghi nhận việc đi vào vùng nguy hiểm rồi rời khỏi vùng. API tách reward và cost, có môi trường vision. Tuy vậy cần xác định chính xác biến thể, input và metric; tên “Safe Vision” hoặc khả năng render không tự bảo đảm planner đang dùng image-only. [Tài liệu chính thức](https://safety-gymnasium.readthedocs.io/en/latest/introduction/basic_usage.html), [ví dụ BuildingGoal với risk-area cost](https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_vision/building_goal.html).

[SafeDreamer](https://arxiv.org/abs/2307.07176) là baseline liên quan vì học planning có reward/cost trong Safety-Gymnasium, gồm cả vision input. Nhưng đây là một hệ RL với cách thu dữ liệu và học policy riêng. Cần một comparison mechanism dùng cùng dataset/candidate bank; số của toàn bộ SafeDreamer có thể báo riêng với khác biệt được công khai.

Đổi reward thành cost cũng tạo nguy cơ planner đứng yên để giảm cost. Báo đồng thời task completion/return và violation/cost, tốt nhất so cost tại mức hoàn thành tương đương. Với một cost cố định, baseline dự đoán trực tiếp chunk reward/cost là bắt buộc về mặt thuyết phục khoa học.

RollBall chưa phải lựa chọn thuyết phục để chứng minh path dependence: [success chính thức](https://maniskill.readthedocs.io/en/latest/api/mani_skill/envs/tasks/tabletop/roll_ball/index.html) kiểm tra vị trí xy của bóng trong bán kính goal. Vận tốc quan trọng cho dự đoán không đồng nghĩa objective phụ thuộc toàn bộ đường đi; endpoint latent có thể chứa vận tốc. Với PegInsertion cũng phải kiểm tra native objective có thực sự đánh giá lịch sử va chạm hay chỉ pose thành công.

Tôi nghiêng Safety-Gymnasium **nếu bổ sung C2**, nhưng giữ PushT làm trục chính. Không có kết quả timing/policy tại workspace này chứng minh việc dựng C2, train PPO và reproduce SafeDreamer sẽ rẻ trong 100–150 giờ MIG.

**9. Lượt phát triển tiếp theo nên trả lời đúng câu hỏi nào?**

Một lượt end-to-end tập trung trên PushT có giá trị hơn việc đổi arena trước khi có comparison sạch:

1. Giữ CTA4 làm mốc; trình bày đúng continuous deployment và supervision. Chạy hard-code variant nếu cần bảo vệ claim bit-budget.
2. So với endpoint codec/predictor cùng kích thước biểu diễn và mức tối ưu, cached DIRECT, cùng một baseline latent theo thời gian đủ mạnh. Khi kiểm tra representation, giữ họ reader/capacity/loss phù hợp; reader weight của hai latent khác nghĩa không thể mặc nhiên dùng chung mà không alignment/training.
3. Control factorized bottleneck cùng kích thước nhưng học task trực tiếp, và distillation/supervision tương ứng khi cần, để tách lợi ích future-code learning khỏi regularization, teacher và model size. DIRECT hiện chỉ nhận ranking, trong khi CTA nhận thêm NLL và consistency: đó là khác biệt thực của method, nhưng chưa cô lập tác dụng của compression.
4. Ghép đánh giá success/native score với deployment latency đúng. Giữ K=8 làm anchor; nếu thêm K=16 với proposal nhiễu, mọi arm dùng cùng bank và báo riêng. Bank lớn có thể tăng lựa chọn tốt lẫn tăng cơ hội chọn sai do model error.
5. Freeze cấu hình, metric, số root và cách gộp seed trước sealed evaluation. Development hiện tại chưa hỗ trợ một claim tổng quát.

Các control được tích hợp trong lượt chạy và dùng để debug; không biến thành chuỗi gate trước implementation. Ablation conditional context, semantic queries và path information chỉ nên mở rộng theo claim cuối cùng, tránh một sweep lớn không có quyết định cụ thể.

Nếu CTA liên tục ngang baseline mạnh nhưng rẻ hơn thực sự, một paper về decision-relevant compression vẫn có cơ sở dù chưa có lợi thế whole-path. Nếu cached DIRECT cũng đạt cùng chất lượng/chi phí thì lý do dùng future code yếu đi; cần xem dữ liệu ít, query transfer hoặc robustness có tạo khác biệt thật không, không tự động bịa thêm claim mới.

**10. Lịch, job state và mức khả thi thực tế.**

Đã kiểm tra cả `squeue` và `sacct` trong lần review này: queue của user trống; 55149, 55169_0–3, 55170 và 55276 COMPLETED. **55278 FAILED sau 30 giây**, exit 1:0.

[Log DINO-WM 55278](/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/dinowm_ti_dinowm_55278.out) ghi `numpy.dtype size changed` và cuối cùng `BrokenPipeError`. Đây là lỗi runtime/dependency, chưa phải kết quả về chất lượng DINO-WM hay CTA. Sửa lỗi đó nằm ngoài yêu cầu review hiện tại.

Một lần train Round 4 bốn mạng mất khoảng 1 giờ 43 phút MIG, và bốn shard closed-loop 100 root tổng khoảng 1 giờ 56 phút MIG. Điều này cho thấy tiếp tục PushT trong phạm vi tập trung có tính khả thi. Nó không dự toán được workload của baseline khác hoặc task mới.

Deadline chính thức là đăng ký paper 10/11/2026 và nộp 16/11/2026, AoE: [CVPR 2027 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates). Kế hoạch bảy tuần có thể dùng để tổ chức một dự án thu hẹp, nhưng cam kết đồng thời C1, arena C2 mới, baseline mới, ba seed và mọi ablation là lạc quan khi baseline đầu tiên chưa chạy thành công.

Đánh giá tách ba mục tiêu:

- **Xây planner học được hoạt động:** khả thi, đã có bằng chứng thực thi và tín hiệu học.
- **Có lợi ích đáng tin trên PushT:** có triển vọng, chưa xác nhận; cần làm rõ surrogate, baseline và chi phí thực.
- **Paper CVPR mạnh với claim trajectory abstraction:** chưa được bằng chứng hiện tại hỗ trợ; có đường đi hợp lý nếu đo được trade-off mới và loại trừ các giải thích đơn giản hơn.

Tôi khuyến nghị tiếp tục CTA, ưu tiên một kết quả PushT có baseline và chi phí thuyết phục, đồng thời điều chỉnh cách diễn đạt theo đúng implementation. C2 là một mở rộng có mục đích khi giữ claim về sự kiện trung gian, không phải điều kiện để CTA có giá trị.
