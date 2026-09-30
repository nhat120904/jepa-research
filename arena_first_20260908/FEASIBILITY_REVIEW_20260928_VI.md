# Review selective stability cho CVPR 2027

Kiểm tra ngày 28/09/2026 theo ngày của workspace; đã tiếp nhận phần điều chỉnh mang ngày 29/09 trong yêu cầu của người dùng. Đọc source/tài liệu tại commit `3c32cb9` cùng working tree đang có thay đổi của các phiên khác; tra cứu nguồn primary trên web; kiểm tra scheduler/accounting và quota. Không huấn luyện, chạy simulator, phân tích lại dữ liệu thô hoặc thay đổi job trong review này. Các số thực nghiệm dưới đây lấy từ báo cáo đã lưu, không phải replication mới.

**Quyết định đề xuất: không chọn bản selective stability hiện tại làm method chính đã chốt cho CVPR. Dừng mở rộng bản PCA + normal-direction penalty thành một chương trình lớn. Chỉ giữ một iteration end-to-end khoảng 5–7 ngày nếu nó thực sự kiểm tra được estimator phân biệt biến thiên cần giữ và sai số cần triệt tiêu. CEM routing tiếp tục là phương án dự phòng chưa được cấp đầu tư, không phải đường lui mặc định.**

Đây là nhận định rủi ro nghiên cứu, không phải xác suất accept định lượng. Chưa đủ dữ liệu để đưa một tỷ lệ phần trăm fail đáng tin.

## 1. Trạng thái và evidence mới hơn audit

- [METHOD_SPEC_EN.md](METHOD_SPEC_EN.md) ghi rõ proposed, not implemented or validated. Thư mục hiện có proposal, status và provenance/config; chưa có pipeline selective stability đã triển khai trong thư mục này. Panel Reacher true endpoint 100% vs predicted 41.5% là local witness-conditioned selection, không phải closed-loop benchmark và không định danh nguyên nhân.
- [CTA ledger](../trajectory_innovation_20260922/JOB_LEDGER.md) đã có learned closed loop: CTA4 142/200 vs P0 122/200 development roots, +10 pp [2,18], một seed. Native-score contrast null; CTA8O 138/200 bằng ENDPOINT8O 138/200. Vì vậy audit cũ nói trajectory reader chỉ có privileged evidence không còn mô tả đủ trạng thái CTA. Đồng thời kết quả mới chưa xác nhận đóng góp riêng của trajectory abstraction.
- CTA Cube offline mới: retained gap CTA .91, DIRECT .94, ENDPOINT .94. Đây là ranking offline, không phải success. Tại lúc kiểm tra, `squeue` và `sacct` cùng xác nhận 55601 và 55614_0 RUNNING, 55616 PENDING; không suy ra kết quả cuối của những job này.
- [Action-response PushT](../action_response_20260927/README.md) +1.3 pp CI [-6,8], [Cube](../action_response_cube_20260928/README.md) 35/50 vs 36/50 và 34/50 vs 37/50 không chứng minh gain. Các null này làm tăng nhu cầu chứng minh cơ chế, không chứng minh mọi intervention dynamics đều vô ích.
- [Rollout repair](../rollout_repair_gate/outputs/stage1/analysis/DECISION.md) đã không qua các tiêu chí ranking/fresh evaluation của chính thử nghiệm. Không đề xuất lại deployment-matched AR fine-tuning như contribution mới.
- [AGENTS.md](../AGENTS.md) ghi CEM routing đã bị bỏ ngày 28/09 vì overlap và latency LeWM PushT khoảng dưới 0.6 s/plan. Review này xác nhận 55392 COMPLETED qua scheduler/accounting nhưng không đo lại latency. Subsecond tự nó không chứng minh không còn giá trị tăng tốc; vấn đề là chưa có nhu cầu control và advantage riêng đủ rõ.

## 2. Prior art trực tiếp và giới hạn overlap

| Nguồn primary | Phần đã có | Phần selective stability còn phải làm |
|---|---|---|
| [Towards Unraveling and Improving Generalization in World Models](https://arxiv.org/html/2501.00195v1), §4; đọc như preprint, không khẳng định main-track acceptance | Phân tích sai số biểu diễn lan truyền trong rollout; Jacobian regularization cho latent dynamics | Chứng minh lựa chọn hướng regularize có ích hơn global Jacobian/noise regularization với compute và data matched |
| [Contractive Auto-Encoders, ICML 2011](https://www-labs.iro.umontreal.ca/~vincentp/Publications/contractive_autoencoder_icml2011.pdf) | Jacobian penalty; contraction chủ yếu ngoài manifold, reconstruction giữ biến thiên dọc manifold | “Co hướng normal, giữ hướng tangent” không phải nguyên lý mới; dynamics/action-conditioned estimation và control phải tạo phần khác biệt |
| [Geometrically constrained vector-field learning, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/318f3ae8be3c97cb7555e1c932f472a1-Abstract-Conference.html) | Ước lượng geometry từ data, tangent vector fields và geometry-preserving integration | Không claim đầu tiên học dynamics trên manifold. Bài này không đồng nhất với visual latent MPC; vẫn có khoảng trống ứng dụng/cơ chế, nhưng chưa tự thành novelty |
| [GRASP](https://arxiv.org/html/2602.00475v1), §3.3; preprint | Latent state gradients có thể bị khai thác để tạo transition không thực; planner dùng grad-cut, Langevin và goal shaping | Selective stability học predictor, còn GRASP đổi planner. Không coi GRASP là cùng training algorithm, cũng không bỏ qua diagnosis/state-sensitivity overlap |
| [A Control Theory of Predictability in Latent World Models](https://arxiv.org/html/2607.10362v1), §4–5; preprint | Planner-reachable distribution, off-manifold divergence và giới hạn prediction loss; lý thuyết đặt dưới linear-control premise | Không dùng bài này như bằng chứng cơ chế đã đúng cho checkpoint Reacher. Chính bài phân biệt Jacobian proxy và residual thực; không cấp một chứng nhận control tổng quát cho neural MPC |
| [Learning to Simulate Complex Physics with Graph Networks, ICML 2020](https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf) | Training với noise để giảm tích lũy lỗi rollout | Rollout-noise/denoising baseline là đối chứng khoa học quan trọng; chỉ đổi noise thành noise anisotropic chưa đủ kết luận contribution |

[HaM-World](https://arxiv.org/abs/2605.05951) cũng là preprint về cấu trúc latent/history cho stability. Không khuyến nghị thêm Hamiltonian, memory hay một kiến trúc mới để né overlap: như vậy mở rộng phạm vi mà chưa giải bài toán nhận diện.

Với phương án efficiency, [surrogate CEM](https://arxiv.org/abs/2009.09043) đã có surrogate và scheduling true evaluations; [iCEM, CoRL 2020](https://proceedings.mlr.press/v155/pinneri21a.html) là baseline giảm sample quan trọng. Influence lên search update có thể là khác biệt cụ thể, nhưng không đủ để bỏ qua các đối chứng này hoặc suy luận 2× latency đồng nghĩa novelty.

## 3. Nút thắt quyết định: phân biệt hai loại biến thiên có xác định được không?

Nhận định sau là phân tích của review, không phải kết quả thực nghiệm mới.

1. **Normal với tập data không đồng nghĩa phi vật lý.** PCA đo geometry và mật độ của history do collection policy sinh ra. Một state/action hợp lệ ít được quan sát có thể nằm ngoài chart đó. Co hướng này có thể xóa chính phương án thoát khỏi thất bại mà planner cần tìm.
2. **Sai số predictor có thể nằm dọc manifold.** Dự đoán một góc khớp sai nhưng hợp lệ vẫn là sai. Projection về manifold có thể tạo một state trông hợp lệ mà sai hậu quả action. Không được đồng nhất plausibility với accuracy.
3. **PCA history không điều kiện đủ theo actions.** Gần nhau theo vector ba frame chưa đảm bảo gần nhau về velocity, latent hidden state hoặc action-conditioned reachability. Giữ các slot quan sát cố định là đúng nhưng chưa giải quyết nhận diện này.
4. **Không thể luôn nhận diện nguyên nhân từ vector latent.** Nếu cùng một perturbation quan sát được có thể xuất phát từ error hoặc một physical change thật, estimator chỉ nhìn vector đó không thể luôn gán đúng hai nhãn trái nhau. Cần giả định quan sát, history hoặc dữ liệu can thiệp bổ sung. Đây là hạn chế thông tin, không phải thiếu một head lớn hơn.
5. **Contact và rare events gây rủi ro xóa tín hiệu.** Một thay đổi nhỏ trước contact có thể tạo hậu quả lớn hợp lệ. Ép contraction trên toàn bộ những thay đổi đó có thể làm giảm error trung bình nhưng giảm success.
6. **Random normal probes chưa đại diện cho lỗi deployment.** Trong 576 chiều, normal complement của rank-12 chart rất lớn. Xóa năng lượng ở đó có thể dễ nhưng không chạm vào directions mà predictor/planner thực sự sử dụng. Cần matched-norm random/noise controls và actual rollout residuals.

Một estimator đáng nghiên cứu cần được định nghĩa bằng dữ liệu có ý nghĩa độc lập, thay vì tự gọi các hướng ngoài PCA là lỗi:

- Cặp predicted/encoded-real histories của cùng trajectory và action prefix cho residual rollout thực; residual vẫn có thể gồm cả sai góc/vận tốc hợp lệ, nên không mặc định là nuisance.
- Cặp history vật lý lân cận, sinh bằng trajectory hợp lệ, chạy cùng continuation cho biết thay đổi nào cần tạo tương lai khác nhau. Không chỉ perturb action ở cùng state rồi lặp lại action-response loss đã thất bại.
- Nếu dùng cùng physical state với thay đổi appearance/rendering để tạo cặp tương đương, phải bảo đảm biến đổi không xóa cue điều khiển và phải xác định target/không gian bất biến phù hợp. Gain chỉ dưới appearance augmentation hỗ trợ robustness claim; chưa chứng minh sửa endogenous rollout error.
- Simulator can thiệp hoặc physical labels trong training phải được khai báo. Deployment chỉ dùng thông tin cho phép. Mọi baseline được dùng cùng transitions/supervision để tránh nhầm gain do data với gain do method.

Ước lượng covariance, local metric hay generalized eigenspaces từ các cặp trên đều là công cụ có sẵn. Contribution nếu có phải là estimator/assumptions phù hợp và advantage đã đo, không phải tên của phép phân rã.

## 4. Thử nghiệm nhỏ nào có giá trị ra quyết định?

Nếu tiếp tục, làm **một iteration end-to-end**, không tuần tự nhiều vòng gate trước khi có planner chạy. Giữ encoder, goal cost, candidate budget và controller; tích hợp các phép đo cơ chế vào cùng training/evaluation.

Các đối chứng tối thiểu cho iteration: released checkpoint; matched AR fine-tuning; noise/consistency; global Jacobian penalty; PCA-normal variant hiện tại; estimator mới nếu định nghĩa được; random-subspace control khi diễn giải phần selective. Có thể triển khai theo stages để tiết kiệm nhưng không tuyên bố thắng khi comparator quan trọng còn thiếu. Một seed là development screen, không phải kết luận paper.

Đo đồng thời:

- Error amplification trên residuals thực, không chỉ artificial isotropic noise.
- Fidelity của response với valid state changes dưới cùng continuation; giảm sensitivity mọi nơi không được tính là giữ physics.
- Closed-loop success/return với fresh CEM reoptimization, matched roots, giới hạn rollout nguyên bản; fixed-pool ranking chỉ là diagnostic.
- Extra training time, inference overhead, số simulator transitions và nguồn supervision.

Chọn Reacher để kiểm tra triển khai; manipulation task có sẵn như Cube là transfer cần thiết cho phạm vi rộng hơn. Không sử dụng 100% vs 41.5% để ước lượng expected closed-loop gain. Không áp cấu hình Reacher sang Cube một cách máy móc.

Không nhất thiết cần gain +10 pp hoặc 2× để một method có giá trị. Cần practical effect đã định nghĩa trước, CI đủ hẹp cho claim, advantage trước comparator mạnh nhất và cơ chế không bị baseline đơn giản giải thích hết. Episode/root là đơn vị uncertainty; nhiều candidates không làm tăng số mẫu độc lập. Pilot 50 episodes có thể loại một cấu hình gây hại lớn nhưng thường không xác nhận cải thiện vài pp.

**Dừng đầu tư thêm cho kỳ CVPR này** nếu sau iteration đã sửa lỗi kỹ thuật, method chỉ bằng noise/global Jacobian/PCA; giảm amplification bằng cách làm mất physical response; hoặc không có benefit khi planner tối ưu lại. Kết quả thiếu power phải ghi inconclusive; đó vẫn có thể là lý do quản trị để không đặt cược deadline, không phải bằng chứng cơ chế sai phổ quát. Không tiếp tục bằng sweep lambda hoặc module mới khi không có một nguyên nhân sửa cụ thể.

## 5. Deadline và compute

[CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) xác nhận registration 10/11/2026, paper 16/11 AoE, supplementary 23/11. Từ mốc 29/09 có 48 ngày đến 16/11. Đủ cho một method gọn trên hạ tầng hiện hữu nếu cơ chế sớm có tín hiệu; không phải dư thời gian cho nhiều lần đổi nền tảng.

Lịch quản trị đề xuất, không phải forecast runtime:

| Mốc | Kết quả cần có |
|---|---|
| 29/09–05/10 | Một implementation + learned closed-loop pilot và cheap controls; nêu được estimator dùng thông tin gì để phân biệt hai loại biến thiên |
| 06–12/10 | Lặp trên seed mới, thêm manipulation setting đã có; so với strong simple baseline và kiểm tra giữ physical response |
| Khoảng 15/10 | Quyết định scope method/paper; không còn contribution chính chỉ dựa vào panel Reacher hoặc MSE |
| 16–31/10 | Khóa method, confirmatory roots/seeds, ablation và đối chiếu data/compute; second WM family nếu claim generality cần nó |
| 01–16/11 | Viết, kiểm tra claim/provenance, figures và các lỗi xác định rõ; không mở hướng chính mới |

Không có quy tắc CVPR bắt buộc chính xác ba môi trường/hai WM. Quy mô bằng chứng phải đủ cho độ rộng claim; hai toy tasks và một checkpoint tạo rủi ro external validity lớn, còn chạy nhiều task không tự bù được thiếu novelty.

Snapshot `sreport -t hours -T gres/gpu,cpu,mem ...` tháng 09:

- GPU: user 173 h; người hạng 5 theo GPU 486 h; 50% là 243 h, chênh 70 h trước dự phòng job hiện có.
- CPU: user 1571 core-h; hạng 5 là 3890, 50% là 1945, chênh 374 core-h trước dự phòng.
- Memory cũng dưới 50% mốc hạng 5 ở snapshot. Đơn vị memory giữ theo accounting, không tự đổi thành GB-h trong review.

Đây **không phải budget được cấp cho selective stability**: chưa trừ toàn bộ remaining wall-time của running/pending jobs, accounting có độ trễ và ranking thay đổi. Ví dụ ceiling 60 GPU-h của proposal cũ, nếu dùng 8 CPU/GPU xuyên suốt, tương ứng 480 core-h, lớn hơn chênh CPU hiện tại trước cả CTA. Phải tính lại resources cụ thể trước submission; không hứa hoàn tất comparison từ GPU-hours đơn lẻ. Không submit job trong review này.

## 6. Nếu bỏ ứng viên này, tìm hướng mới theo cách nào?

Không khuyên tiếp tục tìm một acronym/world-model loss nghe hợp lý. Tài sản mạnh nhất của repo là checkpoint, simulator restore, failures và harness có đối chứng. Dùng chúng để tìm một khoảng trống có thể can thiệp:

1. Chọn một failure lặp lại trong closed loop dưới setting thực, ghi rõ representation, planner và candidate support. Không lấy gap từ panel có witness làm gap benchmark.
2. Xác định một giả định cụ thể của baseline bị vi phạm và một intervention có thể thay đổi nó. Ví dụ selective stability đang giả định variation ngoài data chart là lỗi; nghiên cứu đáng làm phải chỉ ra khi nào giả định này sai và cách nhận diện tốt hơn.
3. Đặt cạnh nearest prior art và baseline rẻ nhất có thể giải thích gain. Nếu câu mô tả contribution chỉ còn “dùng kỹ thuật X cho LeWM” hoặc “thêm trọng số cho loss Y”, rủi ro incremental cao trừ khi có kết quả/cơ chế thực sự mạnh.
4. Chọn một ứng viên, triển khai ngay vòng train–plan–control trên hạ tầng hiện có. Diagnostics phục vụ quyết định sửa cụ thể, không trở thành một audit mới kéo dài nhiều tuần.
5. Tìm transfer đúng cơ chế: đổi goal, rollout horizon, contact regime hoặc representation phù hợp với claim, thay vì thêm dataset chỉ để đếm số benchmark.

Câu hỏi ưu tiên nếu tiếp tục dynamics learning là: **từ history/action và dữ liệu cho phép, khi nào có thể nhận diện một thay đổi latent nên bị sửa mà không xóa một tương lai vật lý hợp lệ?** Đây là bài toán nghiên cứu có nội dung, chưa phải method được xác lập và chưa được đảm bảo mới.

Nếu không có câu trả lời triển khai được cùng advantage closed-loop trong đầu tháng 10, không khuyên ép selective stability thành method paper tháng 11. Giữ kết quả âm có giá trị, hoàn thiện sản phẩm audit riêng nếu thích hợp, và chuyển method mới sang lịch dài hơn. CTA phải được quyết định từ kết quả và comparator riêng của nó; review này không tự hủy hoặc chuyển hướng các job CTA đang được phiên khác thực hiện.
