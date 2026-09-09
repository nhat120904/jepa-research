# Tìm method: world model bảo toàn thông tin phục vụ quyết định

Ngày rà soát: 08/09/2026. Trạng thái: **một ứng viên method để kiểm tra tiền đề; chưa có method đã được xác nhận mới hoặc có lợi trên robot**. Đã đọc literature và kết quả cũ; chỉ chạy một bài toán xác suất hữu hạn bằng linear programming. Không train, không chạy simulator robot, không tải checkpoint.

## 1. Khuyến nghị

Ưu tiên nghiên cứu **học một world model có các nhánh quan sát tương lai tương đương về khả năng ra quyết định với quan sát thật**. Nguyên lý mượn từ thống kê quyết định là **Blackwell comparison / Le Cam deficiency**; tính tuần tự cần ràng buộc không nhìn trước tương lai, liên quan đến adapted optimal transport.

Tên làm việc bằng tiếng Anh: **Feedback-Equivalent World Model Learning**. Đây là mô tả ứng viên, không phải tên một phương pháp đã có kết quả.

Câu hỏi cụ thể: ở cùng ngân sách dữ liệu và dung lượng model, học một kênh quan sát trừu tượng bảo toàn khả năng phân biệt các trạng thái dẫn tới quyết định khác nhau có giúp robot chọn và sử dụng hành động thăm dò tốt hơn full-conditional-likelihood world models không?

Điều đáng nghiên cứu là **learning target và phép chuyển giữa quan sát thật với nhánh tưởng tượng**. Việc thêm memory, ensemble, entropy bonus, hoặc belief-MPC riêng lẻ đều không phải đóng góp mới.

Không khuyến nghị dành ngay cả tháng train. Khuyến nghị một pilot giới hạn để xác định liệu phương pháp có thêm gì ngoài một Bayes filter được học tốt. Cả novelty thực chất lẫn headroom của baseline còn là điều kiện phải vượt.

## 2. Vì sao đổi từ causal OT sang cách đặt câu hỏi này?

COT-GAN đã học sequential generators bằng causal transport; bản conditional đã làm video prediction và có code cho BAIR pushing. Do đó, “thêm action vào conditional COT-GAN” là một đóng góp rất dễ bị coi là mở rộng trực tiếp. WBU đã học belief updates bằng Wasserstein và liên hệ chất lượng model với giá trị quyết định. [COT-GAN](https://arxiv.org/abs/2006.08571), [conditional COT-GAN](https://arxiv.org/abs/2106.05658), [WBU, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d0261e5705d6ec8d54fa1b3c22b69baf-Abstract-Conference.html).

Blackwell/Le Cam đưa ra một đối tượng rõ hơn: **một kênh tín hiệu có thể thay thế kênh kia cho người ra quyết định hay không**, cho phép thay cách mã hóa tín hiệu. Tuy nhiên, bản thân ý tưởng này cũng không mới: Variational Deficiency Bottleneck đã dùng nó trong representation learning; Le Cam Distortion, preprint 2025, đã nghiên cứu directional simulation và có control experiments. Phần còn đáng kiểm tra là một **controlled sequential observation model**, với phép chuyển tín hiệu phải dùng được online, trong learned robot world models. [VDB](https://arxiv.org/abs/1810.11677), [Le Cam Distortion](https://arxiv.org/abs/2512.23617).

**Mức tin cậy:** nguyên lý vững; formulation hữu hạn khả thi; novelty của mở rộng neural/sequential chưa được chứng nhận; lợi ích robot chưa có bằng chứng. Việc search không tìm thấy đúng cụm tên không phải bằng chứng novelty.

## 3. Method làm gì, kể bằng một tình huống robot

Robot chưa biết hướng lệch của một lỗ bị che một phần. Một lần chạm nhẹ tạo tín hiệu giúp chọn hướng chỉnh trước khi insert. World model cần dự đoán cả chuyển động và tín hiệu sau chạm.

Có ba lỗi khác nhau:

- Model xóa quan hệ giữa tín hiệu chạm và hướng lệch: planner đánh giá thấp việc thăm dò.
- Model tạo tín hiệu rõ hơn cảm biến thật: planner tin lần chạm sẽ giải quyết được bất định, rồi hành động quá tự tin.
- Model mã hóa tín hiệu khác thực tế nhưng có thể đổi mã bằng một phép chuyển online: đây có thể là abstraction hợp lệ, không cần phạt như lỗi vật lý.

Method đề xuất học **stochastic observation branches cùng hai phép chuyển tín hiệu không biết hidden state**, để giảm hai loại lỗi đầu và cho phép loại thứ ba. Actor/planner chỉ nhận lịch sử tín hiệu, không nhận hidden-state sample nội bộ của model.

Đây là một giả thuyết về cách phân bổ dung lượng học khi dữ liệu/capacity hữu hạn. Một RSSM đủ mạnh học đúng toàn bộ conditional distribution hoàn toàn có thể giữ được thông tin này. **Không có cơ sở nói vanilla recurrent world models tất yếu vi phạm nguyên lý.**

## 4. Loss tối thiểu có thể viết và kiểm tra

Tại một history và action đã cố định, gọi P là ma trận kênh thật: mỗi hàng là một hidden hypothesis θ, mỗi cột là một tín hiệu quan sát. Q là kênh do model sinh. θ chỉ được dùng làm nhãn hoặc biến nội bộ của generative model; policy không được đọc θ.

Định nghĩa theo quy ước của báo cáo:

`d(P → Q) = min_K max_θ TV(P_θ K, Q_θ)`

K là ma trận xác suất chuyển tín hiệu thật sang tín hiệu model, **cùng một K cho mọi θ**. Nếu K được nhìn θ thì nó có thể tự dựng đáp án và phép kiểm tra vô nghĩa.

- `d(P → Q)` phạt trường hợp model đòi hỏi thông tin mà tín hiệu thật không thể cung cấp.
- `d(Q → P)` phạt model làm mất thông tin hiện có trong tín hiệu thật.
- Hai hướng bằng 0 cho phép hai hệ dùng mã tín hiệu khác nhau nhưng có cùng khả năng giải các bài toán quyết định hữu hạn thích hợp.

Đây là định nghĩa cổ điển, không phải công thức mới của đề xuất. Ở phiên bản hữu hạn, inner optimization là linear program. Bản train tối thiểu giữ dynamics/reward head và thêm channel loss hai chiều trên một tập action và history đã định trước. So trực tiếp với conditional likelihood trên chính các channel rows đó, cùng nhãn θ và cùng dữ liệu.

Phần method cần phát triển mới nằm ở **một cặp phép chuyển có bộ nhớ dùng chung xuyên các bước**: tại bước t chỉ dùng observation/action prefix đến t. Không được fit riêng một phép chuyển cho mỗi thời điểm rồi coi chúng tự động tạo thành một controller nhất quán. Không được dùng toàn bộ future clip để chuyển tín hiệu hiện tại.

Học model dự đoán next physical state, immediate task outcomes và phân phối next signal. Bộ lọc cập nhật belief từ signal và action; model được dùng để xây một cây feedback nông. Tại runtime, chuyển tín hiệu thật qua K rồi dùng cùng bộ lọc và planner. Việc dùng được phép chuyển thật sự là một phần của method, không chỉ là khoảng cách để vẽ chart.

Chi tiết training loop, privilege và giới hạn lý thuyết ở [METHOD_SPEC_EN.md](METHOD_SPEC_EN.md).

## 5. Vì sao có thể thành method paper, và khi nào vẫn chỉ là áp dụng đơn giản?

Một contribution đáng theo đuổi cần cùng lúc có:

1. Formulation và estimator cho **action-dependent, sequential channel matching**, với adapter dùng được online. Không chỉ áp VDB lên embedding hiện tại.
2. Một kết quả về sự chuyển được của feedback decisions, dưới giả định rõ về dynamics, rewards, coverage và approximation. Bound một bước từ thống kê không tự trở thành theorem cho robot nhiều bước.
3. Evidence rằng lỗi về future information tồn tại ở một baseline mạnh và can thiệp sửa nó làm tốt hơn task return/success.
4. Generalization sang chi phí thăm dò, mức nhiễu, hoặc tổ hợp tình huống chưa train, ở cùng lượng dữ liệu và compute được báo cáo đầy đủ.

Nếu công thức cuối chỉ còn `NLL + state classification auxiliary loss`, hoặc chỉ bằng cách tăng trọng số reward đã đạt cùng kết quả, nên bỏ claim về method mới. Nếu một learned likelihood + particle filter nhỏ đã gần oracle thì không có lý do dùng lý thuyết phức tạp hơn.

Không gắn nhãn “đủ top conference” ở thời điểm này. Hướng này phù hợp tiêu chí tìm method hơn selective verification, nhưng độ phức tạp toán học không làm nó tự nhiên có novelty.

## 6. Task–baseline: chọn có điều kiện, không giả vờ đã sẵn sàng

### Task đầu tiên để cô lập mechanism

**Occluded target retrieval**, với hai hoặc ba vị trí che khuất và hành động uncover rồi chọn vật. Có giá phải trả cho thao tác thăm dò, đánh giá success và số thao tác thực thi. Không dùng latent L2 làm task objective.

Task này cần kiểm tra bằng một tầng action primitives đủ ổn định để phân biệt failure về information với failure về grasp. Các primitives và dữ liệu được chia sẻ cho mọi arm. Có thể dùng một biến thể công khai, khai báo rõ trên ManiSkill; đó sẽ là experimental environment variant, không phải đóng góp benchmark. Việc cài và xác minh controller chưa được thực hiện.

IMBench đã mô tả Occluder Push và Mass Sort; Mass Sort cần tín hiệu lực/mô-men khi nhấc vật. Trong lượt rà này xác minh được bài và trang dataset, **chưa xác minh được repository simulator phù hợp để chạy task**. Không coi dataset trajectories là đủ để làm closed-loop evaluation. [IMBench, preprint2026](https://arxiv.org/html/2607.15641v1).

### Task robot thứ hai nếu pilot qua

**Peg insertion với bất định pose và quan sát hạn chế**, dùng visual/proprioception/contact signals. ManiSkill có PegInsertionSide-v1 cùng demos; bản có che khuất và protocol thăm dò sẽ là biến thể cần xây và công bố cấu hình, không phải tính chất đã xác minh của task gốc. [ManiSkill tasks](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html).

BayesContact đã làm particle belief, forward observation simulation và active contact probing. Nó là prior art/baseline mạnh, không phải chứng cứ rằng thêm belief là mới. Cần so bản learned observation model và bản simulator likelihood riêng để tách chất lượng model khỏi planner. [BayesContact, preprint2026](https://arxiv.org/abs/2607.16123).

Không chọn Mass Sort làm task duy nhất: một phép đo lực gần trực tiếp có thể làm Bayes estimation quá dễ. Không lấy success thấp của VLA không có đúng sensor/history làm headroom cho world model.

### Baseline bắt buộc

| Baseline | Nó loại trừ lời giải thích nào? |
|---|---|
| Oracle observation+dynamics model, feedback planner | Task có giải được; mức trần của controller |
| Cùng oracle nhưng certainty-equivalent / không dùng thông tin mới | Có thật sự cần feedback và probing? |
| Learned dynamics + learned observation likelihood + Bayes/particle filter, cùng planner | Method có hơn lời giải model-based trực tiếp? |
| Recurrent stochastic WM, conditional/multistep likelihood; cùng sensors/data/params | Lợi ích có chỉ là thêm memory hoặc stochasticity? |
| Cùng backbone + supervised hidden-state/belief auxiliary head | Lợi ích có chỉ là privileged labels? |
| WBU-style belief learning; conditional COT objective adaptation | Nguyên lý gần nhất có giải quyết được không? |
| Recurrent policy/model-free hoặc imitation baseline có đủ observation history | Có cần world model cho task này không? |
| Giữ architecture, dùng direct conditional channel KL/TV, bỏ garbling | Hai phép chuyển có giá trị vượt ordinary channel fitting? |

WBU/COT adaptations phải ghi là reimplementation/adaptation; không so số trong paper khác task và gọi đó là matched benchmark. Không có checkpoint nào của tổ hợp này đã được tải hoặc reproduce trong lượt nghiên cứu.

## 7. Tránh lặp null cũ

Đã đọc lại `moment_wm_h0/docs/DECISION_REPORT.md`: hidden drag có thể decode tốt, nhưng kernel moment loss không giảm true planning cost. Đây là dấu hiệu rất cụ thể rằng **cứ học hidden physics tốt hơn chưa chắc giúp planning**. Không dùng lại lập luận “vật có hidden parameter nên thêm loss sẽ giúp”.

Trong pilot mới, cần một intervention tách được **kênh quan sát tương lai**: giữ dynamics, reward, action candidates và planner như nhau, chỉ thay observation channel của model bằng oracle channel. Nếu thay như vậy không tăng task return, dừng ứng viên này trên task đó. Dùng oracle full dynamics từ đầu để che hết lỗi không phải là evidence method học được world model tốt hơn.

Null về PS-JEPA không chứng minh mọi task partial observability đều vô ích; đồng thời cũng không cho phép mặc định task mới có aliasing. Phải kiểm tra performance của frame-only và history baseline với đúng sensors và labels.

Null về local geometry không phải bằng chứng ủng hộ information-channel loss. Chỉ một phép thay channel có hiệu quả thực nghiệm mới tạo được cầu nối đó.

## 8. Pilot hữu hạn, có điểm dừng

**P0 — kiểm tra task/controller trước training lớn.** Oracle feedback controller phải giải được task ở mức đủ cao; một baseline thăm dò cố định đơn giản phải được đánh giá. Nếu scripted uncover-all gần tối ưu cả success lẫn execution cost, task không phù hợp. Ngưỡng practical effect được chọn trước, không lấy từ paper khác.

**P1 — tìm failure ở baseline mạnh.** Train learned-likelihood Bayes baseline và recurrent WM có đủ thông tin. So channel-only oracle replacement trên cùng test episodes. Chỉ đi tiếp nếu lower confidence bound cho cải thiện vượt một mức hữu ích đã khóa trước (ví dụ 5 điểm phần trăm success, hoặc 10% normalized return gap; đây là đề xuất thiết kế, không phải ngưỡng thống kê phổ quát).

**P2 — thử phiên bản method nhỏ nhất.** Action và hidden hypotheses hữu hạn, 2–3 observation stages, 4–8 signal codes; cùng data/capacity/labels; ba training seeds để screen, chưa đủ kết luận paper. So channel loss với matched NLL, auxiliary belief head, và direct conditional channel divergence.

**P3 — chỉ mở robot expansion khi có task gain.** Khóa hyperparameters bằng validation, rồi mở held-out test. Dùng paired reset seeds và bootstrap theo episode, đồng thời báo variation theo training seed. Số episode final cần power analysis từ variance pilot; không chọn tùy ý một số nhỏ rồi kết luận null mạnh.

Dừng nếu metric channel tốt hơn nhưng return không tăng; gain biến mất khi baseline nhận cùng privileged labels; gain chỉ có khi adapter nhìn hidden state/future; hoặc lợi ích đến từ nhiều simulator samples hơn. Không lặp vòng vá metric vô hạn.

## 9. Kết quả đã có trong lượt này

[Ví dụ hữu hạn](channel_example.py) giải chính xác các LP so sánh kênh hai hidden states; [JSON](channel_example.json) lưu kết quả. Mọi kênh trong ví dụ có cùng marginal của tín hiệu. Kênh mất thông tin có accuracy tối ưu 0.5 thay vì 0.9; một kênh quá lạc quan dự đoán 0.99; kênh chỉ đổi nhãn vẫn giữ 0.9 và deficiency hai chiều bằng 0.

Đó là kiểm tra minh họa **nguyên lý đã biết**, không phải kết quả tích cực mới cho research. Nó không chứng minh NLL thất bại, không chứng minh algorithm neural train được, và không cho sequential guarantee.

## 10. Các nhánh không chọn làm hướng chính

| Nhánh | Lý do hạ ưu tiên |
|---|---|
| Causal OT loss thuần túy | COT-GAN/conditional COT-GAN, WBU, OT-MDPs đã sát; cần contribution controlled/sequential rõ hơn |
| Saltation/contact event derivatives | ContactNets, Neural Hybrid Automata và event-aware derivatives đã có; gradient đúng chưa chắc tối ưu tốt; simulator soft contact có thể không thỏa giả định |
| Port-Hamiltonian WM | PH-Dreamer preprint2026 đã chiếm formulation gần nhất |
| Learned coarse/fine propagation để tăng tốc planning | Parareal với learned coarse model đã áp dụng robotic manipulation từ 2019/2020 |
| Thêm belief và information-gain probing | Active tactile perception và BayesContact đã có lời giải sát |

Nguồn và trạng thái publication ở [SOURCE_LEDGER.md](SOURCE_LEDGER.md). Research cũ không bị xóa. Kết quả lần này là **một thiết kế method hẹp hơn cùng các phép thử có thể bác bỏ nó**, không phải lời bảo đảm đã tìm được đề tài thắng.
