# Audit target progress của CTA: coverage plateau và credit assignment

2026-09-25. Đã kiểm tra source và metadata Round 1; không thay đổi cấu hình
Round 2 đang chạy. Những mục đề xuất bên dưới chưa được thực nghiệm xác nhận.

## Kết luận

**Có lỗ hổng thật trong objective và cách lấy mẫu của implementation hiện tại.**
Coverage cuối chunk không cung cấp thứ tự đúng giữa các bước chuẩn bị có
cùng overlap. Co-design chỉ tối ưu code theo những target đang có, không tự
tạo ra long-horizon supervision. Điều này hạn chế claim tổng quát của CTA.

Đây chưa phải bằng chứng conditional trajectory abstraction không thể work:
source/WM/reader là kiến trúc, còn query target xác định thông tin nào cần
được giữ. Goal-conditioned continuation value là một hướng cần qualification,
**chưa phải hướng sửa ưu tiên đã được dữ liệu hỗ trợ**. Cần đối chiếu kết quả
GR00T và PushT bên dưới trước khi đầu tư thêm vào target này.
Không khuyến nghị viết thêm các hàm distance/grasp/stage shaping riêng từng task.

### Cập nhật sau khi đối chiếu thí nghiệm GR00T trước đây

Lo ngại về long continuation đã được phân tích trong
[`VALUE_LEARNING_REASSESSMENT_20260917_EN.md`](../../latent_scope_20260909/docs/VALUE_LEARNING_REASSESSMENT_20260917_EN.md).
Đề xuất TD critic dưới đây không phải giải pháp mới chưa từng cân nhắc trong repo.

- [GR00T confirmation 52573](../../latent_scope_20260909/docs/BASELINE_AUDIT_52597_AND_CONFIRMATION_RESULT.md):
  selected 6/16, baseline 5/16; năm wins, bốn losses, trên hai prefix đã chọn.
  Không confirm rõ screening gain; không phải 16 scene độc lập và không chứng
  minh mọi continuation objective đều vô ích. Run mất 02:10:47 nên không coi
  branching dài là một audit rẻ mặc định.
- [PushT Gate A–C](GATE_ABC_RESULT_53806.md): single-intervention continuation
  estimate với split seeds +2 pp [-3, +7.5], còn repeated PHYS8 +15 pp [5, 25]
  trên 100 qualification roots. Protocol khác nhau; không diễn giải số +2 là
  upper bound của repeated selection. Naive hindsight estimate +22.75 pp là
  cảnh báo winner's curse, không phải achievable expected-value headroom.

Phải phân biệt noise của một label với target thật sự gần phẳng. Lặp nhiều
seed có thể giảm uncertainty của Q^pi nhưng không tạo ra gap khi policy phía
sau xóa khác biệt, hoặc không thực hiện được bước tiếp theo mà candidate cần.
Common random numbers không cố định feedback actions; TD giảm nhu cầu rollout
dài bằng bootstrap nhưng thêm critic error, không sửa target phẳng.

Khuyến nghị sửa lại: giữ Round 2 là kiểm tra WM trên coverage target đã có
bằng chứng repeated-control headroom. Với target mới, qualification là một
nhánh riêng, có budget/stop rule; chưa scale terminal-label collection hay
train CTA value chỉ vì coverage có plateau. Một learned direct critic có thể
là baseline qualification, nhưng chưa phải bằng chứng abstraction mang lợi ích.
Nếu thay continuation bằng một selector mạnh hơn, phải version policy và
relabel/evaluate nhất quán; đó là một intervention mới, không tự động rescue.

## 1. Chính xác implementation đang học gì

Native environment đã pin:
`prepare_53776/deps/gym_pusht/envs/pusht.py`, `_get_coverage`:

    coverage = area(block intersect goal) / area(goal)

Không phải IoU và không phải khoảng cách tới goal. Goal là vùng hình học trên
mặt phẳng, không phải vật thể mà agent phải chạm. Block chưa overlap goal
thì coverage=0; tiếp xúc giữa agent và block là chuyện khác.

`scripts/d_collect.py` lưu `cov8 = b.coverage` của mỗi sibling sau tối đa 8
action steps, hoặc lúc kết thúc sớm. Không phải max coverage toàn episode và
không phải cumulative future success. Vì mọi sibling cùng context, dùng
cov8−cov_now vẫn cho cùng thứ tự và không sửa plateau.

`ti_wm/sibling.py:rank_loss` chỉ dùng cặp có label difference >1e-3. Nếu không
có cặp như vậy, loss=0 và không có ranking gradient. Ví dụ minh họa:

| Candidate | Hệ quả thật sau chunk | cov8 | Tín hiệu ranking hiện tại |
|---|---|---:|---|
| A | agent tiến đến phía đẩy hữu ích | 0 | hòa |
| B | block tiến gần goal nhưng chưa overlap | 0 | hòa |
| C | agent đi xa block | 0 | hòa |

Không có nghĩa model chắc chắn xuất ba score bằng nhau: network có thể
generalize từ những state khác và cho điểm khác. Nhưng supervision hiện tại
không xác định thứ tự đúng ở bank này. Sai số tùy ý cũng có thể chọn action
kém hơn P0. PHYS8 dùng coverage thật nên hòa chính xác và chọn candidate 0;
learned CTA không tự động được bảo vệ bởi cùng fallback đó.

Plateau không chỉ xuất hiện ở zero coverage. Agent thay đổi vị trí tiếp cận
mà block chưa dịch chuyển thì mọi coverage có thể bằng nhau và khác zero.
Ngược lại, coverage giảm tạm thời vẫn có thể là bước chuẩn bị hữu ích cho
rotation/contact tiếp theo; do vậy nonzero coverage cũng không là value dài hạn.

## 2. Sampling làm thiếu tín hiệu rộng hơn ranking loss

`scripts/cta_train.py:Split` lập `rank_pool` từ các bank có ptp(cov8)>1e-3.
Stage 1 lấy batch từ pool này rồi cùng lúc update source encoder, reader,
feature decoder, FULL và DIRECT. Round-2 source blocks cũng dùng pool đó.
WM/prior stages lấy trên tất cả decisions.

Metadata `cta_train_54717/train_report.json`:

- Tổng training decisions: 22,680.
- Bank vào ranking pool: 11,773 (khoảng 51.9%).
- Bank không vào pool: 10,907 (khoảng 48.1%).

**48.1% không phải tỷ lệ all-zero coverage.** Nó gồm mọi bank có spread
không vượt margin. Cập nhật 2026-09-26, CPU audit 54975 đã hoàn tất: train
2,570 all-zero (11.3%), 7,649 nonzero exact-flat (33.7%), 688 near-tie (3.0%).
Dev tương ứng 332 (11.0%), 981 (32.6%), 88 (2.9%), trên 3,011 decisions.
Xem [failure map và hướng sửa](CTA_FAILURE_MAP_AND_REPAIR_20260926_VI.md).

Hệ quả được chứng minh từ code: các bank đó không trực tiếp đóng góp cả
feature reconstruction/source-code updates trong stage 1, dù mục đích của
feature decoder là giữ thông tin rộng hơn ranking. Không được suy diễn rằng
representation không chứa gì về các phase đó: DINO cố định và generalization
vẫn có thể giữ thông tin. Nhưng sampling không bảo đảm điều này.

Sửa sampling nên tách rõ:

- Reconstruction/representation: sample từ mọi phase/decision.
- Coverage ranking nếu còn giữ như auxiliary: mask cặp có label rõ.
- Long-horizon value: sample mọi phase, với terminal masks và đúng targets.

Đây là một ablation riêng ở cùng update/compute budget. Chỉ bỏ filter không
tạo được thứ tự hữu ích khi labels vẫn hòa; nó chỉ sửa exposure của representation.

## 3. Ladder cũ có thể không nhìn thấy lỗ hổng này

`ti_wm/cta_eval.py:ranking_metrics` chỉ tính Spearman ở bank có ptp(label)>0.
Retained gap dùng gain trên oracle coverage. Một bank all-equal đóng góp 0
cho cả numerator và denominator của retained gap, bất kể chosen action tốt
hay xấu cho thành công về sau. Các bank gần hòa nhưng ptp>0 có thể xuất hiện
trong metric trong khi bị loại khỏi train vì margin 1e-3.

Vì vậy code giữ 89% gap của FULL là retention của **metric coverage có
thể phân biệt**, không có nghĩa giữ 89% thông tin quyết định cho cả task.
Điều này không làm các số cũ sai; nó giới hạn điều có thể kết luận từ chúng.

Closed-loop success vẫn đo hệ quả tổng thể. Việc một selector theo coverage
thành công hơn P0 trên PushT là bằng chứng nó hữu ích ở một phần trạng thái,
không phải chứng minh coverage cung cấp tín hiệu cho mọi bước.

## 4. Có phải mỗi task phải tự viết progress không?

Không bắt buộc viết dense progress. Nhưng không thể bỏ mọi mô tả về mục tiêu.
Cần ít nhất một trong: native success predicate; successful demonstrations
được xác nhận; goal observations với định nghĩa đạt goal; preferences; hoặc
task instruction gắn với dữ liệu học tương ứng.

Phân biệt hai mức:

1. **Task specification:** thế nào là thành công. Benchmark thường đã cung
   cấp `is_success`. Đây là giả định hợp lệ và phải công khai.
2. **Dense shaping:** tự đặt trọng số distance, grasp, lift, alignment, milestones.
   Không cần nếu value học được credit assignment từ outcomes/transitions.

Một cùng objective học value có thể áp dụng qua task khác với goal/language
conditioning, nhưng vẫn cần data coverage, grounding và evaluator tương ứng.
Không gọi đây là universal progress tự sinh từ một goal image bất kỳ.

### Các hướng đã có trong literature

- [V-GPS](https://arxiv.org/html/2410.13816v2): học Q bằng offline RL rồi
  rerank proposal của policy. Dữ liệu demonstration được gán binary reward
  ở các bước cuối; không cần dense geometric shaping riêng từng task.
  Giả định endpoint của demonstration biểu thị task completion là quan trọng.
  Với rollout lẫn thành công/thất bại của CTA, không được gán mọi final frame
  là success như thể tất cả đều là demonstration thành công.
- [VIP](https://arxiv.org/abs/2210.00030): học representation/value từ video
  để tạo visual rewards. Hữu ích như reference/control, không bảo đảm resolve
  được các khác biệt nhỏ giữa siblings ở contact.
- [HER](https://arxiv.org/abs/1707.01495): relabel achieved goals để học từ
  sparse binary rewards. Vẫn cần cách xác định goal và recompute reward cho
  goal đã relabel; không tự giải quyết grounding hay mọi task logic.
- [Potential-based shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf):
  có invariance dưới điều kiện phù hợp khi tối ưu full return. Thêm một
  potential vào greedy short-horizon selector không tự có bảo đảm đó; phải
  xử lý terminal/boundary và bootstrap nhất quán.

Trong repo, C2 đã thử regression khoảng thời gian tới một future/goal frame.
Hiệu ứng closed-loop không replicate rõ ở C2-confirm. Vì vậy không đề xuất
chỉ đổi tên lại “temporal distance reader”. Elapsed time trong một demo có
thể chứa chờ, đi vòng và variation của policy; nó không đồng nhất với
counterfactual value khi đổi action ở một state cố định.

## 5. Hướng cần qualification: reader ước lượng khả năng thành công dài hạn

Giữ frozen proposal policy pi_ref. Định nghĩa đối tượng cần dự đoán:

    Q^pi_ref(h, A, g, b)
      = P(success trong b bước còn lại | history h,
          thực thi chunk A, sau đó theo pi_ref, goal g).

Đây là value của continuation policy đã xác định, không phải optimal Q*.
Khởi đầu với policy evaluation này giúp tránh tự xây ngay một offline-RL
actor phức tạp. Nó không bảo đảm đủ signal để rerank các chunk rất giống nhau.

Với finite-horizon success objective, có thể dùng gamma=1, success chỉ tính
một lần, và budget còn lại b là input của value/reader. Target cho sibling:

    y_k = u_k + (1 - d_k) * V_bar(h'_k, g, b - n_k)

Trong đó:

- u_k=1 nếu chunk lần đầu đạt success, ngược lại 0;
- d_k=1 nếu success hoặc hết episode budget, ngược lại 0;
- n_k là số bước thực thi thực tế, kể cả chunk dừng sớm;
- h'_k là observed history thật sau branch, chỉ dùng ở training/diagnostic;
- V_bar là target value đã học từ actual trajectories, không lấy target từ
  imagined WM futures của chính nó ở bước đầu.

Các state chưa overlap vẫn có y khác nhau nếu xác suất thành công về sau
khác nhau. Credit truyền qua transition bằng Bellman/n-step targets hoặc
Monte Carlo returns, thay vì cần dense reward ở từng bước. Nếu gamma<1,
objective thêm ưu tiên hoàn thành sớm; phải nói rõ vì không còn đúng bằng
xác suất success không chiết khấu trong thời hạn.

CTA reader vẫn nhận (C,S,g,b), không nhận candidate actions trực tiếp.
WM vẫn dự đoán code từ context/action trước execution. Future images,
native success và V teacher được dùng để tạo training targets; không gọi
simulator hay đọc future thật khi CTA chọn action ở test.

### Dữ liệu sẵn có và còn thiếu

Feature cache giữ root, decision, t, cov8, done8, context và sibling end
features. Candidate 0 nối thành P0 trajectories; có thể bắt đầu học V trên
đó với root split, episode outcomes và timeout handling đúng. Phải kiểm tra
next-history alignment và phần episode bị cap trước khi dùng TD targets.

Các nondefault siblings chỉ có short branch, **không có terminal outcome
cho continuation tương ứng**. Không được gán outcome của candidate-0 root
cho mọi sibling. Dùng V bootstrap là estimate, rồi cần một tập nhỏ continuation
rollouts độc lập để đánh giá estimate đó. Nếu V cần hai frame tại endpoint,
cached nondefault futures chưa mặc định chứa đủ đúng hai frame như context;
phải chọn observation contract rõ hoặc bổ sung dữ liệu.

Để tránh target và evaluation tự xác nhận lẫn nhau, fit V theo train roots,
freeze trước held-out sibling audit, và đánh giá gain bằng continuation outcomes
thật. Một teacher fit tốt trên training states vẫn có thể bị max-over-K exploit.

## 6. Thử nghiệm kế tiếp nên tách từng câu hỏi

### A. Audit không train model mới

Trên train và dev hiện có, thống kê:

- all-zero cov8;
- exact nonzero-flat cov8;
- near ties với spread <=1e-3;
- informative banks;
- sự khác biệt theo phase/time trong episode;
- trên closed-loop dev, score spread và tỷ lệ CTA đổi candidate ở các bank
  coverage-flat, báo cáo mỗi controller trên visited-state distribution của nó.

Không diễn giải outcome của cả root là causal effect của một override tại
một bank; cần paired branching/intervention riêng để nói điều đó.

### B. Qualify learned value với actual futures

Đây là nhánh có điều kiện, không tự động là bước chạy kế tiếp. Phải dùng kết
quả continuation cũ làm prior bất lợi, profile chi phí và khóa budget trước;
không lặp lại một campaign terminal-success labels một mẫu/candidate. Khi
chưa có evidence đủ để resolve within-bank gaps, không diễn giải fit tốt
trên episode outcomes là value ranking đã được qualify.

Freeze policy, goal protocol và root splits. Học value bằng actual P0
trajectories với sparse success và terminal/budget handling, dùng các phase.
So reader trên actual future với P0, coverage reader và DIRECT value scorer.

Ở tập bank coverage-flat được chọn trước, đo ranking bằng replicated
continuations. Dùng seeds khác nhau cho selection và evaluation; báo cáo
uncertainty theo root. Không hard-rank hai candidates từ một sample success
duy nhất khi khác biệt nhỏ hơn noise.

Đánh giá thêm full dev distribution: sửa vùng zero nhưng làm hỏng final
alignment vẫn có thể làm giảm overall success. AUC/Brier/calibration trên
states không thay cho within-bank ranking và closed-loop test.

### C. Nếu B có tín hiệu, mới đo value qua CTA code/WM

Ladder mới: actual futures -> actual source code -> predicted code.
Train DIRECT bằng **cùng value targets, data và optimizer budget**.
So endpoint code và conditional per-frame code cùng bit budget.

Đây là đổi objective có chủ đích, không phải chỉ reader adaptation cho
predicted-code distribution. Phải ghi protocol trước khi chạy và tính vào
giới hạn debug rounds đang có; không âm thầm thay target trong Round 2.

## 7. Những cách sửa không đủ

- Dùng delta coverage: current coverage là hằng số chung trong bank.
- Dùng max/sum coverage trong chunk: mọi frame zero thì vẫn zero.
- Chỉ tăng K: thêm candidate có thể giúp, nhưng nếu tất cả vẫn zero thì
  label không có thứ tự; không đảm bảo giải quyết cost.
- Chỉ tăng prediction horizon: có thể gặp reward sớm hơn, nhưng model khó
  hơn, vẫn có delayed outcomes và chưa giải quyết logic nhiều bước.
- Tự thêm Euclidean distance: dense hơn nhưng có thể thưởng đi sai phía,
  bỏ contact feasibility, hoặc phạt bước lùi cần thiết.
- Dùng VLM chấm mọi frame: thay reward engineering bằng teacher có sai số;
  phải đo precision và ranking dưới candidate selection, không coi là oracle.
- Chỉ fallback về P0 khi “coverage=0”: test-time CTA không biết coverage của
  candidate chưa thực thi; learned gating cần calibration riêng, và chỉ giảm
  thiệt hại chứ không tạo progress knowledge.
- Chỉ tăng lambda co-design: làm target hiện tại dễ predict hơn không cung
  cấp thứ tự dài hạn còn thiếu; có thể nén mất thêm thông tin cần cho value mới.

## 8. Ý nghĩa cho contribution

Học long-horizon value không phải novelty của CTA. V-GPS và offline RL đã
cung cấp nền tảng này. Novelty phải nằm ở information/compute frontier của
conditional code so với direct value và frame/endpoint representations.

Nếu terminal objective và trạng thái endpoint Markov đầy đủ đã đủ xác định
value, whole path không bắt buộc về lý thuyết. CTA phải chứng minh lợi ích về
compression, partial observability hoặc nhiều query có phụ thuộc trajectory;
không mặc định “task dài” đồng nghĩa “phải nén whole trajectory”.

Đánh giá cuối: nên giải quyết target gap trước khi dựng nhiều arena mới.
Coverage vẫn là baseline hợp lệ đã cho tín hiệu PushT; nó không phải nền tảng
đủ để claim task-general progress. Round 2 tiếp tục là ablation hữu ích về
predictability, nhưng kết quả dương của nó sẽ không tự đóng lỗ hổng này.
