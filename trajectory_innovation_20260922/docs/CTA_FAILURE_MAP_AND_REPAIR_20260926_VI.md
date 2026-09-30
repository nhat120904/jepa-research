# CTA: lỗi đã đo được và hướng sửa có căn cứ

2026-09-26. Research assessment, không phải preregistration Round 3. Không
thay đổi model/job đang chạy. Đã đọc source, offline Round 2, CPU audit 54975
và primary literature. Tại lần kiểm tra squeue+sacct: audit 54975 COMPLETED;
54976_0 đang chạy preflight, task 1 chờ; closed loop 54977 và aggregate 54978
chờ dependency. Chưa có kết quả closed-loop Round 2.

## 1. Những điều đã biết

Nguồn: `cta_r2_compare_54934/comparison.json` và
`cta_r2_audit_54975/audit.json` dưới shared trajectory_innovation results.

| Thành phần | Bằng chứng | Kết luận được phép |
|---|---|---|
| Candidate bank trên PushT | W2, 400 roots: PHYS8 .765 vs P0 .625 | Có repeated-selection headroom dưới coverage oracle; bank không phải bức tường đã chứng minh ở đây |
| Code actual future, lambda=0 | FULL gap .785 -> CODE .706 | Code giữ được phần lớn khả năng ranking theo coverage; chưa chứng minh task-general abstraction |
| WM -> reader | CODE .706 -> greedy .293; expected .501 | Mất nhiều signal khi thay code thật bằng code dự đoán/decoding hiện tại |
| Greedy hard decisions | 71.9% label-distinct pairs có predicted code giống nhau ở panel 600 decisions | Với cùng C, reader không có cách phân biệt các cặp code giống hệt nhau |
| Co-design lambda=.1 | CODE .458, source perplexity 1.37 vs control 131.20; collisions 91.7% | Cấu hình này làm mất thông tin trong lúc giảm prediction loss; không là WM improvement |
| Coverage target | 48.1% train banks không vào rank pool | Supervision bỏ qua nhiều candidate differences; chưa đo causal harm trên terminal success |

Các gap là retained coverage gap, không phải success rate. Source retention
drop method-control -.248 [-.297,-.200] có CI rõ. Greedy drop -.050
[-.108,+.007] chưa có CI rõ. Không gọi toàn bộ CTA đã fail closed loop:
Round 1 66% vs P0 62%, paired CI [-5,+13] chưa quyết định được.

## 2. Audit coverage mới: không chỉ zero overlap

| Category | Train, 22,680 decisions | Dev, 3,011 decisions |
|---|---:|---:|
| Tất cả coverage bằng zero | 2,570 / 11.3% | 332 / 11.0% |
| Coverage khác zero nhưng bằng nhau chính xác | 7,649 / 33.7% | 981 / 32.6% |
| Có khác biệt nhưng spread <=1e-3 | 688 / 3.0% | 88 / 2.9% |
| Informative theo margin train | 11,773 / 51.9% | 1,610 / 53.5% |

Ở 64 bước đầu dev, all-zero là 34.6%, informative 46.1%. Đây là dữ liệu
P0-visited, không phải phân bố state của CTA closed loop.

Ngay ở all-zero dev, R1 greedy đổi khỏi P0 19.6%, expected reader 81.6%.
Điều này cho thấy learned scores không tự hòa khi target hòa. Không được
suy ra các override đó là đúng hoặc sai cho terminal success.

Sampling hiện tại dùng rank_pool cho source/reader/reconstruction cùng lúc;
WM học mọi decisions. Tách reconstruction ra mọi phase sửa exposure, không
tạo long-horizon ranking labels. Phải làm một ablation riêng với budget
khớp, không ghép nó vào sửa WM rồi mất khả năng attribution.

## 3. Chẩn đoán WM: chứng cứ mạnh và chỗ còn chưa biết

CodeWM hiện dùng 16-token autoregression; train nhận true prefix, inference
nhận prefix tự sinh. Sibling contrast đã có trong Round 1/2 và cũng dùng
true prefixes. Không đề xuất 'thêm action contrast' như thể chưa từng thử.

Ở control, privileged true-prefix reader gap .695 trên 600 decisions.
Greedy .293 là full-dev result, nên không so hai số như một paired effect.
Quan trọng hơn, true prefix chứa thông tin về actual future. Một prior không
dùng action cũng có thể hưởng lợi. Kết quả này **chưa phân biệt** exposure
mismatch, prior copying, hay thông tin không dự đoán được từ C,A.

Trước khi chốt sửa WM, cần diagnostic trên cùng panel (đề xuất, chưa chạy):

- WM và context-only prior, cả true-prefix và free-running, cùng reader.
- WM với action đúng và action hoán vị trong bank, giữ true target prefixes
  như nhau; xem prediction/ranking có thực sự dùng action.
- Tách collision, ranking và token errors theo vị trí code; free-running
  reader gap là metric chính, teacher-forced CE chỉ là diagnostic.

Không dùng true prefix ở test-time và không gọi diagnostic này là oracle
upper bound. Source-code .706 là empirical ladder tier, không phải trần
toán học cho mọi scorer học trên predicted evidence.

## 4. Hướng sửa ưu tiên

### A. Giữ code tốt; sửa predictor/decoding trước

Freeze encoder tốt của lambda=0 và reader trong phép thử WM. Giữ cùng target
codes, data, 16x256 nominal code space và update/compute budget. Một đối
chứng đáng thử là dự đoán song song 16 vị trí bằng learned queries cross-attend
C,A, không đưa true code prefix vào input. Mỗi vị trí học categorical CE;
giữ action contrast trên target codes của cùng bank để so công bằng.

Mục đích: loại bỏ dependency train-only vào prefix thật, tránh lỗi đầu chuỗi
truyền sang cuối, giữ nguyên discrete-code interface. Đây là hypothesis,
không phải fix đã có kết quả. Factorized marginals có thể làm mất dependency
giữa tokens, vẫn có mode collapse khi argmax, hoặc sinh tổ hợp code bất hợp
lý. Một masked/iterative predictor là phương án khác nhưng thêm complexity;
không thử tất cả trong debug round cuối.

True-prefix diagnostic phải được đối chiếu với prior trước khi coi đây là
nguyên nhân chính. Nếu nó không chỉ ra action-sensitive information thì
đổi decoder có thể chỉ sửa triệu chứng. Không tăng model size như mặc định.

### B. Reader adaptation là nhánh hẹp, không cứu hard-code collisions

Control expected-code reader .501 vượt matched DIRECT .388: paired gap
+.113 [.061,.168] offline. Đây là evidence có signal trong distribution
chưa được greedy khai thác, nhưng không đủ để suy ra closed-loop gain.

Nếu closed-loop CTA8E củng cố signal đó, có thể freeze encoder+WM, train reader
trên đúng predicted evidence dùng ở inference và so reader tiếp tục học
source codes với cùng update budget. Existing adapt() chỉ mix greedy codes;
muốn adapt CTA8E cần implementation khớp expected-code path rõ ràng.

E[D(S)] không bằng D(E[S]) với nonlinear reader. Expected-code hiện tại dùng
distribution dưới greedy prefixes, không phải exact joint posterior marginal.
Không gắn nhãn 'Bayesian expected value' cho nó. Multi-sample scoring đã thử
4 mẫu (.398 control), chưa tự vượt expected code và có chi phí cao hơn.

Soft vectors cũng đổi bandwidth contract: hard sequence 16x8=128 bits, còn
16x3 expected scalar values cần 48 floats nếu lưu/truyền trực tiếp (768 bits
ở fp16). Internal dequantization của hard code không làm tăng message bits,
nhưng truyền mean vectors thay indices thì có. Phải báo bytes/latency đúng;
không nhận gain của soft variant rồi claim nguyên xi hard-code 128-bit method.

### C. Sửa progress target là research branch riêng

All-phase representation exposure là sửa rõ ràng. Học value dài hạn có thể
tạo thứ tự ở plateau nhưng chưa được qualify, và GR00T/PushT single-chunk
continuation đã cảnh báo tín hiệu yếu. Không thay coverage bằng một terminal
label duy nhất rồi coi vấn đề được giải quyết. Cần actual-future/direct scorer
qualification và terminal-control evidence độc lập trước khi gắn vào CTA.

Không tự viết thêm weighted distance/grasp shaping cho từng task để che
target gap rồi claim task-general. Success predicates/demo goals vẫn là task
specification hợp lệ; học credit assignment từ chúng là vấn đề còn phải chứng minh.

## 5. Vì sao không tiếp tục sweep lambda hay chỉ ép dùng nhiều code?

Một code gần hằng số rất dễ dự đoán. Khi source được phép đổi để giảm NLL,
giảm information là một đường tắt có thể xảy ra. Reconstruction/ranking
terms hiện tại không ngăn được đường tắt ở lambda=.1. Đây là cơ chế phù hợp
với kết quả, chưa phải phân tích gradient chứng minh toàn bộ nguyên nhân.

FSQ có codebook cố định, nên lỗi này là encoder sử dụng quá ít giá trị, không
phải learned VQ codebook vectors chết. Chỉ thêm entropy/diversity có thể giữ
variation không liên quan action/task. Nếu quay lại co-design về sau, cần
giữ decision-relevant retention bằng teacher/constraint đã qualify và đánh
giá dưới free-running prediction; tăng entropy không là tiêu chí đủ.

## 6. Primary literature đã đối chiếu

- [FSQ](https://arxiv.org/abs/2309.15505): fixed scalar product codebook;
  kết quả của paper không là guarantee chống encoder collapse dưới objective
  co-design mới của repo.
- [Scheduled Sampling](https://arxiv.org/abs/1506.03099): mô tả mismatch giữa
  true-prefix training và generated-prefix inference. Có cùng dạng rủi ro
  trong code CTA, nhưng chưa chứng minh là toàn bộ nguyên nhân.
- [Huszar, 2015](https://arxiv.org/abs/1511.05101): phân tích inconsistency của
  scheduled-sampling objective; không khuyến nghị bật nó như fix mặc định.
- [MaskGIT](https://arxiv.org/abs/2202.04200): tiền lệ prediction song song và
  masked refinement trên image tokens, không phải bằng chứng nó giải CTA.
- [Value equivalence](https://arxiv.org/abs/2011.03506): động lực cho giữ
  thông tin liên quan quyết định; không cung cấp miễn phí value/progress target
  chính xác trong task của mình.

Đề xuất này giữ trọng tâm: bảo vệ code tốt, kiểm chứng prediction không có
true future inputs, rồi mới thay đổi reader hay task target. Nó không phải
cam kết kết quả dương hoặc đảm bảo CVPR acceptance.
