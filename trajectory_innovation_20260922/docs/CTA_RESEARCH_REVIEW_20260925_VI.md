# Đánh giá CTA và quyết định Round 2 — 2026-09-25

## Kết luận

CTA hiện có bằng chứng tốt rằng **code của tương lai thật giữ được phần lớn
thông tin phục vụ ranking**, nhưng chưa có bằng chứng chắc chắn rằng code do
WM dự đoán giúp closed-loop vượt policy hoặc direct scorer. Can thiệp ưu tiên
cho Round 2 là co-design có đối chứng từ đúng checkpoint Round 1. Không có
cơ sở để hứa rằng bất kỳ can thiệp nào chắc chắn cho kết quả dương.

## Idea thực chất là gì

History tạo context C dùng chung. Source encoder chỉ dùng khi train, đọc C
và cả đoạn tương lai để tạo 16 code FSQ, tổng nominal 128 bit. WM đọc C và
chunk 8 hành động, dự đoán code trước khi thực thi. Reader đọc C, code và
ảnh goal, không nhận hành động. Vì WM không nhận goal, cùng code có thể
dùng lại cho nhiều truy vấn. Feature decoder giữ thông tin tương lai ngoài
một scalar task score; ranking reader hiện được train bằng coverage labels.

Đây là conditional trajectory abstraction: chỉ bắt code truyền phần thông
tin tương lai còn cần sau khi reader đã có C, đồng thời làm phần thông tin
đó dự đoán được từ action. Chỉ đạt success tốt trên PushT vẫn chưa đủ chứng
minh segment abstraction hay khả năng dùng lại cho nhiều goal.

## Chuỗi bằng chứng và những gì đã bị loại

- Gate A/B: runtime và bank policy có headroom; oracle tốt hơn P0.
- Gate C: khoảng cách SSL trên tương lai thật không giữ được headroom.
- C2: hindsight reader có tín hiệu ở lượt đầu nhưng không replicate rõ.
- S1: label-free sibling objective không học được; reader có coverage labels
  giữ được ranking tốt hơn. Vì vậy §5a cho phép task labels ở reader.
- W2, 400 root: reader trên tương lai thật tăng 8.5 pp [2.75,14.25]. Đây là
  tín hiệu của scorer có privileged future, chưa phải deployable CTA.
- Round 0: FSQ bão hòa và WM gần như không phân biệt action. Codec gap .462,
  greedy prediction .186, closed-loop CTA chỉ +1 pp.
- Round 1: saturation penalty và sibling contrast cải thiện codec lên .709,
  greedy .323, expected-code .454. Closed-loop greedy 66% so với P0 62%,
  chênh +4 pp [−5,+13]. Kết quả vẫn ở dev, một training seed.
- LIBERO L3: oracle dense progress trên chunk hiện tại không có closed-loop
  headroom (−1 pp [−10,+7]); chuyển CTA sang đó ngay không giải quyết được vấn đề.

## Vì sao ưu tiên co-design

Code thật giữ .709/.798 ≈ 89% gap của FULL, nhưng code dự đoán chỉ giữ
.323/.709 ≈ 46% (greedy), hoặc .454/.709 ≈ 64% (expected-code). Source codes
phân biệt 79% cặp siblings; greedy codes chỉ phân biệt 29%. Đây là động cơ
thử co-design, không phải chứng minh WM capacity là nguyên nhân duy nhất.

Adapt reader là can thiệp rẻ và hợp lý cho distribution shift. Tuy nhiên,
với cùng C và code giống nhau, mọi reader xác định phải cho cùng điểm.
Nó không thể tạo lại phân biệt action đã bị mất. Nếu co-design giữ được
thông tin mà chỉ làm reader lệch phân phối, adaptation sẽ là lựa chọn Round 3.

Không chỉ bật λ trên implementation cũ: loss khoảng cách tới expected
codeword không bằng categorical NLL và có thể kéo hai mode về vùng code
ít xác suất. Round 2 dùng nội suy NLL trên lưới FSQ, cùng ranking và feature
reconstruction. Đây vẫn là surrogate có bias, cần kiểm nghiệm.

Chi tiết đã khóa trong [CTA_ROUND2_PROTOCOL.md](CTA_ROUND2_PROTOCOL.md):
λ=0 và .1, cùng checkpoint 54717, 3k source updates + 6k WM updates + 2k
WM alignment với source cuối. FULL và DIRECT cũng được thêm 3k updates.
Không sweep, không chọn checkpoint đẹp nhất, không đọc sealed roots.

## Những cách đọc kết quả cần tránh

1. **Gap 2→3 không đồng nghĩa chỉ WM bị lỗi.** Sai số dự đoán, code trùng,
   teacher-forcing mismatch và reader distribution shift cùng tác động.
2. **CE tốt hơn không đủ.** Code ít thông tin cũng dễ dự đoán. Phải giữ
   tier-2 ranking và feature reconstruction, đồng thời tăng tier-3 ranking.
3. **pred_soft không phải kỳ vọng của score.** D(E[S]) khác E[D(S)]; bản
   hiện tại còn conditioning trên greedy prefix. Đây là readout riêng.
4. **Code prior CE trừ WM CE không phải đo chính xác mutual information.**
   Cả hai model đều hữu hạn và có sai số fitting.
5. **FULL không phải trần vật lý cứng của toàn episode.** Nó là reference
   dùng tương lai thật và một reader/myopic horizon cụ thể; mỗi controller
   tạo phân phối state khác nhau.
6. **DIRECT Round 0 không thay được DIRECT matched Round 2.** Chênh lệch
   giữa hai training run trên cùng dev roots không thể giải thích chỉ bằng
   việc có 100 root. Cần kiểm tra training/numerics và paired comparisons.
7. **Thêm vài trăm root không bảo đảm CI sạch.** Power phụ thuộc số root
   hai controller bất đồng và effect thật. Với tỷ lệ bất đồng khoảng .2–.3,
   effect .04–.05 có thể cần khoảng 600–1,500 root để đạt gần 80% power
   theo xấp xỉ chuẩn, chứ không mặc định 400 là đủ. Phải tính lại từ paired
   discordance trước confirmation; không mở test dần đến khi p nhỏ.

## Kết quả nào sẽ có ý nghĩa với method

Co-design chỉ được credit nếu vượt continuation λ=0 ở predicted-code
retention mà không làm hỏng code thật. Sau đó cần so hai model cố định trên
cùng dev closed-loop roots, rồi khóa cấu hình và confirm ba training seeds.
Các CI trên dev đã được dùng để phát triển method, không phải confirmatory CI.

Để claim về CTA mạnh hơn một scorer thành công trên PushT, còn phải có:
conditional per-frame code với cùng bit budget; path-vs-endpoint; reader có
và không có C; nhiều goal/query có held-out thật; latency đầy đủ gồm policy,
encoding và reader, với bank/compute matched. Ảnh khác nhau của cùng goal
PushT không tự tạo thành bài toán generalization sang goal mới.

Những yếu tố nền đã có trong [FSQ](https://arxiv.org/abs/2309.15505),
[GCQ](https://arxiv.org/abs/2510.16039) và
[DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html). Điểm CTA cần
chứng minh là sự kết hợp conditional segment code, retention cho truy vấn
và predictability mang lại lợi ích đo được so với các đối chứng thích hợp.
