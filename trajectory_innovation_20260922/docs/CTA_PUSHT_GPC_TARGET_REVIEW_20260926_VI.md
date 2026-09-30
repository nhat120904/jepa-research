# PushT vẫn là phép thử CTA hợp lệ: cần audit target trước khi train tiếp

2026-09-26. User yêu cầu đối chiếu GPC, arXiv 2502.00622. Đây là source review
và thứ tự công việc đề xuất, không phải kết quả relabel/oracle/train mới.

## Điều chỉnh định hướng

CTA không chỉ dành cho task có history-dependent progress. S có thể giữ endpoint
information khi đó là thông tin cần cho quyết định. PushT kiểm tra conditional
compression/predictability/control-cost; task temporal kiểm tra khả năng giữ thông
tin giữa đoạn. Không yêu cầu S phải dùng hết trajectory trên mọi task.

Do đó không cần bỏ PushT hoặc chờ RinseBowls qualify mới tiến bộ được. Ưu tiên
tiếp theo trên PushT là kiểm tra scorer/target lấy từ baseline có sẵn. RinseBowls
vẫn là ứng viên cho temporal claim; chưa mở training hoặc task collection ở đó.

## Phát hiện từ paper và official code

Paper v4 (2026-03-11), §V-A/B: PushT uses vertex-registration distance; image
reward is learned. Table III: vision DP score .642; RANK K=100 score .739;
decision times .457 vs 11.745 seconds. These are reported scores, not our success
rates or a matched speed comparison. §III reports prediction horizon 16 and action
horizon 9. WM training also uses exploration data.

Official release `gpc_rank_evaluation/eval_baseline.py`:
`estimate_reward_torch` computes 0.01 times summed corresponding-vertex distances;
inference decodes xy and angle from the final predicted image and minimizes cost.
The checked script sets K=50, uses a 15-action future suffix, and executes 9 actions
with the checked config despite `action_horizon: 8`. Environment evaluation uses
normalized coverage; the script records maximum episode reward. Pin and audit the
release before attempting paper reproduction; do not treat all settings as identical.

Nguồn:
- [Paper v4](https://arxiv.org/pdf/2502.00622v4).
- [Official evaluation](https://github.com/han20192019/gpc_code/blob/main/gpc_rank_evaluation/eval_baseline.py).
- [Official config](https://github.com/han20192019/gpc_code/blob/main/gpc_rank_evaluation/configs/gpc_rank_evaluation_config.yml).
- [Official environment](https://github.com/han20192019/gpc_code/blob/main/gpc_rank_evaluation/pusht_env.py).

## Ý nghĩa đối với coverage plateau

Hai vật chưa overlap goal vẫn có thể khác độ lệch hình học. Registration target
có thể phân biệt nhóm này, trong khi coverage không thể. Nếu mọi candidate chỉ
di chuyển agent và vật không đổi, cả hai target vẫn có thể hòa. Registration còn
có thể ưu tiên một bước tiến gần ngắn hạn gây khó cho bước sau. Không gọi đó là
value tối ưu hoặc một cách sửa toàn bộ credit assignment.

Target là lựa chọn task-specific, không phải novelty của CTA. Giữ cùng target
cho CTA và các baseline để lợi ích có thể quy cho representation/prediction.
Không dùng thêm số hạng contact/distance/stage tùy biến để cứu một kết quả âm.

## Bước kế tiếp có chi phí thấp nhất

1. CPU audit trên raw collection CTA đã có. `scripts/d_collect.py` lưu `phys8`;
   `physical_state` đặt block xy/angle ở các cột 4:7. Feature-cache `meta.npz`
   không có trường này nên phải đọc raw shard, chỉ load các array cần thiết.
   Dùng native geometry và goal của gym-pusht đã pin; kiểm tra vertex transform
   khớp simulator trước khi tính labels. Không sao chép mù vertices/rotation của
   GPC vì hai implementation có local geometry/frame conventions khác nhau.
2. Tách all-zero coverage, nonzero flat và informative banks. Đo registration
   spread, block-pose spread, agreement/disagreement với coverage. Định tolerance
   vật lý trước khi đọc kết quả; số khác nhau do floating-point không phải headroom.
   Không suy ra lợi ích closed-loop từ việc nhãn có spread hoặc immediate regret.
3. Nếu có phần tín hiệu mới đáng kể, chạy một pilot oracle repeated-selection
   có cap riêng: P0 / coverage oracle / registration oracle, cùng K=8 và execute8.
   Native success và episode coverage vẫn là outcome; không đổi metric cuối sang
   registration để tự chứng minh target vừa chọn tốt. Chưa có pilot này.
4. Nếu oracle có ích, qualify reader trên true endpoint với nhãn đó, rồi mới
   dùng source code và WM. Source phải giữ ranking target mới; predictor và reader
   phải được đánh giá lại. Không chỉ đổi reader trên code cũ rồi mặc định target
   mới đã được giữ. Giữ all-phase data, không ép preference cho label ties.
5. So sánh P0, DIRECT, endpoint/compact-frame WM và CTA cùng bank, context,
   supervision và horizon. Báo cáo native success, ranking, latency toàn hệ thống.
   GPC-RANK là system reference; không lấy số paper làm matched baseline và không
   credit lợi ích K=100 cho CTA K=8.

Chỉ audit horizon dài hơn nếu H8 còn không phân biệt candidate có ích. Có thể
look ahead qua phần còn lại của **cùng proposal đã có**, vẫn thực thi 8 rồi replan;
đây khác chạy policy tiếp đến terminal. Hiện PolicyRunner cắt suffix thành 8, nên
dữ liệu hiện có không đủ để đánh giá H15/H16: cần collection version mới và clone
checks. Không tự nâng execute horizon hay nối tail dài mà không khóa protocol.

Không bật lại co-design lambda=.1, tăng bank, đổi target và horizon cùng một lúc.
Round 2 source collapse và free-running WM collisions vẫn là vấn đề riêng; target
mới không tự chữa chúng. Sau khi target được qualify, ladder quyết định sửa codec,
WM hay reader, thay vì mặc định WM là nguyên nhân duy nhất.

## Phạm vi kết luận

W2 đã xác nhận coverage oracle có headroom ở operating point hiện tại; không
phủ nhận kết quả đó. Phát hiện GPC tạo thêm lý do cụ thể để kiểm tra target trên
PushT trước khi chuyển arena. Nó chưa chứng minh registration oracle tốt hơn,
CTA sẽ thắng, query reuse hữu ích trên một fixed goal, hoặc đã reproduce GPC.

Vòng source review này không submit compute mới.
