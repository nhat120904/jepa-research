# Kiến trúc tiếp theo: quyết định theo lỗi, không đổi module hàng loạt

2026-09-22. Đây là kế hoạch có điều kiện, không phải kết quả thực nghiệm mới.

## Điều đã biết

53698: ở H48, compact16 có normalized bin MSE .533 so với .881 của no-action;
shuffle action làm lỗi tăng lên 1.009. Model có học thông tin action. Tuy nhiên
A-before-B MSE .03211 gần no-action .03175; selection regret .13713 còn tệ hơn
.02369. Không được gọi toàn bộ predictor là action-blind.

No-action có thể hưởng lợi từ tie-breaking: tất cả candidate cùng score thì argmax
chọn candidate đầu, cũng chính là default mạnh. Diagnostic mới báo cả uniform ties.
Các số trên là score/regret theo query ảnh, KHÔNG phải success rate vật lý.

Target fixed-bin của tương lai quan sát thật giữ query tốt, nhưng điều đó chưa chứng
minh action-conditioned predictor có thể dự đoán target hoặc dùng nó để control.
24 prefix train và 12 prefix validation chưa đủ chốt khả năng của cả lớp method.

## Thử ngay: frozen-predictor readout diagnostic

Xem READOUT_DIAGNOSTIC_PROTOCOL.md. Không train lại WM. So sánh kernel cũ, reader học
từ observed future, reader học từ frozen forecast. Tách missed visits và timing/order.
Giữ matched frame và no-action controls. Không dùng validation để fit reader.

Đây vẫn là supervision tự sinh từ RGB với thiết kế affinity theo task, không có
simulator state/reward. Nó chưa chứng minh một biểu diễn self-supervised task-agnostic.

## Nếu phép thử này không đủ: ba nhánh sửa khác nhau

### A. Latent có tín hiệu, nhưng readout không chuyển được từ thật sang dự đoán

Dấu hiệu: reader trên observed future tốt; reader source-adapted khôi phục query và
ranking trên readout holdout/validation, tốt hơn no-action, không chỉ giảm train loss.

Thay đổi tối thiểu: train readout trên cả observed và imagined summary; đưa query loss
làm mục tiêu chính thay vì bắt model khớp mọi tọa độ latent như nhau. So sánh cùng
supervision/budget: summary, full-frame, direct-query; mỗi loại với và không latent
alignment loss. Mọi dự đoán tương lai chỉ nhận history + proposed actions, không goal.
Query được cấp ở readout để summary có thể dùng lại.

Không chỉ tăng kernel bandwidth: với cùng global bandwidth và score chỉ gồm max/min,
đó là một phép biến đổi đơn điệu chung nên giữ nguyên thứ hạng (trừ sai số số học/ties).
Nó có thể đổi calibration, không tự sửa ordered selection. Điều này không áp dụng
cho mọi query dạng trung bình hoặc tổng.

### B. Visit được đoán đúng, nhưng thứ tự/thời điểm sai

Giữ temporal bins làm baseline. Thử summary gồm token nội dung + khoảng thời gian,
với supervision thời gian lấy từ video; ưu tiên giữ sự kiện ngắn thay vì chỉ average
pool. Chỉ thêm nhánh này khi error decomposition cho thấy timing thực sự là lỗi chính.
So sánh với tăng số bin ở cùng số byte/compute; learned token không tự là novelty.
Không đưa lại learned composer hay Chen product.

### C. Forecast không giữ được event, kể cả sau readout adaptation

Đây là nhánh nên ưu tiên nếu diagnostic thất bại, nhưng thất bại của một MLP không
chứng minh latent tuyệt đối không chứa thông tin.

1. Kiểm tra predictor ở horizon 4/8/16/32: input observed khi teacher forcing so với
   tự rollout. Nếu bước ngắn đã sai, đừng đổ lỗi cho long-horizon composition.
2. Dùng predictor chuyển tiếp ngắn giữ spatial patch tokens làm baseline năng lực,
   thay vì chỉ dự đoán thẳng toàn đoạn qua bottleneck hiện tại. Đây là giả thuyết
   sửa inductive bias, không phải đã chứng minh bottleneck gây lỗi.
3. Nếu train tốt/validation kém, đo learning curve theo số PREFIX độc lập (ví dụ
   24/96/192), không chỉ tăng window hay candidate ở cùng prefix. Chưa tự động thu
   thêm dữ liệu trước khi xem diagnostic.
4. Chỉ khi full-frame reference dự đoán được events, mới train jumpy summary để giữ
   đáp án nhiều query với ít bộ nhớ/compute hơn. Dùng local/multihorizon latent loss
   như auxiliary, và luôn có ablation để biết lợi ích đến từ đâu.

Sơ đồ giả thuyết (không phải code đã chạy):

    history + actions -> local patch transition model -> predicted frame latents
                  \-> jumpy summary predictor --------> compact temporal summary
                                     queries + readout -> answers/ranking

Full-frame branch là đối chứng/teacher tùy thí nghiệm, không bắt buộc chạy đồng thời
khi inference của summary. Không train policy/VLA. Chưa thay proposal hay chạy MPC.

## Cơ sở literature và giới hạn novelty

- [DINO-WM](https://arxiv.org/abs/2411.04983) dự đoán spatial DINOv2 patch features
  từ trajectories offline và action, không cần reward model. Đây là tham chiếu cho
  baseline giữ cấu trúc không gian. Code predictor DINOv3 hiện tại của repo KHÔNG phải
  reproduction đầy đủ baseline DINO-WM; dùng environment gốc không làm nó thành baseline đó.
- [PlaNet, ICML 2019](https://proceedings.mlr.press/v97/hafner19a.html) đưa ra latent
  overshooting, một objective biến phân nhiều bước. Có thể mượn nguyên lý kiểm tra
  transition ở nhiều horizon; không gọi một loss MSE mới là cùng thuật toán. PlaNet
  gốc dự đoán reward, còn nhánh đề xuất ở đây không thêm reward supervision.
- [RaMP, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/b048dd19ba6d85b9066aa93b4de9ad4a-Paper-Conference.pdf)
  đã học cumulative feature predictions có điều kiện theo state và action sequence.
  Vì vậy action-conditioned summary tự nó không mới. Cần matched additive/multi-bin
  feature baseline; phần phân biệt cần đo là query có thứ tự, không cộng được, và
  reuse cho query chưa dùng thiết kế proposal. Chưa có bằng chứng điều đó thắng.

Các thay đổi A/B/C là engineering để có phép thử hợp lệ, chưa tự tạo thành contribution.

## Điều kiện để trở lại hướng method paper

- Query anchors độc lập với A/B dùng sinh proposal; held-out anchor pairs/time windows.
  Hiện 8 outputs quanh cùng A/B chưa chứng minh general query reuse.
- Matched frame/direct predictors dùng cùng dữ liệu và auxiliary supervision.
- Report theo prefix, vài seed, tập test chưa dùng thiết kế; byte và latency gồm đọc
  history/cache. Không dùng token count làm đại diện duy nhất cho compression.
- Selection trên cùng candidate bank: oracle support đo trực tiếp, uniform ties,
  default, no-action và direct scorer. Chỉ sau đó mới kiểm tra closed-loop control.
- RQ giữ lại: summary nào dự đoán trước được đáp án nhiều future queries không cộng
  được, với accuracy/compute tốt hơn matched frame/direct alternatives?

Một vòng sửa có kiểm soát; không cam kết mọi nhánh A/B/C đều được chạy. Nếu matched
frame predictor cũng không qua được event/ranking gate, tạm dừng claim abstraction và
giải quyết năng lực baseline/dữ liệu. Nếu frame/direct đã tốt nhưng summary không có
lợi thế, đóng kiến trúc summary này; không diễn giải thành toàn bộ predictive
abstraction bất khả thi, và cũng không tiếp tục đổi arena để tìm số dương.
