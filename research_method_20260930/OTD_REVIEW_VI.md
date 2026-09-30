# Kiểm chứng phản biện OTD

Ngày: 30/09/2026. Nguồn đầu vào: bản phản biện người dùng đính kèm; source/ledger trong repo; paper và proceedings chính thức. Không có run mới.

## Quyết định

**Rút OTD khỏi vị trí khuyến nghị chính cho CVPR.** Đề xuất thay thế ở [PGC_RECOMMENDATION_VI.md](PGC_RECOMMENDATION_VI.md). Lý do là tổng hợp rủi ro: chưa có decision gap của student mạnh trong arena hiện có, thiếu prior art quan trọng trong lượt khảo sát đầu, và cần kiểm chứng gradient WM cùng tích hợp baseline/arena trong lịch quá ngắn. Không gọi một proposal chưa chạy là thất bại thực nghiệm.

## Những phản biện có cơ sở

- [OneDP, ICML 2025](https://proceedings.mlr.press/v267/wang25ba.html), [bản tác giả](https://arxiv.org/html/2410.21257), Table 1: OneDP-S gần teacher hoặc hơn teacher ở nhiều task. PushT dùng coverage, không phải native threshold-success của repo. Paper nói rõ Lift/Can đã bão hòa nên không đưa vào bảng. Khuyến nghị dùng các arena đó trước đây thiếu cơ sở.
- [ILD, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Imitation_Learning_As_State_Matching_via_Differentiable_Physics_CVPR_2023_paper.html) học bằng khớp state qua differentiable physics; Chamfer matching xét các thời điểm trong trajectory. Đây là prior art phải đối chiếu cho ý tưởng học từ hậu quả, dù không phải cùng conditional balanced teacher–student distribution.
- [IMLE Policy, RSS 2025](https://imle-policy.github.io/) là baseline một bước quan trọng về giữ mode; [FQL, ICML 2025](https://proceedings.mlr.press/v267/park25f.html) đã dùng distillation sang policy một bước với Q-guidance. Không thể đặt vấn đề như thể action-MSE là baseline tốt nhất.
- [Protocol PushT cũ](../trajectory_innovation_20260922/docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md) ghi 97,8% standard-policy sibling pairs có block displacement khác nhau dưới 1 pixel. Điều này làm arena đó kém thuyết phục cho outcome-distribution novelty.
- Loss so teacher future thật với student future dự đoán bị bất đối xứng. Với F là WM và y là future thật, ngay tại action teacher, gradient là 2 J_F(A)^T(F(A)−y), có thể khác 0. Loss có thể đẩy student bù sai số WM thay vì giữ hậu quả teacher.
- Repo chưa có một baseline OneDP đã tái lập. Có [implementation không chính thức](https://github.com/aminamazlin/onestep-diffusionpolicy), nhưng tác giả ghi configs chưa được thêm; không đồng nghĩa có một baseline sẵn sàng cho so sánh paper.
- Đường dẫn tuyệt đối tới máy Mac trong artifact gốc gây vấn đề khi đọc trên cluster. Các link nội bộ trong hai artifact gốc được chuyển sang đường dẫn tương đối; assistant vẫn dùng absolute links khi trả lời trên desktop.

## Những kết luận cần thu hẹp

- Không có định lý student không thể hơn teacher; chính bảng OneDP có một số trường hợp student hơn. Khoảng hụt nhỏ là rủi ro thực nghiệm, không phải chứng minh gain bằng 0.
- SD coverage của OneDP không trực tiếp cho sample size của native success trong repo. Ước lượng khoảng 25 training seeds/arm dựa trên giả định kiểm định độc lập, SD 0,05 và effect 0,04; paired variance và cách gom roots có thể làm khác.
- Số 97,8% thuộc bank/protocol cũ. Nó không chứng minh bank perturbation mới hoặc action của student có cùng đặc tính. Object gần đứng yên cũng không khiến outcome-distance bằng action-distance về toán.
- Decoder R² 0,35/0,47 là kết quả round cũ trong [debug log](../trajectory_innovation_20260922/docs/CTA_DEBUG_LOG.md), không phải xác nhận chất lượng decoder hiện tại. R² không so trực tiếp được với pixel displacement. Chưa đo gradient error để kết luận nó chiếm phần lớn loss OTD.
- DDIM 2/4 là đối chứng tốc độ có ích trong [protocol đã lưu](../trajectory_innovation_20260922/docs/FAST_POLICY_HEADROOM_PROTOCOL.md), chưa phải method learned mới. Tỷ lệ số denoising steps không bằng tỷ lệ latency toàn pipeline.
- Các budget tháng 10 dự phóng không thay thế sreport hiện tại. Lượt này không kiểm tra quota hoặc submit job, nên không khẳng định ngân sách còn dùng được.

## Điều sửa trong quyết định nghiên cứu

Không tiếp tục chọn method chỉ vì có loss đẹp và tài sản WM. Đề xuất thay thế dùng arena có khoảng hụt quyết định được ghi nhận, tích hợp trực tiếp vào planner đang chạy, không cần WM action-gradient hay tái lập một teacher mới. Đây vẫn là một giả thuyết phải so end-to-end với baseline mạnh, không phải bảo đảm có paper.
