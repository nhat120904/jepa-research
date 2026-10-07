# Khuyến nghị cho learned events từ offline play

Ngày 06/10/2026. Phân tích thiết kế, không phải kết quả thử nghiệm hoặc cấu hình đã được triển khai. Không thay source/config của các run STATE đang chạy.

## Mục tiêu và lựa chọn

Mục tiêu là discover các tương tác object có thể thực thi lại từ video + action offline, không có event label hay danh sách discrete actions cho sẵn. Một event cần giúp WM dự đoán trạng thái sau, giúp skill thực thi, và tạo không gian ứng viên hữu hạn để search nhiều bước. Video segmentation đẹp hoặc action reconstruction thấp riêng lẻ chưa chứng minh đạt mục tiêu này.

Tôi chọn object-centric event discovery được ràng buộc bởi cả prediction và control. Bắt đầu từ proposals/tracks của SAM2 hiện có, học representation thay centroid/RGB và học segmentation/termination; không bắt đầu bằng việc huấn luyện lại toàn bộ object discovery từ đầu. Pretraining SAM2 là prior bên ngoài, phải công khai và dùng cùng frontend trong đối chứng attribution. Không dùng simulator object labels, tên cube/button hoặc ánh xạ cột observation làm đầu vào pipeline ảnh.

| Hướng | Vì sao chưa chọn làm đường chính ngay |
|---|---|
| Mask biến đổi ảnh kiểu EAWM | Tín hiệu hỗ trợ học trạng thái; chưa cho object identity, interaction hoàn chỉnh hoặc controller |
| Skill latent trên toàn scene với đoạn cố định | Có thể mã hóa arm/wander, phụ thuộc bố cục; đoạn dễ cắt giữa press/place. Cần đối chứng nhưng thiếu object anchoring cho hướng hiện tại |
| Object + segmentation học từ prediction/control | Phù hợp event WM hiện tại, giữ mục tiêu cục bộ để skill thực thi và chỉ học effects của toàn scene trong WM |

## Bằng chứng đã có và giới hạn

- `JOB_LEDGER.md:761–774`: job 57202, compact SlotContrast-style K=8, clip 4 frames, 10k updates, không bind được cube; một slot chiếm phần lớn ảnh. Đây là thất bại của cấu hình ngắn này, không chứng minh SlotContrast nói chung không dùng được.
- `JOB_LEDGER.md:793–798`: SAM2 video propagation mất object sau occlusion, cube tracked khoảng .59–.67 số frame; puzzle event recall .47 trong pilot. Chọn SAM2 bootstrap không có nghĩa tracking đã được giải quyết.
- `u_events.py` hiện ghép các thay đổi chồng nhau, gán acted object ưu tiên object trung tâm nhóm thay đổi rồi xét gần agent. Giả định tâm effects trùng cause không đủ chung. Cần học phân biệt acted object/side effects từ lịch sử tương tác; trường hợp không rõ cần giữ uncertainty.
- `u_wm.py` có state D=6 và identity embedding theo K; `u_skill.py` có cùng identity embedding. Frontend latent mới cần cập nhật interface, không thay encoder là xong.

## State và identity cần phục vụ các mục đích khác nhau

Đề xuất token gồm feature identity để association, feature trạng thái để planning, thông tin vị trí/mask và visibility/confidence. Identity phải ổn định khi vật di chuyển hoặc đổi màu; state phải thay đổi khi điều kiện điều khiển thay đổi. Cần giữ memory qua occlusion và association giữa observation với goal image. Không dùng ordinal slot index như một tên object toàn cục.

[SlotContrast](https://arxiv.org/html/2412.14295) dùng temporal contrast và feature reconstruction cho object consistency; đây là tham khảo cho identity branch. Việc tách identity/state thành hai head và giữ state sáng/tắt, depth, fixture position là đề xuất cho bài toán này, chưa được kiểm chứng ở OGBench.

Planning state nên compact và có cơ chế nhận lại trạng thái tương đương. Có thể học prototypes/quantization cho các chế độ lặp lại, giữ thông tin liên tục cho geometry. Không ép mọi object thành binary bits hoặc chỉ RGB; không coi latent distance nhỏ là đủ để kết luận goal vật lý đã đạt.

## Event record khác event command

Event record để train gồm: khoảng tương tác, trạng thái trước và sau của mọi object, action segment, acted-object posterior và confidence. WM được train dự đoán full successor từ trạng thái trước và command.

Command đề xuất: `e = (object pointer i, interaction code c, local target q)`.

- `i` chọn token/mask trong scene hiện tại, không phải tên semantic hoặc một index cố định trên benchmark.
- `q` là mục tiêu cục bộ về vị trí/trạng thái hoặc quan hệ của object, không phải toàn bộ ảnh/state sau. Dùng target đạt được trong offline segment là hindsight supervision hợp lệ nếu planner có thể tạo cùng loại target lúc inference.
- `c` là code compact học từ interaction/action segment, giúp phân biệt các cách tác động có cùng object/target. Không đặt sẵn nhãn press/pick/place. Không ép codebook có số entry bằng số nút. Nếu `(i,q)` đã đủ, code cần được phép không thêm thông tin thay vì bắt model phát minh các loại vô ích.
- Không cho encoder command nhìn full-scene successor rồi nén tất cả side effects vào c. Giới hạn posterior vào action và thông tin agent/acted object cần thiết, dùng bottleneck và prior từ trạng thái hiện tại. Đây là hạn chế nguy cơ shortcut, không phải bằng chứng causal identification.

Ví dụ mô tả để hiểu, không phải nhãn training: yêu cầu tác động object i bằng code c tại vùng q; WM dự đoán object i và các object lân cận thay đổi thế nào. Với một lần chuyển vật, q xác định đích cục bộ; WM vẫn phải dự đoán che phủ, va chạm hoặc ảnh hưởng lên object khác. Skill nhận ảnh/history hiện tại và command, trả action chunks.

## Học boundary từ offline video/actions

1. Dùng lịch sử visibility và thay đổi object state để khởi tạo các khoảng tương tác với confidence. Thay đổi mask/pixel chỉ là một tín hiệu; che khuất không đồng nghĩa rest. Quy trình chung dùng mọi family, không có quy tắc riêng cho Lights Out hoặc stacking.
2. Train event encoder, event WM và BC skill trên các khoảng đó. Giữ các đoạn approach–interaction–release đủ để skill kết thúc và reader thấy successor; phân biệt khoảng object thay đổi với toàn đoạn controller cần thực thi. Không tự động gán toàn bộ phần robot đi lang thang thành interaction.
3. Học/refine boundary sao cho đoạn vừa mô tả được bằng một command compact, vừa dự đoán đúng endpoint, vừa tái tạo được hành động với policy có conditioning đúng. Dùng proposal confidence/rest như prior mềm; không đóng băng toàn bộ pseudo-boundaries rồi gọi đó là learned segmentation.
4. Tradeoff đề xuất: prediction endpoint, BC, chi phí số boundary/độ phức tạp code và hỗ trợ identity/state/visibility đều cần được xét. Không tối ưu joint tất cả gradients mặc định: cập nhật ngày 07/10 theo OPOSM, WM loss có thể đẩy event code mã hóa uncontrolled outcomes. Cần phân biệt event inference theo policy/prior với dynamics fitting, rồi kiểm tra predicted successor sau khi skill thực thi. Chỉ giảm số event sẽ gộp cả episode; chỉ giảm prediction loss có thể cắt mỗi frame. Chỉ BC loss có thể học wander hoặc bỏ qua command.
5. Offline posterior có thể nhìn toàn segment để gán event; termination dùng ở runtime chỉ nhận history hiện tại và command. Cần train/distill causal termination riêng, không gọi future frames lúc inference. Khi visibility chưa đủ để xác nhận endpoint, cần tiếp tục quan sát hoặc recovery theo budget.

Đây là định hướng objective, chưa chốt optimizer, trọng số hay codebook size. Không có bảo đảm từ offline reconstruction rằng skill sẽ thực thi tốt ở states mới: closed-loop là phép kiểm tra cuối cùng.

Cập nhật 07/10: xem [cân nhắc global latent và ràng buộc causal](GLOBAL_LATENT_ASSESSMENT_20261007.md). Object-centric là lựa chọn representation, không phải điều kiện bắt buộc hoặc bằng chứng novelty. Chưa có matched end-to-end comparison để kết luận nó tốt hơn global latent.

[LOVE](https://arxiv.org/html/2212.04590) là prior art quan trọng cho learned boundaries và skills từ offline state/action sequences: compression giúp tránh các decomposition vô ích dù likelihood tốt. Chính paper vẫn nêu dữ liệu offline nhiễu/không hữu ích là thách thức. Vì vậy không copy nguyên objective rồi coi atomic object interaction đã được học; thêm object grounding và endpoint prediction là đề xuất của mình.

## Sinh candidates và kiểm tra khi triển khai

Inference không chạy posterior cần future. Sinh candidates từ các code/local targets đã học trong TRAIN, goal-object matches và prior conditioned on current state. Search chỉ thử tập hữu hạn có data support; không tối ưu latent tùy ý rồi coi output WM là executable. Support prior cho biết phù hợp dữ liệu, chưa phải calibrated probability of execution success.

[Hi-LeWM](https://arxiv.org/html/2607.12547) cho thấy search có thể chọn macro-actions có dự đoán hấp dẫn nhưng subgoal khó thực thi khi lệch support. Kết quả này là động cơ kiểm tra candidate support, không chứng minh mọi learned latent action đều thất bại.

Run end-to-end đầu tiên cần log cùng lúc object identity, state trước–sau, boundary, command, predicted successor và actual successor sau skill. So sánh trên held-out episodes/layouts với cùng dữ liệu và frontend: đoạn cố định so với learned boundary; `(i,q)` so với thêm c nếu c được dùng; backend theo timestep/subgoal so với event WM để đo đóng góp temporal abstraction. Đây là những đối chứng tập trung, không yêu cầu một sweep lớn trước khi triển khai.

## Vị trí nghiên cứu

[OPOSM, ICML 2023](https://proceedings.mlr.press/v202/freed23a.html) đã học skills cùng skill-conditioned temporally abstract WM hoàn toàn offline. Vì vậy riêng "offline skill + macro WM" không đủ mới.

Claim cần đánh giá của mình cụ thể hơn: discover object-grounded executable events từ pixel play, tạo action vocabulary hữu hạn, dự đoán state-dependent side effects, rồi search/replan cho các task combinatorial dài. Tính mới chưa được xác nhận đầy đủ bằng review này. Phải có matched baselines và held-out learned closed-loop, không chỉ offline event purity hoặc reconstruction loss.
