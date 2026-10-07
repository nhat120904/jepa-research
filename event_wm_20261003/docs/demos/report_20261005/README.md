# Bộ demo báo cáo Event WM — 05/10/2026

Mở `index.html` trong browser. Trang chạy offline khi giữ nguyên các thư mục asset.
Chọn **Chế độ báo cáo** để ẩn lời dẫn; các nút đổi event, task, bước prediction và tốc độ video vẫn dùng được.

## Kịch bản 4–5 phút

1. **Architecture — 30 giây.** SAM2 tạo pseudo-label offline; reader đọc trạng thái; event WM và heuristic phục vụ planner; skill thực hiện event. Các module train riêng theo môi trường.
2. **Event — 45 giây.** Cube: khối đỏ E2 từ phải sang trái. Nêu nhãn `(entity, target)` và khoảng event; phần trước/sau là ngữ cảnh. Chuyển sang puzzle: một thao tác có thể đổi nhiều đèn. Đây là nhãn suy luận, không phải GT đã xác nhận.
3. **Planning — 60 giây.** Chọn task 1 rồi task 2; nhấn từng event để đi qua trạng thái WM dự đoán. Chọn task 5 để cho thấy full plan chưa được tìm thấy. Những chấm trạng thái là prediction, không phải ảnh tương lai hay execution thực tế.
4. **Closed-loop — 30 giây.** Phát task 1: ảnh quan sát ở trái, goal ở phải, event được chọn ở thanh dưới. Replay CPU thành công 66 bước; reference GPU cũng thành công 69 bước.
5. **Failure — 60 giây.** Phát task 5 ở 4×; có thể nhảy tới các mốc replan. Sáu lần gọi planner không tìm full plan; fallback và thực thi chưa đạt goal. Chưa xác định một nguyên nhân duy nhất.
6. **Số liệu — 30 giây.** Reference GPU unified cube-triple 8/30, theo task 6/6, 1/6, 1/6, 0/6, 0/6. Đây là development result, một training seed. Nêu sự khác biệt với baseline về dữ liệu, SAM2 pretraining và evaluation.

## Các chú thích phải giữ khi dùng ảnh/video trong slide

- Event: **offline play, inferred labels**, không gọi là policy rollout.
- Planning: **WM-predicted entity state**, không gọi là rendered future observation hay control success.
- Video: **CPU replay, same checkpoint**, không cộng lại vào tỷ lệ GPU reference.
- Task 2 CPU thành công nhưng GPU reference thất bại: chỉ đặt trong phần bổ sung.
- Task 3/4: mới có planning trên ảnh reset, chưa có execution demo mới.
- Flag contamination và reader-space error không phải GT vật lý.
- Kết quả puzzle chuyên biệt 99/100 và 100/100 thuộc pipeline khác; trình bày riêng.

## Source và provenance

- Job mới: `57577`, main CPU, 2 CPU, 16 GB, time limit 20 phút; không GPU, không training.
- Replay gốc: `57497`; event/mask export gốc: `57539`.
- Source exporter: `source/export_report.py`; scheduler source: `source/demo.sbatch`.
- Checkpoint SHA256, nguồn reference, event metadata và các lệnh planner: `job57580/report_data.json`.
- Task 1/2/5 dùng first plan đã được ghi trong replay gốc. Task 3/4 chạy lại planning CPU, cùng checkpoint / seed formula / budget, chỉ reset environment và đọc ảnh.
- WM predictions được tính bằng checkpoint hiện có. Không fit model, không dùng trạng thái simulator privileged làm input.

## File để chèn vào slide

- `assets/architecture.png`
- `job57580/event_cube_before_after.png`, `event_puzzle_before_after.png`, `event_excluded_before_after.png`
- `job57580/event_cube.mp4`, `event_puzzle.mp4`, `event_excluded.mp4`
- `job57580/plan_task1.png` … `plan_task5.png`
- `job57580/rollout_task1.mp4`, `rollout_task5.mp4`; task 2 là phần bổ sung

Các video được resample / tăng tốc để trình bày, không dùng thời lượng playback để báo cáo tốc độ điều khiển thật.

Bản cuối đã xác nhận: job **57580** COMPLETED, 7 giây. Dùng một renderer OSMesa và LP_NUM_THREADS=1; ảnh initial/goal của task 3/4 đều đã kiểm tra hình ảnh. Các bản trung gian 57577/57578/57579 có ảnh reset lỗi ở một số task, không dùng trong báo cáo.
