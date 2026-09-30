
## Đính chính (2026-09-28, sau khi đọc BDDL và code)

- Chỉ **13/45** task có vật cản thực sự di chuyển (TSA L1: 5, FSHOA L1: 5, HRI "banana ... in my hand" L0–L2: 3).
  Các vật cản `(:dynamics)` còn lại có `motion_travel_dist 0`, tức mocap đứng yên.
- Motion generator là class iterator thường (không có `yield`), `copy.deepcopy` sao chép được; không cần viết lại.
  Clone còn phải chép `model.body_pos/body_quat` (LIBERO đặt fixture bằng cách ghi vào model lúc reset).
- Số liệu π0.5 đã đọc được từ bảng trên trang dự án: violation rate chỉ khoảng 6–16% episode; phần lớn thất bại là
  không va chạm nhưng không hoàn thành. Chi tiết: `docs/CTA_LIBSAFE_PROTOCOL.md`.
