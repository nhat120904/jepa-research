# Event extraction từ pixel (scene memory v2): tình hình ngày 08/10/2026

Phạm vi: front end trích event từ offline pixel play (OGBench visual scene, cube-triple, puzzle-4x5), theo hướng đã
thống nhất "learned scene representation + rule-based event boundaries". Tất cả số liệu là **phát triển trên VAL**
(100 episode/family), chấm bằng **state privileged của simulator** (chỉ để chấm, không đưa vào training), **một seed**,
chạy trên máy local. **Chưa có closed-loop**. Chi tiết lệnh, đường dẫn run và số đầy đủ: `JOB_LEDGER.md`, mục
"Scene memory v1" và "Scene memory v2".

## 1. Câu hỏi

Từ ảnh 64×64 và action của play data (đúng những gì mọi phương pháp OGBench có; không proprio, không nhãn), có tạo được:
- một trạng thái cảnh rời rạc **không chứa cánh tay**, giữ được trạng thái vật kể cả khi bị tay che;
- **thời điểm event** (vật đổi trạng thái do tương tác), dùng cùng một bộ hyper-parameter cho cả ba family?

Thước đo: 4 tiêu chí đã pre-register ngày 04/10/2026 cho state component (c1 trạng thái rời rạc ≥ .98; c2 độ đầy đủ
keypoint ≤ 1.2× pixel; c3 agent-free arm R² ≤ .2; c4 event recall ≥ .9 và precision ≥ .8 trong cửa sổ 15 frame, định
nghĩa tham chiếu như `slowmap.py`).

## 2. Phương pháp hiện tại (v2)

| Thành phần | Học hay luật | Dữ liệu | Vai trò |
|---|---|---|---|
| AgentNet | học | frame + action | dự đoán frame kế tiếp có/không có action; vùng thay đổi mà action giải thích được = agent (phiên bản pixel của luật agent trong `g_entities.py`) |
| Teacher | luật chung | đầu ra AgentNet | token đổi pixel và action giải thích phần lớn thay đổi (ngưỡng 2-means tự fit: .31/.31/.33 ở ba family), nở theo vùng đổi liền kề |
| Segmenter | học | frame | mask agent từ một ảnh (dùng được cho ảnh goal) |
| SeeThrough | học, tự giám sát | frame | đọc mã cảnh **phía sau** cánh tay trong suốt và bóng, kèm xác suất; nhãn lấy từ chính dữ liệu: lượt tay che mà mã trước và sau giống nhau |
| Memory | luật | mã + xác suất / mask | memory "tươi": nhận mã SeeThrough khi p > .5 trong 3 frame liền (đọc trạng thái); memory "xác nhận": chỉ nhận khi tay đã rời (event) |
| Event | luật | memory | thay đổi đã xác nhận, lùi thời điểm về lúc memory tươi lần đầu thấy giá trị mới |

Mã cảnh: mã FSQ 5^6 theo patch 4×4 học bằng autoencoder; đối chứng là token pixel thô (màu trung bình 8 mức). Code mới:
`scripts/sm2_model.py`, `sm2_train.py`, `sm2_diag.py`, `local/run_sm2.ps1`, `tests/test_sm2.py` (9 test pass). v1 giữ nguyên.

## 3. Kết quả

| VAL | c1 trạng thái rời rạc | c3 cánh tay (R² max) | c4 event, cửa sổ 15 (rec / prec) | event, cửa sổ 30 | thay đổi giả lúc vật đứng yên |
|---|---|---|---|---|---|
| **scene** | nút min .956 (frame ổn định: min .934, TB .965) — chưa đạt | .12 — đạt | .893 / .833 — thiếu .007 recall | .968 / .976 | 12% frame |
| **cube** | không có nút | -.11 — đạt | .950 / .667 — chưa đạt | .988 / .996 | 9% frame |
| **puzzle** | đèn min .815 (ổn định: min .908, TB .989) — chưa đạt | -.03 — đạt | .927 / .961 — **đạt** | .990 / 1.0 | 15% frame |

c2 (độ đầy đủ của cube) **chưa chấm được**: probe keypoint cũng sai 7–15 cm trên pixel thô (lần đo 57184 là ~4 cm),
nên probe không đáng tin.

So với các front end trước (cùng định nghĩa event và cửa sổ 15 với slow map):
- v1 (gate L0 học được, 4 cấu hình): memory hoặc đóng băng, hoặc ghi lại mọi frame kèm cánh tay (arm R² .8–.98); không có event dùng được.
- Slow map 57185: scene event .89 / .31 và cánh tay rò (R² .13–.51). v2: scene .893 / .833, không có cánh tay.
- SAM2 entities (57271–57291, chấm theo định nghĩa khác nên chỉ tham khảo): cube recall .89–.95 / precision ~1.0, scene recall .39–.59.

## 4. Điều đã chứng minh và chưa

**Đã có bằng chứng (dev, VAL):**
- Tách agent khỏi cảnh bằng action hoạt động **cùng một bộ hyper-parameter trên cả ba family**; ngưỡng tự fit gần như
  trùng nhau. Đây là vấn đề cốt lõi khiến v1 và slow map thất bại.
- Memory chứa trạng thái vật và gần như mọi thay đổi thật đều được bắt (cửa sổ 30: recall .97–.99, precision .98–1.0).
- Đọc xuyên cánh tay trong suốt làm trạng thái "tươi" hơn hẳn: đèn puzzle khi đã ổn định từ min .57 / TB .83 lên min .91 / TB .99.

**Chưa đạt / chưa chứng minh:**
1. **Thời điểm và độ vụn của event**: một lần di chuyển cube cho ra nhiều event (nhấc ở A, đặt ở B, các token của vật
   cập nhật lệch nhau), cube có 46 event/episode so với 11.8 đoạn tham chiếu, nên precision cửa sổ 15 chỉ .67. Luật gộp
   thành tương tác còn chưa tin được (gộp quá tay ở puzzle, thiếu ở cube/scene).
2. **Trạng thái rời rạc theo từng frame** (c1) chưa tới .98 cho mọi nút/đèn; mức trần đọc từ frame hiện tại là .9996.
3. **Mã học được chưa hơn token pixel thô**. Phần học tạo khác biệt là tách agent và SeeThrough, không phải autoencoder.
   Với token pixel, trạng thái tốt hơn chút nhưng cánh tay rò (R² .22–.36).
4. Chưa có closed-loop, chưa có held-out, một seed, dev trên VAL. Các lựa chọn thiết kế trong phiên này đều được chọn khi
   nhìn VAL của scene; muốn claim chung phải giữ nguyên và đánh giá trên dữ liệu khác.
5. SeeThrough dựa vào việc OGBench vẽ cánh tay gần trong suốt. Với cánh tay đục (robot thật), phần "đọc xuyên" sẽ
   giảm vai trò và hệ thống quay về mask (vẫn chạy được nhưng memory cập nhật chậm hơn).

## 5. Đánh giá khả năng

**Có triển vọng cho phần "pixel → trạng thái cảnh không có agent + thời điểm thay đổi"**: tổng quát qua ba family,
không nhãn, không luật theo task, và đã vượt rõ các front end trước ở đúng chỗ chúng hỏng (cánh tay, precision).
**Chưa đủ để thay event record của track STATE**: còn thiếu bước ghép thay đổi thành *tương tác* (cái gì bị tác động,
từ đâu đến đâu: ghép "nhấc ở A" với "đặt ở B", gộp mảnh của cùng một vật). Bước này gắn với interaction code c (đang
hoãn) và logic acted-object/contact của `u_events`.

Đề xuất thứ tự tiếp theo:
1. Ghép tương tác dựa trên agent: một tương tác kéo dài từ lúc agent chạm vùng đó đến lúc rời đi; ghép nhấc/đặt bằng
   vật đi cùng agent (vùng contingent mang theo vật). Đo bằng số event/episode so với tham chiếu và span recall/precision.
2. Sửa probe completeness (c2) để chấm được cube.
3. Nối event record vào backend (event WM / skill / cost-to-go) và chạy closed-loop trên visual puzzle trước (event
   đã đạt c4), rồi scene và cube; so sánh với kết quả STATE và baseline cùng protocol.
4. Held-out (seed và episode khác) trước khi đưa bất kỳ số nào vào báo cáo.
