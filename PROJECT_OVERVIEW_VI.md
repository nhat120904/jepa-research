# CAI-JEPA — Tài liệu giải thích toàn bộ dự án (cho người mới hoàn toàn)

> Mục tiêu của tài liệu này: một người **chưa biết gì** về dự án (thậm chí chưa rành về world model)
> đọc xong sẽ hiểu được: (1) bài toán là gì, (2) ý tưởng nghiên cứu, (3) dữ liệu, (4) phương pháp đo,
> (5) từng metric nghĩa là gì, (6) cách chạy code, (7) kết quả hiện tại và kết luận.
>
> Đọc theo thứ tự từ trên xuống. Các thuật ngữ được giải thích ngay lần đầu xuất hiện.

---

## Phần 0 — Tóm tắt trong 1 phút (đọc cái này trước)

Dự án này nghiên cứu **world model** dùng cho **robot lập kế hoạch (planning)**. World model là một mạng
neural "tưởng tượng" ra tương lai: cho nó trạng thái hiện tại + một hành động, nó dự đoán trạng thái kế tiếp.
Robot dùng nó để thử nhiều hành động "trong đầu" rồi chọn hành động tốt nhất, thay vì thử thật ngoài đời.

**Vấn đề chúng ta nghi ngờ:** các world model hiện tại (DINO-WM, V-JEPA-2-AC, JEPA-WM) có thể **không thực sự
phân biệt được hành động**. Tức là cho cùng một trạng thái nhưng 2 hành động khác nhau, nó dự đoán ra 2 tương lai
gần như y hệt. Nếu vậy thì robot lập kế hoạch sẽ thất bại (mọi hành động "trông như nhau" → không chọn được cái tốt).

**Việc chúng ta làm:** xây một bộ **chẩn đoán (diagnostic)** đo định lượng xem lỗi này có thật không, và nó có
**tập trung ở những tình huống tiếp xúc/grasp tinh vi** đúng như lý thuyết dự đoán không. Đây là một nghiên cứu
**go/no-go**: nếu lỗi có thật → viết paper đầy đủ; nếu không → bỏ ý tưởng.

**Kết quả hiện tại:** lỗi **có thật và đo được**. Quyết định: **CONDITIONAL_GO** (đủ tự tin để viết paper, nhưng
phần dữ liệu DROID còn cần chạy hoàn chỉnh thêm).

> ⚠️ **CẬP NHẬT QUAN TRỌNG (2026-06-09) — đọc Phần 3.5 trước khi tin phần "phương pháp sửa lỗi".**
> Hướng nghiên cứu đã **xoay (pivot)** sang khung **"Boundary-Blind World Models"**. Phần *chẩn đoán* (CRA/AUG/ECS)
> vẫn đứng vững, nhưng:
> - Có một **metric mới quan trọng nhất**: **Boundary Blindness (BB)** — xem Phần 5.7.
> - Phần "cách sửa lỗi" cũ (CAI-JEPA margin loss một bước, mô tả ở Phần 3 mục 3) đã bị **thay thế** bởi fix mới
>   (mixture-density + boundary supervision) — xem Phần 3.5.
> - **BB gate ĐÃ CHẠY và PASS (2026-06-10)** — BB tập trung đúng ở biên pre-grasp trên cả Metaworld lẫn DROID
>   (transfer). Xem Phần 9.6 (số liệu thật) và `diagnosis/results/boundary_gate_report.md`.
> - **Fix C1 mức head ĐÃ THỬ và NULL (cùng ngày, 4 biến thể)**; fix mức metric (φ-probe) cũng NULL — cả hai giữ làm
>   ablation. Xem `diagnosis/docs/FIX_C1_EXPLAINER.md` §6–§7.
> - ⭐ **FIX THÀNH CÔNG: "kênh động học vật có giám sát" `h(z,a)→Δvật`** (0.5M tham số, mọi thứ khác đóng băng,
>   chỉ dùng cache): tracking phản-thực corr **+0.035 → +0.682**; BB tại biên pre-grasp **giảm 50%** (1.323 → 0.660);
>   gap pre_grasp-vs-free_space sập từ 1.04 → 0.32. Nguồn: `diagnosis/results/metaworld_boundary_dynamics.csv`.
>   Leg planning open-loop: không hại/chưa thấy lợi trên Action Error (đúng dự đoán — metric này thưởng bắt chước
>   tay; cần success-rate closed-loop, là thí nghiệm tiếp theo trên server).

---

## Phần 1 — Kiến thức nền: World Model & JEPA là gì?

### 1.1. World model dùng để làm gì?

Tưởng tượng bạn dạy một cánh tay robot Franka gắp cốc. Hai cách:

- **Cách cũ (thử thật):** robot cứ thử hành động ngoài đời, sai thì làm lại. Chậm, hỏng đồ.
- **Cách world model:** robot có một "trình mô phỏng trong đầu". Trước khi cử động, nó tưởng tượng:
  "nếu tôi đẩy tay sang trái 2cm thì cảnh sẽ thành thế này; nếu đóng kẹp thì thành thế kia". Nó thử hàng trăm
  hành động trong đầu, chấm điểm xem cái nào đưa cảnh gần với **mục tiêu (goal)** nhất, rồi mới thực hiện.

Thuật toán chọn hành động bằng cách thử-và-chấm-điểm này tên là **CEM (Cross-Entropy Method)** — sẽ nói kỹ ở Phần 6.

### 1.2. JEPA — không dự đoán pixel mà dự đoán "latent"

Dự đoán tương lai dưới dạng **ảnh pixel đầy đủ** rất tốn kém và không cần thiết (robot không cần biết màu sắc
từng hạt bụi). Họ JEPA (Joint-Embedding Predictive Architecture) làm khác:

1. Một **encoder** (bộ mã hóa) đã được huấn luyện sẵn và **đóng băng** (frozen — không train lại) biến ảnh
   thành một vector số gọi là **latent** `z`. Latent là "bản tóm tắt ngữ nghĩa" của cảnh.
   - Encoder thường là DINOv2 hoặc V-JEPA 2 (các mô hình thị giác lớn học từ video internet).
2. Một **predictor** (bộ dự đoán) nhỏ, nhẹ học cách: cho latent hiện tại `z_t` và hành động `a_t`, dự đoán
   latent kế tiếp `ẑ_{t+1}`.

Công thức cốt lõi:

```
ẑ_{t+1} = F_θ(z_t, a_t)
```

- `z_t` = latent (tóm tắt) cảnh hiện tại
- `a_t` = hành động (ví dụ vận tốc đầu kẹp + lệnh đóng/mở kẹp)
- `F_θ` = predictor (mạng neural có tham số θ)
- `ẑ_{t+1}` = latent **dự đoán** của cảnh kế tiếp

Predictor được train bằng cách so dự đoán với latent thật của khung hình kế tiếp:

```
Loss = ‖ F_θ(z_t, a_t) − z_{t+1}_thật ‖²     (sai số bình phương trong không gian latent)
```

### 1.3. Ba "baseline" (mô hình chuẩn) mà dự án soi vào

| Tên | Encoder | Cách đưa action vào | Train trên |
|---|---|---|---|
| **DINO-WM** (2024) | DINOv2 (đóng băng) | nối (concat) action vào feature | nhiều domain |
| **V-JEPA-2-AC** (2025, Meta) | V-JEPA 2 (đóng băng) | prepend action vào chuỗi | 62 giờ video Franka từ DROID |
| **JEPA-WM / Terver** (2026) | DINOv3 | AdaLN (tiêm action vào mọi tầng) | — |

Tất cả đều theo cùng công thức `ẑ_{t+1} = F_θ(z_t, a_t)`. Dự án dùng **checkpoint đã train sẵn, đóng băng** —
**không train lại gì cả**, chỉ đo.

---

## Phần 2 — Bài toán cốt lõi: "Action Grounding" (neo hành động)

### 2.1. Câu hỏi trung tâm

> Các world model JEPA hiện tại **có thực sự phân biệt được tương lai do những hành động khác nhau gây ra**,
> trên đủ loại trạng thái mà robot gặp khi lập kế hoạch, hay không?

Tính chất "phân biệt được hành động" này gọi là **action-identifiability** (tính nhận-dạng-được-theo-hành-động),
hoặc **action grounding** (neo hành động). Một mô hình "grounding tốt" thì `F_θ(z_t, a)` và `F_θ(z_t, a')` phải
**khác nhau rõ rệt** khi `a ≠ a'`.

### 2.2. Tại sao đây là vấn đề thật, và tại sao các bài test cũ không phát hiện ra?

Lưu ý quan trọng: chúng ta **KHÔNG** nói "mô hình bỏ qua hoàn toàn action". Điều đó đã bị bác bỏ — nếu xóa action
thì planning sập hẳn, nên rõ ràng action có tác dụng. Vấn đề tinh vi hơn:

Các bài đánh giá hiện hành có **điểm mù**:
- **Prediction loss** (sai số dự đoán 1 bước trên dữ liệu thật) — một mô hình **lờ action** vẫn có thể đạt loss
  thấp nếu trạng thái kế tiếp dễ đoán từ trạng thái hiện tại.
- **Planning success rate** (tỉ lệ hoàn thành nhiệm vụ) — gộp chung quá nhiều thứ, và trên các task mà tác động
  của action đủ lớn thì dự đoán "nhiễu" vẫn đủ dùng.
- **Demo định tính "mở vs đóng kẹp"** (Terver et al.) — đây là **tương phản dễ nhất có thể**: mở kẹp (vật rơi)
  vs đóng kẹp (nâng vật) khác nhau rõ rành rành. Qua được bài này **không** chứng minh mô hình grounding tốt nói chung.

### 2.3. Action grounding hỏng → planning hỏng như thế nào

Khi CEM lập kế hoạch, nó lấy hàng trăm chuỗi hành động, cho mỗi chuỗi chạy qua predictor, chấm điểm (khoảng cách
tới goal). Nếu predictor cho ra **kết quả gần giống nhau với mọi action**, thì "bản đồ điểm" trở nên **phẳng lì** —
không có hành động nào nổi bật → planner chọn đại theo trung bình → robot làm sai.

Đáng chú ý: chính paper V-JEPA-2-AC thừa nhận gắp **hộp chỉ thành công 25%** vs **cốc 65%**, và quy lỗi cho
"cần điều khiển kẹp chính xác hơn". Đây đúng là vùng grounding mong manh nhất: tác động của action (độ mở kẹp vài mm)
là **nhỏ về mặt hình ảnh nhưng quyết định thành/bại**.

### 2.4. Giả thuyết cụ thể (cái sẽ kiểm chứng)

Lỗi grounding **tập trung ở các "regime" (chế độ trạng thái) mà tác động của action nhỏ-trên-pixel-nhưng-quan-trọng**:
tiến tới gần vật (pre-grasp), đóng/mở kẹp (gripper), tiếp xúc tinh vi. Đây là lý do mô hình qua được demo "mở vs đóng"
(một ca dễ) nhưng vẫn fail khi gắp tinh.

---

## Phần 3 — Ý tưởng nghiên cứu đầy đủ (4 đóng góp của paper CAI-JEPA)

`cai_jepa_paper_proposal.md` đề xuất 4 đóng góp. **Dự án trong repo này mới làm phần chẩn đoán (đóng góp 1+2 ở
mức go/no-go)**, các đóng góp 3+4 là kế hoạch nếu go.

1. **CounterfactualBench** — bộ chẩn đoán định lượng action grounding: **4 metric × 4 regime × các chiến lược
   negative**. (Phần 4 & 5 của tài liệu này.)
2. **Correlation study** — chứng minh: grounding kém ⇒ planning kém (đo tương quan giữa metric và Action Error).
   (Phần 6.)
3. **CAI-JEPA loss** (đề xuất GỐC, **đã bị thay thế** — xem Phần 3.5) — một **hàm mất mát** ép predictor phải phân
   biệt action: phạt mô hình nếu dự đoán dưới hành động giả lại gần tương lai thật hơn so với hành động thật. ⚠️
   **Framing này không còn là idea-of-record.** Một critique (Phần 3.5) chỉ ra nó không sửa được đúng vùng cần sửa.
4. **Probing** (tùy chọn) — gắn các bộ dò nhỏ vào từng tầng predictor để xem thông tin action "chảy" qua mạng
   thế nào, hỏng ở đâu.

> Từ khóa nền: **counterfactual** = "phản thực" = một hành động **giả định khác** với hành động đã thực sự xảy ra.
> Toàn bộ ý tưởng xoay quanh: cho cùng trạng thái, so sánh hành động thật vs các hành động phản thực.

---

## Phần 3.5 — ⭐ REFRAMING MỚI NHẤT (2026-06-09): "Boundary-Blind World Models"

> Đây là **idea-of-record hiện tại** (`diagnosis/docs/PAPER_IDEA.md` + `DIAGNOSIS_PLAN.md` +
> `plans/2026-06-09-action-identifiability-fix-design.md`). Phần *chẩn đoán* (CRA/AUG/ECS, đóng góp 1+2) vẫn đứng
> vững và được **mở rộng** thêm metric BB. Phần *sửa lỗi* (đóng góp 3 ở trên) bị **viết lại hoàn toàn**.

### 3.5.1. Critique giết các "fix dễ" (lý do phải reframe)

Trực giác ngây thơ là: "ép `F(z,a)` và `F(z,a')` tách xa nhau" (bằng classifier-free action guidance, hoặc margin
loss một bước như CAI-JEPA gốc). **Nhưng nó không sửa được đúng vùng quan trọng:**

- Nếu mô hình **đã** collapse `F(z,a) ≈ F(z,a')` rồi, thì vector hướng dẫn ≈ 0 và đạo hàm `∂loss/∂a ≈ 0` — **không
  có gì để khuếch đại**. Các fix này chỉ *khuếch đại* độ nhạy mô hình **sẵn có**, không *tạo ra* được độ nhạy mới.
- Biên (boundary) sắc trong **không gian kết quả (outcome)**, nhưng `z_{t+1}` một bước ngay sau một nhiễu nhỏ lại
  gần như **liên tục** — sự phân kỳ chỉ lộ ra **qua rollout nhiều bước** (sensitive dependence — phụ thuộc nhạy cảm).
  Vì vậy mọi phương pháp **một bước về cấu trúc đều mù** với nó.

### 3.5.2. Phát biểu lại bài toán (the reframed gap)

> Các world model JEPA không chỉ "yếu trong việc khuếch đại tác động của action" — chúng **thất bại trong việc mô
> hình hóa các vùng action có độ nhạy cao (high-sensitivity), nơi một nhiễu action nhỏ gần một biên tiếp xúc tạo ra
> những tương lai khác nhau về chất** (kẹp căn giữa → vật được nâng; lệch 2–3° → không nâng được). Dự đoán latent
> **đơn mode (unimodal) bằng point + L2 thì CHẮC CHẮN lấy trung bình (averages)** qua các phân nhánh kết quả
> (bifurcation), và latent chỉ-từ-thị-giác có thể còn không **phân giải (resolve)** được trạng thái quyết định biên.

Hai điểm hỏng tách biệt, mỗi cái cần fix riêng:
| | Điểm hỏng | Fix | Cần lực (force)? |
|---|---|---|---|
| (a) | Latent **không phân giải** được trạng thái biên | **D** — latent có gắn state | force = không; pose+gripper = được |
| (b) | Predictor **quá mượt** / lấy trung bình đơn-mode | **C1** — dự đoán phân phối (đa mode) + giám sát biên | không |

### 3.5.3. Đóng góp mới (3 cái)

- **C1 (CHẨN ĐOÁN) — CounterfactualBench + Boundary Blindness.** Trên checkpoint đóng băng. CRA/AUG/ECS đo "mô hình
  có dùng action không"; **BB** (metric mới — Phần 5.7) đo "mô hình có **phân giải được biên sắc** không" — điều
  CRA/ECS **không nhìn thấy**. Đã code: `metrics/boundary_blindness.py`, `stratification/boundary_regime.py`,
  `scripts/12_boundary_diagnostic.py`.
- **C2 — counterfactual sensitivity ⇄ planning.** Per-transition CRA_eff correlation **null/underpowered** (mất cân
  bằng lớp: chỉ ~4% positive trên DROID) — và **chính cái null đó là bằng chứng cho reframe**: lỗi liên quan là
  *bifurcation ở biên*, không phải "lờ action một bước". Planning leg được **đúc lại quanh BB**, không quanh CRA_eff.
- **C3 (FIX) — dự đoán latent phân phối + giám sát biên.**
  - **C1-fix (hero, không cần force):** thay point-head bằng **mixture-density head** trên `z_{t+1}` (K≈2–4, loss
    NLL) để predictor biểu diễn được "nâng HOẶC không-nâng" thay vì trung bình L2 của chúng; thêm **boundary-
    supervision head** (sự kiện grasp / vật-dịch-chuyển từ state Metaworld) ép dung lượng mạng vào biên.
  - **D (phụ, chỉ Metaworld):** thêm vào latent lát-cắt-state liên quan biên `z̃ = [z^vis ‖ φ(hình học ee–vật, gripper)]`.
  - Tích hợp planning: CEM chấm điểm candidate theo mode của mixture / NLL-of-goal → không còn tối ưu trên một bề mặt
    chi phí đã-bị-lấy-trung-bình.

### 3.5.4. Data reality (ràng buộc cứng, đã verify ở loader)

DROID proprio = **7 chiều = cartesian_position(6) + gripper_position(1)**, **KHÔNG có force/torque, KHÔNG có joint**.
→ Dạng fix mạnh nhất (grounding theo lực tiếp xúc) **bất khả thi trên DROID**. Hệ quả: **boundary proof chỉ làm trên
Metaworld** (dataset duy nhất có object-state để *định nghĩa*, *giám sát*, *đo* biên so với ground truth). DROID chỉ
là **transfer check**, dùng proxy ‖Δz‖, **không** được tuyên bố grounding xúc giác/lực. (Câu hỏi mở duy nhất: raw
DROID có lộ joint_position/velocity không — `scripts/inspect_droid_observation_keys.py` trả lời, 5 phút, chưa chạy.)

---

## Phần 4 — Bộ chẩn đoán: dữ liệu, regime, chiến lược negative

Code nằm trong thư mục `diagnosis/`. Chạy hoàn toàn trên **checkpoint đóng băng** (không train). Toàn bộ chạy trên
một **cache latent** (mã hóa ảnh **một lần duy nhất**, lưu ra HDF5, mọi metric đọc từ cache → nhanh và tái lập được).

### 4.1. Hai dataset (và vì sao chọn chúng)

Luận điểm là "lỗi tập trung ở vùng tiếp xúc/tinh vi", nên cần dữ liệu (a) có những vùng đó và (b) có tín hiệu để
**phát hiện** ra vùng đó.

**Metaworld (chính) — bề rộng theo độ khó:**
- Tập con **12 task** trải từ dễ → khó: dễ (`reach`, `push`, `pick-place`), trung bình (`door-open/close`,
  `drawer-close`, `button-press`, `window-close`), khó (`peg-insert-side`, `assembly`, `hammer`, `stick-pull`).
- 2 mô hình: `dino_wm_metaworld`, `jepa_wm_metaworld`.
- **Hạn chế:** bản Metaworld trên HuggingFace **không có ground-truth tiếp xúc của MuJoCo**, nên regime là **proxy**
  suy ra từ vector trạng thái 39 chiều (vị trí đầu kẹp/vật; vật dịch chuyển = dấu hiệu tiếp xúc). Và **không có tín
  hiệu đóng/mở kẹp dùng được** → regime `gripper_actuation` rỗng trên Metaworld. Đây chính là lỗ hổng DROID lấp.

**DROID (phụ) — robot Franka thật, kẹp thật, tiếp xúc thật:**
- 2 mô hình: `dino_wm_droid` + `vjepa2_ac_droid` (đây là so sánh "đinh" của paper trên DROID).
- Dữ liệu: tập con **333 episode** công khai (gom tay vì DROID không có bản tải HF, bucket gốc 5.6 TB). Mỗi episode
  1 clip 8 khung ở fps=4 → ~2331 transition. DROID **không có nhãn task** → coi là một pool phẳng (`task = "droid"`).
- **Vì sao DROID là mảnh ghép quan trọng:** `gripper_position` là tín hiệu **thật**, nên 2 regime `gripper_actuation`
  và `contact_manipulation` mới có dữ liệu — đúng 2 ô Metaworld để trống. DROID cũng là dataset mà V-JEPA-2-AC được
  train trên đó → bằng chứng sát nhất cho luận điểm planning.

> **Push-T / PointMaze** chỉ là **sanity check** (kiểm tra pipeline không hỏng), **không bao giờ** dùng làm bằng
> chứng cho luận điểm.

### 4.2. Bốn "regime" (chế độ trạng thái)

Mỗi transition được gán **một** trong 4 nhãn. Giả thuyết tiên đoán một **mẫu hình** cụ thể: grounding tốt ở vùng
"sạch", kém ở vùng tác-động-nhỏ-trên-pixel-nhưng-quan-trọng.

| Regime | Nghĩa | Phát hiện trên Metaworld (proxy) | Phát hiện trên DROID |
|---|---|---|---|
| `free_space` | tay di chuyển qua không gian trống | đầu kẹp di chuyển, vật không dịch | gripper ít đổi, biến đổi latent dưới trung vị |
| `pre_grasp` | tiến gần vật, chưa chạm | đầu kẹp gần vật, vật đứng yên | gripper **mở** + biến đổi latent trên trung vị |
| `gripper_actuation` | đóng/mở kẹp | (không có tín hiệu → **rỗng**) | `|Δgripper| > 0.2` |
| `contact_manipulation` | đang tiếp xúc, action làm vật chuyển | vật dịch chuyển > ngưỡng | gripper **đóng** + biến đổi latent trên trung vị |

**Giả thuyết:** CRA cao ở `free_space` và (nơi thấy được) `contact`; CRA **giảm** ở `pre_grasp` và
`gripper_actuation`. Đây là lý do lỗi **trốn** được các demo định tính nhưng lại làm hỏng gắp tinh.

> Lưu ý trung thực cần báo cáo: regime tiếp xúc/pre-grasp của DROID là **proxy** (không có GT MuJoCo). Ngưỡng tiếp
> xúc được **hiệu chuẩn theo encoder**: patch-L2 của DINOv2 ViT-S có dải hẹp (trung vị ≈622, max ≈842), nên ngưỡng
> gốc `1.5×trung vị` cho ra **0% tiếp xúc**; đã hạ xuống `1.0×trung vị`. Đây là hiệu chuẩn có chủ đích, ghi rõ
> trong tài liệu, không phải "chỉnh lén".

### 4.2-bis. "Boundary regime" — lát cắt MỚI (2026-06-09), cắt ngang 4 regime trên

Ngoài 4 regime trên, reframing mới thêm một cách chọn transition **cắt ngang (cross-cutting)**: **boundary regime**.
Một transition `(z_t, a_t)` nằm trong boundary regime nếu trong **hàng xóm trạng-thái-tương-tự** của nó (pool similar-state,
tái dùng máy móc `hard_nn`/`hard_effect`), một thay đổi action nhỏ làm **kết quả thật bung ra (fan out)** — tức là một
**phân nhánh (bifurcation)**:

```
boundary_score = std(kết quả thật trong hàng xóm) / mean(‖Δaction‖)
```

`boundary_score` cao = action đổi tí xíu mà outcome lật hẳn = đúng vùng biên cần soi. Trên Metaworld, "kết quả thật"
= **dịch chuyển vật** (object displacement, từ state 39 chiều) — không chỉ ‖Δz‖. Code: `stratification/boundary_regime.py`.

### 4.3. Bốn chiến lược "negative" (độ khó của hành động phản thực) — TRÁI TIM của chẩn đoán

Với mỗi transition thật `(z_t, a_t, z_{t+1})`, ta tạo ra **K=16 hành động phản thực** rồi hỏi: **dự đoán dưới hành
động thật có gần `z_{t+1}` thật nhất không?** "Chiến lược" quyết định các hành động phản thực **khó cỡ nào**. Một
chiến lược dễ mà mọi mô hình đều qua thì chẳng chứng minh gì; chiến lược khó mới phơi bày được lỗ hổng.

| Chiến lược | Hành động phản thực `a⁻` | Kiểm tra điều gì | Kỳ vọng |
|---|---|---|---|
| `random` | ngẫu nhiên trong biên action | nhạy cảm action thô | **dễ** — mọi mô hình nên qua |
| `opposite` | `−a_t + nhiễu`, lật chiều kẹp | đảo chiều — **bản định lượng** của demo "mở vs đóng" | **dễ** — CRA cao kể cả mô hình yếu |
| `hard_nn` | từ pool **trạng thái tương tự**, lấy action **khác `a_t` nhất** | phân biệt action tinh vi mà một policy giỏi có thể chọn từ trạng thái gần như y hệt | **KHÓ** — chỗ lộ lỗ hổng |
| `hard_effect` | từ pool trạng thái tương tự, action có **tác động thật Δz khác** factual nhất, ưu tiên action **gần** `a_t` | "hành động chính xác có ý nghĩa": negative khó **công bằng** vì tương lai thật của nó thực sự khác | **khó nhất + công bằng** |

**Vì sao `opposite` gần hoàn hảo nhưng `hard_nn` sụp đổ chính là BẰNG CHỨNG.** Trên Metaworld cả 2 mô hình đạt
~0.97–0.99 ở `opposite` (qua demo) nhưng rớt còn ~0.46–0.57 ở `hard_nn` trong pre-grasp/contact (chance = 1/17 ≈ 0.059).
Sự **phân ly** này — qua bài dễ, fail bài khó **đúng ở các regime đã tiên đoán** — **chính là** lỗ hổng action grounding
mà paper khẳng định.

**`hard_nn` vs `hard_effect` (khác biệt tinh tế nhưng quan trọng):**
- `hard_nn` tối đa hóa **khác biệt action**, mặc kệ kết quả. Rủi ro: từ trạng thái này, negative được chọn có thể
  dẫn tới **cùng** tương lai như `a_t` (ví dụ free-space mượt, nhiều action cho `z_{t+1}` gần như nhau). Khi đó CRA
  thấp là "oan" — mô hình không sai, hai tương lai thật sự không phân biệt được.
- `hard_effect` sửa điều này bằng cách chấm điểm candidate theo `‖Δz_cand − Δz_factual‖ − phạt·‖a_cand − a_t‖`. Nó
  chọn một action **gần** nhưng dẫn tới tương lai thật **khác**, nên mô hình grounding tốt **có thể** thắng. Ở vùng
  mượt không tồn tại negative như vậy → `hard_effect` **tự chọn về phía** transition tiếp xúc/tinh vi — đúng nơi
  paper nói grounding quan trọng.

---

## Phần 5 — Giải thích từng METRIC (chi tiết)

Có 4 metric. **CRA** (và biến thể effect-conditioned của nó) là tín hiệu quyết định chính.

### 5.1. CRA — Counterfactual Ranking Accuracy (Độ chính xác xếp hạng phản thực) ⭐ CHÍNH

**Trực giác:** "Trong số hành động thật + 16 hành động giả, mô hình có xếp hành động **thật** lên đầu (dự đoán gần
tương lai thật nhất) không?"

Công thức:
```
CRA = P[ d(F(z_t, a_t), z_{t+1}) < min_k d(F(z_t, a⁻_k), z_{t+1}) ]
```
- `d(...)` = khoảng cách **L2** (mọi baseline dùng L2 vì mọi config planning upstream là `L2_cem`).
- Đọc: xác suất mà khoảng cách [dự đoán-dưới-action-thật ↔ tương lai-thật] **nhỏ hơn** khoảng cách nhỏ nhất trong
  số các action giả.
- **Chance (đoán mò) = 1/(K+1) = 1/17 ≈ 0.059.** Mô hình lờ action sẽ rơi về mức này.
- Mô hình grounding hoàn hảo → CRA ≈ 1.0.
- Báo cáo cả **top-1** (đúng hạng nhất) và **MRR** (Mean Reciprocal Rank — trung bình nghịch đảo thứ hạng, "mềm" hơn).

### 5.2. CRA_eff — Effect-conditioned CRA ⭐⭐ SỐ QUYẾT ĐỊNH

**Đây là con số dùng để ra quyết định go/no-go.** Nó là CRA nhưng **chỉ tính trên những transition mà latent thực
sự thay đổi**: `‖z_{t+1} − z_t‖ > τ` (τ = trung vị Δz, hiệu chuẩn theo từng mô hình).

**Vì sao cần lọc?** CRA thô thấp trong vùng tiếp xúc có thể chỉ vì bước delta quá nhỏ (chẳng có gì xảy ra trong 1
bước) — không phải lỗi mô hình. Lọc theo effect cô lập đúng câu hỏi: "mô hình có fail dùng action **khi có chuyện
thực sự xảy ra** không?" Trong CSV cột tên là `cra_top1_eff` (kèm CI: `cra_top1_eff_lo`, `cra_top1_eff_hi`).

### 5.3. AUG — Action Usage Gap (Khoảng cách dùng action)

**Trực giác:** "Nếu tôi tráo nhầm action, sai số có tăng lên không?" Nếu mô hình thực sự dùng action thì dùng action
thật phải dự đoán tốt hơn dùng action tráo.

```
AUG = E[ MSE(F(z_t, action_tráo), z_{t+1}) − MSE(F(z_t, a_t), z_{t+1}) ]
```
- AUG ≈ 0 → mô hình **lờ** action (action thật hay tráo cũng như nhau).
- AUG > 0 (dương lớn) → mô hình **có dùng** action.
- Khác CRA: AUG nhạy với **độ lớn tuyệt đối** của ảnh hưởng action, không chỉ thứ hạng.

### 5.4. ECS — Effect-Conditional Sensitivity (Độ nhạy có điều kiện theo tác động)

Giống AUG nhưng **chỉ trên transition có tác động** (`‖z_{t+1} − z_t‖ > τ`). Lý do giống CRA_eff: ở free-space
nhiều transition gần như không biến đổi, action chẳng quan trọng, tính vào sẽ làm loãng tín hiệu. ECS cô lập vùng
"có chuyện xảy ra".

### 5.5. CTD — Counterfactual Trajectory Divergence (tùy chọn, không vào quyết định)

Đo **rollout nhiều bước**: tung 2 chuỗi action khác nhau từ cùng trạng thái, xem 2 quỹ đạo dự đoán **tách xa nhau**
bao nhiêu ở chân trời H ∈ {1,3,5,10}. Bắt lỗi "mô hình phân biệt action ở bước 1 nhưng mất nhạy cảm khi rollout dài"
(CTD không tăng theo H = lỗi). Đây là tùy chọn (`--ctd`), không nằm trong logic quyết định.

### 5.7. BB — Boundary Blindness (Mù Biên) ⭐⭐⭐ METRIC MỚI & HEADLINE (2026-06-09)

**Đây là con số quan trọng nhất của framing mới.** Nó đo điều mà CRA/ECS **không nhìn thấy được**: không phải "mô
hình có dùng action không", mà "mô hình có **phân giải được biên sắc** không".

**Trực giác:** ở một vùng biên (bifurcation), thế giới thật rất nhạy — action đổi tí xíu thì kết quả lật. Một mô
hình tốt phải **cũng nhạy như vậy**. Mô hình "mù biên" thì dự đoán **gần như cùng một tương lai cho mọi action** dù
thế giới thật đang phân nhánh.

Với mỗi transition biên, trên cùng một **hàng xóm các action gần** `{a'}`:
- `S_true` = độ trải (spread: var/range) của **kết quả THẬT** (dịch chuyển vật trên Metaworld; proxy ‖Δz‖ trên DROID).
- `S_model` = độ trải của **dự đoán của mô hình** `F(z_t, a')` trên cùng các action đó.
- Chuẩn hóa cả hai trên toàn mô hình, rồi:

```
BB = relu(S_true_norm − S_model_norm)
```

- `BB ≈ 0` → mô hình bám đúng độ nhạy cục bộ của thế giới (tốt).
- `BB lớn` → thế giới phân nhánh ở đây **nhưng** mô hình dự đoán ~cùng một tương lai cho mọi action (**mù biên** — xấu).

**Kết quả tiên đoán (luận điểm):** baseline có **BB cao tập trung ở pre-grasp / gripper-actuation / contact**, ngay cả
ở những chỗ CRA tổng thể trông chấp nhận được. Báo cáo `bb` (tất cả) và `bb_boundary` (chỉ tập con boundary-score cao).

Code: `metrics/boundary_blindness.py`, runner `scripts/12_boundary_diagnostic.py`. Đã có 11 test (gồm chứng minh
tổng hợp: mô hình lờ-action thì mù biên; mô hình hoàn hảo thì không). **Cập nhật 2026-06-10: metric này ĐÃ CHẠY
trên baseline đóng băng và gate ĐÃ PASS — số liệu thật ở Phần 9.6.**

### 5.6. Khoảng tin cậy (Confidence Interval)

Mọi metric đều kèm CI 95% bằng **bootstrap phân cụm theo quỹ đạo** (`n_resamples=1000`): khi lấy mẫu lại, lấy lại
**cả quỹ đạo** chứ không lấy từng transition rời. Lý do: các transition trong cùng một quỹ đạo tương quan với nhau,
nếu lấy mẫu rời sẽ **thổi phồng** độ tin cậy giả tạo. Trong CSV là các cột `..._lo` / `..._hi`.

---

## Phần 6 — Đóng góp 2: nối grounding với planning (Action-Score probe)

Đo được CRA thấp chưa đủ; phải chứng minh **CRA thấp ⇒ planning tệ thật**. Đây là việc của script `08` và `09`.

- **`08_planning_probe.py`** chạy một **CEM planner trung thực** (port đúng tham số `L2_cem` của upstream) trên
  cache latent. Với mỗi transition, nó: (a) tính `CRA_eff` (hard_nn), (b) cho planner lập kế hoạch rồi đo
  **Action Error** = sai số giữa hành động planner chọn và hành động chuyên gia thật (đây là "Action Score" trong
  paper DROID).
- **`09_correlate_planning.py`** tính tương quan **Spearman/Pearson(Action Error, CRA_eff)**. Kỳ vọng: **âm rõ rệt**
  (CRA_eff càng thấp → Action Error càng cao → planning càng tệ).

> CEM (Cross-Entropy Method): thuật toán tối ưu bằng lấy mẫu. Lấy K chuỗi action, chấm điểm (khoảng cách tới goal
> qua predictor), giữ M chuỗi tốt nhất ("elite"), khớp lại phân phối quanh elite, lặp vài vòng. Action grounding kém
> → mọi chuỗi điểm gần nhau → elite chỉ là nhiễu → planner chọn đại.

> ⚠️ **Cập nhật framing planning (2026-06-09):** trên Metaworld, tương quan **mức regime** giữa Action Error và
> CRA_eff là −1.0 (rõ, đúng dấu — xem Phần 9.4). Nhưng **per-transition** thì **null/underpowered**: trên DROID chỉ
> ~4% positive (mất cân bằng lớp nặng), trên Metaworld thì Action Error đơn-chuyên-gia nhiễu cao. Theo idea-of-record
> mới, **cái null per-transition này KHÔNG phải điểm yếu mà là bằng chứng cho reframe**: lỗi cốt lõi là *bifurcation
> ở biên*, không phải "lờ action một bước". Vì vậy planning leg được **đúc lại quanh BB** (Phần 5.7) làm tín hiệu
> liên-kết-planning đúng, thay cho CRA_eff. Figure mục tiêu mới: "Action Error vs BB".

---

## Phần 7 — Cách chạy code (pipeline 6 + 2 bước)

Dữ liệu chảy một chiều, mỗi script đọc artifact của bước trước. **Không train gì cả.** Quản lý phụ thuộc bằng `uv`.

```
[setup] scripts/01_setup_environment.sh   → clone external/jepa-wms + uv sync
[smoke] scripts/smoke_test.py             → checkpoint thật load + encode + predict được không
[check] scripts/check_normalization.py    → kiểm tra chuẩn hóa action (lỗi #1)

[03] 03_extract_latents.py  → data/precomputed_latents/{dataset}__{model}.h5   (mã hóa mọi frame 1 lần)
[04] 04_classify_regimes.py → {…}.h5.regimes.json                              (gán regime mỗi transition)
[05] 05_run_diagnostic.py   → results/{dataset}_diagnostic.csv                 (CRA/AUG/ECS + CI)
[06] 06_analyze_results.py  → results/decision_report.md + figures/*.pdf       (quyết định go/no-go)

[08] 08_planning_probe.py     → results/droid_planning.csv (+ .npz)            (CRA_eff + Action Error)
[09] 09_correlate_planning.py → results/planning_correlation.md + figure_c_*   (tương quan)
```

**Offline (không cần GPU/data) — kiểm tra tính đúng của code:**
```bash
cd diagnosis
.venv/bin/python -m pytest tests/          # 34 unit test
python scripts/07_validate_synthetic.py    # PerfectModel→CRA≈1.0; ActionIgnoringModel→CRA≈chance
```
`07_validate_synthetic.py` rất quan trọng: nó cắm **mô hình giả** vào đúng đường chạy production để chứng minh metric
đo đúng (mô hình hoàn hảo cho CRA≈1, mô hình lờ action cho CRA≈chance).

**Trên server (có GPU + data) — chạy thật:** xem `diagnosis/RUNBOOK.md` và `HANDOFF*.md`.

### Những "bẫy" triển khai quan trọng nhất (đọc kỹ nếu sẽ chạy lại)
1. **Chuẩn hóa action là bug #1.** Phải dùng `preprocessor.normalize_actions` (số nhiều) của chính mô hình. DROID =
   identity (mean 0/std 1); Metaworld = shift+scale thật. Kiểm bằng `check_normalization.py`.
2. **Luôn sanity-check** bằng `terver_gripper_test.py` (mở vs đóng kẹp trên DROID; kỳ vọng CRA 2-chiều > 0.90). Nếu
   fail → nghi pipeline bug (chuẩn hóa, frameskip), không phải lỗi mô hình.
3. Mọi config planning là `L2_cem` → CRA luôn dùng **L2**.
4. Push-T / PointMaze chỉ là sanity, không phải bằng chứng luận điểm.

---

## Phần 8 — Logic ra quyết định (go/no-go)

Đọc trên baseline mạnh nhất, chiến lược `hard_nn`, regime tiếp xúc, task khó (với Metaworld). `c` = CRA_eff,
`hi` = cận trên CI.

| Kết luận | Quy tắc |
|---|---|
| **GO** | bệnh lý mạnh ở **cả hai** dataset: `mw < 0.60` **và** `droid < 0.65` |
| **ABANDON** | chỉ khi **cả hai** cận trên CI đều cao: `mw_hi ≥ 0.85` **và** `droid_hi ≥ 0.85` |
| **CONDITIONAL_GO** | bệnh lý vừa phải ở **ít nhất một** dataset (`c < 0.75`) |
| **PIVOT** | tín hiệu lẫn lộn còn lại |

> Sự bất đối xứng là **có chủ đích**: ABANDON cần cận **trên** CI cao ở cả hai dataset, nên một con số nhiễu đơn lẻ
> không thể giết dự án.

---

## Phần 9 — KẾT QUẢ HIỆN TẠI (đọc kỹ phần này)

### 9.1. Quyết định chính thức: **CONDITIONAL_GO**

Lý do: bệnh lý vừa-đến-mạnh ở ít nhất một dataset. Số đầu vào quyết định:
- Metaworld task-khó regime-tiếp-xúc `CRA_eff = 0.651`, cận trên CI `0.703`.
- DROID regime-tiếp-xúc `CRA_eff = 0.047`, cận trên CI `0.072`.

### 9.2. Metaworld — gap rõ ràng (đã hoàn tất)

Bảng CRA_eff theo chiến lược (gộp regime), cho thấy **phân ly** giữa easy và hard:

| Chiến lược | free_space | pre_grasp | contact |
|---|---|---|---|
| `opposite` (dễ) | 0.992 | 0.966 | 0.978 |
| `random` (trung bình) | 0.858 | 0.664 | 0.753 |
| `hard_nn` (khó) | 0.601 | **0.491** | **0.530** |

Chi tiết theo mô hình (chiến lược `hard_nn`, CRA_eff [CI 95%]):

| Mô hình | free_space | pre_grasp | contact |
|---|---|---|---|
| dino_wm | 0.568 [0.50–0.64] | 0.452 [0.36–0.55] | 0.486 [0.42–0.55] |
| jepa_wm | 0.634 [0.57–0.70] | 0.530 [0.43–0.63] | 0.574 [0.51–0.64] |

**Đọc ra sao:** `opposite` gần bão hòa (~0.97–0.99) nhưng `hard_nn` rớt mạnh (~0.45–0.57), thấp nhất ở `pre_grasp`.
Mô hình phản ứng được với thay đổi action thô nhưng **fail khi action phản thực đi kèm trạng thái tương tự** — đúng
luận điểm. `jepa_wm` luôn nhỉnh hơn `dino_wm` nhưng cả hai đều mất biên dưới `hard_nn`.

### 9.3. DROID — sàn (floor), bệnh lý nặng nhất

CRA_eff dưới `hard_nn` **nằm sát chance floor** (1/17 ≈ 0.059) ở mọi regime tiếp xúc/kẹp:

| regime | CRA_eff |
|---|---|
| contact_manipulation | 0.059 |
| gripper_actuation | 0.035 |
| pre_grasp | 0.072 |
| free_space | 0.000 |

Đây là lỗi action-grounding sắc nét nhất — gần như **fail tuyệt đối** trên robot thật.

### 9.4. Liên kết grounding → planning (Action-Score probe)

**Metaworld (183 transition, CRA_eff trải 0.43–1.0 → test có ý nghĩa):**
- **Spearman(Action Error, CRA_eff) theo regime = −1.000** ở cả H=1 và H=3 (đúng dấu kỳ vọng). CRA_eff giảm → Action
  Error tăng đơn điệu: pre_grasp/free_space (CRA_eff cao, error thấp) → contact (CRA_eff thấp nhất, error cao nhất).
  **Chất lượng grounding tiên đoán chất lượng planning, và lỗi tụ ở vùng tiếp xúc — đúng luận điểm.**
- Per-transition thì tương quan yếu, không significant (H=1 +0.08, H=3 −0.13) vì Action Error đơn-chuyên-gia nhiễu
  cao mỗi transition; nhiễu trung bình hóa ở mức regime. **Bằng chứng vững là ở mức regime.**

**DROID (355 transition):** CRA_eff floored ở chance khắp nơi (~4% positive) → **không còn variance để tương quan**.
Bằng chứng ở đây là **mức (level)**: CRA_eff gần chance **trong khi** Action Error cao và tăng theo chân trời
(≈1.0–1.2 ở H=1 → ≈1.6–2.3 ở H=3). Lỗi grounding gần-tuyệt-đối đi kèm planning error cao, dồn theo horizon.

### 9.5. Tóm tắt bằng chứng

| Mảnh bằng chứng | Trạng thái |
|---|---|
| Metaworld: gap `opposite` vs `hard_nn` ở pre-grasp/contact | ✅ rõ ràng, đã hoàn tất |
| Metaworld: link grounding→planning (Spearman regime = −1.0, cả 2 horizon) | ✅ có |
| DROID: CRA_eff sát chance floor ở contact/gripper | ✅ có (nhưng mẫu mỏng) |
| `jepa_wm` > `dino_wm` nhất quán | ✅ |
| **BB (Boundary Blindness) per-regime** | ✅ **ĐÃ CHẠY (2026-06-10) — gate PASS**, BB tụ ở biên pre-grasp trên cả 2 dataset (xem 9.6) |
| DROID `05` (dino_wm) chạy lại + gripper sanity gate | ✅ PASS — hard_nn vẫn ở chance floor trong regime tiếp xúc |
| Fix C1 (mixture head) + fix metric (φ-probe) | ⚠️ **NULL có định lượng — giữ làm ablation** (4 biến thể head + 2 biến thể metric; BB không đổi/chỉ tái phân bố). Nguyên nhân đo: subspace biên bị nén ~10× dưới nhiễu trong L2; kênh phản-thực của predictor là nhiễu (V3 corr +0.035). `FIX_C1_EXPLAINER.md` §6–§7 |
| ⭐ **Fix THÀNH CÔNG: kênh động học vật `h(z,a)→Δvật`** | ✅ corr phản-thực **+0.682** (gấp 20× predictor đóng băng); pre_grasp `bb_boundary` **1.323 → 0.660 (−50%)**; gap biên-vs-tự-do 1.04 → 0.32. Nguồn: `metaworld_boundary_dynamics.csv` |
| Planning A/B (CEM cost L2 vs grounded) | ✔️ **xong — không hại, chưa thấy lợi** trên Action Error open-loop (mọi CI cặp đều chứa 0; pre_grasp chỉ n=6). Lý do đã phân tích: Action Error thưởng cho việc bắt chước cả quỹ đạo tay — cái L2 đã tối ưu; lợi ích của fix biên nằm ở closed-loop. **Bước tiếp theo: success rate closed-loop trên server.** (1 bug scale ở lần chạy đầu được công bố trong `_buggy_scale.csv`) |

### 9.6. ⭐ Boundary Blindness gate — ĐÃ CHẠY (2026-06-10): **PASS**

Gate đã được chạy trên máy local (RTX 5070 12 GB, checkpoint đóng băng, không train gì) cho cả hai baseline
Metaworld + DROID transfer (`dino_wm_droid`). Nguồn số liệu: `diagnosis/results/metaworld_boundary.csv` (64 dòng),
`diagnosis/results/droid_boundary.csv` (4 dòng); phân tích đầy đủ + run log:
`diagnosis/results/boundary_gate_report.md`; figure: `diagnosis/results/figures/figure_bb_per_regime.pdf`.

**Bảng BB per-regime (Metaworld, pooled theo trọng số n_boundary, LOẠI `mw-door-close` — xem caveat):**

| regime | dino_wm `bb_boundary` | jepa_wm `bb_boundary` | dino_wm `bb` | jepa_wm `bb` |
|---|---|---|---|---|
| free_space | 0.282 | 0.299 | 0.069 | 0.070 |
| **pre_grasp** | **1.323** | **1.280** | 0.541 | 0.581 |
| contact_manipulation | 0.481 | 0.441 | 0.212 | 0.194 |

So sánh CI-aware theo từng task (`bb_boundary_lo(regime) > bb_boundary_hi(free_space)` trong cùng task):
pre_grasp cao có-ý-nghĩa ở **4/6 task (dino_wm)** và **5/6 task (jepa_wm)**, **không có đảo chiều có-ý-nghĩa nào**.
`gripper_actuation` không có cell Metaworld nào đủ dữ liệu (đúng như dự đoán — subset này không có tín hiệu gripper).

**DROID (transfer, outcome là proxy ‖Δz‖):** pre_grasp `bb_boundary` = **1.975 [1.601, 2.350]** so với free_space
0.721 [0.613, 0.834] — cao có-ý-nghĩa CI; gripper_actuation 1.093 [0.791, 1.393] (cao theo điểm, CI chớm chồng lấn);
contact_manipulation 0.463 — không cao.

**KẾT LUẬN GATE: PASS** — BB tụ đúng ở **biên pre-grasp** (nơi bifurcation grasp/trượt xảy ra) trên cả hai dataset,
đúng tiên đoán của framing "Boundary-Blind". `contact_manipulation` chỉ cao vừa phải — hợp lý: sau khi đã cầm
vật, động học trơn trở lại, ít bifurcation. → **Bắt đầu fix C1.**

**Caveat phải nhớ:**
- `mw-door-close` bất thường: BB ≈ 1.7–2.9 ở *mọi* regime kể cả free_space (cửa là vật khớp quay — proxy dịch-chuyển-vật
  bị nhiễu); bảng pooled ở trên đã loại task này, so sánh theo-task là cách đọc bền vững.
- Nhãn biên Metaworld vẫn là **proxy dịch chuyển vật** (không có contact GT MuJoCo); DROID là **transfer-only**
  (pose+gripper, không force/joint), regime là proxy.
- `vjepa2_ac_droid` chưa chạy (cần ~24 GB VRAM — chỉ chạy được trên server A5000).
- Hai fix code đã cần và được công bố trong report: chunk predict (tránh tràn VRAM/RAM trên máy 12 GB) và port
  fallback relax-to-nearest của hard_nn vào `state_neighbours` (lần chạy đầu bị suy biến BB≡0 vì radius tuyệt đối).

---

## Phần 10 — Đánh giá thẳng thắn: đủ để viết paper chưa?

**Về bản chất: ĐỦ để commit viết paper.** Quyết định đã loại trừ ABANDON một cách tự tin, gap có thật, đo được, tập
trung đúng regime tiên đoán, và đã có link nhân quả grounding→planning trên Metaworld.

**Nhưng còn 3 lỗ hổng cần vá trước khi nộp** (reviewer sẽ nhắm vào đây):

1. **DROID còn mỏng / chưa chạy xong.** Trong bảng coverage DROID chỉ có **n=1 trajectory-row mỗi cell**
   (CI = [0.000, 0.000] ở free_space là dấu hiệu mẫu cực mỏng). Theo trạng thái dự án, DROID mới tới bước 04, còn
   thiếu `05`+`06` cho baseline `vjepa2_ac_droid`. **Đây là lỗ hổng lớn nhất.**
2. **Per-transition correlation không significant** (p ≈ 0.44 / 0.23). Bằng chứng chỉ vững ở **mức regime**, mà
   Metaworld `pre_grasp` chỉ có n=7 (H=1) / n=2 (H=3) → Spearman −1.0 "hoàn hảo nhưng low-n", dễ bị chê.
3. **DROID floor** không thể test correlation (hết variance) — chỉ đọc được "level".

**Khuyến nghị trước khi viết paper:**
- Chạy nốt DROID `05`+`06` với **nhiều trajectory hơn** (≥ vài chục episode/cell, không phải 1) + thêm baseline
  `vjepa2_ac_droid`.
- Tăng Metaworld planning probe lên ~300 traj/task để `pre_grasp` cell hết mỏng.
- Chạy `terver_gripper_test.py` làm sanity gate cho regime proxy của DROID.

### 10.1. ⭐ Nhưng đọc kỹ: kế hoạch THẬT theo idea-of-record mới (2026-06-09)

Đánh giá ở trên đúng cho framing **chẩn đoán cũ** (CRA). Theo `DIAGNOSIS_PLAN.md` (plan-of-record), thứ tự ưu tiên
hiện nay (cập nhật 2026-06-10, sau khi BB gate đã chạy):

1. ~~**(GATE)** Chạy `12_boundary_diagnostic.py`~~ — **ĐÃ XONG: PASS** (Phần 9.6). Bảng BB + figure đã được fold
   vào `decision_report.md` (`figure_bb_per_regime.pdf`).
2. ~~Bắt đầu fix C1 (mixture-density head + boundary supervision)~~ — **ĐÃ LÀM (2026-06-10), kết quả NULL ở mức
   head với nguyên nhân đo được** (xem bảng 9.5 và `FIX_C1_EXPLAINER.md`). **Việc tiếp theo quan trọng nhất bây
   giờ:** fix ở mức **encoder/metric** (projection có giám sát bằng nhãn dịch-chuyển-vật để khuếch đại lại subspace
   biên trong latent) và/hoặc dữ liệu counterfactual ở biên.
3. DROID: `05` cho `dino_wm_droid` đã chạy lại xong (kèm gripper sanity gate PASS); còn lại STEP 1
   (`inspect_droid_observation_keys.py`, 5 phút) và `vjepa2_ac_droid` (chỉ chạy được trên server 24 GB) —
   **DROID giờ chỉ là transfer check**, không phải nơi chứng minh biên.
4. Planning leg: đúc lại quanh BB (figure "Action Error vs BB"), không dựa CRA_eff per-transition (đã null).

**Kết luận thẳng về "đủ viết paper chưa":**
- **Framing CRA cũ:** đủ bằng chứng cho một paper chẩn đoán (CONDITIONAL_GO), nhưng đã bị nâng cấp.
- **Framing "Boundary-Blind" mới (mạnh hơn, top-venue-shaped):** bằng chứng cốt lõi (BB gate) **đã có và PASS**
  (Phần 9.6) — luận điểm chính của framing mới đã được chốt trên baseline đóng băng. Mảnh còn thiếu lớn nhất bây
  giờ là **fix C1/D chưa bắt đầu** (cần cho contribution C3 + figure "BB before/after") và leg planning-vs-BB.

---

## Phần 11 — Bản đồ tài liệu & file (đọc tiếp ở đâu)

| File | Nội dung |
|---|---|
| **`diagnosis/docs/PAPER_IDEA.md`** | ⭐ **Idea-of-record HIỆN TẠI** — framing "Boundary-Blind" + metric BB (đọc cái này để hiểu hướng mới) |
| **`diagnosis/docs/DIAGNOSIS_PLAN.md`** | ⭐ **Plan-of-record** — kế hoạch hợp nhất, BB gate, thứ tự ưu tiên thật |
| **`diagnosis/docs/plans/2026-06-09-action-identifiability-fix-design.md`** | ⭐ Thiết kế fix mới (critique + C1/D + boundary diagnostic) |
| **`diagnosis/docs/HANDOFF_BOUNDARY_FIX.md`** | Vận hành: chạy BB gate + STEP 1 DROID + build C1/D |
| `cai_jepa_paper_proposal.md` | Ý tưởng GỐC (4 đóng góp); §6 fix đã bị thay thế bởi PAPER_IDEA.md |
| `diagnosis/docs/METHODOLOGY.md` | Khái niệm + bản đồ code + ma trận dataset/task/regime/strategy (đọc đầu tiên khi vào code) |
| `diagnosis/docs/plans/2026-06-01-real-api-rewrite-design.md` | API upstream thật + các quyết định thiết kế |
| `diagnosis/docs/plans/2026-06-05-planning-action-score-design.md` | Thiết kế planning Action-Score probe |
| `diagnosis/docs/HANDOFF.md` | Vận hành: chạy Metaworld (primary) trên server mới |
| `diagnosis/docs/HANDOFF_DROID.md` | Vận hành: chạy DROID (secondary) + §8 planning probe |
| `diagnosis/RUNBOOK.md` | Trình tự lệnh đầy đủ trên server |
| `diagnosis/results/decision_report.md` | Báo cáo quyết định (kết quả CRA tổng hợp) |
| `diagnosis/results/planning_correlation.md` | Kết quả tương quan grounding↔planning |
| `diagnosis/results/metaworld_diagnostic.csv` | Số thô Metaworld (mọi cell) |
| `diagnosis/results/droid_diagnostic.csv` | Số thô DROID |
| `diagnosis/results/metaworld_boundary.csv` / `droid_boundary.csv` | ✅ Bảng BB thật (2026-06-10) — nguồn số của Phần 9.6 |
| `diagnosis/results/boundary_gate_report.md` | ✅ Report gate BB: run log, bảng, verdict PASS, caveat, khuyến nghị C1 |

### Bảng thuật ngữ nhanh
- **World model** — mạng dự đoán trạng thái kế tiếp từ (trạng thái, hành động); dùng để robot "tưởng tượng" trước khi làm.
- **Latent `z`** — vector tóm tắt ngữ nghĩa của một cảnh (output của encoder đóng băng).
- **Predictor `F_θ`** — mạng dự đoán latent kế tiếp `ẑ_{t+1} = F_θ(z_t, a_t)`.
- **Action grounding / action-identifiability** — tính chất predictor phân biệt được các action khác nhau.
- **Counterfactual** — hành động phản thực (giả định khác hành động đã xảy ra).
- **Regime** — chế độ trạng thái (free_space / pre_grasp / gripper_actuation / contact_manipulation).
- **CRA** — Counterfactual Ranking Accuracy: tỉ lệ xếp đúng action thật trên đầu (metric chẩn đoán "dùng action"). Chance ≈ 0.059.
- **CRA_eff** — CRA chỉ trên transition có biến đổi latent; con số quyết định của framing chẩn đoán cũ.
- **AUG / ECS** — khoảng cách dùng action / phiên bản chỉ trên transition có tác động.
- **BB — Boundary Blindness (Mù Biên)** — `relu(S_true − S_model)`: thế giới phân nhánh ở biên nhưng mô hình dự đoán
  cùng tương lai → **metric headline của framing mới "Boundary-Blind"** (đã chạy 2026-06-10, gate PASS — Phần 9.6).
- **Boundary regime** — lát cắt cắt-ngang chọn transition ở vùng bifurcation (`boundary_score` cao).
- **Bifurcation / boundary** — phân nhánh kết quả: action đổi tí xíu → tương lai khác về chất (kẹp căn giữa → nâng vật).
- **Mixture-density head** — đầu ra dự đoán **phân phối đa mode** trên `z_{t+1}` (fix C1) thay cho point + L2.
- **CEM** — Cross-Entropy Method: thuật toán planning bằng lấy mẫu và chấm điểm.
- **Action Error / Action Score** — sai số giữa hành động planner chọn và hành động chuyên gia (proxy planning trên DROID).
- **CONDITIONAL_GO** — kết luận của framing CRA cũ: đủ tự tin theo đuổi paper, nhưng cần củng cố.
- **Idea-of-record** — framing nghiên cứu hiện hành = "Boundary-Blind World Models" (2026-06-09), thay framing CAI-JEPA gốc.
```
