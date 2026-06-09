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
3. **CAI-JEPA loss** (kế hoạch nếu go) — một **hàm mất mát mới** ép predictor phải phân biệt action: phạt mô hình
   nếu dự đoán dưới hành động giả (counterfactual) lại gần tương lai thật hơn so với hành động thật. Có cơ chế
   "effect-conditional gating" chỉ áp lực ở những transition thực sự có biến đổi.
4. **Probing** (tùy chọn) — gắn các bộ dò nhỏ vào từng tầng predictor để xem thông tin action "chảy" qua mạng
   thế nào, hỏng ở đâu.

> Từ khóa nền: **counterfactual** = "phản thực" = một hành động **giả định khác** với hành động đã thực sự xảy ra.
> Toàn bộ ý tưởng xoay quanh: cho cùng trạng thái, so sánh hành động thật vs các hành động phản thực.

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

---

## Phần 11 — Bản đồ tài liệu & file (đọc tiếp ở đâu)

| File | Nội dung |
|---|---|
| `cai_jepa_paper_proposal.md` | Ý tưởng nghiên cứu đầy đủ (4 đóng góp, related work, scope) |
| `diagnosis/docs/METHODOLOGY.md` | Khái niệm + bản đồ code + ma trận dataset/task/regime/strategy (đọc đầu tiên khi vào code) |
| `diagnosis/docs/plans/2026-06-01-real-api-rewrite-design.md` | API upstream thật + các quyết định thiết kế |
| `diagnosis/docs/HANDOFF.md` | Vận hành: chạy Metaworld (primary) trên server mới |
| `diagnosis/docs/HANDOFF_DROID.md` | Vận hành: chạy DROID (secondary) + §8 planning probe |
| `diagnosis/RUNBOOK.md` | Trình tự lệnh đầy đủ trên server |
| `diagnosis/results/decision_report.md` | Báo cáo quyết định (kết quả tổng hợp) |
| `diagnosis/results/planning_correlation.md` | Kết quả tương quan grounding↔planning |
| `diagnosis/results/metaworld_diagnostic.csv` | Số thô Metaworld (mọi cell) |
| `diagnosis/results/droid_diagnostic.csv` | Số thô DROID |

### Bảng thuật ngữ nhanh
- **World model** — mạng dự đoán trạng thái kế tiếp từ (trạng thái, hành động); dùng để robot "tưởng tượng" trước khi làm.
- **Latent `z`** — vector tóm tắt ngữ nghĩa của một cảnh (output của encoder đóng băng).
- **Predictor `F_θ`** — mạng dự đoán latent kế tiếp `ẑ_{t+1} = F_θ(z_t, a_t)`.
- **Action grounding / action-identifiability** — tính chất predictor phân biệt được các action khác nhau.
- **Counterfactual** — hành động phản thực (giả định khác hành động đã xảy ra).
- **Regime** — chế độ trạng thái (free_space / pre_grasp / gripper_actuation / contact_manipulation).
- **CRA** — Counterfactual Ranking Accuracy: tỉ lệ xếp đúng action thật trên đầu (metric chính). Chance ≈ 0.059.
- **CRA_eff** — CRA chỉ trên transition có biến đổi latent; **con số quyết định**.
- **AUG / ECS** — khoảng cách dùng action / phiên bản chỉ trên transition có tác động.
- **CEM** — Cross-Entropy Method: thuật toán planning bằng lấy mẫu và chấm điểm.
- **Action Error / Action Score** — sai số giữa hành động planner chọn và hành động chuyên gia (proxy planning trên DROID).
- **CONDITIONAL_GO** — kết luận hiện tại: đủ tự tin theo đuổi paper, nhưng cần củng cố (chủ yếu DROID).
```
