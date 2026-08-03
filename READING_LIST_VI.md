# 20 paper bắt buộc đọc cho người mới vào hướng nghiên cứu này

**Ngày lập:** 2026-08-03 · **Neo theo:** `paper/main.tex` — *"Decoding Accuracy Does Not Certify
Plannability: An Optimizer-Conditioned Audit of World-Model Costs"*
· **Đối tượng:** người mới join, chưa biết gì về JEPA/world-model planning.

> **Phạm vi:** bám đúng trục của paper đang viết — **cost geometry & optimizer-conditioned audit**.
> Phần belief compression **ngoài phạm vi**, đã loại. Trục action-grounding/CRA (metric CRA,
> boundary blindness, Phase-H) **cũng đã bị loại khỏi main text của paper**, nên chỉ còn ở mục đọc thêm.

---

## 0. Quy tắc uy tín: ba lý do đọc, ba chuẩn khác nhau

Đây là phần quan trọng nhất của tài liệu. Đừng đọc mọi paper với cùng một mức tin cậy.

| Vai trò | Chuẩn uy tín | Cách đọc |
|---|---|---|
| **(A) Hệ thống ta chạy** | Không phải venue, mà là **checkpoint/code đã phát hành và ta tái lập được** | Đọc như tài liệu kỹ thuật. Tin phần mô tả kiến trúc, tự verify phần số liệu. |
| **(B) Nền tảng khái niệm ta dựa vào** | **Bắt buộc peer-review ở venue mạnh.** Đây là những claim ta *đứng lên trên* | Đọc kỹ, được phép trích như sự thật đã kiểm chứng. |
| **(C) Công trình đồng thời (concurrent)** | **Uy tín không liên quan.** Đọc vì *bắt buộc phải phân định ranh giới*, kể cả khi nó là preprint yếu | Đọc để trả lời: "họ đã làm gì rồi, phần nào của ta còn mới?" Không được trích như bằng chứng. |

**Áp dụng vào trường hợp UWM-JEPA (arXiv:2605.25313):** preprint tháng 5/2026, chưa qua bình duyệt,
setting toy (density-matrix latent, task hidden-velocity), không có robot thật, không có planning.
Nó không đủ chuẩn cho vai trò (B). Nó *từng* thuộc vai trò (C) khi paper còn trục counterfactual
objective — nhưng `main.tex` hiện tại **không cite nó**, vì Phase-H đã bị đẩy xuống secondary. Nên
nó rơi khỏi cả ba vai trò → **loại khỏi core 20**, chỉ giữ ở mục đọc thêm với ghi chú rõ.
Lý do y hệt áp dụng cho **ATM** (arXiv:2606.09028): nó là anh em song sinh của metric CRA, mà CRA
không còn trong main text.

**Cân đối của list sau khi sửa:** 12/20 là peer-reviewed ở venue mạnh (ICML, ICLR, NeurIPS, CVPR,
TMLR, CoRL, L4DC, Management Science); 8/20 là preprint — và **7 trong 8 preprint đó nằm ở vai trò
(C)**, tức là đọc để phân định chứ không phải để tin.

---

## 1. Câu chuyện của paper, tóm trong 6 câu

1. World model kiểu JEPA học dự đoán **latent** tương lai `z_{t+1} = f(z_t, a_t)` thay vì pixel.
2. Planning = chạy optimizer (CEM) tìm chuỗi action tối thiểu hoá **cost cuối**, thường là
   `‖z_T − z_goal‖₂`.
3. Paper hỏi một câu rất hẹp: **nếu mọi candidate future đều chính xác tuyệt đối, thì khi nào một
   cost dẫn xuất từ representation vẫn không đỡ nổi planning contact-rich?**
4. Thiết kế: snapshot MuJoCo → thực thi từng candidate CEM trong **simulator thật** → render endpoint →
   encode bằng encoder đóng băng. Dynamics chính xác theo cấu tạo; chỉ còn representation + cost.
5. Kết quả: cost simulator-state giải push 64/64; latent L₂ giải **0/64**; stateprobe (probe có nhãn
   privileged, sai số median chỉ 2.1 cm trên held-out) đạt tối đa **5/64**.
6. Lý do chỉ lộ ra **dưới áp lực search**: tương quan hạng giữa stateprobe và reference tụt từ
   0.43–0.55 (quần thể ban đầu) xuống **0.11–0.16** sau khi CEM refit.

Ba cụm từ phải nắm: **predictive ≠ plannable**, **coverage vs selection failure**,
**cost exploitation dưới refitting**.

---

## 2. Bảng 20 paper

Cột **Cite?** = có xuất hiện trong `paper/refs.bib` / `main.tex` không.

| # | Paper | Venue | Vai trò | Cite? | PDF local |
|---|---|---|---|---|---|
| 1 | A Path Towards Autonomous Machine Intelligence (LeCun) | *Position paper, không bình duyệt* | B⚠ | — | `world_model/jepa.pdf` |
| 2 | I-JEPA | **CVPR 2023** | B | — | `world_model/i-jepa.pdf` |
| 3 | DINO-WM | **ICML 2025** | A | ✅ | `world_model/dino-wm.pdf` |
| 4 | What Drives Success in Physical Planning with JEPWM (Terver) | **TMLR 2026** | A | ✅×3 | `world_model/jepa-success.pdf` |
| 5 | V-JEPA 2 / V-JEPA 2-AC | *arXiv tech report* | A | — | `world_model/v-jepa2.pdf` |
| 6 | Meta-World | **CoRL 2020** | A | ✅ | `world_model/metaworld.pdf` |
| 7 | iCEM — Sample-efficient CEM for Real-time Planning | **CoRL 2020** | B | (CEM: ✅) | *tải về* |
| 8 | TD-MPC2 | **ICLR 2024** | B | — | `world_model/TD-MPC2.pdf` |
| 9 | Objective Mismatch in Model-based RL (Lambert) | **L4DC 2020** | B | ✅ | *tải về* |
| 10 | When to Trust Your Model: MBPO (Janner) | **NeurIPS 2019** | B | ✅ | *tải về* |
| 11 | The Optimizer's Curse (Smith & Winkler) | **Management Science 2006** | B | ✅ | *tải về* |
| 12 | Defining and Characterizing Reward Hacking (Skalse) | **NeurIPS 2022** | B | ✅ | *tải về* |
| 13 | Scaling Laws for Reward Model Overoptimization (Gao) | **ICML 2023** | B | ✅ | *tải về* |
| 14 | Goodhart's Law in RL (Karwowski) | **ICLR 2024** | B | ✅ | *tải về* |
| 15 | Closing the Train-Test Gap in WM for Gradient-Based Planning | *arXiv 2512.09929* | B⚠ | ✅ | `world_model/train-test-gap-wm.pdf` |
| 16 | TRM: Beyond Euclidean Proximity | *arXiv 2605.22164* | **C** | ✅ | *tải về* |
| 17 | IMWM: Intuition Models Complement World Models | *arXiv 2606.01626* | **C** | ✅ | `world_model/imwm.pdf` |
| 18 | ACID: Action Consistency via Inverse Dynamics | *arXiv 2607.02403* | **C** | ✅ | `world_model/ACID.pdf` |
| 19 | Predictive but Not Plannable: RC-aux | *arXiv 2605.07278* | **C** | ✅ | `world_model/Predictive-but-Not-Plannable.pdf` |
| 20 | Latent Geometry Beyond Search | *arXiv 2605.08732* | **C** | ❌ *(thiếu — xem §6)* | `world_model/latent-geometry.pdf` |

---

## TẦNG 1 — Nền tảng: JEPA là gì và tại sao (tuần 1, nửa đầu)

### 1. A Path Towards Autonomous Machine Intelligence — LeCun 2022 · *(B⚠ position paper)*
`world_model/jepa.pdf`

- ⚠ **Đây là tuyên ngôn, không phải kết quả thực nghiệm, và không qua bình duyệt.** Đọc nó như
  *tập hợp giả định*, không phải bằng chứng. Không bao giờ trích nó để chứng minh điều gì.
- **Trọng tâm:** §3 (kiến trúc JEPA, collapse), §4 (cost-based planning: actor tìm action tối thiểu
  hoá cost qua world model). Bỏ qua phần intrinsic motivation/emotion.
- **Take-away:** giả định ngầm "latent tốt để **dự đoán** ⇒ latent tốt để **tối ưu hoá**" nằm ở đây.
  Toàn bộ paper của ta là phép thử nhân quả cho giả định đó — và kết quả là **không**.

### 2. I-JEPA · *(B, CVPR 2023)*
`world_model/i-jepa.pdf` · arXiv:2301.08243

- **Ý chính:** dự đoán representation của block bị che từ block ngữ cảnh; không augmentation thủ công,
  không tái tạo pixel.
- **Phải nắm:** context/target encoder, EMA target, **representation collapse** và các cơ chế chống nó.
- **Take-away:** giải thích vì sao latent JEPA **không có decoder** — và do đó vì sao ta không có cách
  nào "nhìn vào" xem cost latent đang thực sự đo cái gì. Đó chính là lý do paper phải dựng stateprobe.

---

## TẦNG 2 — Bốn thứ bạn sẽ trực tiếp chạy (tuần 1, nửa sau) · *vai trò A*

> Bốn mục này đọc như **tài liệu hệ thống**. Uy tín đến từ checkpoint đã phát hành mà ta tái lập được,
> không từ venue. Nhưng cũng vì thế: **mọi con số họ báo cáo phải tự verify**, đừng trích lại.

### 3. DINO-WM: World Models on Pre-trained Visual Features Enable Zero-Shot Planning · *(ICML 2025)*
`world_model/dino-wm.pdf` · arXiv:2411.04983 · cited: `zhou2024dinowm`

- **Ý chính:** đóng băng encoder DINOv2, chỉ học predictor trên patch token, planning zero-shot bằng
  CEM với cost = L2 tới latent ảnh goal.
- **Phải nắm:** latent là **patch token** (không phải vector 1 chiều), cost L2 trên toàn token, CEM
  receding-horizon, goal cho bằng **ảnh**.
- **Take-away:** đây chính là "latent $L_2$" trong Bảng kết quả của paper — cái cho **0/64**.
- **Tự kiểm tra:** vẽ được sơ đồ ảnh → encoder → predictor → cost → CEM → action, và chỉ ra paper của
  ta cắt bỏ khối nào (predictor) rồi thay bằng gì (MuJoCo rollout).

### 4. What Drives Success in Physical Planning with JEPWM — Terver et al. · *(TMLR 2026, đã bình duyệt)*
`world_model/jepa-success.pdf` · arXiv:2512.24497 · cited: `jepawms2025` (**3 lần — nhiều nhất paper**)

- **Quan trọng nhất tầng này:** phát hành đúng checkpoint `dino_wm_metaworld` + `jepa_wm_metaworld` và
  đúng stack planning mà paper audit, qua interface `EncPredWM`.
- **Phải nắm:** các ablation design choice, cấu hình `L2_cem`, và benchmark họ dùng.
- **Take-away:** họ trả lời "cái gì làm planning **thành công**"; ta trả lời câu bổ sung —
  "khi thất bại thì hỏng ở **khâu nào**". Đây là quan hệ bổ trợ, không phải đối đầu.
- ⚠ **Vận hành:** PushT / PointMaze / Wall là sanity check **bão hoà**, cấm dùng làm bằng chứng luận điểm.

### 5. V-JEPA 2 / V-JEPA 2-AC · *(⚠ arXiv tech report, không qua bình duyệt)*
`world_model/v-jepa2.pdf` · arXiv:2506.09985

- **Vì sao vẫn giữ dù không peer-review:** nó là *hệ thống*, không phải *claim* — và là action-conditioned
  JEPA duy nhất có kết quả robot thật. Đọc phần **-AC**: cách đưa action vào predictor, cost khi planning.
- ⚠ **Bug đã biết:** bản Meta gốc lỗi action-norm; repo dùng bản Terver-fixed (`vjepa2_ac_droid`).
  Xem `CLAUDE.md` §"Critical pitfalls".
- **Cảnh báo về claim scale:** V-JEPA 2.1 (arXiv:2603.14482, `world_model/v-jepa2-1.pdf`) báo cáo
  **+20 điểm grasping** nhờ dense feature. Đây là **lời giải thích thay thế** mạnh nhất cho kết quả của
  ta, và là lý do paper phải giới hạn phạm vi vào "hai checkpoint đã phát hành", không nói "JEPA nói chung".

### 6. Meta-World · *(CoRL 2020)*
`world_model/metaworld.pdf` · arXiv:1910.10897 · cited: `yu2020metaworld`

- **Đọc để trả lời:** `mw-push` và `mw-pick-place` định nghĩa success thế nào? state 39 chiều gồm gì?
- **Phải nắm chính xác:** paper dùng **task flag của simulator ở bước cuối**, không phải any-step latch;
  bán kính object-to-goal là **5 cm (push)** và **7 cm (pick-place)**. Đọc nhầm chỗ này là đọc sai
  toàn bộ bảng kết quả.
- **Liên quan repo:** stratification MetaWorld dùng **proxy** từ state 39-chiều (dịch chuyển vật thể =
  proxy tiếp xúc) vì dataset HF không có contact GT của MuJoCo.

---

## TẦNG 3 — Optimizer: đối tượng bị audit (tuần 2, nửa đầu) · *vai trò B*

### 7. iCEM — Sample-efficient Cross-Entropy Method for Real-time Planning · *(CoRL 2020)*
Pinneri et al. · arXiv:2008.06389 *(tải về — verify mã arXiv khi tải)* · CEM gốc cited: `rubinstein1997cem`

- **Bắt buộc, không thể bỏ.** Cả paper xoay quanh từ **"optimizer-conditioned"**. Không hiểu vòng lặp
  refit thì không đọc được kết quả cốt lõi.
- **Phải nắm:** elite set, **refitting phân phối qua từng iteration**, colored noise, momentum, shift-init.
- **Take-away:** CEM **không** lấy mẫu i.i.d. — nó chủ động dịch phân phối về phía vùng cost thấp.
  Nếu vùng cost thấp là *sai số dư của readout*, CEM đi thẳng vào đó. Cấu hình của paper:
  100 candidate, elite top-10%, **6 vòng refit**, horizon H=6.
- **Tự kiểm tra:** giải thích vì sao "median 2.1 cm trên held-out expert frame" **không** kéo theo
  "xếp hạng đúng trên quần thể do chính CEM sinh ra". Trả lời được câu này là hiểu được cả paper.

### 8. TD-MPC2 · *(ICLR 2024)*
`world_model/TD-MPC2.pdf` · arXiv:2310.16828

- **Ý chính:** latent dynamics **hướng theo task** (học cùng reward + value), planning bằng MPPI có
  terminal value học được.
- **Vì sao đọc:** đây là **đối chứng khái niệm**. TD-MPC2 không gặp lỗi này theo cùng cách, vì latent
  của nó được huấn luyện *cho điều khiển*, không phải cho dự đoán chung.
- **Take-away:** chứng minh "cost geometry" là thứ **học được**, và là lập luận phản biện trực tiếp
  triết lý encoder-đóng-băng của DINO-WM.

---

## TẦNG 4 — Vì sao proxy sụp dưới áp lực tối ưu (tuần 2, nửa sau + tuần 3) · *vai trò B, toàn bộ peer-reviewed*

> Năm mục 9–14 là **xương sống lý thuyết** của paper và đều đã bình duyệt. Nếu chỉ có thời gian đọc
> 6 paper trong cả list, đọc nhóm này.

### 9. Objective Mismatch in Model-based RL — Lambert et al. · *(L4DC 2020)* · cited: `lambert2020objective`
- Mục tiêu huấn luyện model (likelihood/MSE) **không** tương quan đơn điệu với hiệu năng điều khiển.
- **Take-away:** phát biểu gốc, ở phía *dynamics*. Paper của ta chuyển đúng lập luận đó sang phía
  *cost/representation*, và chứng minh nó vẫn đúng **ngay cả khi dynamics hoàn hảo tuyệt đối** — đó là
  phần mới.

### 10. When to Trust Your Model (MBPO) — Janner et al. · *(NeurIPS 2019)* · cited: `janner2019trust`
- Policy **exploit sai số của learned dynamics**; giới hạn rollout theo độ tin cậy của model.
- **Take-away:** thiết lập kênh exploit "cổ điển" ở phía dynamics. Paper ta **đóng kênh này bằng cấu tạo**
  (oracle rollout) để cô lập kênh còn lại. Cite để phân định, không phải để đồng nhất.

### 11. The Optimizer's Curse — Smith & Winkler · *(Management Science 2006)* · cited: `smith2006optimizer`
- **Bị đánh giá thấp nhất nhưng cực kỳ load-bearing.** Chọn max trên các ước lượng có nhiễu thì giá trị
  thật của cái được chọn **luôn** thấp hơn ước lượng — kể cả khi ước lượng hoàn toàn không thiên lệch.
- **Take-away:** đây chính là **null hypothesis** của paper. Kết quả "stateprobe chọn sai" chỉ có ý nghĩa
  khi vượt qua được **matched-residual null** — tức là sai lệch còn *nhiều hơn* mức mà optimizer's curse
  thuần tuý gây ra. Không hiểu paper này thì không hiểu vì sao paper phải dựng cái null đó.

### 12. Defining and Characterizing Reward Hacking — Skalse et al. · *(NeurIPS 2022)* · cited: `skalse2022defining`
- Hình thức hoá khi nào cải thiện proxy **làm giảm** true reward; điều kiện "unhackable" chặt đến mức
  thực tế gần như không đạt được.
- **Take-away:** từ vựng chuẩn. Trong paper: "proxy" = latent L₂ / stateprobe; "true" = cost trên
  simulator state.

### 13. Scaling Laws for Reward Model Overoptimization — Gao et al. · *(ICML 2023)* · cited: `gao2023overopt`
- Đo proxy vs gold khi tăng áp lực tối ưu (RL và best-of-n); xuất hiện dạng **tăng rồi giảm**.
- ⚠ **Quy tắc viết bài, không phải gợi ý:** đây là *hình dạng đường cong* phải chứng minh trước khi được
  dùng chữ **"overoptimization"**. Chưa đủ power thống kê thì viết là
  **"selection into residual cost-error pockets"**.

### 14. Goodhart's Law in RL — Karwowski et al. · *(ICLR 2024)* · cited: `karwowski2024goodhart`
- Định lượng phân kỳ proxy/true khi áp lực tối ưu tăng; early stopping và reward worst-case.
- **Take-away:** cho định nghĩa **đo được** của "áp lực tối ưu" — chính là trục hoành của các thí nghiệm
  quét budget.

### 15. Closing the Train-Test Gap in WM for Gradient-Based Planning · *(⚠ arXiv 2512.09929)* · cited: `parthasarathy2025traintest`
`world_model/train-test-gap-wm.pdf`

- **Ý chính:** planner dùng gradient truy vấn model **khác hẳn lúc train** → sinh chuỗi action
  OOD/adversarial; họ vá bằng tổng hợp dữ liệu lúc train.
- **Take-away:** kênh exploit **thứ ba** (gradient + dynamics). Paper ta dùng sampling planner và đóng
  kênh dynamics → cite để **vạch ranh giới**. ⚠ Preprint, nên chỉ dùng làm đối chiếu định tính.

---

## TẦNG 5 — Công trình đồng thời (tuần 4) · *vai trò C — đọc để phân định, KHÔNG phải để tin*

> Cả 5 mục đều là preprint 2026 chưa bình duyệt. Chúng ở đây **không phải vì uy tín** mà vì reviewer
> sẽ hỏi "cái này khác gì paper X?". Nhiệm vụ khi đọc: với mỗi paper, viết đúng một câu
> *"họ giữ cố định cái gì, thay đổi cái gì, và ta khác chỗ nào"*.

### 16. TRM — Beyond Euclidean Proximity · *(arXiv:2605.22164)* · cited: `trm2026`
*(chưa có PDF local — tải về; repo đã có baseline: `diagnosis/results/trm_heldout_summary.json`)*

- **Gần paper của ta nhất. Đọc kỹ nhất trong cả list.**
- **Họ làm gì:** giữ nguyên encoder, dynamics, sampler, CEM, budget — **chỉ thay terminal cost**.
  Latent thô xếp hạng sai dù XY giải mã tuyến tính gần hoàn hảo; cost state thật đạt 100%; tăng mạnh
  search *không* cứu được objective sai; **SCSA** so sánh thứ tự cost trên cùng một tập candidate;
  can thiệp subspace: rowspace XY hữu ích chiếm **<1%** latent MSE.
- ⚠ **Để không viết sai:** TRM **đã** audit selection và **đã** triển khai cost sửa bên trong CEM.
  Câu "prior work chỉ báo cáo aggregate success" là **sai**. Phần còn lại của ta: contact-rich Franka
  trên checkpoint đã phát hành, cộng giao thức **mine → repair → re-mine** trên episode tách rời.

### 17. IMWM · *(arXiv:2606.01626)* · cited: `imwm2026`
`world_model/imwm.pdf`

- **Họ làm gì:** thay predictor bằng **dynamics thật của môi trường**, giữ nguyên mọi thứ còn lại. Rồi
  hỏi: quần thể CEM có candidate nào đạt goal không? Kết quả: ở các ô thất bại, **gần như không có** →
  lỗi **coverage**, không phải lỗi ranking.
- **Take-away kép:**
  1. **Oracle dynamics không phải phát kiến của ta** — IMWM làm trước. Phải cite đúng chỗ.
  2. Đây là **giả thuyết đối lập phải bác bỏ**: nếu thất bại của ta cũng chỉ vì không có candidate tốt
     nào, kết luận "cost chọn sai" sụp đổ. Vì thế paper bắt buộc đo **exact-success availability**.
- **Tự kiểm tra:** phát biểu khác biệt *coverage failure* vs *selection failure* trong một câu, và chỉ
  ra số liệu nào trong `main.tex` dùng để tách hai cái đó.

### 18. ACID · *(arXiv:2607.02403)* · cited: `acid2026`
`world_model/ACID.pdf`

- **Họ làm gì:** thêm residual inverse-dynamics cycle theo từng bước vào cost planning, scale thích ứng;
  cải thiện trên 4 world model, 6 task.
- **Câu hỏi thí nghiệm quan trọng:** chạy ACID trên **cả learned và oracle dynamics**. Lợi ích biến mất
  dưới oracle dynamics ⇒ nó chữa failure surface *khác*. Vẫn còn ⇒ phản ví dụ đáng kể cho ta.

### 19. RC-aux — Predictive but Not Plannable · *(arXiv:2605.07278)* · cited: `rcaux2026`
`world_model/Predictive-but-Not-Plannable.pdf`

- **Họ làm gì:** candidate có thể **gần trong latent nhưng không tới được trong horizon hữu hạn**;
  thêm head reachability có điều kiện budget, nhãn từ offset quỹ đạo + hard negative theo thời gian.
- **Take-away:** cùng chẩn đoán, khác cách chữa (họ huấn luyện thêm; ta audit checkpoint đã phát hành).
  ⚠ Không bao giờ viết "chúng tôi là người đầu tiên chỉ ra predictive ≠ plannable".

### 20. Latent Geometry Beyond Search · *(arXiv:2605.08732)* · **❌ chưa được cite — xem §6**
`world_model/latent-geometry.pdf`

- **Họ làm gì:** bỏ hẳn search online, thay bằng inverse-dynamics có điều kiện goal
  `(z_t, z_goal, horizon) → a`; đạt/vượt CEM ở 7/8 setting, rẻ hơn 100–130×. Sweep của họ gồm
  CEM, MPPI, iCEM và gradient planner.
- **Take-away kép:**
  1. Nếu "planner là kẻ tấn công" thì **bỏ planner** là cách chữa hiển nhiên. Repo đã thử (E1) và nhận
     **null** — controller amortized cũng không vượt bức tường contact. Điều này *thu hẹp* kết luận.
  2. Sweep nhiều optimizer của họ là **cảnh báo trực tiếp**: kết luận chỉ dựa trên CEM sẽ bị reviewer
     đánh. Đây là lý do tồn tại của `slurm_planner_generality.sh`.

---

## 3. Lộ trình 4 tuần

| Tuần | Paper | Sản phẩm phải nộp |
|---|---|---|
| 1 | 1–6 | Sơ đồ khối ảnh → encoder → predictor → cost → CEM → action; khoanh khối paper cắt bỏ và cái thay vào |
| 2 | 7–11 | Giải thích bằng lời: vì sao sai số 2.1 cm trên held-out không đảm bảo xếp hạng đúng dưới CEM refit |
| 3 | 12–15 | 1 đoạn: khi nào được viết "overoptimization", khi nào phải viết "selection into error pockets" |
| 4 | 16–20 + `paper/main.tex` | Bảng 5 dòng: mỗi paper vai trò C — giữ cố định gì / đổi gì / ta khác chỗ nào |

Sau tuần 4, đọc trong repo theo thứ tự: `2026-06-25-oracle-ladder-results.md` →
`2026-07-13-iclr-literature-positioning-audit.md` (**văn bản quan trọng nhất repo**) →
`paper/main.tex` → `diagnosis/docs/CLAIMS_EVIDENCE.md`.

---

## 4. Đã loại khỏi core 20, và lý do

| Paper | Lý do loại |
|---|---|
| **UWM-JEPA** (arXiv:2605.25313) | Preprint chưa bình duyệt; setting toy (density-matrix latent, hidden-velocity task), không robot thật, không planning. Chỉ liên quan trục Phase-H counterfactual objective — **`main.tex` không cite**. Đọc thêm nếu quay lại trục objective. |
| **ATM** (arXiv:2606.09028) | Preprint; là anh em song sinh của metric CRA/boundary-blindness — mà CRA đã **bị bỏ khỏi main text**. Chỉ thành must-compare nếu CRA quay lại. |
| World Model Survey (2026) | Preprint; hữu ích để định hướng nhưng làm loãng phạm vi. `world_model/world-model-survey.pdf` |
| WAV, MiraBench, What-If World | Preprint; thuộc trục action-reliability đã đóng. |
| V-JEPA 2.1 (arXiv:2603.14482) | Không phải để học, mà là **lời giải thích thay thế** cần nêu trung thực. Đã ghi ở mục #5. |
| Decision-centric WM evaluation (arXiv:2606.15032) | Preprint, được cite (`yu2026wmeval`) nhưng chỉ đóng vai trò một câu framing. |
| PLDM, LeWorldModel, VICReg, Temporal Straightening | Chiều sâu tuỳ chọn, không load-bearing cho paper hiện tại. Có PDF trong `world_model/`. |

---

## 5. Ghi chú nguồn

- Mã arXiv của các mục có PDF local đã verify bằng `pdftotext` trực tiếp trên file.
- **Chưa có PDF trong repo:** iCEM (#7), Lambert (#9), Janner (#10), Smith & Winkler (#11),
  Skalse (#12), Gao (#13), Karwowski (#14), TRM (#16). Mã arXiv của iCEM (2008.06389) và
  Lambert (2002.04523) ghi theo trí nhớ — **verify khi tải**. Các mục còn lại tra theo
  `paper/refs.bib` (đã có đầy đủ entry).
- **TRM nên tải về đặt vào `world_model/`** cho nhất quán, vì repo đã chạy baseline của nó.

---

## 6. Hai việc phát hiện khi đối chiếu list với `main.tex`

1. **`main.tex` chưa cite Latent Geometry Beyond Search** (arXiv:2605.08732), trong khi audit văn liệu
   §"Must cite prominently" xếp nó vào nhóm 1 cùng TRM/IMWM/ACID/RC-aux. Đây là **khoảng trống citation**,
   không phải quyết định loại bỏ — nó cũng là paper duy nhất sweep nhiều optimizer, tức là chính chỗ
   reviewer sẽ chọc vào rủi ro "CEM-only". Nên bổ sung vào Related Work đoạn "Closest concurrent diagnoses".
2. **`refs.bib` hiện có 16 entry; 9/20 mục của list này đã được cite.** Phần chưa cite (LeCun, I-JEPA,
   V-JEPA 2, iCEM, TD-MPC2) là kiến thức nền cho người đọc list, không nhất thiết phải vào paper —
   trừ **iCEM**, đáng cân nhắc thay cho `rubinstein1997cem` vì paper mô tả CEM có refitting lặp,
   gần iCEM hơn là CEM gốc 1997.
