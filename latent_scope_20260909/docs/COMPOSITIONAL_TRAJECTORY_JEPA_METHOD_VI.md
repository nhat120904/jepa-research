# Compositional Trajectory JEPA: method, flow planning và bằng chứng hiện có

> Tài liệu giải thích hướng nghiên cứu tại ngày 15/09/2026. Tài liệu chỉ dùng các
> thí nghiệm có kết quả khoa học còn dùng được; không dùng job lỗi setup, restore,
> render, replay hay lỗi readout làm bằng chứng.

## 1. Ý tưởng trong một câu

Giữ GR00T là policy sinh các action sequence ngắn hợp lý; dùng trajectory JEPA có
history để dự đoán *hiệu ứng* của từng sequence; dùng value/score có điều kiện theo task
để chọn sequence tốt nhất; thực thi nó và quan sát lại để planning tiếp.

Giả thuyết khoa học ban đầu không chỉ là “JEPA dự đoán future frame tốt”. Giả thuyết là:

> Một summary biểu diễn **hiệu ứng bên trong** action segment và ghép được qua nhiều
> segment sẽ giúp chọn action cho manipulation task phụ thuộc history tốt hơn endpoint,
> frame rollout, memory tuần tự thông thường, hay direct value baseline.

`ScrubCuttingBoard` được chọn vì native evaluator lưu các scrub contact đã chấp nhận theo
thời gian: action có ích bao nhiêu phụ thuộc cả vào sponge đã chạm những vị trí nào trước
đó, chứ không chỉ endpoint image hiện tại.

## 2. Phân biệt đúng vai trò từng component

| Component | Input | Output | Dùng để làm gì | Trạng thái trong project |
|---|---|---|---|---|
| **GR00T N1.5** | Ba ảnh RGB hiện tại, robot state, language task description | Một chunk 16 native actions | Sinh các robot action hợp lý; behavioral prior/policy | Đã chạy frozen, chưa fine-tune bởi method này |
| **Observation encoder** | Camera feature và proprio history | Latent hiện tại `z_t`, history/memory `m_t` | Biểu diễn robot đã ở đâu trong task mà không dùng simulator-only state khi deploy | Đã dùng trong pilot offline |
| **Action-conditioned segment predictor** | `z_t`, `m_t`, candidate action segment `a[t:t+H]` | Endpoint dự đoán `z_hat[t+H]`, segment summary `s_hat` | Tưởng tượng candidate action cụ thể sẽ gây hậu quả gì | Đã test offline, chưa nối với GR00T planning |
| **Memory update** | Memory hiện tại và predicted segment effect | Memory tiếp theo `m_hat[t+H]` | Mang thông tin tích lũy qua segment | Đã test offline |
| **Learned composer** | `s_hat(left)`, `s_hat(right)` | Summary cho đoạn ghép | Ghép effect các đoạn ngắn để suy luận đoạn dài | Đã test offline; đóng trên cấu hình Scrub này |
| **Value / score head** | Memory hiện tại, predicted outcome/summary, task signal | Scalar score, dự kiến là xác suất eventual success | Biến “hậu quả dự đoán” thành “candidate này tốt cho task không?” | Có scaffold, chưa train/evaluate end-to-end |
| **Planner / selector** | Score của `K` GR00T candidates | Index candidate score cao nhất | Chọn chunk, thực thi, quan sát lại và replan | Chưa chạy end-to-end |
| **Simulator oracle** | Counterfactual branch rollouts | Native progress/success thật | Upper bound phục vụ thí nghiệm; robot deploy không có | Đã dùng ở Stage B |

Điểm quan trọng nhất:

```text
GR00T không phải JEPA world model.
JEPA không phải action generator.
Value head không phải planner.
Oracle không phải component khi deploy.
```

## 3. Input và output chi tiết

### 3.1 GR00T sinh action proposal

`PandaOmronDataConfig` đã pin cho GR00T các input:

```text
ảnh RGB left agent-view hiện tại
ảnh RGB right agent-view hiện tại
ảnh RGB eye-in-hand hiện tại
end-effector / gripper / base state hiện tại
human task description bằng ngôn ngữ
```

GR00T **không** nhận goal image, native contact history, native task-progress variable,
simulator state, hay một history dài được đưa tường minh. Một policy query trả về 16 native
action indices. Trong Stage B, các random seed bị khóa khác nhau sinh `K = 8` proposal;
8 action đầu (và trong một kiểm tra có kiểm soát, đủ 16 action) được dùng làm intervention.

Nói cách khác, GR00T biết goal qua language instruction và có thể suy progress nhìn thấy
qua ảnh/proprio. Nó cũng có thể thành công nhờ imitation: học pattern hành động thường có
trong demo thành công. Nhưng interface của nó không có exact hidden record về các accepted
scrub contacts.

### 3.2 JEPA dự đoán gì

Với candidate action sequence được cung cấp:

\[
a^j_{t:t+H-1}, \qquad j \in \{1,\ldots,K\},
\]

model dự kiến dự đoán:

\[
(\hat z^j_{t+H},\hat s^j,\hat m^j_{t+H}) =
P_\theta(z_t,m_t,a^j_{t:t+H-1}).
\]

Trong đó:

- `z_hat`: latent endpoint dự đoán sau segment;
- `s_hat`: summary của những gì xảy ra **bên trong** segment, không chỉ endpoint;
- `m_hat`: history/memory dự đoán sau segment.

Lý do cần `s_hat`: hai trajectory có thể tới endpoint gần giống nhau nhưng có interior effect
khác nhau. Với Scrub, một đoạn có thể chà vùng board mới, còn đoạn khác chỉ chà lặp lại vùng
đã chà; endpoint arm pose có thể không phân biệt được hai trường hợp.

### 3.3 Composition làm gì

Với segment dài gồm hai đoạn liên tiếp,

\[
a_{t:t+H-1}=a_{t:t+h-1}\Vert a_{t+h:t+H-1},
\]

learned composition mong muốn:

\[
C_\phi(s_{left},s_{right}) \approx s_{whole}.
\]

Mục tiêu là học segment ngắn rồi ghép effect để tưởng tượng horizon dài, thay vì bắt model
nhảy trực tiếp sang sequence dài chưa từng train. Nó không đơn giản là cộng số contact:
summary phải giữ đủ thông tin không gian/history để contribution của đoạn phải phụ thuộc
vào đoạn trái.

### 3.4 Value/score tính như thế nào

Chỉ dự đoán effect chưa nói effect nào tốt cho task. Vì thế Stage C ban đầu có value head:

\[
q_j=V_\psi(m_t,\hat z^j_{t+H},\hat s^j,\text{task}),
\]

với `q_j` là dự đoán xác suất eventual native success. Khi train, nhãn là eventual success
của rollout; khi deploy, head chỉ thấy context hiện tại, task và predicted consequence của
candidate.

Selector chọn:

\[
j^*=\arg\max_j q_j.
\]

Trong scaffold sớm có hai task cố định nên task signal là `task_id`; hệ tổng quát nên dùng
language-goal representation. Score “contact tăng nhiều” không đủ: Scrub còn đòi sweep và
release đúng. Vì vậy value head là component bắt buộc của method planning hoàn chỉnh.

## 4. Flow planning dự kiến khi deploy

Method dự kiến là **receding-horizon selector**, không thay GR00T và không chỉ dự đoán một
lần đến terminal.

```text
quan sát cameras + robot state + language task
        │
        ├── frozen GR00T lấy mẫu K action chunks
        │
        ├── với từng candidate:
        │     JEPA dự đoán endpoint + path summary + next memory
        │     value head score hậu quả dự đoán theo task
        │
        ├── thực thi chunk có score lớn nhất (hoặc prefix ngắn của nó)
        │
        └── quan sát lại và lặp
```

Nó chỉ có thể giúp khi candidate bank của GR00T có variation liên quan task. Nếu mọi
candidate gần như tương đương, selector hoàn hảo cũng gần như không cải thiện được gì.

Method ban đầu cũng **không cải thiện GR00T action generation**: không fine-tune GR00T và
không optimize action sequence. Thêm value-guided action optimization, diffusion guidance
hay action-head fine-tuning sẽ là một method rộng hơn, khác claim ban đầu.

## 5. Stage B thực sự đã đo gì

Stage B là một diagnostic rẻ hơn, gọi là **one-intervention oracle**, chứ không phải flow
deploy ở mục 4. Tại một prefix, nó làm:

```text
lấy mẫu K=8 GR00T candidate chunks từ cùng state
với từng candidate j:
    chạy j đúng H=8 native actions
    giao phần còn lại cho frozen GR00T đến native horizon
    dùng native RoboCasa evaluator ghi terminal success Y[i,j,s]
```

`s` là continuation seed. Candidate 0 là GR00T behaviour baseline đã khóa trước. Single-seed
availability oracle là:

\[
\max_j Y_{i,j,s}-Y_{i,0,s}.
\]

Oracle **không** lấy gold action từ demonstration. Nó chạy mọi candidate branch trong
simulator và dùng outcome counterfactual thật. Vì vậy candidate “success” chỉ có nghĩa:

> Candidate này cộng với continuation seed GR00T này đã thành công.

Nó không chứng minh 8 action đầu intrinsically tốt, vì phần lớn episode sau đó do GR00T
quyết định. Dùng common continuation seed giữa candidates giảm nhiễu so sánh, nhưng không
loại bỏ nhiễu continuation.

Do đó Stage B chỉ đo headroom cho protocol hẹp:

```text
thay một chunk 8 action của GR00T, sau đó để GR00T chạy nốt.
```

Nó **không** đo headroom của repeated GR00T-proposal + JEPA-selection planning. Full terminal
oracle lặp ở mọi decision time sẽ branch gần theo cấp số mũ. Một future protocol thực tế cần
hoặc (a) local native-progress oracle ở mỗi lần replan, hoặc (b) ước lượng expected terminal
value của candidate qua nhiều continuation seed rồi mới chạy closed-loop reranking.

## 6. Các thí nghiệm khoa học hợp lệ đã chạy

### 6.1 Baseline GR00T — job 51947

- Official target-posttrained GR00T checkpoint, current RoboCasa runtime.
- `ScrubCuttingBoard`: 3/10 success; `RinseSinkBasin`: 6/10 success.
- Mục đích: xác nhận policy có thể tạo successful trajectory ở runtime hiện tại để dùng làm
  proposal/continuation policy.

**Kết luận:** GR00T đủ dùng cho diagnostic giới hạn. 10 episode không phải paper replication
hay benchmark estimate; thí nghiệm này không test composition, planner hay value model.

### 6.2 One-shot candidate branches hợp lệ — job 52399 và qualification sau đó

Profile Stage B hợp lệ tại event-aligned prefix thấy candidate post-state đa dạng. Ở một
Scrub prefix với `H=8`, 7/8 branch success, nhưng candidate 0 cũng success, nên gain oracle
so với candidate 0 là 0 pp. Các ô còn lại bị bão hòa: mọi candidate đều success.

Qualification sau đó ở partial-contact prefix tìm thấy vài apparent recovery có điều kiện:

- Một screen tám prefix có hai prefix mà candidate khác success còn candidate 0 fail:
  +25 pp single-seed availability screen result.
- Khi khóa candidate đã chọn và chạy seed continuation mới, gain không ổn định. Một lượt có
  selected 6/16 so với baseline 5/16, gồm năm wins và bốn losses; đây là lặp trên hai prefix
  đã chọn, không phải 16 scene độc lập.

**Kết luận:** GR00T candidate có thể khác terminal outcome ở một số state. Chưa có population
estimate đáng tin của one-shot selection headroom, và chưa có bằng chứng về repeated-planning
headroom. Đây không phải positive headroom result, cũng không chứng minh headroom bằng 0.

### 6.3 Scrub data và composition audit — job 52617

CPU audit offline dùng đủ 504 Scrub demonstrations và native-style playback label. Trong 442
episode có terminal label playback khớp recorded label:

- cộng native contact count từng phần khác whole-segment count ở khoảng 12–22% window hai
  phần; tỷ lệ cao hơn với split ba phần;
- khi sai, chênh thường khoảng một contact;
- contribution mới của segment phụ thuộc history trước đó trong khoảng 14–34% window;
- spatial contact map với fixed union compose gần như chính xác.

**Kết luận:** native Scrub evaluator có một composition/history problem thật nhưng vừa phải.
Headroom của learned composer bị giới hạn, vì spatial-map-plus-union là strong baseline không
cần học composer.

### 6.4 Self-supervised offline composition pilot và corrected re-evaluation — jobs 52634/52642

Model dùng frozen DINOv2-S visual feature pool 4x4, proprio, action segment dài 32/64, và
matched frame/no-composition/composition/supervised-map controls. Corrected re-evaluation
cho thấy:

- raw DINO/proprio input chỉ đọc native contact information ở mức vừa;
- proprio ít nhất cũng nhiều thông tin bằng pooled visual feature cho target này;
- learned composition không tốt hơn matched no-composition trên held-out composed split.

**Kết luận:** self-supervised target/feature setup ban đầu không bảo vệ được learned-composer
claim. Đây là representation test offline trên demo, không phải GR00T planning.

### 6.5 Một spatial-target correction đã khóa — job 52644

Một sửa target giữ spatial structure tốt hơn đã được chạy đúng một lần theo protocol khóa
trước. Nó cải thiện nhẹ mức contact information summary giữ được ở một số length, nhưng
composition so với matched no-composition vẫn lẫn lộn và mọi decisive interval chứa 0.

**Kết luận:** sửa target một lần không cứu được learned composition. Nhánh Scrub-demo này
được đóng thay vì tiếp tục tune target đến khi có kết quả dương.

### 6.6 Grounded shared-teacher prototype — jobs 52655 và 52662

Teacher frozen có privileged physical supervision khi train; student chỉ thấy permitted
visual/proprio history và action candidate. Đây là test thuận lợi hơn cho effect summary.

- Learned composer không vượt sequential no-composition memory rollout; nó tệ hơn ở primary
  long 128-step split.
- Learned memory update đi qua short segment tốt gần bằng continuing teacher memory từ
  observed data.
- Segment predictor có edge nhỏ, một seed, dưới ngưỡng đặt trước so với frame predictor ở
  128 bước; không qua accuracy criterion hay 2x end-to-end speed criterion.

**Kết luận:** trên Scrub setup này, sequential memory rollout là cơ chế có ích; thêm learned
composer thì không. Kết quả này vẫn không test GR00T reranking hay action generation.

## 7. Những gì đã và chưa được kết luận

### Điều đã thiết lập trên setting này

1. GR00T có thể cung cấp candidate chunk và successful trajectory ở current runtime, dù local
   10-episode rate không phải paper parity.
2. Candidate post-state, và ở vài prefix có điều kiện terminal outcome, có thể khác nhau.
3. Scrub có history-dependent accumulation signal, nhưng spatial map + fixed union là một
   alternative/baseline mạnh.
4. Chia long horizon thành segment có length đã train rồi advance memory tuần tự giúp
   long-horizon effect prediction.

### Negative result có phạm vi rõ ràng

Trên Scrub demonstration với feature, target, length và one-seed pilot budget đã dùng,
learned composer không vượt matched no-composition hoặc sequential-memory control. Vì vậy
learned-composer claim được **đóng cho cấu hình này**.

Điều này không chứng minh mọi compositional trajectory representation đều bất khả thi ở mọi
arena. Nhưng không thể dùng kết quả Scrub hiện tại để claim learned composition cải thiện
planning.

### Những điều vẫn chưa được test

1. Goal-conditioned value head đã train để rank GR00T candidates.
2. End-to-end loop replan sau mỗi selected chunk.
3. One-shot selection headroom đáng tin trên prefix representative và nhiều continuation seed.
4. Frozen GR00T internal representation có decode progress đủ tốt để loại nhu cầu history/
   value model riêng hay không.
5. Method cải thiện chính GR00T candidate generator.

## 8. Kết luận cuối cùng

Full idea ban đầu là:

```text
GR00T đề xuất action
→ trajectory JEPA dự đoán candidate effect
→ composition/memory biểu diễn effect dài và phụ thuộc history
→ value score effect theo task
→ planner chọn và replan
```

Phần được test kỹ nhất là mệnh đề ở giữa: learned composer có cho segment-effect
representation tốt hơn trên Scrub hay không. Nó không qua. Value-based GR00T selector và
repeated planning loop mới được thiết kế, chưa từng chạy.

Vì thế, phát biểu chính xác không phải “toàn bộ GR00T + compositional trajectory-JEPA planner
đã fail”. Kết luận đúng là:

> **Learned-composer mechanism hiện tại không chứng minh được vai trò của nó trên Scrub;
> one-shot GR00T branch test cũ cũng không thiết lập stable headroom để justify train selector
> chưa được test.**

## 9. Các record nguồn chính

- `docs/BASELINE_POLICY_RESULT_51947.md`
- `docs/STAGE_B_PROTOCOL.md`
- `docs/STAGE_B0_2_RESULT_52399.md`
- `docs/QUALIFICATION_RESULT_52419.md`
- `docs/BASELINE_AUDIT_52597_AND_CONFIRMATION_RESULT.md`
- `docs/SCRUB_COMPOSITION_AUDIT_52617.md`
- `docs/COMP_PILOT_REEVAL_RESULT_52642.md`
- `docs/COMP_PILOT_TARGET_FIX_RESULT_52644.md`
- `docs/COMP_GROUNDED_RESULT_52655.md`
- `docs/SEGMENT_VS_FRAME_RESULT_52662.md`
