# Kết quả refinement CEM stopping — 29/09/2026

Đã thực hiện một vòng cải tiến có giới hạn trên development data. **Chưa có
bằng chứng method có lợi thế trước baseline mạnh nhất.** Test outcomes chưa
được mở; các kết quả dưới đây là development, không phải held-out paper claim.

## Thay đổi và lý do

- V1 dùng adjacent elite gap và tune success, tie-break compute: một cải thiện
  rất nhỏ về success có thể được mua bằng nhiều iteration. Thêm utility có giá
  compute (`success - .002 * iterations`) và giữ lambda=0 làm sensitivity.
- Thêm baseline fixed `(K1,K2)` để không nhầm lợi ích của hai ngân sách khác
  nhau với lợi ích của thích ứng theo state.
- Fitted continuation: depth-2 trees học lợi ích chạy tiếp từ prefix CEM;
  backward targets theo policy tương lai đã fit, không theo oracle best leaf.
- Giảm capacity thành một contextual budget stump/plan. Sau đó sửa mismatch
  của plan-2 training: thay uniform hypothetical branches bằng hai lần policy
  update trên branches thực sự được policy hiện tại chọn, bắt đầu từ best
  fixed pair. Không mở sweep depth/lambda/features để săn kết quả.
- Live runner thực sự ngắt upstream solver ở callback sau update được chọn;
  các mean/cost/outcome được so với cây. Fixed baselines có chế độ chạy solver
  nguyên bản theo n_steps, không gắn recorder, để kiểm tra timing công bằng.
- Analysis mặc định từ chối thiếu root/summary và chỉ mở files của split cần
  đọc. `--allow-partial` phải được gọi rõ, output gắn nhãn PARTIAL.

Các policy mới dùng simulator success labels trên dev để fit. Đây là supervised
decision calibration, không phải reward-free method hay chứng nhận reliability.
World models/checkpoints không được train lại. Runtime chỉ đọc cost statistics
và trạng thái proposal/mean của CEM, không đọc future simulator labels.

## So sánh cùng fold

100 dev roots/task, 5-fold GroupKFold theo dataset episode. Việc chia fold khác
v1 giải thích fixed Reacher từ 66% ở báo cáo trước thành 72% ở đây; đây không
phải gain do thay checkpoint. Các phương pháp trong bảng dùng cùng folds.
Compute là tổng iteration của những plan thật sự được execute trong episode.

| Policy | Cube success / iterations | Reacher success / iterations |
|---|---:|---:|
| Fixed K, quality tuned | 67% / 4.32 | 72% / 25.50 |
| Fixed pair, lambda .002 | 71% / 6.00 | 64% / 22.25 |
| V1 gap, quality tuned | 68% / 19.36 | 68% / 42.15 |
| Fitted continuation, lambda .002 | 65% / 2.92 | 67% / 13.15 |
| Stump uniform branches, lambda .002 | 67% / 3.94 | 62% / 21.57 |
| Stump reached branches, lambda .002 | 71% / 6.00 | 61% / 19.65 |

Quality-only continuation: 68% Cube, 62% Reacher. Quality-only stump after the
branch correction: 71% Cube / 7.40 iterations, 63% Reacher / 23.72 iterations.
Every negative variant is retained in the saved reports.

Paired episode-bootstrap development intervals:

- Continuation .002 vs fixed K: Cube -2 pp [-6,+2]; Reacher -5 pp [-14,+3].
  Compute drops but non-inferiority at the predeclared -5 pp margin is not
  established. Do not label a CI crossing zero as equality/non-inferiority.
- Reached-branch stump .002 vs fixed pair: Cube exactly the same outcomes and
  iterations. The Cube gain relative to same-K CEM is explained by `(1,10)`.
- Reached-branch stump .002 vs fixed K on Reacher: -11 pp [-20,-3].

CV uncertainty is exploratory after iterative method design. It does not
include all model-selection uncertainty and cannot replace an unopened test.

## Observed bottlenecks

1. **Overfitting:** Reacher continuation fit on all dev has 89% training success,
   but grouped OOF is 67%. High oracle tree success is not evidence that the
   available score statistics can identify the good paths on unseen roots.
2. **Plan-specific budgets:** Cube's robust improvement is fixed `(1,10)`.
   Calling its 71% a success of adaptive rank resolution would be misleading.
3. **Training distribution:** fitting plan 2 uniformly on all branches can pick
   a constant budget inferior to fitting the actual controller. Correcting this
   recovers the Cube baseline but not an adaptive advantage.
4. **Scope:** these are two-plan LeWM tasks and one checkpoint per task. 47 Cube
   dev roots were trivial under every choice in v1. No outcome-based filtering
   was used; no test roots or additional tasks were opened to seek a win.

## Artifacts

- `refine_v2_55949/results/report.json`: continuation + matched controls.
- `budget_v3_55951/results/report.json`: low-capacity uniform-branch variant.
- `budget_v3_55952/results/report.json`: reached-branch correction.
- `live_v2_55950/`: actual early-stop validation on all 100 dev roots/task.
- `live_v2_55953/`: 16 roots/task, corrected budget policies and plain fixed
  baselines without instrumentation overhead; small deployment integrity check.

All paths are under `/mnt/data/nhatnc129/jepa/cem_stopping/`. Code is copied into
each run before execution. Scheduler state and final live results are recorded
in `JOB_LEDGER.md`.

## Decision

Do not promote any adaptive variant as a CVPR result, open the test, or launch
a larger threshold sweep on this evidence. The implementation now supports
real early stopping and fairer controls, but a new scientific advantage needs
a better deployable signal of physical benefit, not merely another threshold
on the same score summaries. Keep the held-out split untouched. Before any
future confirmatory run, exclude dev source episodes from its test sampling.
