# Nối tiếp CTA trên H100 — 02/10/2026

User đã yêu cầu tiếp tục implement/train/closed loop từ phiên bị ngắt, sau đó xác nhận DNS H100 đang lỗi và cho phép hoàn thiện code trước. Chưa có job mới được nộp trong lượt nối tiếp này.

## Trạng thái thật

- SSH tới `login-restricted-1` và filesystem dùng được. `/etc/resolv.conf` có kích thước 0 byte; `squeue` không kết nối controller và `sacct` không phân giải/kết nối accounting service. Đây là lỗi đường truy cập scheduler, chưa xác minh được trạng thái job hiện tại.
- Checkpoint L15 hiện có: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_replan15_train_56504/train/cta_v2.pt`. Source/config review cho thấy tương thích strict-load với trainer mới; chưa tải model để kiểm chứng runtime.
- Training data vẫn là các bank L15 đã lưu từ sampler lịch sử. Pilot thay sampler ở evaluation thành canonical; chưa recollect train data canonical. Cần ghi rõ khác biệt này và dùng closed loop để đo hệ quả.

## Code đã chuẩn bị

- `scripts/cta_hit_train.py`: sáu model CTA/ENDPOINT/DIRECT × continuation control/native-hit intervention; mỗi cặp bắt đầu từ cùng checkpoint, cùng số update, optimizer và gradient clipping riêng. Codec, reader code và FULL reader được giữ cố định. Control dùng sampler/loss cũ; intervention kết hợp standard-heavy sampling, mixed-hit replay và native-hit loss. Hai nhánh dùng cùng selection rule và toàn bộ 16 goal images. So sánh này đo can thiệp kết hợp, chưa cô lập riêng hit loss.
- `scripts/cta_hit_closed.py`: 11 acting arms `P0, GEOM8, CTA/END/DIR × BASE/CTRL/HIT`; K8/L15, canonical proposals, scorer theo từng root. CTA reader chỉ nhận `C,S,g`; learned scores được tính trước candidate simulation. Qualification tích hợp kiểm tra proposal prefix/grouping, score grouping, native-hit labels và discrete episode outcomes. Report lưu clone method, smoke/preparation hashes, dependency versions, visual weights thực đã tải, visual source và runtime hashes.
- `scripts/cta_hit_aggregate.py`: matched roots, từ chối partial/truncated/identity mismatch, candidate index sai và nhãn không hợp lệ. Ratio không xác định được xuất JSON `null`, Markdown `n/a`.
- `ti_wm/cta_hit_contracts.py`: validation tham số, shape K8/L15, miền candidate/native-hit và JSON/Markdown thuần Python; không import model hoặc simulator.
- `scripts/slurm_cta_hit.sh`: các mode `tests`, `smoke`, `train`, `closed`, `aggregate`; CPU defaults an toàn, GPU modes đòi allocation rõ, source snapshot/run directory riêng. Bắt buộc truyền `CTA_SOURCE_ROOT` vì Slurm spool script ở một đường dẫn khác source gốc.

## Verification đã làm và còn thiếu

- 8 unit tests của helper thuần Python đã thấy fail trên stub rồi pass sau implementation; sau đó thêm feature-cache shape test, tổng 9 tests pass local trong khoảng 1 ms. Không giả `SLURM_JOB_ID` và không chạy model/physics/analysis trên login node.
- Python AST và Bash syntax đã kiểm tra. Rà soát độc lập đã bắt và sửa phép trừ boolean trong qualification native-hit logs.
- `cta_tests/test_cta_hit_aggregate.py` có 6 regression tests cho aggregate thực và qualification logs: zero headroom/strict JSON, boolean native hits, candidate/label sai, episode outcome sai type, clone/dependency mismatch và partial/truncated outputs. Các test này **chưa chạy**, phải chạy bằng CPU `sbatch`.
- Toàn bộ existing tests, GPU forward/backward smoke, frozen-reader invariance và learned closed-loop result **chưa kiểm chứng**. Không có kết quả success mới hoặc claim cải thiện.

## Chạy tiếp khi DNS hồi phục

Source release và prior source backup được ghi trong `JOB_LEDGER.md`. Luôn dùng đường dẫn release `/code` cho `CTA_SOURCE_ROOT`; không chạy lại từ checkout mutable.

Trước mỗi submission: kiểm tra duplicate/peer work và cả `squeue` lẫn `sacct`. Trước mọi GPU submission và CPU array: chạy query month-to-date theo AGENTS; cộng planned job ở full time limit cùng các job live/pending, giữ ≤50% của user xếp thứ 5 cho GPU/CPU/memory. Không được suy ngân sách từ ledger cũ. Các limits dưới đây là đề xuất ban đầu, chỉ nộp khi quota thật cho phép; giảm scope/time hoặc chờ nếu không đủ.

```bash
squeue -u nhatnc129 -o '%i %j %T %M %l %D %R'
sacct -u nhatnc129 --starttime 2026-10-02 -X --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,AllocTRES -P
sreport -t hours -T gres/gpu,cpu,mem cluster UserUtilizationByAccount start=$(date +%Y-%m-01) end=now -P -n
```

Các lệnh sau là mẫu chưa thực thi. Đặt `CTA_RELEASE_CODE` thành đường dẫn release `/code` trong ledger, dùng job ID thật trả về và record ngay vào ledger:

```bash
# 1. CPU tests: 2 CPU, 8 GB, 10-minute cap.
sbatch --parsable --export=ALL,CTA_SOURCE_ROOT="$CTA_RELEASE_CODE" "$CTA_RELEASE_CODE/scripts/slurm_cta_hit.sh" tests

# 2. GPU smoke, chỉ sau khi CPU tests pass và quota cho phép: 1 MIG slice, 4 CPU, 64 GB, 15-minute cap.
sbatch --parsable --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=64G --time=00:15:00 --export=ALL,CTA_SOURCE_ROOT="$CTA_RELEASE_CODE" "$CTA_RELEASE_CODE/scripts/slurm_cta_hit.sh" smoke

# 3. Bounded continuation: 2400 updates, full selection split, 60-minute cap; không queue trước khi biết smoke/runtime/quota.
sbatch --parsable --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=64G --time=01:00:00 --export=ALL,CTA_SOURCE_ROOT="$CTA_RELEASE_CODE" "$CTA_RELEASE_CODE/scripts/slurm_cta_hit.sh" train

# 4. Pilot closed loop: 20 development roots, 2 × 10-root shards, concurrency 1, 40 minutes/shard.
sbatch --parsable --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=64G --time=00:40:00 --array=0-1%1 --export=ALL,CTA_SOURCE_ROOT="$CTA_RELEASE_CODE" "$CTA_RELEASE_CODE/scripts/slurm_cta_hit.sh" closed "$CTA_TRAIN_JOB_ID"

# 5. CPU aggregate: completed full episodes only; expected roots 2200–2219.
sbatch --parsable --cpus-per-task=2 --mem=8G --time=00:10:00 --export=ALL,CTA_SOURCE_ROOT="$CTA_RELEASE_CODE" "$CTA_RELEASE_CODE/scripts/slurm_cta_hit.sh" aggregate "$CTA_CLOSED_JOB_ID"
```

GPU budget của mẫu train+closed là tối đa 140 phút ngoài smoke. Timing thực chưa được đo; không mở rộng trước khi có pilot. 20 development roots chỉ là pilot để debug và chọn bước tiếp, không phải xác nhận headline hay sealed evaluation. Diagnostic branching/cross-scoring không phải deployment latency.
