# Chạy event WM trên máy Windows này (RTX 5070, 12 GB VRAM)

Thiết lập ngày 2026-10-07. Mọi thứ nặng (dataset, run, cache) nằm ngoài git ở `E:\jepa-data`; `.venv` nằm trong
repo (đã được `.gitignore`).

| Thành phần | Vị trí |
|---|---|
| venv (Python 3.10, torch 2.11+cu128, mujoco 3.15, ogbench 1.2.1) | `event_wm_20261003\.venv` |
| OGBench `*.npz` (play + val) | `E:\jepa-data\ogbench\data` (`EVENT_WM_DATA`) |
| Run root, cache, log | `E:\jepa-data\event_wm` (`EVENT_WM_RUNS`) |
| HF cache | `E:\jepa-data\hf_cache` (giữ ổ C: không bị đầy) |

## Dùng hằng ngày

```powershell
cd E:\code-project\jepa-research\event_wm_20261003
. .\local\env.ps1                         # biến môi trường + $EventWmPy
& $EventWmPy local\check_env.py           # GPU, env OGBench, dataset
```

Dựng lại venv từ đầu: `.\local\setup_env.ps1` (pin trong `local\requirements.txt`, bản đã cài trong `requirements.lock.txt`).
Tải/kiểm tra dataset: `& $EventWmPy local\fetch_data.py [--list] [--groups ...]` (resume được, bỏ qua file đã đủ dung lượng).

## Quan trọng: `scripts/` KHÔNG phải code đã chạy ra kết quả mới nhất

`scripts/` là bản cũ: thiếu `event_support.py`, `skill_segments.py`, `train_event_support.py`; `u_wm.py`, `u_skill.py`,
`u_events.py`, `s_entities.py`, `u_closed_loop.py` khác bản đã freeze. Bản mới nhất, tự chứa, nằm trong `docs/`:

| Profile | Source | Drivers |
|---|---|---|
| `adapter` (unified rerun + fix `u_events` 07/10, layout adapter `s_entities.py`) | `docs/generic_state_20261007/base_source` | `docs/unified_rerun_20261006` |
| `generic` (front end không biết layout, `g_entities.py`) | `docs/generic_state_20261007/source_generic` | `docs/generic_state_20261007` |

`local/make_run_root.py` copy chúng thành một run root (đúng quy ước cluster: `source/` + `protocol.json` +
`SOURCE_SHA256SUMS` + driver). Chỉ `data_root` và `evaluation.workers` khác protocol đã freeze (ghi trong
`local_overrides.json`); công thức, số bước, seed giữ nguyên. Các script chỉ có trong `scripts/`
(`sm_*.py` scene memory, `repair_wm_labels.py`, `resume_unified_h.py`, `diagnose_wm_state.py`) chạy riêng, xem cuối file.

## Chạy pipeline STATE (cube-triple / puzzle-4x5 / scene)

```powershell
. .\local\env.ps1
& $EventWmPy local\make_run_root.py --profile adapter --name unified_local     # 1 lần
$root = "$env:EVENT_WM_RUNS\unified_local"

# 1) preparation: s_entities -> u_events -> verify, cả 3 family, chỉ CPU
.\local\run_stage.ps1 -Root $root -Script run_prepare.py -Name prep  -ScriptArgs '--root',$root
# 2) training một family (WM+h 150k, skill 40k+10k, support 6k): MỘT job GPU mỗi lần
.\local\run_stage.ps1 -Root $root -Script run_train.py   -Name train -ScriptArgs '--root',$root,'--family','cube'
# 3) closed-loop eval (job id = id của lần train, lấy từ tên thư mục runs\cube\train_<id>)
.\local\run_stage.ps1 -Root $root -Script run_eval.py    -Name eval  -ScriptArgs '--root',$root,'--family','cube','--training-job','<id>','--seed','6'
```

`run_stage.ps1` thay `sbatch`: chạy dưới watchdog RAM (kill khi RAM trống < 1.5 GB, tránh treo máy như 2026-06-10),
ưu tiên BelowNormal, ghi log vào `E:\jepa-data\event_wm\logs`, in wall time và peak GPU. Nó đặt `SLURM_JOB_ID=local<timestamp>`
cho mỗi stage vì các script của repo từ chối chạy khi thiếu biến này (rào chắn dành cho login node cluster; máy này không
phải login node) và driver đặt tên thư mục run theo nó.

Profile `generic` cần run root `adapter` đã prepare xong (so sánh ghép cặp):
`make_run_root.py --profile generic --name generic_local --adapter-root unified_local`, rồi `run_prepare_generic.py`.

## Giới hạn của máy này (đọc trước khi chạy dài)

- **Một job GPU tại một thời điểm**, 12 GB VRAM, ~24 GB RAM. `run_train.py` tự chạy WM/h và skill song song trên cùng GPU;
  nếu hết VRAM thì chạy tuần tự bằng cách gọi trực tiếp `source\u_wm.py` và `source\u_skill.py` (lệnh nằm trong `run_train.py`).
- GPU python từng chết native không traceback và Monitor nền chết âm thầm (xem memory): đừng dựa vào Monitor; kiểm tra bằng
  file output/ckpt. `resume_unified_h.py` resume giai đoạn h.
- Quy tắc quota hàng tháng của `nhatnc129` chỉ áp dụng cho cluster; chạy trên máy này không tính vào đó.
- Tốc độ đo được (RTX 5070, cube, 2026-10-07; `local\smoke.ps1` chạy được cả chuỗi wm → skill → support → closed-loop):
  - h (cost-to-go) ≈ 0.048 s/bước (~21 bước/s, nghẽn ở vòng Python `M.candidates`, không phải GPU) → 150k bước ≈ **2 h** cho cube.
    Puzzle (K=20) có nhiều candidate hơn nên chậm hơn; trên cluster puzzle từng timeout 4 h ở giai đoạn h (`resume_unified_h.py`).
  - skill ≈ 0.02 s/bước (300 bước ≈ 0.1 phút); support 6000 bước chỉ vài chục giây; một episode closed-loop thất bại 1000 bước ≈ 0.2 phút
    (thời gian tìm kiếm A* trên model tốt hơn sẽ lâu hơn).
  - Tạo pool đi bộ tưởng tượng của h (`--h-walk-pool 300000`) là vòng Python; 20000 mẫu mất ~2 phút, nên pool đầy đủ có thể mất vài chục phút.
- `smoke.ps1` chỉ kiểm tra chạy được (model 300 bước, success 0% là đúng). Không dùng số của nó làm kết quả.

## Chưa cài (chủ ý)

- Pixel front end SAM2 (`sam2_*.py`, `transformers` + `facebook/sam2.1-hiera-small`) và track pixel puzzle
  (`config_generalization_20261001`, cần stable-worldmodel/stable-pretraining, không có ở máy này).
- Baseline JAX (OGBench CRL, `docs/generic_state_20261007/baseline`): JAX CUDA không hỗ trợ Windows native; cần WSL2 (Ubuntu đã có)
  hoặc cluster.
- Cache `.npy` giải nén cho dữ liệu visual (`cache_data.py`): 37–61 GB mỗi puzzle 4x5/4x6; chỉ tạo khi cần.

## Scene memory v1 (visual, `scripts/sm_*.py`)

Đã chạy thử trên GPU với `visual-scene` (60 episode, 200 bước): cache → `sm_train` → `sm_diag` đều exit 0.
Cần hai sửa nhỏ trong `scripts/` (đã áp dụng, chưa commit), vì code mới chỉ được test trên CPU:
- `sm_train.py`: `F.binary_cross_entropy` bị torch từ chối dưới CUDA autocast → viết thẳng `-log(alpha)` (cùng giá trị, target toàn 1).
- `sm_diag.py` `occlusion_test`: chạy 200 episode × 64 frame một lượt cần > 12 GB → chia khối 50 episode như `export()` (cùng kết quả, các episode độc lập).

**VRAM (nguyên nhân chậm):** một lượt batch 16 × clip 32 cần 12.6 GiB reserved, vượt 11.9 GiB của card nên Windows tràn sang RAM hệ thống:
2.2–3.2 s/bước, GPU chỉ ~77 W dù báo 100% util. Chia batch thành 2 micro-batch 8 (`--accum 2`, cùng effective batch 16, VRAM đỉnh 5.8 GB) chạy
~0.55 s/bước. Không bit-giống batch 16 một lượt: loss/gradient là trung bình của 2 micro-batch (mẫu số theo mask chuẩn hoá theo micro-batch) và chuỗi
RNG khác; ghi rõ khi báo cáo. `run_sm.ps1` mặc định `-Accum 2`. Chạy một mình, qua `run_stage.ps1`.

Quy mô cluster (`slurm/sm_pipeline.sh`: STEPS=20000, EPISODES=1000) bằng một lệnh; cache → train tự phục hồi → diag:

```powershell
Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','E:\code-project\jepa-research\event_wm_20261003\local\run_sm.ps1','-Family','scene','-DataEnv','visual-scene-play-v0' `
  -RedirectStandardOutput E:\jepa-data\event_wm\logs\run_sm_scene_launcher.log -RedirectStandardError E:\jepa-data\event_wm\logs\run_sm_scene_launcher.log.err
# tiếp tục sau khi bị ngắt: thêm -Out <thư mục run cũ>   (train đọc resume.pt, mỗi 1000 bước)
```
- Thêm vào `sm_train.py` (mặc định tắt, cluster không đổi): `--mmap` (frame đọc từ đĩa; bản gốc copy 12.3 GB vào RAM, máy này chỉ còn ~13 GB trống),
  `--ckpt-every N` + `--resume` (lưu model, optimizer, scheduler, RNG; kiểm chứng trên CPU: chạy ngắt-rồi-resume cho trọng số **giống hệt** chạy liền, chênh 0.0).
- Cache 1000 episode: scene ~12.6 GB, `.npy` tạo trong 0.8 phút. `visual-cube-triple`/`visual-puzzle-4x5` cũng ~12 GB train + ~3.7 GB val.
- **Tốc độ đo được** (RTX 5070, một bước tối ưu với mẫu tổng hợp): batch 16 một lượt 2.25 s (3.17 s ở run thật; reserved 12.6 GiB, tràn VRAM); batch 8 0.27 s;
  batch 4 0.20 s (~0.15 s/bước là chi phí cố định: kernel nhỏ + vòng lặp Python theo thời gian); fp32 không chậm hơn bf16 (0.28 s ở batch 8).
  Với `--accum 2`: ~0.55 s/bước → 20000 bước ≈ **3.1 giờ** mỗi family, rồi diag ~10 phút.
- `tests\test_scene_memory.py` chạy được trên CPU (6 test pass).

## Scene memory v2 (`scripts/sm2_*.py`, thay cho v1 ở trên)

v1 (gate L0 học được) không tạo ra event ở cả 4 cấu hình đã thử (xem `JOB_LEDGER.md`); run v1 `scene_20261007195148` đã dừng có chủ ý, không resume.
v2: tách agent bằng action (AgentNet + teacher + segmenter), mã cảnh theo patch, SeeThrough đọc cảnh phía sau cánh tay trong suốt, memory/event
theo luật. Một lệnh cho mỗi family (cùng hyper-parameter), khoảng 25 phút trên RTX 5070, peak RAM < 6 GB:

```powershell
.\local\run_sm2.ps1 -Family scene  -DataEnv visual-scene-play-v0
.\local\run_sm2.ps1 -Family cube   -DataEnv visual-cube-triple-play-v0
.\local\run_sm2.ps1 -Family puzzle -DataEnv visual-puzzle-4x5-play-v0
# tiếp tục / thêm stage cho run cũ: -Out <thư mục run> (stage đã xong được bỏ qua); -DiagName đặt tên thư mục diag
```

Kết quả (VAL, chấm privileged) nằm trong `<run>\<diag>\diag.json`: các biến thể `learned` / `pixel_tokens` (mask), `see_*` (SeeThrough),
`conf_*` (event đã xác nhận + trạng thái tươi). Tóm tắt và đánh giá: `docs/scene_memory_v2_20261008/REPORT.md`.
`tests\test_sm2.py` chạy trên CPU (9 test).
