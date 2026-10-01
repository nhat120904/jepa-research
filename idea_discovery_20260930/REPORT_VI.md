# Kết quả vòng tìm hướng latent world model — 30/09/2026

**Quét sáu vùng đã hoàn tất; ba probe không cần train đã chạy thật. Quyết định chọn hướng ở bước 3: chưa có ứng viên được bằng chứng hiện tại hỗ trợ đủ để bắt đầu một method CVPR.** Không có kết quả vượt SOTA được chứng minh trong vòng này. Những kết quả dưới đây loại một số cách áp dụng cụ thể; chúng không bác bỏ cả application hay latent world model nói chung.

Đây là một vòng screening trong phiên làm việc, không phải tuyên bố đã hoàn tất bảy ngày nghiên cứu hay đã tìm được paper. Không train method mới khi premise của ba ứng viên chưa được hỗ trợ.

Tài liệu: [bảng sáu vùng](PRIOR_ART_MATRIX_VI.md), [nguồn failure/data](research/failure_data.md), [nguồn streaming/occlusion](research/stream_occlusion.md), [nguồn physics/robustness](research/physics_shift.md), [ledger và cấu hình](JOB_LEDGER.md), [kết quả tổng hợp](results/summary_final.json).

## 1. Prior art: giữ câu hỏi còn mở, loại claim đã trùng

Accepted top-tier papers được dùng để học cách đặt vấn đề và đánh giá contribution. Preprint chỉ bổ sung kiểm tra va chạm novelty; không được gọi là accepted paper hoặc baseline đã reproduce. Đây là khảo sát có giới hạn, không phải chứng nhận rằng phần còn lại chưa ai làm.

| Vùng | Claim rộng đã có trước | Câu hỏi cụ thể của vòng này | Tình trạng |
|---|---|---|---|
| Robot failure monitoring | Sentinel/FAIL-Detect/SAFE đã dùng consistency, uncertainty và policy features; FARM preprint dùng frozen WM state + small readout | External frozen predictor có thêm evidence về outcome ngoài feature change/motion, không train readout? | Đã chạy PushChair. Chưa thấy lợi ích của residual đã thử. |
| Streaming perception | Deep Feature Flow, Accel, F2MF, Skip-Convolutions, CoVOS đã propagate/forecast/skip computation | Thay feature ở frame không quan sát bằng prediction có cải thiện correspondence tại cùng observation budget? | Đã chạy đủ 30 video TAP-Vid-DAVIS sau sửa lỗi. Kém repeat. |
| Occlusion/object permanence | XMem, TCOW, SAM2Long và Diffusion-VAS đã xử lý memory/occlusion/amodal | Predictive state có giúp khôi phục identity/hidden state vượt native memory không? | Probe tracking chỉ là feasibility. Không có visibility/amodal head nên câu hỏi đầy đủ **chưa được kiểm tra**. |
| Physical/counterfactual reasoning | Physion và Causal-JEPA; IntPhys2/GeoPhys/WMReward đã dùng physical prediction/surprise/geometry | Center residual hoặc giữ patch residual lớn có sửa paired plausibility decision không? | Đã chạy pilot IntPhys2. Chưa có gain; không phải counterfactual QA. |
| Demonstration selection | DemInf, DataMIL, CUPID/ReMix đã xét predictability/diversity hoặc learning utility | Predictive signal có giữ rare useful behaviors và tăng student control sau chọn data? | Chưa chạy. No-training quality correlation không đo được learning utility. |
| Robustness của video perception | ViTTA/TeCo đã dùng temporal consistency/statistics để thích ứng | Frozen predictive consistency có chọn evidence tốt hơn uniform/confidence dưới corruption? | Chưa chạy. Cần đúng released classifier head/encoder; Giant256 đang cached không có head SSv2 phù hợp. |

Nguồn accepted tiêu biểu: [FAIL-Detect, RSS2025](https://www.roboticsproceedings.org/rss21/p073.html), [F2MF, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Saric_Warp_to_the_Future_Joint_Forecasting_of_Features_and_Feature_CVPR_2020_paper.html), [SAM2Long, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf), [Causal-JEPA, ICML2026](https://icml.cc/Downloads/2026), [DemInf, RSS2025](https://www.roboticsproceedings.org/rss21/p023.html), [ViTTA, CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Video_Test-Time_Adaptation_for_Action_Recognition_CVPR_2023_paper.html).

Một va chạm rất gần: [FARM, preprint 10/09/2026](https://arxiv.org/html/2609.11445v1) đã có frozen V-JEPA2/VLA-JEPA states, supervised readout nhỏ, causal prefixes và transfer qua policy/platform. Vì vậy “dùng latent WM để monitor robot” chưa đủ là contribution mới. Nó không chứng minh mọi câu hỏi trong vùng này đã đóng.

## 2. Ba probe đã chạy

Tất cả dùng released checkpoint, không train, không simulator. Model loading/inference, decode và statistical analysis đều qua `sbatch`. Strict loading xác nhận toàn bộ keys của encoder, EMA target encoder và predictor khớp. Future RGB bị loại trước context-transformer attention; future target trong plausibility chỉ dùng để chấm điểm sau quan sát.

| Probe | Thước đo cuối | Kết quả chính | So sánh có thể kết luận |
|---|---|---|---|
| IntPhys2, 24 scenes / 96 videos | Possible–impossible paired accuracy | Global **54,2%**, scene-bootstrap CI95% **[41,7%;66,7%]**; centered **50,0%**; top10% patches **50,0%** | Hai sửa residual chưa cải thiện baseline trong pilot. Chưa reproduce SOTA. |
| TAP-Vid-DAVIS, đủ 30 videos | Average Jaccard, thang 0–100 | Predicted features **8,30**; repeat **10,46**; copy-position **10,43**. Pred−repeat **−2,17 điểm**, video-bootstrap CI95% **[−3,55;−0,52]** | Cùng reader và RGB quan sát, prediction làm xấu tracking của adapter này. CoTracker3 tham chiếu **64,09**, khác information/reader budget. |
| Real PushChair, 10 calibration + 20 test | Balanced accuracy / outcome AUROC | WM **35% / 0,20**; feature change **70% / 0,83**; released STAC **85% / 0,96** | Quy tắc cố định high surprise = failure thua hai đối chứng trên cùng episodes. Không chứng minh pre-onset anticipation. |

### IntPhys2: chưa có hiệu ứng, chưa có phép so SOTA

Protocol: original Giant256; 16 frame lấy cách 8 source frames; 12 context frames; tối đa 8 cửa sổ/video; temporal-max là aggregation chính. Scene chọn deterministic theo strata trước khi chấm, ghép đúng `SceneIndex` và prefix 1/2, bootstrap theo scene vì hai pairs có tương quan. Pilot bao gồm continuity, immutability, permanence; **chưa có solidity**.

Centering loại residual trung bình theo patch trong từng future tubelet; top10 giữ spatial losses lớn. Chúng là can thiệp thăm dò đơn giản, không phải learned method. Published extraction dùng 48 frame, sampling/context khác và hyperparameter selection khác. Kết quả 54,2% không được so trực tiếp với bảng SOTA hay gọi là reproduce đầy đủ V-JEPA2. Không có lý do từ dữ liệu này để thiết kế thêm một loss hình học.

Chi tiết: [result.json](results/physics_56178/result.json). Job56178 COMPLETED, exit0, 2m41s.

### Tracking: correction đã chạy lại; reader còn là giới hạn thực

RGB quan sát ở stride4 cộng frame có query, giống nhau cho repeat/prediction/copy. Query chỉ dùng nhãn first-visible mà benchmark cấp để khởi tạo; không đọc future positions. Future maps dự đoán từ context quan sát, không dùng RGB của skipped frame. Observed EMA maps được non-affine layer-normalized để khớp target-space trong official training, sau đó cả observed/predicted maps L2-normalized cho cosine reader.

Một lỗi trước đó thiếu target normalization đã được tìm ra bằng audit độc lập. Job56223 chỉ là debug; job56224 bị hủy ngay khi phát hiện; số của chúng không dùng để chọn hướng. Job56226 là lượt đã sửa, đủ 30 videos; prediction thắng repeat ở **4/30 videos**.

Adapter dùng grid16×16, fixed query prototype và patch offset; không có visibility head. Vì vậy nó không đo amodal completion, và không tách được reader floor khỏi giới hạn predictor. Sparse duplicated-frame inputs cũng khác native training clips. Negative result chỉ áp dụng cho interface này.

Compute thực đo cho 1.999 frames, 577 frames quan sát: EMA anchor encoding **29,17s**; prediction wrapper thêm **30,84s** vì phải chạy student encoder và predictor; reader repeat/prediction **0,51/0,36s**. CoTracker3 full observations **13,91s**. Đây là timing của các implementation trên cùng allocation, không phải FLOPs-matched comparison. Adapter prediction hiện chưa có lợi thế chất lượng/latency. CoTracker3 dùng full-video forward sliding windows, có native visibility head, không phải per-frame-causal observation-matched baseline.

Chi tiết: [summary.json](results/tracking_56226/summary.json). Job56226 COMPLETED, exit0, 1m45s.

### PushChair: surprise có thông tin nhưng hướng của signal không khớp giả thuyết

Labels là final success/failure, không có verified failure-onset labels. Residual chỉ có sau khi ảnh tương lai thật đã đến. Threshold dùng 10 successful calibration episodes; không fit readout hoặc chọn threshold trên test. STAC dùng precomputed official scores, kiểm tra lại đúng episode IDs và native time units; không chạy lại policy checkpoint.

Với high score = failure cố định, WM residual outcome AUROC **0,20**, feature change **0,83**. Paired difference **−0,63**, stratified episode-bootstrap CI95% **[−0,85;−0,35]**. RGB change là **0,21**. Nếu đảo chiều WM sau khi thấy test labels, AUROC thành **0,80**, nhưng đảo RGB cũng thành **0,79**; đó không phải lựa chọn hợp lệ để báo một method win, cũng không chứng minh prediction thêm giá trị ngoài motion. Không kết luận rằng WM không chứa thông tin.

Ở shared prefix **3s**, WM/feature-change/STAC AUROC là **0,23/0,38/0,39**. 6/9/12s không có common coverage cho mọi test episode; không bỏ các episode kết thúc sớm để tạo một gain giả. Whole-episode scores dùng mỗi episode tới native timestep<40 hoặc kết thúc, nên có thể chịu duration/termination confounding; chúng không thay cho một common-prefix anticipation test.

STAC native detection tái lập từ released scores: TP8, TN9, FP1, FN2, BA85%, mean first true alarm9,83s. Native stored AUROC0,95 dùng first-alarm score; số **0,96** trong bảng là outcome AUROC được tính lại từ temporal-max cumulative score theo cùng window, tránh gộp hai metric. STAC có action samples, trong khi WM probe chỉ đọc video; đây là một strong task baseline, không phải input/compute-matched ablation.

Chi tiết: [failure_probe.json](results/failure_56193/failure_probe.json). Job56193 COMPLETED, exit0,40s.

## 3. Quyết định chọn hướng

**Không giữ ứng viên nào từ ba probe này để bắt đầu train một method CVPR ngay.** IntPhys2 chưa có gain đáng tin; streaming adapter giảm metric cuối và tăng compute; failure residual chưa cung cấp lợi ích ngoài observed signal hay native monitor. Không đặt tên method mới từ các số này.

Occlusion, curation và robustness không bị bác bỏ: câu hỏi đầy đủ của chúng chưa được chạy. Curation cần student training để đo utility; robustness cần native classifier; occlusion cần đúng head/metric. Xếp chúng thành “ba hướng fail nữa” sẽ lặp sai lầm mà người dùng đã yêu cầu tránh.

Quy trình ban đầu cần hai sửa thực chất:

1. **Prior art loại một claim, không loại cả vùng.** “Có paper dùng WM” không có nghĩa không còn problem; cần xác định chính xác input, supervision, deployment budget, metric và failure mode khác gì.
2. **Không yêu cầu một released frozen WM phải thắng SOTA trước khi được thiết kế learned method.** Đó là bộ lọc phù hợp cho reuse/training-free contribution. Với nhiệm vụ cần reader hoặc dynamics interface mới, zero-shot mismatch có thể che cả năng lực; phải bắt đầu từ native task baseline và một bottleneck đo được, rồi làm một can thiệp tối thiểu vào đúng interface.

Sau vòng này, dữ liệu chưa phân biệt được application nào có tiềm năng CVPR cao nhất. Vì vậy không chọn robustness hoặc curation chỉ vì chúng chưa chạy. Việc tiếp theo có ích phải làm rõ lỗi của **native downstream method**; tiếp tục thử raw residual hay đặt thêm tên loss chưa có cơ sở.

## Vì sao method đơn giản vẫn có contribution mạnh: điều học được từ accepted papers

Không thể suy ra đầy đủ lý do acceptance chỉ từ một paper. Các cấu trúc bằng chứng dưới đây là những điều có thể quan sát được trong method và experiments:

| Accepted paper | Gap cụ thể | Design khớp gap | Thí nghiệm bảo vệ contribution |
|---|---|---|---|
| F2MF, CVPR2020 | Warp không sinh được vùng mới xuất hiện; direct forecasting thiếu motion constraint | Ghép feature-motion và feature-feature heads | Final future segmentation, short/mid-term, ablate hai thành phần; không lấy feature error làm endpoint. |
| SAM2Long, ICCV2025 | Greedy memory lưu mask sai rồi lỗi lan theo video | Giữ vài hypotheses, search/filter memory pathways bằng native mask scores | Reproduce SAM2 cùng settings; nhiều model sizes/datasets; long-video gain, số pathways và runtime. Frozen backbone vẫn có native task interface. |
| FAIL-Detect, RSS2025 | Failure data khó thu; uncertainty proxies không đồng nhất với failure | Success-only scalar signals/density + conformal monitoring | Detection accuracy/time trên nhiều robot tasks; so learned/post-hoc signals và prior monitors. Có training, không đòi zero-shot scalar thắng trước. |
| DemInf, RSS2025 | Chọn data chỉ dễ đoán có thể mất action diversity | Mutual-information estimate trong VAE embeddings | Human quality association rồi **train policies trên data được lọc**, đo simulation/real control. Correlation chưa tự là learning utility. |

Đối với nghiên cứu của người dùng, bài học dùng được là: chọn một lỗi của baseline task đang hoạt động, sửa bằng một cơ chế tối thiểu, và đo lợi ích độc lập của cơ chế đó trên task metric cùng chi phí. Số module nhỏ không làm giảm yêu cầu về protocol, strong controls và replication.

## Tình trạng bàn giao

Scripts được ghi rõ là exploratory probes, không phải production learned method. Source pins, checkpoint/config paths, job IDs và lỗi debug ở [JOB_LEDGER.md](JOB_LEDGER.md). Tất cả jobs đã kết thúc và được kiểm tra bằng cả `squeue` lẫn `sacct`; không còn GPU giữ idle. Tổng GPU allocation **0,10694 GPU-h**, gồm cả retry/debug, nằm trong monthly cap. Kết quả thô và prediction arrays được giữ trong từng run directory; không ghi đè run cũ.
