# Review và chạy thử native-hit CTA — 03/10/2026

Pipeline native-hit đã chạy end-to-end; **154 CPU tests**, GPU smoke, training, audit và fallback check đã pass. Pilot closed loop trên 20 development roots đã hoàn tất, **chưa cho thấy lợi ích của can thiệp native-hit**: CTA HIT chọn lại BASE, ENDPOINT HIT bằng CONTROL về số success, DIRECT HIT thấp hơn CONTROL. Cải thiện offline nhỏ chưa chuyển thành closed-loop gain. Audit xác nhận gap giữa source oracle và predicted code; kết quả của reader HIT không ủng hộ giải thích thất bại chỉ do reader quên source semantics.

## Thay đổi và kiểm chứng

- Sáu model CTA/ENDPOINT/DIRECT × CONTROL/HIT, mỗi cặp cùng trọng số khởi đầu. CONTROL tiếp tục objective/sampler cũ; HIT kết hợp standard-heavy sampling, replay bank có cả hit/miss và native-hit loss. Hai nhánh cùng update budget và selection rule. Đây là can thiệp kết hợp, chưa cô lập từng thành phần.
- Giữ nguyên `rank_loss` gốc cho DIRECT. Reader/source của CTA và ENDPOINT giữ cố định; gradient vẫn đi từ reader về world model. CTA deployment dùng `WM(C,A) -> S` rồi `reader(C,S,g)`, không đọc future và reader không nhận A.
- Nhãn native hit dùng `done8` của simulator, không suy từ coverage float32. Canonical proposal và scorer theo từng root hạn chế sai khác do batch shape. Đã lưu backend/source/checkpoint/visual hashes, exposure thực và selected step độc lập của từng model.
- Recovery checkpoint có optimizer, scheduler, NumPy/Torch/CUDA RNG. Chưa implement hoặc kiểm chứng resume CLI.
- Job **56912**: 127 tests pass trên CPU compute node, COMPLETED 0:0 trong 23 giây.
- Job **56913**: GPU smoke COMPLETED 0:0 trong 1 phút 50 giây. Strict checkpoint load, forward/backward sáu model, source frozen, bounded rollout 11 arms và numerical qualification pass. Smoke chỉ chạy hai quyết định nên không dùng làm success comparison. Slice train smoke không có mixed-hit bank; bước đầu của full training xác nhận native-hit loss thực nhận mixed labels và có gradient hữu hạn.

## Pilot và các job đã chạy

- **56914 COMPLETED 0:0 trong 40:22**: 2.400 updates/model, selection toàn bộ 1.476 banks/100 roots và 16 goal images tại bước 0/1.200/2.400; source/readers unchanged bằng exact tensor check.
- **56915_0 và 56915_1 COMPLETED 0:0 trong 35:02 và 34:36**: 20 development roots 2200–2219, hai shard chạy lần lượt, 11 acting arms `P0, GEOM8, CTA/END/DIR × BASE/CTRL/HIT` và 9 cross-scorers. K8/L15, cùng proposal/candidate protocol. Đây là lần chạy canonical được đối sánh mới trên roots development đã dùng trong programme 2200–2399 trước đây; không phải dữ liệu test mới hoặc sealed held-out. Roots này tách khỏi training hiện tại 34000–35199 và selection 2000–2099.
- **56916 COMPLETED 0:0 trong 14 giây**: CPU aggregation đủ 20 roots; kiểm tra incomplete/truncated/identity mismatch và báo paired confidence intervals.
- **56918 COMPLETED 0:0** (thay 56917 trước allocation): 10 tests của Factorized DIRECT/readout pass; artifact readout tái dựng selected scores, exposure và uncertainty theo root. Factorized task-only DIRECT mới chuẩn bị source và CPU tests, **chưa train**, không nằm trong 11 acting arms.
- **56920 COMPLETED 0:0 trong 5 giây**: 9 fallback tests pass và artifact check PASS cho score/trajectory/outcome của hai arm chọn bước 0. Tổng tests là **154 = 127 + 10 + 5 + 3 + 9**.
- **56934 COMPLETED 0:0 trong 3 giây**: dựng figure từ kết quả pilot đã aggregate. Trạng thái các job hoàn tất đã được kiểm tra bằng cả `squeue` và `sacct`.

Nguồn chạy immutable: `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_hit_review_20261003_v1/code`. Mỗi job có run directory riêng; job IDs, resources và kiểm tra budget được ghi ở [JOB_LEDGER.md](/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922/JOB_LEDGER.md). Toàn bộ model, physics và bulk analysis chạy qua Slurm compute nodes. Recipe và ngân sách thực của training nằm trong [config job56914](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_train_56914/train/config.json).

## Kết quả selection cuối

Selection rule đã khai báo trước: geometry retention + 0.25 × mixed hit capture, cho phép giữ checkpoint bước 0 khi continuation kém hơn. Không đổi rule theo kết quả. Selected step có thể khác nhau dù mọi model đều nhận đủ 2.400 updates.

| Model | Selected step | Geometry retention | Mixed hit capture |
|---|---:|---:|---:|
| CTA CONTROL | 2400 | 0.3844 | 22/49 |
| CTA HIT | 0 | 0.3739 | 22/49 |
| ENDPOINT CONTROL | 2400 | 0.4545 | 19/49 |
| ENDPOINT HIT | 2400 | 0.4983 | 24/49 |
| DIRECT CONTROL | 0 | 0.2195 | 22/49 |
| DIRECT HIT | 1200 | 0.2416 | 20/49 |

CTA HIT giữ BASE; không có CTA native-hit gain. ENDPOINT HIT tăng retention 0.0586 so với BASE, descriptive paired95%CI [0.007,0.127]; so với continuation CONTROL tăng 0.0437, CI [-0.020,0.123]. Những CI này là selection uncertainty, không sửa selection bias hoặc thay cho closed loop.

Exposure thực: HIT 80.14% standard, 28.32% mixed; CONTROL 42.74% standard, 5.57% mixed. Native-hit signal thực đã được dùng. Source CODE capture 44/49 mixed-hit banks, predicted CTA 22/49; centered normalized RMSE giữa source/predicted **scores** khoảng 0.95. Readout này dùng archived source batch4 và prediction batch1, metadata provenance có caveat. Audit56932 bên dưới so sánh lại source/mean/MAP bằng cùng bank batch1 và cùng16 goals.

Readout đầy đủ: [bản tiếng Việt](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_training_readout_56918/training_readout_vi.md) và [JSON](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_training_readout_56918/training_readout.json).

## Kết quả closed loop trên 20 roots

| Arm | Success | Mean score | Mean max coverage |
|---|---:|---:|---:|
| P0 | 10/20 | 0.933 | 0.893 |
| GEOM8 — privileged physics oracle | 17/20 | 0.976 | 0.938 |
| CTA BASE | 14/20 | 0.931 | 0.892 |
| ENDPOINT BASE | 11/20 | 0.933 | 0.890 |
| DIRECT BASE | 13/20 | 0.934 | 0.893 |
| CTA CONTROL | 11/20 | 0.858 | 0.820 |
| ENDPOINT CONTROL | 13/20 | 0.932 | 0.890 |
| DIRECT CONTROL — chọn BASE | 13/20 | 0.934 | 0.893 |
| CTA HIT — chọn BASE | 14/20 | 0.931 | 0.892 |
| ENDPOINT HIT | 13/20 | 0.908 | 0.871 |
| DIRECT HIT | 10/20 | 0.932 | 0.891 |

CTA HIT đạt14/20 chính là kết quả BASE được chọn lại, **không phải learned improvement**. So với P0 chênh lệch+20pp, paired bootstrap95%CI[-10,+50], exact McNemar p=0.3438; pilot này chưa xác lập CTA vượt P0. CTA HIT hơn CTA CONTROL15pp nhưng CI[-5,+35] vẫn rộng. ENDPOINT HIT−CONTROL là0pp, CI[-30,+25.1]; DIRECT HIT−CONTROL là−15pp, CI[-30,0]. Không có bằng chứng can thiệp HIT cải thiện closed-loop success trong pilot.

Selection offline tăng không bảo đảm rollout tốt hơn: CTA CONTROL retention0.3844 so với BASE0.3739 nhưng success giảm từ14/20 xuống11/20 và mean score từ0.931 xuống0.858. ENDPOINT HIT retention0.4983 so với CONTROL0.4545 nhưng cùng13/20 success; mean score HIT0.908 thấp hơn CONTROL0.932. Các arm tạo ra acting-state distributions khác nhau, nên không dùng geometry retention trên trajectory riêng của từng arm để kết luận scorer nào tốt hơn.

Trên **cùng P0 states**, chỉ có4 native crossing opportunities trong325 decisions: mọi CTA/ENDPOINT scorers capture3/4, DIRECT scorers1/4. Native evidence ở pilot này rất thưa. Horizon bias hoặc lệch action distribution là giả thuyết cần kiểm chứng, chưa phải nguyên nhân được xác lập.

Fallback check xác nhận `CTA_HIT=CTA_BASE` và `DIR_CTRL=DIR_BASE`: cross-scores bitwise exact trên states của **mọi 11 acting arms**, cùng các trường trajectory `root, decision, t, chosen, geom, native_hit` và discrete episode outcomes. Coverage/episode float reductions chỉ khác ở mức roundoff, tối đa khoảng6.7e-16, trong tolerance1e-12. Check này dùng declared step-zero provenance và equivalence trên artifact; **không độc lập load/attest checkpoint weight tensors**, cũng không bảo đảm mọi hardware/backend sẽ bất biến. [Fallback JSON](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_fallback_56920/fallback_check.json).

Nguồn kết quả và paired contrasts đầy đủ: [summary job56916](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_aggregate_56916/aggregate/summary.md). Figure dùng Wilson95%CI riêng từng arm; khoảng này khác paired CIs ở trên. [PNG](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_plot_56934/plot/success_closed_loop.png) · [SVG](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_plot_56934/plot/success_closed_loop.svg).

![Closed-loop success trên 20 development roots; CTA HIT và DIRECT CONTROL chọn lại BASE](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_plot_56934/plot/success_closed_loop.png)

## Iteration reader đã chạy

Job **56923** chạy 800 updates cho hai CODE readers, dùng chung data/matched dropout và source-hard + frozen-WM expected-code dense supervision; HIT thêm native-hit loss. CPU job **56922** có 5 tests pass, đưa tổng tại thời điểm đó lên142. GPU job hoàn tất 0:0 trong 5:35; encoder/WM exact unchanged, không có gradients. CUDA bước đầu cho cùng dense loss giữa hai readers và hit gradients hữu hạn. Selection chỉ đọc predicted code; source branch là averaged supervision, không phải constraint bảo đảm giữ semantics. [Config reader calibration](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_reader_train_56923/train/config.json).

Recipe này chưa giúp: cuối bước800 CTRL retention0.3075, HIT0.2964, mixed capture đều22/49; cả hai chọn lại bước0. Không nộp closed job trùng với BASE.

## Audit code prediction đã hoàn tất

CPU job **56930** có3 tests pass: MAP đúng FSQ grid, không phụ thuộc future và score-label IDs khớp; tổng **145 tests**, chưa tính fallback56920. GPU job **56932 COMPLETED 0:0 trong 2:48**, full1.476 selection banks/100 roots, cùng bank batch1 và16 goal images. Encoder/WM kiểm tra exact unchanged. CTRL/HIT dưới đây là **reader state cuối bước800**, không phải hai checkpoint được chọn lại bước0.

| Reader/code | Geometry retention | Mixed hit capture | Domain |
|---|---:|---:|---|
| BASE/source hard | 0.6679 | 44/49 | Privileged future oracle |
| BASE/expected mean | 0.3739 | 22/49 | Predicted từ C,A |
| BASE/MAP hard | 0.0978 | 24/49 | Predicted từ C,A |
| CTRL/source hard | 0.6392 | 46/49 | Privileged future oracle |
| CTRL/expected mean | 0.3075 | 22/49 | Predicted từ C,A |
| HIT/source hard | 0.6689 | 45/49 | Privileged future oracle |
| HIT/expected mean | 0.2964 | 22/49 | Predicted từ C,A |

Source-vs-mean gap của BASE là0.2940, paired95%CI[0.1810,0.4120]. Reader HIT vẫn giữ source quality gần BASE: chênh lệch0.0010, CI[-0.0128,0.0189], nhưng mean retention giảm0.0775, CI[-0.1558,-0.0150]. Vì vậy không thể giải thích thất bại reader HIT chỉ bằng việc quên source semantics. Reader CTRL có source geometry retention giảm0.0287, CI[-0.0680,-0.0054]; đây là một phần riêng của kết quả, không phải giải thích chung cho cả hai readers.

MAP không sửa được gap: so với mean, geometry retention giảm0.2761, CI[-0.3989,-0.1548]. Mixed capture tăng2 banks, tương đương4.08pp nhưng CI[-12.50,19.57] rộng. Kết quả không có cơ sở để triển khai MAP như một cải thiện hoặc mở thêm closed-loop MAP.

Expected-code within-bank variance chỉ bằng **4.23% source variance**; candidate-centered code MSE/source variance là0.9516. NLL trung bình mỗi coordinate1.2173 nats; entropy1.1972 nats. Đây là dấu hiệu predicted codes phân biệt action candidates yếu hơn source codes trong tọa độ đang dùng. **4.23% là tỷ lệ variance, không phải phần trăm task information giữ được**; những chỉ số này chưa chứng minh đây là nguyên nhân duy nhất, hoặc mọi khác biệt future code đều dự đoán được từ C,A.

Source conditions đọc tương lai thật để làm oracle diagnostic; mean/MAP không đọc future và mọi reader chỉ nhận C,code,goal. Những CI trên là descriptive selection uncertainty, không thay cho closed-loop hoặc test độc lập. Nguồn đầy đủ: [audit tiếng Việt](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_code_audit_56932/audit/audit_vi.md) và [audit JSON](/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_code_audit_56932/audit/audit.json).

## Giới hạn của kết luận

- 20 development roots đã từng dùng trước đây và một training seed là pilot; chưa đủ cho headline paper hoặc kiểm định held-out đa seed. Selection CI vẫn là mô tả trên labels dùng chọn checkpoint; closed-loop paired CI không khắc phục việc reuse development roots.
- CONTROL/HIT cùng số continuation updates, initialization trong từng family và selection rule; HIT đổi cả sampler lẫn native objective. Tổng training compute giữa CTA/ENDPOINT/DIRECT chưa được đối sánh: CTA/ENDPOINT kế thừa thêm stage world-model prediction, còn DIRECT BASE là scorer từ stage1. Vì vậy không dùng bảng này để khẳng định CTA thắng DIRECT với cùng tổng training compute hoặc cô lập lợi ích source-code supervision.
- Cache training dùng sampler lịch sử; deployment dùng canonical sampler. Cặp C,A,future trong cache vẫn factual, nhưng proposal distribution có thể lệch. Pilot đo hệ quả, chưa recollect canonical training data.
- Qualification kiểm tra proposal prefixes, score grouping ban đầu của chín learned scorers và bounded P0/GEOM trajectories; chưa chứng minh mọi learned trajectory/hardware bất biến.
- Cross-scoring và physics diagnostic tăng runtime. Timing này không phải deployment latency. Deployed expected code là48 continuous scalars; **128 bits lý thuyết chỉ mô tả16 source FSQ tokens**, không phải dung lượng kênh expected-code deployment.
- Audit hiện hướng chẩn đoán vào predicted action contrasts; source oracle vẫn có quality hữu ích. Chưa có bằng chứng iteration tiếp sẽ tăng closed-loop success. Không quy mọi failure cho WM hoặc kết luận phải đổi arena.
