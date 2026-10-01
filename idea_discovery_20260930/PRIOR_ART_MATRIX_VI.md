# Sáu vùng nghiên cứu ngoài LeWM/CTA

Ngày khảo sát:30/09/2026. Mục tiêu: tìm một khoảng hụt quan trọng có thể phát triển thành method, không mặc định quay về planner cũ. Vòng này dùng accepted top-tier papers để học cách đặt vấn đề, và kiểm tra thêm preprint để tránh bỏ sót va chạm novelty.

## Điều chỉnh cần thiết khi thực hiện quy trình

1. Có prior art không loại cả application; loại **claim đã trùng**, rồi tìm câu hỏi hẹp còn đáng đo.
2. Phép thử frozen/no-training là thăm dò rẻ, không phải điều kiện cần cho mọi learned method. Một task thiếu readout phù hợp có thể cần train; probe bất khả thi không phải hypothesis bị bác bỏ.
3. Chọn vùng dựa trên metric cuối, uncertainty, baseline mạnh cùng protocol, khả năng triển khai và phần contribution còn lại. Không so trực tiếp “5AJ points” với “0,05AUROC” để tìm effect lớn nhất; không bắt buộc có winner.
4. Không dùng con số “99preprint” trong bản trao đổi trước như census đã kiểm chứng. Mật độ prior art được thể hiện bằng các nguồn cụ thể ở dưới.

| Vùng | Accepted work gần nhất | Câu hỏi còn có thể thử | Quyết định trong vòng này |
|---|---|---|---|
| Robot failure/outcome monitoring | [Sentinel,CoRL2024](https://arxiv.org/html/2410.04640v2); [FAIL-Detect,RSS2025](https://www.roboticsproceedings.org/rss21/p073.html); [SAFE,NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html) | Predictor đóng băng, không train target readout, có bổ sung evidence ngoài observed features/motion ở prefix sớm không? | ProbePushChair bằng release có native labels và precomputedSTAC. FARM/Foresight preprints đã chiếm broad WM-monitor claim. |
| Efficient streaming perception | [DeepFeatureFlow,CVPR2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhu_Deep_Feature_Flow_CVPR_2017_paper.html); [Accel,CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jain_Accel_A_Corrective_Fusion_Network_for_Efficient_Semantic_Segmentation_on_CVPR_2019_paper.pdf); [F2MF,CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Saric_Warp_to_the_Future_Joint_Forecasting_of_Features_and_Feature_CVPR_2020_paper.html) | Frozen future features có giữ được correspondence giữa các quan sát tốt hơn repeat/copy ở cùng observation budget không? | ProbeTAP-Vid-DAVIS; đếm totalcompute. Predict rồi correct/skipencoder không phải novelty mới. |
| Occlusion/object permanence | [TCOW,CVPR2023](https://tcow.cs.columbia.edu/); [SAM2Long,ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html); [Diffusion-VAS,CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Using_Diffusion_Priors_for_Video_Amodal_Segmentation_CVPR_2025_paper.html) | Future representation giúp vị trí/persistence chỗ bị che không, vượt learned memory? | Dùng cùng trackingprobe như feasibility, không gọi visible-point tracking là amodalcompletion. Không có visibilityhead là giới hạn quan trọng. |
| Physical/counterfactual reasoning | [Physion,NeurIPS2021](https://physion-benchmark.github.io/); [Causal-JEPA,ICML2026](https://proceedings.mlr.press/v306/nam26c.html) | Surprise so possible/impossible có bị nuisance chung/background áp đảo; correction rẻ có cải thiện finalpairedaccuracy không? | ProbeIntPhys2; plain surprise đã có. Action-freepredictor không giả lập tùy ý intervention removeobject. GeoPhys/WMReward preprints va chạm broad scoringclaims. |
| Demonstration selection | [DemInf,RSS2025](https://www.roboticsproceedings.org/rss21/p023.html); [DataMIL,ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/033d9e8dbbbc0ad90e59222cf1db0fc2-Abstract-Conference.html); [CUPID,CoRL2025](https://cupid-curation.github.io/) | Dự đoán được có chỉ ra datautility mà vẫn giữ rare useful behaviors không? | Defer: qualitycorrelation không thay thế studentcontrol; no-trainingconstraint không đo finalutility. Không phải vùng bị bác bỏ. |
| Robustness/domain shift của video perception | [ViTTA,CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Video_Test-Time_Adaptation_for_Action_Recognition_CVPR_2023_paper.html) | Predictive consistency có giữ actionrecognition dưới corruption hơn matchedviews/strongadaptation không? | Có publicH265archive127MB nhưng cần đúngLarge256encoder+SSv2head~5,33GB; cachedGiant256không khớpheadGiant384. Không thêm download/run chỉ để đủ số vùng. |

Chi tiết source/API/data: [failure/data](research/failure_data.md), [streaming/occlusion](research/stream_occlusion.md), [physics/robustness](research/physics_shift.md).

## Các giới hạn kỹ thuật ảnh hưởng trực tiếp việc chọn idea

- V-JEPA2 action-free predictor là masked-feature model, không tự là autoregressive simulator cho mọi action. Xóa future tokens phải diễn ra trước encoder attention; cắt output từ fullclip không đảm bảo causality.
- Native V-JEPA2.1 dense propagation đã có; frozenencoder không đồng nghĩa readout không train. Predictors distilled2.1 còn có outputteacher dimension khácencoder dimension.
- Publicbenchmark nhỏ không đảm bảo còn headroom trên metric mới. PushChairfulldetection mạnh nhưng fixedprefix chưa biết; phải đo đúngprefix với sharedcoverage và không đổi tên thành anticipation trước onset.
- Một no-training reader16×16thiếuvisibilityhead có thể yếu vì interface task, không đủ để bác bỏ latentWM cho tracking nói chung.
- Các probe không reproduce fulloptimizedSOTA protocol chỉ cho evidence feasibility/negative của implementation đã thử. Không chọn method với claim vượtSOTA dựa trên chúng.
