# Paper ledger: acceptance, phạm vi và vai trò đối với quyết định nghiên cứu

Rà soát ngày 07/09/2026. “Main accepted” dựa trên proceedings/conference program chính thức, không dựa vào cụm “submitted to”, bibtex tự khai hoặc một workshop PDF. Nội dung kỹ thuật lấy từ paper/project tác giả. Các hàng là tập chọn lọc, không phải tổng điều tra toàn bộ hội nghị. Một số poster ICML/ICLR không đọc được qua web parser nhưng trang HTML chính thức truy cập trực tiếp được; không suy status từ lỗi parser.

## A. 20 công trình đã xác minh main conference

| # | Công trình | Venue xác minh / nguồn | Cơ chế và phạm vi thực nghiệm cần nhớ | Ý nghĩa cho nghiên cứu này |
|---|---|---|---|---|
| 1 | DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning | [ICML 2025, PMLR](https://proceedings.mlr.press/v267/zhou25t.html) | Frozen visual patch features, learned dynamics, visual goal planning | Nền tham chiếu; không mặc định là baseline mạnh nhất mọi regime |
| 2 | Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models — PLDM | [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e7cf447f21cd11c846463affefce665-Abstract-Conference.html) | Latent predictive learning + planning; navigation và generalization theo dữ liệu/layout | Cần khớp coverage, reward-free setting và planner |
| 3 | seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models | [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f63d2963526bdd9ff1b8bcc2dc9905a-Abstract-Conference.html) | Tách invariant/equivariant representations; transform/view-related prediction | Không chuyển thành claim manipulation success |
| 4 | Time-Aware World Model for Adaptive Prediction and Control | [ICML 2025, PMLR](https://proceedings.mlr.press/v267/nhu25a.html) | Δt-aware dynamics/control trên nền TD-MPC2 | Baseline bắt buộc của hướng B |
| 5 | TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning | [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html), [oral program](https://iclr.cc/virtual/2026/poster/10009366) | Policy-dependent long-term latent predictions, TD và successor-style structure; ExoRL/OGBench | Không phải chỉ dự đoán next-frame embedding tốt hơn |
| 6 | Temporal Straightening for Latent Planning | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/64904), [project](https://agenticlearning.ai/temporal-straightening/) | Temporal trajectory geometry hỗ trợ gradient planning | Không tương đương action-space curvature hoặc CEM |
| 7 | Causal-JEPA: Learning World Models through Object-Level Latent Masking | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/63623), [paper](https://arxiv.org/abs/2602.11389) | Object masking, counterfactual reasoning; control và representation efficiency | Có cả bản workshop; masking quan sát không tự động là physical do-intervention |
| 8 | GeoWorld: Geometric World Models | [CVPR 2026, CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_GeoWorld_Geometric_World_Models_CVPR_2026_paper.html) | Hyperbolic JEPA/geometric RL; procedural planning CrossTask/COIN | Không phải robot continuous-action MPC |
| 9 | VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/61579) | Variational predictive distributions, Bayesian filtering interpretation | Novelty threat cho probabilistic/belief JEPA tổng quát |
| 10 | Compositional Planning with Jumpy World Models — CompPlan | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62821) | Policy composition; geometric-horizon occupancy models và horizon consistency; OGBench | Baseline trung tâm của hướng A |
| 11 | Parallel Stochastic Gradient-Based Planning for World Models | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/64851) | Tối ưu virtual states với soft dynamics constraints và stochastic gradient planning | Nếu làm planner mới, CEM/GD thường chưa đủ làm baseline |
| 12 | Sparse Imagination for Efficient Visual World Model Planning | [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008222), [project](https://nikriz1.github.io/sparse_imagination/) | Điều chỉnh token computation trong rollout | Tiết kiệm compute cần gắn với control nếu theo ưu tiên của người dùng |
| 13 | Learning to Be Uncertain: Pre-training World Models with Horizon-Calibrated Uncertainty | [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10007319) | Probabilistic ensemble, random prediction horizons; downstream control | “Thêm horizon uncertainty” đã có đối thủ |
| 14 | Latent Particle World Models | [ICLR 2026 poster/oral](https://iclr.cc/virtual/2026/poster/10007676), [project](https://taldatech.github.io/lpwm-web) | Object/particle structure, stochastic latent actions, multi-entity video và control | Generic object-centric world model chưa đủ mới |
| 15 | R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation | [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010184) | Redundancy reduction cho decoder-free MBRL; DMC/MetaWorld | Baseline khi đổi sang online reward-based representation learning |
| 16 | SVL: Goal-Conditioned Reinforcement Learning as Survival Learning | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/61425) | Survival/censoring và hierarchical goal-conditioned policy | Loại claim “lần đầu dùng time-to-goal survival learning” |
| 17 | The Surprising Difficulty of Search in Model-Based Reinforcement Learning — MRS.Q | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62452) | Search/value overestimation, pessimistic value ensemble | Đối thủ gần cho “search khai thác model/value error”, nhưng regime khác frozen cost |
| 18 | Value Flows | [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10011734) | Distributional Bellman learning với flow matching và uncertainty-based prioritization | Không nhận flow-based distributional critic làm ý tưởng mới |
| 19 | Factored Latent Action World Models — FLAM | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/61299) | Factored action representations theo entities | Đối thủ cho latent action factorization |
| 20 | The Cost of Commitment in Option-Based Hierarchical RL | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/66480) | Tradeoff option duration, deliberation cost, model error | Adaptive duration vì uncertainty không phải khoảng trống nguyên vẹn |

Nhóm 1–9 sát JEPA; các hàng sau mở sang world modeling/control và đối thủ của các câu hỏi ứng viên. Không gọi cả 20 là “JEPA papers”.

## B. Không trộn vào tập main accepted ở trên

| Công trình | Status có thể khẳng định trong lần kiểm tra này | Lý do vẫn cần đọc |
|---|---|---|
| [LeWorldModel](https://arxiv.org/abs/2603.19312) | Preprint; chưa xác minh main acceptance từ nguồn chính thức được kiểm tra | Invited talk ở hội nghị không thay thế acceptance của paper |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Chưa xác minh main conference acceptance | Baseline/release quan trọng vẫn có thể là novelty threat |
| [VLA-JEPA](https://arxiv.org/abs/2602.10098) | Chưa xác minh main acceptance; không gán ICRA từ kết quả tìm kiếm lân cận | VLA/control khác image-goal MPC |
| [Policy-Guided World Model Planning for Language-Conditioned Visual Navigation — PiJEPA](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/html/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.html) | CVPR 2026 **workshop**, theo CVF WDFM-EAI | Không ghi thành CVPR main |
| [ODEWorld](https://arxiv.org/abs/2607.27924) | Preprint trong tập kiểm tra | Physical-time dynamics; không bỏ qua chỉ vì chưa có main acceptance |
| [AgentOWL](https://arxiv.org/abs/2602.02799) | Preprint theo [trang tác giả](https://www.cs.cornell.edu/~wp237/) | Options + abstract world model sát hướng A |

Không kết luận những bài “chưa xác minh” bị reject hoặc chắc chắn không accepted. Không có hàng ICRA main được xác nhận trong tập này; đây là giới hạn coverage, không phải khẳng định ICRA không có công trình liên quan. Không dùng NeurIPS 2026 như một tập acceptance đã xác minh trong báo cáo này.

Tên dễ nhầm: **TD-JEPA** trong hàng 5 là bài của Bagatella và cộng sự, không phải preprint “Temporal-Distance JEPA” ra sau. **VJEPA**, **Var-JEPA**, **V-JEPA 2** cũng là các công trình khác nhau. Var-JEPA có ICML 2026 [poster riêng](https://icml.cc/virtual/2026/poster/61472), nhưng tabular predictive/generative SSL không phải evidence robot control nên không chọn vào 20 hàng chính.

## C. Code/data readiness

“Tìm thấy official repo” không có nghĩa đã cài đặt, có đủ checkpoint hoặc đã reproduce số paper.

| Thành phần | Nguồn tác giả | Mức sẵn sàng được xác minh |
|---|---|---|
| OGBench | [seohongpark/ogbench](https://github.com/seohongpark/ogbench) | Có repo môi trường, hướng dẫn dataset và reference agents |
| TAWM | [anh-nn01/Time-Aware-World-Model](https://github.com/anh-nn01/Time-Aware-World-Model) | Có official code; chưa chạy reproduction |
| HSVL | [Simple-Robotics/hierarchical-survival-value-learning](https://github.com/Simple-Robotics/hierarchical-survival-value-learning) | Có code, config OGBench và launcher; chưa chạy |
| MRS.Q | [facebookresearch/MRSQ](https://github.com/facebookresearch/MRSQ) | Có source, usage và published results directory |
| DINO-WM | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) | Official repo; checkpoint access là vấn đề riêng, không kiểm tra tải lại ở đây |
| Causal-JEPA | [galilai-group/cjepa](https://github.com/galilai-group/cjepa) | Có repo tác giả |
| VJEPA | [YongchaoHuang/VJEPA](https://github.com/YongchaoHuang/VJEPA) | Có repo tác giả |
| R2-Dreamer | [NM512/r2dreamer](https://github.com/NM512/r2dreamer) | Có repo tác giả |
| CompPlan | [paper và pseudocode](https://arxiv.org/html/2602.19634v1) | **Chưa xác minh được official implementation**; không thay bằng repo người khác tên gần giống |

## D. Prior art ngoài cửa sổ 2025–2026 bắt buộc cho novelty

- [Universal Option Models, NIPS 2014](https://papers.nips.cc/paper/5590-universal-option-models): option models và reward-independent abstraction đã có từ lâu.
- [Generalised Policy Improvement with Geometric Policy Composition, ICML 2022](https://proceedings.mlr.press/v162/thakoor22a.html): nền tảng geometric composition, không phải nguyên lý mới năm 2026.
- [Learning Abstract World Model for Value-preserving Planning with Options](https://arxiv.org/abs/2406.15850): abstraction và value preservation cho option planning.

Một contribution có thể mới dù nguyên lý cũ, nhưng phải mới ở estimator, guarantee với giả định rõ, phạm vi giải được, hoặc cơ chế control có bằng chứng. “JEPA chưa dùng nguyên lý này” tự nó không đủ.
