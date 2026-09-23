# TargetedFedPhoenix research decision audit

审计基点：`98345ced583894285338bfdd2acd5181de57e499`；2026-09-23。本文是**已完成实验的审查和后续实验计划**，没有运行新的训练、修改核心算法或启动其他 seed。百分数差异均为百分点（pp）。主要复算来源为同目录 `offline_metrics.json`；可用 `D:\software\minicoonda\envs\fd\python.exe experiments/analyze_research_decision.py` 复现。下文的“支持”均限于当前单 seed、固定 CIFAR-10 划分和已测试配置。

## 1. Executive conclusion

**值得继续做一个严格的因果辨别阶段，但不值得继续 ratio/gap/start 的 seed=1 搜索；也尚未达到可直接宣布 E5 优于 FedPhoenix 并进入正式多 seed 主表的程度。** 最紧缺的不是更高的单次 peak，而是证明“同一个 returning client 的历史”比相同年龄的其他 client 历史或群体历史更有用，以及用当前代码生成一个 1000-round 配对 FedPhoenix 基线。E5 的 1000-round peak 只比历史日志高 **0.01 pp**，last100 低 **0.64 pp**；历史日志又不是逐轮配对轨迹。若下一阶段的身份对照失败，应停止宣称 client-specific dispatch 的有效性，重新审视 reset-high 的动作方向，而不是继续调参。

## 2. Verified current evidence

| 来源 / 协议 | Peak (round) | Final | Last20 | Last100 | 证据解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| 当前 FedPhoenix observer-only，300 轮 | 78.35 (218) | 73.93 | 69.665 | 71.004 | 当前代码、相同 seed 的干净短程轨迹；不是 1000 轮 |
| Corrected TargetedFedPhoenix V1，ρ=.25，gap≤20，300 轮 | 78.33 (284) | 73.54 | 70.7595 | — | peak 基本持平 |
| E2，ρ=.125/all/start1，300 轮 | 78.56 (218) | 75.16 | 71.883 | — | 单 seed 提升 0.21 pp |
| E4，ρ=.25/strong/start1，300 轮 | 77.88 (232) | 73.28 | 71.186 | — | 同层即时 intervention 不好 |
| E5，ρ=.25/strong/start151，300 轮 | 78.91 (232) | 74.27 | 72.1315 | — | 本次 screen 最好；与 E4 相比还少干预 6,566 slots |
| P1 / P2，start151/all，300 轮 | 77.92 / 78.34 | 74.15 / 73.46 | 71.820 / 71.188 | — | 延迟本身并非普遍增益 |
| 历史 FedPhoenix 日志**前 1000 轮** | 82.61 (997) | 80.59 | 80.5935 | 79.2956 | 非当前代码逐轮配对 baseline；完整日志实际有 1200 轮 |
| L2/E5，当前代码，1000 轮 | 82.62 (997) | 79.19 | 80.9620 | 78.6558 | peak +0.01、final −1.40、last100 −0.64 vs 历史日志 |
| L3/E2，当前代码，1000 轮 | 82.15 (783) | 81.86 | 79.8825 | 77.5759 | final 最高；peak / last100 不如 E5 |

300 轮 screen 的 E1–E6 均见 `results/targeted_fedphoenix_overnight6/comparison.csv`；P1/P2 见 `results/targeted_fedphoenix_delayed_pair/comparison.csv`；1000 轮见 `results/targeted_fedphoenix_longrun1000/comparison.csv` 与各 run 的 `round_metrics.csv`。E5 1000 轮前 150 轮对当前 FedPhoenix observer 的 selected clients、task seeds、global hashes **150/150 逐项一致**；这验证了共同前缀，却不提供后 850 轮的 FedPhoenix 对照。1000 轮 L2/E5 与 L3/E2 的阶段均值差（L2−L3）为 1–150: −0.101、151–300: −0.083、301–500: +0.233、501–750: −0.185、751–1000: +0.371 pp；效果没有单调增强。

E5 1000 轮总计 23,755 个 targeted slots、23,380 个实际位置替换；第 771–1000 轮 targeted slots **为零**。因此后期准确率差只可能是早先介入所改变轨迹的延续、随机波动或两者，不能称为持续 intervention 的即时收益。E2 仍至第 1000 轮有 targeting。E5 的历史内存峰值约 2.11 MB，两个 1000 轮 run 均未记录失败 client update。

## 3. What is actually established

Phase-A observer 的 `update_norm` 共有 37,700 个 client-layer 再参与事件：同 client top-k overlap **0.1453**，跨 client 均值 **0.0723**，随机基准 **0.0161**；Spearman 分别 **0.4755/0.3022**。四个 E5 层的同/跨 overlap 为 `features.20` **.185/.079**、`.24` **.212/.079**、`.27` **.193/.078**、`.30` **.231/.107**；`features.0` 仅 **.047/.044**。同一 client 的 magnitude/location 排名存在时间可预测性，且层间异质。`angular` 同/跨 overlap 只有 **.0220/.0181**，并无同等强度的方向证据。`update_norm` 同 client 重合率随 gap 增大而下降（原 `gap_summary.csv`：1–10 为 .164，11–20 为 .112，21–40 为 .101，>40 为 .087），说明新鲜度重要。

代码上，当前 Targeted 路径先生成 FedPhoenix baseline task，再以确定性的 SHA256 priority 保留随机 reset slots、将 displaced donor 的**已有初始化样本**搬到新 target，恢复 donor 的 global weight。每层 reset **数量**和初始化值 multiset 保持，reset **位置、样本与通道的对应关系及训练轨迹**改变。local training 使用 `deepcopy(task_model)`；训练后读取**未参与训练、保持原样的** `task_model.state_dict()` 作为 `actual_dispatch_state`，与训练所得 `returned_state` 区分；reset-exposed 的输出 kernel 被置为历史无效；observer 不参与聚合。10-round `ρ=0` parity 与 E5 首 150 轮 hash parity 已通过，说明实现的基线回退和共同前缀可信。

## 4. What is NOT established

1. Phase-A **没有证明**高 `update_norm` kernel 应该被 reset。高幅度可能是有害的 client-specific overfitting，也可能是有用的非 IID 特征，日志缺少同一候选 kernel 被 reset/protect 的反事实效用标签。
2. 同/跨 client overlap 差值**不是严格身份因果效应**。跨 client 对照按 client ID 升序选前 10 个，未逐事件匹配 history gap、数据量或标签组成；日志只保存跨 client overlap 的**平均值**，无法从现存 CSV 精确重建 gap-matched overlap。离线读取 `cross_history_rounds` 发现 own gap 均值 **9.63** 轮、跨历史逐比较均值 **10.12** 轮，只有 **38.2%** 的事件在这 10 个跨 client 中存在至少一个与 own gap 完全相同的比较。均值接近不能消除逐事件不匹配。
3. `topk_comparison` 用 `k=min(reset_budget, common_valid)`，随机期望 `k/common_valid`；总体 chance 约 .016，但 64-channel/K=1 层的 overlap 是 0/1 离散量，不能把各层的原始均值当成同方差观测或把 37,700 个相关事件当成独立样本。
4. 现有结果**不能区分** identity、历史 salience、启动时间、被干预层的活动窗口、intervention 数量和非线性轨迹偏移。E4→E5 改变了 start，同时前 300 轮 targeted slots 从 **13,461** 降为 **6,895**；P2→E5 在同 start 下改变层集也把 slots 从 **14,775** 降为 **6,895**。这些是所选策略的组合效应，非各因素独立效应。
5. 多轮历史目前是**每个 client 仅保留最近一次 interaction 的 scalar scores**，每次参与就覆盖；这证明 repeated-client lookup，而非利用完整 longitudinal trajectory 学得稳定 operator。方法新意必须以“历史身份真正决定下一次 reset、且带来可复现效益”来支持。

## 5. Most likely current bottleneck

首先是**因果识别和公平性能证据**，其次才是评分/动作设计。高幅度的时间持久性与良好的 reset 位置并非逻辑等价；E5 的 300 轮表现很可能利用了 warm FedPhoenix 前缀与四个中层尚活跃的窗口，但其 1000 轮 peak 几乎与历史 FedPhoenix 并列，last100 反而更低。单 seed、多配置看 test peak 决策使 0.01–0.56 pp 的数值尤其脆弱。若身份对照显示同/错配历史等效，当前方法只剩“FedPhoenix + top-k salience heuristic”，不应继续用搜索包装为 personalized longitudinal dispatch。

## 6. Code / methodology risks

- `TargetedFedPhoenixController.history[client_id]` 覆盖语义丢弃累计证据；gap 只限制最近一次交互的年龄，未刻画跨参与一致性或置信度。此阶段不建议为此马上增加 controller。
- `floor(ρK)` 使 ρ=.25 对 K=1/2/4/8 层分别 target **0/0/1/2** 个，ρ=.125 则 **0/0/0/1** 个；所谓 all layers 实际很多浅层没有 targeted reset。`target_layers` 与 ratio 不能脱离预算解释。
- 13 个 VGG conv 层按 `current_iter=round−1 < (depth+1)×1000/13` 阶梯停止。E5 四层最后活跃轮分别为 `.20`: **539**、`.24`: **616**、`.27`: **693**、`.30`: **770**。因此 E5 的后 230 轮是自然形成的有限窗口，不是显式设计的持续 targeting。
- SHA256 private priority 避免固定小索引的 prefix-retention 偏差；corrected V1 的 displaced donor 归一化索引均值 **.497**、random-kept **.499**，未见粗大的索引偏差。但固定哈希和 donor-pairing 仍引入确定性的样本—通道配对；“相同 reset 数量”不等于干预完全可交换，需在后续对照中保持同一算法及 priority 规则。
- reset-exposed kernel 排除的是**本次 final reset index**，符合防止 reset 初始化污染历史 score；但合法 `update_norm` 仍受本地样本量、label skew、模型成熟度和参与频率影响。Phase-A 的 residual/normalized/angular 较弱，不能自动证明 update_norm 是最优效用代理。
- 对 test accuracy 的连续筛选存在选择偏差。300-round clean FedPhoenix 第 295 轮曾降到 **22.56%**（历史日志同轮约 72.10%），说明轨迹也有强瞬态波动；peak 与末轮均须辅以固定窗口均值。此异常不应被解释成方法优劣或 baseline bug 的证据。

## 7. Baseline provenance audit

历史 `result_other_method/cifar10/0.3/vgg_FedPhoenix.log` 只含 round accuracy/peak 等文本，没有完整 args、partition 文件 hash、逐轮 selected clients、task seeds、reset masks、global model hash 或代码 commit。其**前 1000 轮**数字可作为参考；完整文件有 **1200** 轮，不能误把完整日志 peak 当 1000-round peak。历史与当前干净 observer 的前 300 轮仅第 1 轮 accuracy 在 0.01 pp 内，**299/300 不同**，第 2 轮即分叉；绝对差中位数 **0.99 pp**、均值 **1.73 pp**。这些差异**证明不配对，不证明是哪一个代码/随机性/数据源造成**。现有 JSON 分割文件存在，当前训练在 `generate_data=0` 时读取它；无法从历史纯文本反证其使用了同一文件。原始 FedPhoenix 公开实现与本仓当前 task-bank 的随机实现也不能仅凭 accuracy 视为同一轨迹。

所以历史日志**不适合作为论文主表唯一的 paired FedPhoenix baseline**。应在冻结代码、commit、数据分割 SHA256、模型、optimizer、reset、`FP_conv`、seed、device 与评估协议后，重新生成 current-code FedPhoenix 1000-round 同 seed 轨迹，并对齐 selected clients/task seeds；已有 300 轮 observer 和 E5 前缀不足以替代。未来正式多 seed 要**逐 seed 配对重跑 FedPhoenix 与冻结方法**，同轮通信预算与 split，报告 seed-level 差、均值/离散度和固定后期窗口，不用最优 seed 的 test peak 作主结论。

## 8. Skill(s) used

未使用第三方 skill：当前可用 skill 未提供比直接审计本仓 traces、标准统计和原论文更针对性的 FL 因果/复现审计流程，也无需执行不透明安装脚本。文献只用于界定新意：[FedPhoenix 原论文](https://papers.neurips.cc/paper_files/paper/2025/hash/16e71d1a24b98a02c17b1be1f634f979-Abstract-Conference.html)及[官方代码](https://github.com/UniString/FedPhoenix)已涵盖随机局部 reset；[FedLFH 论文](https://www.sciencedirect.com/science/article/abs/pii/S0893608025009517)说明使用 client 历史本身不是新概念；[FedSelect](https://arxiv.org/abs/2306.13264)研究个性化参数选择，但目标为 fine-tuning 而非此处的 server dispatch/reset。因而潜在贡献必须是**用反复 server-client 交互的身份条件证据，决定未来同 budget 的干预位置，并证明比身份置换和群体 salience 更有效**，不能只声称“首次使用历史”。

## 9. Offline analyses performed

`experiments/analyze_research_decision.py` 只读 CSV/日志，不导入训练代码，生成 `offline_metrics.json`：重新计算历史日志**前 1000**和 current 轨迹准确率、前 300 轮不匹配数量、E5 前 150 的 3 项 parity、分阶段均值和 targeted slots、四层理论停止轮、Phase-A 同/跨/随机 overlap、跨历史年龄分布及 exact-age donor 可得性。对于 E5 的 151–770 活跃窗口，已有 selected-client 序列中 **4,032** 个有 gap≤10 的候选参与，**4,028（99.90%）** 可在全部 100 个历史 client 中找到至少一个**相同 age**且不同身份的 donor；exact-age 可用 donor 中位数为 **7**。这仅验证匹配可行性，**未**检查标签分布匹配或 score validity 的严格共同支持，也**未**进行任何反事实训练。不能从现存 Phase-A 跨 client 均值 CSV 推导精确 gap-matched identity advantage。

## 10. Candidate decision matrix

定性等级相对本仓当前证据；“提升 peak 机会”不是虚构的概率。成本按本机 L2 已记录 **13,421 秒≈3.73 小时/1000-round run** 粗估，显存并发可能改变 wall time。

| 候选 | 信息增益 | 提升 peak 机会 | 新意重要性 | 审稿稳健性 | 实现风险 | 计算成本 | seed1 过拟合风险 | 负结果可解释性 | 决定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E5 直接多 seed 1000 轮 | 中 | 未知 | 中 | 高 | 低 | 高 | 低于再搜索 | 中：基线仍不配对 | **暂缓** |
| current-code FedPhoenix 同 seed 配对基线 | 高 | 不适用 | 低 | 极高 | 低 | 中/高 | 低 | 高 | **纳入** |
| 同 client vs exact-age shuffled history | 极高 | 未知 | 极高 | 极高 | 中 | 高 | 中 | 高 | **纳入** |
| 同 client vs age-matched population history | 高 | 未知 | 高 | 高 | 中 | 高 | 中 | 高 | **纳入** |
| reset-high vs protect-high/低分 reset | 高 | 未知 | 中 | 高 | 中 | 高 | 中 | 高 | 下一阶段再做 |
| 自适应 activation 取代固定151 | 中 | 未知 | 中 | 中 | 高 | 高 | 高 | 低：阈值再调会混淆 | 暂缓 |
| 人工有限 intervention window | 中 | 未知 | 中 | 中 | 低 | 高 | 高 | 中；E5 已自然有限 | 不单独做 |
| age/label/size 匹配的 Phase-A 诊断 | 高 | 不适用 | 高 | 高 | 中 | **低（离线）** | 低 | 高 | 随下一阶段做，不计训练 run |
| 换 residual/angular/新 score | 低/中 | 未知 | 低/中 | 低 | 中 | 高 | **高** | 低 | 不做 |

## 11. Reviewer attack and response

**“这不就是 FedPhoenix + heuristic top-k？”** 目前确实可能是；只有同一 reset budget、位置配对规则、history age、score-validity 支持下，same-client 显著胜过 shuffled 和 population，且胜过 current-code FedPhoenix，才能构成非平凡身份条件效应。否则叙事必须降级。

**“为什么 client identity 必要？”** 现有 Phase-A 跨 client 对照未严格匹配。下一阶段以 age-exact deterministic derangement 和群体 ranking 为反事实，target client 与全部本地训练不变，仅替换历史分数来源；若无差异，承认 identity 非必要。

**“为什么不是 test-set tuning？”** E5 已从多个 test-peak 观察选出，只能视为 development hypothesis。下一阶段冻结 E5 定义、共同掩码和终点，不挑最佳 checkpoint；本阶段 seed1 的正结果仍只是机制提示，之后需独立种子/分割确认。

**“固定 151 凭什么？”** 目前只是开发选择，不能把它说成经证明的 client confidence threshold。它还与 FedPhoenix stair schedule 和干预预算交织。先在 E5 固定151上识别历史身份的必要性，再考虑无需 test 调参的基于参与次数/置信度 gate；若身份不重要，gate 创新没有基础。

**“只看 seed1，baseline 又不配对？”** 接受。下一阶段先补同代码 1000 轮基线，再在固定单 seed 下检验机制；任何性能主张必须进入预先锁定的 paired multi-seed，不能用历史日志 + 现有 0.01 pp peak 宣称胜利。

## 12. Recommended next stage

**一个 focused stage：current-code baseline provenance + history-source causal ablation。** 计划 4 个 1000-round 训练配置，**此文仅提出，完全不执行**。保留 CIFAR-10/VGG/100 clients/frac .1/local 5/BS50/SGD lr .01 momentum .5 wd0/α=.3/seed1/`FP_conv=1000`/reset 1/64/`ori_normal`/同 partition/同 1000 轮。冻结 E5 的 ρ=.25、gap≤10、四层和 start151。预先生成参与序列与 age-exact donor 匹配、共同资格 mask；4,032 个候选中的 4 个无 exact-age peer，所有历史驱动臂在这些事件**统一回退原 FedPhoenix reset**，避免不等量 exposure；若再有共同 valid-score 支持不足，亦对全部臂应用同一事先冻结的回退规则。不得从 test 结果反调匹配或 start。

对每轮全部 100 个**已经有历史**的 client 按最新 history age 分组；组内采用独立于全局训练 RNG 的确定性 derangement，保证 donor≠target、age 精确相同。Same、shuffled、population 三臂每个 target/layer 使用相同 score-validity 共同支持、相同 `K=floor(ρ×baseline_reset_slots)`、相同 baseline task seed、SHA256 retention 和 reset-value relocation，仅改变用于 top-k 的 score 来源。Population 用同 age 组其他 client 的 layer 内 percentile ranks 平均（排除 target）；same 和 shuffled 也先转成 percentile rank，单 source 排名不变。记录 age、client 数据量/label-profile 距离和逐层来源；score 来源带来的 label-skew 混杂仍须诚实报告，不能声称已证明超越数据分布的“身份本体”。

正式 run 前仅做无训练的匹配可行性/共同支持预检和小规模 correctness gate：`ρ=0` 原 FedPhoenix parity、三历史臂前 150 轮 global hash 一致、每轮 selected clients/task seeds 对齐、三臂在相同 client/layer 的 target slot 数及 reset tensor multiset 相同、全局 RNG 未改变。**预注册主要描述指标**为 rounds 901–1000 的逐轮 accuracy 均值以及与配对 baseline 的差；同时报告 151–1000 平均、peak/peak round、final、last20/50/100、best rolling5/10、阶段均值、per-layer/client targeted slots、实际位置替换、fallback、内存/运行时间。这些终点已受 seed1 既有观察影响，只能用于开发判断，不能作为无偏显著性测试。解释时以**run/seed 为推断单位**，不把 1000 个自相关 rounds 当成 1000 个独立样本。

**决策门槛（定性、预注册）**：若 same 在固定后期均值与大部分阶段同时优于 shuffled 和 population，且不低于配对 FedPhoenix，才冻结方法进入 paired multi-seed；若 same≈controls，identity 叙事失败，即使都优于 FedPhoenix也更像 cohort salience；若 same>controls 但≤FedPhoenix，机制有信号而性能价值不足，应优先做 reset-high/protect-high 单一符号检验；若全臂均≤FedPhoenix，停止此版本，不继续比例搜索。单 seed 差异不作显著性或“证明有效”表述。下一批**不启动 seed2/3**；进入多 seed 是本阶段正结果后的另一项独立决定。

## 13. Exact proposed experiments (plan only)

| ID | 训练配置 | 与其他臂的锁定关系 | 预算 |
| --- | --- | --- | --- |
| B0 | 当前 main 的纯 FedPhoenix，1000 轮，seed1 | 同划分/模型/optimizer/task generator/evaluation；保存逐轮 hashes、selected clients、reset masks | 约 3.7 GPU-h |
| H | E5 同 client 最近一次 `update_norm`，age-exact 共同 mask | E5 本体；四层、ρ、start、gap 固定 | 约 3.7 GPU-h |
| S | H 的 score source 改为**同 age、不同 client** 的 deranged history | H 其余路径、资格和目标预算不变 | 约 3.7 GPU-h |
| P | H 的 score source 改为**同 age peers 的 leave-one-out mean percentile rank** | H 其余路径、资格和目标预算不变 | 约 3.7 GPU-h |

总计约 **14.9 GPU-hours**；若安全并行，wall time 不一定线性缩短，任何 OOM 只允许按相同配置顺序重跑。已有 L2/E5 作为外部可复核结果，不替代 H：共同资格 mask 预计排除至少 4 个事件，H 必须与 S/P 使用完全相同 mask 才能严格解释身份差。不得在上述四臂之外暗中加入 ratio/start/score search。

## 14. Per-experiment hypotheses, controls and interpretations

| Run | Hypothesis | Control | 唯一变化 | 正结果解释 | 负结果解释 | 计算估计 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 当前 FedPhoenix 1000 轮提供可审计同协议参照 | 当前 300-round observer 前缀、历史 1000-round 日志仅作 provenance 参照 | **无算法改动**；把当前实现的 baseline 从 300 轮扩展到公平的 1000 轮 | 首段 hashes/任务一致且元数据完整，能合法作 paired baseline | 若前缀不一致，先定位运行/配置差异，后续性能比较暂停 | ≈3.7 GPU-h |
| H | 固定 E5 在共同资格 mask 上仍有有效的 target-history 效果 | B0；S/P 为更强机制对照；已有 L2 只作复核 | 相对原 E5 仅加三历史臂共享资格/validity mask | 若优于 B0 且身份对照，支持继续正式检验 | 不及 B0：现版本缺乏性能价值；若与旧 L2 大幅不同，先审计 mask 与轨迹 | ≈3.7 GPU-h |
| S | 错配身份会降低性能，说明 target 的自身 repeated history 有增量价值 | H | **只把 score 来源**改为 age-exact 异 client，budget/mask/action 相同 | H>S 支持 client-conditioned 排名超越普通历史 salience | H≈S 或 H<S：identity 非必要或错配反更好；不能宣称 client-specific mechanism | ≈3.7 GPU-h |
| P | 同 client 历史优于同时龄群体排名 | H（S 作交叉参照） | **只把 score 来源**改为 leave-one-out 群体分位排名 | H>P 且 H>S 支持个体历史增量 | H≈P 或 H<P：群体时序 salience 足够或更稳健；新意/机制叙事受损 | ≈3.7 GPU-h |

上述“正/负”只是对固定 seed 机制的解释，不是统计显著性结论。三历史臂的全局训练轨迹在首次不同 intervention 后自然分叉，这正是长期 treatment effect；但逐轮 selected clients/task seeds、eligible budget 必须保持一致，防止比较混入其他因素。

## 15. Explicitly NOT recommended now

- **立即 E5 多 seed 主表**：baseline provenance 与 identity 对照都未闭合；先跑会把昂贵计算用在未识别机制上。正结果后应尽快多 seed，而非无限延后。
- **继续 seed1 ratio/gap/start/layer/score 网格**：已有 test-guided screen 足够多，继续搜会放大 winner's curse；E5 的 0.01 pp 1000-round peak 没有容错空间。
- **立即把固定 151 改 adaptive gate**：会同时改变开始时间和干预量，负结果难解释；先验证历史源有用。
- **单独测试手工 stop round 770**：现有阶梯调度已经产生自然 stop，新增 cutoff 主要复制现象并增加搜索维度。
- **现在改 optimizer、数据划分、augmentation、通信预算或 aggregation**：破坏与 FedPhoenix 的公平性，也不回答身份因果问题。
- **只做 300 轮替代 1000 轮**：E5 与其他臂的晚期排序和活动窗口会改变，不能代表主论文预算。
- **把 Phase-A overlap 当 reset 效用的替代指标**：这是预测排名而非 intervention 的因果收益；必要时下一阶段之后再做 reset-high vs protect-high 符号检验。

**最终判断：**当前 E5 是一个值得用严格对照检验的研究假设，而不是已经成立的 client-specific 性能方法。论文框架可保留；方法性能与身份必要性须由上述四臂、随后配对多 seed 的结果决定。
