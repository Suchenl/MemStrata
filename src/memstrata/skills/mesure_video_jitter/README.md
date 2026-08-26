# 运动稳定性 Skill (Motion Stability)

> **核心定义**：把"画面一直在抖"从主观感受变成可比较的数字，用于对比不同生成器、LoRA、步数配置的
> 帧间微抖动（micro-jitter）。
> **典型用途**：判断某条视频生成路线是否可用于生产；A/B 对比蒸馏模型与全步数模型；回归监控。

---

## 🔑 为什么不能只看位移 (Why Displacement Is Not Enough)

**位移无法区分"运镜"和"抖动"**：
- 一个缓慢的横摇（pan）位移很大，但完全不抖；
- 一个锁死机位的镜头如果每帧微微颤动，位移几乎为零，但看起来很难受。

真正的判别信号是运动的**二阶导数（加速度）**：
- 真实摄影机（或真实胶片扫描）的加速是平滑的，加速度远小于速度；
- 逐帧重新"决定"构图的生成器会产生随机游走，其加速度与速度相当甚至更大——观众读到的就是"画面一直在抖"。

逐帧位移用 **相位相关（phase correlation）** 在 Hann 窗灰度图对上求得，跟踪全局主位移，忽略局部主体运动。

---

## 📏 三个指标与判定 (Metrics and Verdict)

| 指标 | 含义 |
|---|---|
| `speed_px` | 逐帧位移幅度的中位数（像素） |
| `accel_px` | 逐帧位移**变化量**幅度的中位数（像素）——抖动信号本体 |
| `ratio` | `accel / speed`，≥1 表示运动被逐帧噪声主导 |

判定顺序**先看 ratio**：被噪声主导是最坏情况，即使绝对幅度不大——眼睛跟的是不一致性，不是幅度。

| verdict | 条件 |
|---|---|
| `noise-dominated` | `ratio >= 1.0` |
| `jittery` | `accel_px > 3 x 0.05px`（真实胶片底线的 3 倍） |
| `steady` | 其余 |

---

## 🚨 最重要的一条：位移大 ≠ 有问题 (Big Pixel Diff Is Not Automatically A Defect)

帧间像素差大，很可能只是**运动本身大**。这就是本 skill 不用位移、而用 `accel` 与 `ratio` 的原因，
也是为什么**报告时必须把 `speed` 和 `accel` 一起给出**。

实测例子（`seg_023`）：位移 1.906px，是所有素材里最大的，但 `ratio = 0.70` → 判为 `jittery` 而非
噪声主导，因为它的运动确实是真实的画面移动。反例是 `seg_013`：位移只有 0.347px（画面几乎静止），
但 `accel` 高达 0.448px、`ratio = 1.29` → 这才是"明明没怎么动却一直在颤"。

**唯一站得住的比较方式**：同一个镜头、同一条提示词、运动量相近的两个臂之间对比 `accel`。
跨素材横比绝对值没有意义。

---

## 📌 实测记录 (Measured Reference, 2026-07-28)

全部用本 skill 测得，命令见文末。`cuts` = 检出的内容突变对数；真实长片是拼接的，须用
`within_shot=True` 取无切镜窗口。

### 生成素材（Track B story 0001，同一部片）

| 素材 | speed px | accel px | ratio | verdict | cuts |
|---|---|---|---|---|---|
| A14B distill4step + morphic + 双关键帧 | 0.293 | 0.194 | 0.66 | jittery | 18 |
| A14B distill4step + 双关键帧，无 LoRA | 0.821 | 0.560 | 0.68 | jittery | 18 |
| A14B distill4step + morphic + 单首帧 | 0.135 | 0.079 | 0.59 | **steady** | 0 |
| Turbo-5B seg_013（与上面同镜头同提示词） | 0.347 | **0.448** | **1.29** | noise-dominated | 0 |
| Turbo-5B seg_005 | 0.942 | 0.919 | 0.98 | jittery | 0 |
| Turbo-5B seg_016 | 1.105 | 0.772 | 0.70 | jittery | 0 |
| Turbo-5B seg_023 | 1.906 | 1.327 | 0.70 | jittery | 0 |
| A14B distill4step + 5 帧运动前缀（续写探针 P5） | 0.239 | 0.211 | 0.88 | jittery | 0 |

**可用的结论**：seg_013 与 A14B 双关键帧臂运动量相近（0.347 vs 0.293），此时 Turbo 的逐帧加速度是
A14B 的 2.3 倍——同等运动量下 Turbo 明显更抖。这是同镜头同提示词的对照，站得住。

### 本 skill 的第一个实战成果：定位到 Turbo 抖动的根因是分辨率

用同一张 FLUX 关键帧、同一条提示词、同一个种子 2026、同一个常驻服务，只改渲染分辨率：

| 渲染分辨率 | speed px | accel px | ratio | verdict | 生成耗时 |
|---|---|---|---|---|---|
| **704x1280（checkpoint 原生）** | 1.858 | **0.170** | **0.09** | jittery | 25.4 s |
| 480x832（原 Track B 设置） | 0.435 | **0.490** | **1.13** | noise-dominated | 14.6 s |

原生分辨率下画面运动量是 4.3 倍，逐帧加速度反而低 2.9 倍，ratio 差 12.5 倍。抖动来自偏离
checkpoint 原生分辨率，而不是模型本身。修复方式是原生渲染 + 单次降采样交付，见
`methods/MemStrata/configs/video_gen/wan22_ti2v5b_turbo.toml`。

注意 704x1280 那条仍被标为 `jittery`：绝对 accel 随运动量水涨船高，而它的运动量本来就大。看
`ratio = 0.09` 才知道它是干净的运动——**这就是"位移大 ≠ 有问题"在真实决策里的样子**，也是为什么
本 skill 坚持 speed 与 accel 必须一起报。

证据在 `experiments/results/probe/turbo_resolution_jitter/`。

**副产品**：双关键帧两臂各检出 18 个内容突变，单首帧臂 0 个。因为那两臂被要求在 81 帧内从
seg_013 的构图morph到 seg_014 的构图（跨段插值），画面会成段跳变。

### 真实影片（无切镜窗口）

| 素材 | 窗口 | speed px | accel px | ratio | verdict | cuts |
|---|---|---|---|---|---|---|
| Casablanca（固定机位老片） | 61 帧 | 0.952 | 0.095 | 0.59 | jittery | 4 |
| Big Buck Bunny（CGI） | 155 帧 | 0.145 | 0.109 | 0.58 | steady | 1 |
| **American Beauty（手持实拍）** | 91 帧 | 0.373 | **0.546** | **1.44** | **noise-dominated** | 4 |

**这一行是本 skill 最重要的校准事实**：一部真实电影在完全无切镜的窗口内也能测出 ratio 1.44，
比当初触发这套度量的 Turbo 片段（1.29）还"差"。真实手持摄影本身就是小幅随机游走。
所以 **verdict 不是真假/好坏的绝对闸门**，只能用于同条件 A/B 排序。

### 排除掉的混淆项：胶片颗粒

怀疑相位相关会把颗粒/编码噪声读成逐帧位移，于是对所有素材先做 σ=2 高斯模糊再测：
American Beauty 0.546→0.439、Casablanca 0.095→0.072、Turbo seg_013 0.448→0.472。
**量级和排序都没变，颗粒不是原因**，所以本 skill 不做去噪预处理。

---

## 🛠️ 用法 (Usage)

```python
from memstrata.skills.mesure_video_jitter import measure_jitter, compare

report = measure_jitter("outputs/seg_013.mp4")
print(report.speed_px, report.accel_px, report.ratio, report.verdict, report.cuts_detected)

# 拼接长片必须限定在无切镜窗口内，否则不可与单个生成段比较
film = measure_jitter("0001_American_Beauty.mp4", within_shot=True)
print(film.window, film.accel_px)

for r in compare(["a.mp4", "b.mp4"]):
    print(r.as_line())     # 不可读的片子会被跳过并告警，不会拖垮整批
```

命令行（本次实测就是这条）：

```bash
PYTHONPATH=methods/MemStrata/src python -m memstrata.skills.mesure_video_jitter \
  <clip.mp4> ... [--within-shot] [--max-frames 200]
```

数值内核与视频 I/O 是分开的，便于测试和复用：

```python
from memstrata.skills.mesure_video_jitter import translations_from_frames, jitter_from_translations

shifts = translations_from_frames(my_gray_frames)   # 逐帧全局位移
report = jitter_from_translations(shifts)
```

---

## ⚠️ 注意事项 (Precautions)

1. **段内比段间**：跨镜头切换处像素差本来就很大。`cuts_detected` 会始终报出检出的内容突变数，
   看到非 0 就用 `within_shot=True` 限定到最长无切镜窗口，并连同 `window` 一起报数。
   默认**不**自动限定，避免悄悄用一个被截短的窗口出数。
   实测发现：中位数本身就抗单个切镜（一个切镜几乎不改变结果），真正会毁掉统计的是密集快切——
   此时本 skill 会直接报错拒绝出数，而不是给一个假数。
2. **静止片段**：完全静止时 `speed_px = 0`，`ratio` 为 `inf`，此时只看 `accel_px` 的绝对值。
3. **至少 3 帧**：加速度需要两个位移差分，不足会直接报错而不是返回可疑的 0。
4. **重编码不改结论**：抖动来自生成器时，原始段文件与拼接后的视频测出的量级一致；
   若两者差异很大，先查拼接/编码环节。
