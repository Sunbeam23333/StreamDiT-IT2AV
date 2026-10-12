# TTS 评测与 CosyVoice "非人类语言" 排查

> CosyVoice 系列在公开榜单上 CER / SIM 都很漂亮，但在我们的环境里生成出来的却是"像人声但不是人话"的音频。本文回答三个问题：
> 1. 榜单上的 TTS 指标到底在测什么、测不到什么；
> 2. 我们用什么方法定位到了问题所在的环节；
> 3. 如果要公平地评测 TTS，应该怎么做。

---

## 1. 常用 TTS 指标

| 指标 | 怎么算 | 衡量什么 | 盲区 |
|---|---|---|---|
| **CER / WER** | 用强 ASR（Paraformer-zh / Whisper-large-v3）把生成音频转回文字，与输入文本算字 / 词错误率 | 内容是否读对（漏字、错字、多字） | ASR 对"音素大致对但韵律 / 共振峰崩坏"的音频容忍度很高，口齿不清的乱读也可能被识别回正确文本 |
| **SIM / SECS** | 用说话人验证模型（ERes2Net / CAM++ / WavLM-Base-Plus）分别提取参考音频和生成音频的 embedding，算余弦相似度；0.6–0.8 算好 | 音色像不像参考人 | speaker embedding 编码的是声道 / 基频范围 / 共振峰，**对内容不敏感**——"外星语"只要音色像，SIM 照样高 |
| **MOS**（NMOS / SMOS / QMOS） | 人工 1–5 分打分 | 自然度 / 相似度 / 质量 | 最可靠，但最贵，只在论文最终结果里出现 |
| **UTMOS / NISQA** | 训练好的 MOS 预测网络 | 近似自然度 | 离"是不是人话"最近的自动指标，但仍是近似 |
| **DNSMOS P.835** | 微软语音质量预测（SIG / BAK / OVRL） | 语音 / 背景噪声质量 | 偏降噪场景 |
| **RTF** | 推理耗时 / 音频时长 | 效率 | — |

**为什么 CER 和 SIM 可以同时很高，音频却是"非人话"**：CER 只看 ASR 能不能识别回文本，SIM 只看音色——而音色主要来自参考音频的 speaker embedding，几乎不受 LLM 输出质量影响。**没有一个常用自动指标直接衡量"像不像人话"**，这是 TTS 评测长期存在的盲区，也是 CosyVoice3 的 GitHub issue 里大量"效果是噩梦"反馈与榜单分数并存的原因。

---

## 2. 我们的诊断方法

### 2.1 CosyVoice2 的链路

```text
                  ┌──────────────┐                   ┌──────────────────┐             ┌────────┐
text + prompt ──► │  LLM (Qwen)  │ ── speech token ─► │ Flow Matching    │ ── mel ───► │  HiFT  │ ──► 24 kHz wav
prompt speech tok └──────────────┘  (0–6560 的整数,   │ (chunk-aware CFM)│             │vocoder │
       ▲                             每个约 40 ms)    └──────────────────┘             └────────┘
       │
speech_tokenizer_v2.onnx (prompt_wav → token)
```

"乱读"理论上可能出在三个位置：

| 位置 | 表现 | 怎么验证 |
|---|---|---|
| ① prompt speech tokenizer（ONNX） | LLM 看到的 prompt 上下文就是噪声 | 检查 `llm_prompt_speech_token` 的唯一值率 |
| ② LLM 生成 | 输出 token 退化（重复）或越界 | **token 序列统计（Step 4）** |
| ③ Flow + HiFT 解码 | token 正常但 mel / wav 异常 | **信号层检查（Step 5）** |

### 2.2 信号层检查（不需要参考音频、不需要神经网络）

`debug_cosyvoice_tts.py` / `debug_cosyvoice_weights.py` 对生成的 wav 做纯信号统计：

| 统计量 | 算法 | 判断 |
|---|---|---|
| RMS | `sqrt(mean(wav²))` | <0.001 是静音，>0.5 是爆音；只判断"活着没有" |
| 波形范围 | `[min, max]` | 是否截断 / 全零 / clip |
| FFT 频段能量 | `rfft` 后按 次低频 / F0 / 低谐波 / 中频 / 高频 / 超高频 累加占比 | 能量是否集中在语音频段（全在 0–85 Hz 是嗡嗡声，全在 4 kHz 以上是嘶嘶声） |
| F0 峰值 | 85–255 Hz 内的频谱最强点 | 男声约 130 Hz、女声约 200 Hz |

它像"心跳监测"：能告诉你信号是不是像语音，但不知道说的是什么、像不像目标说话人。

### 2.3 LLM speech token 统计（关键断点）

```python
tokens = np.array(model.tts_speech_token_dict[uuid])
print("范围:", tokens.min(), tokens.max())                  # 正常应覆盖 0–6560 的大部分
unique_ratio = len(np.unique(tokens)) / len(tokens)          # 正常 50%–85%
repeat_ratio = np.sum(np.diff(tokens) == 0) / (len(tokens) - 1)   # 正常 <20%
out_of_range = np.sum((tokens < 0) | (tokens >= 6561))       # 必须为 0
print(tokens[:30], tokens[-30:])                             # 肉眼看: 正常是"乱"的, 退化时是 100,100,100... 或 100,200,100,200...
```

| 统计 | 正常 | 退化信号 |
|---|---|---|
| 唯一值率 | 50%–85% | <10%（模型在少数 token 间循环） |
| 相邻重复率 | <20% | >50%（卡在某个 token 上，解码成嗡嗡声或拖长的元音） |
| 越界 | 0 | >0 说明 LLM head 维度与码本不匹配（权重加载错 / 版本错配） |

### 2.4 我们的实测结果

```text
speech RMS:                    0.04 – 0.1     ✅
F0 峰值:                        ~180 Hz       ✅ 女声范围
中频共振峰 (500–2000 Hz) 占比:   30%+           ✅ 像语音
LLM token 唯一值率:             < 5%           ❌
LLM token 相邻重复率:           > 50%          ❌
```

**结论**：信号层一切正常（所以"听起来像人声、有元音、有音色"），但 LLM 输出的 speech token 严重退化，内容就是乱码。按这个形态推断，SIM 大概率仍在 0.7 以上，CER 也未必很差——这正是第 1 节说的盲区。

### 2.5 排除过的假设

| 假设 | 结果 |
|---|---|
| `fp16=True` 数值问题 | 改 `fp16=False`，无效 |
| CosyVoice3 本身质量差 | 退回 CosyVoice2，无效 |
| CPU 版 onnxruntime 让 speech tokenizer 静默跑在 CPU 上 | 换 `onnxruntime-gpu` 1.18 → 1.20.1，无效 |
| whisper 版本导致 `mel_filters` 不一致 | 读源码排除 |

**还没验证的方向**：① 先看 `llm_prompt_speech_token` 本身的唯一值率，确认 ONNX tokenizer 的输出是否已经异常；② 校验 `speech_tokenizer_v2.onnx` 和 `llm.pt` 是否完整（是否误拉到了 LFS 指针或截断文件，参考 [踩坑 E3](environment_pitfalls.md#e3-new_auto_promptpt-是-git-lfs-指针) 的教训）；③ 检查输入 mel 的数值范围与官方实现是否一致；④ 调整 LLM 采样参数（top-k / temperature / repetition-aware sampling）。

```bash
bash script/digit/run_debug_tts.sh      # 跑完整诊断, 重点看 Step 4 (token 统计) 与 Step 5 (信号统计)
```

---

## 3. 建议的 TTS 评测协议

只报 CER + SIM 不足以发现这类问题。建议至少：

1. **CER / WER**（Paraformer-zh 测中文、Whisper-large-v3 测英文）——内容层；
2. **SIM**（ERes2Net / WavLM）——只在有参考音频的克隆任务上报；
3. **UTMOS 或 NISQA**——自然度近似；
4. **DNSMOS P.835**——质量；
5. **RTF**——`run_audio_benchmark.py` 已经在 `benchmark_report.json` 里记录；
6. **人工抽听**：每个模型至少抽 10 条。这是发现"非人话"问题唯一可靠的手段；
7. 对 LLM 类 TTS（CosyVoice / Spark-TTS / Fish-Speech）额外记录 **token 唯一值率与重复率**，作为廉价的退化预警。
