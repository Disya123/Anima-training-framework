# AnimaTrainer v0.1

Нативный architecture-aware trainer для CircleStone Labs Anima. Это отдельный проект: он не использует L2P, не обучает B1 как универсальный style-код и не зависит от кода `nl_align`.

## License

Released under the [GNU GPL v3](LICENSE).

Главная идея — выбирать режим обучения по типу изменения модели:

| Режим | Что должен нести trigger | Основной train scope |
| --- | --- | --- |
| `style` | визуальный способ рендеринга | DiT LoRA |
| `character` | инвариантную identity персонажа | DiT LoRA |
| `object` | форму, материал и identity объекта | DiT LoRA |
| `general` | trigger не нужен; меняется базовое распределение | широкий LoRA или full weights |

Текущая версия реализует минимальный практический контур: строгий JSONL-датасет, aspect-ratio buckets, нативные Qwen Image VAE latents, нативный Qwen/LLMAdapter conditioning, Anima FLOW objective, выбор scope, prior preservation, checkpoint/resume, детерминированную latent-space validation, ComfyUI-совместимый LoRA export и измерение update geometry (`analyze`): per-layer спектры ΔW, энергия по блокам/компонентам, functional truncation — чтобы rank и scope выбирались по данным, а не вслепую.

## Что именно обучается

VAE, Qwen text encoder и штатный Anima LLMAdapter заморожены. Они используются один раз при кэшировании. Во время train на GPU остаётся DiT и выбранные обучаемые параметры.

Для clean latent `x0`, шума `epsilon` и уровня `sigma` используется нативная flow-параметризация Anima:

```text
x_sigma = (1 - sigma) * x0 + sigma * epsilon
target  = epsilon - x0
L_flow  = weighted_mse(DiT(x_sigma, sigma, conditioning), target)
```

При наличии regularization-датасета итоговая функция:

```text
L = L_concept + prior_loss_weight * L_prior
```

Это не дополнительный perceptual loss. В v0.1 ставка сделана на правильную декорреляцию датасета, sampling и held-out проверку.

## Требования и установка

- Python 3.11+
- CUDA GPU с поддержкой `bfloat16`
- локальные Anima DiT, Qwen Image VAE и Qwen 3 0.6B checkpoint
- локальный Hugging Face-кэш токенизаторов `Qwen/Qwen3-0.6B` и `google/t5-v1_1-xxl`

Фреймворк полностью автономен: он не импортирует и не использует установку ComfyUI ни для VAE, ни для conditioning, ни для чего-либо ещё. Qwen Image VAE портирован в `anima_trainer/wan_vae.py` (оригинал — Apache-2.0 Wan-Video/Wan2.1), токенизаторы берутся из локального HF-кэша и дают идентичные продакшену token id и conditioning (проверено bit-exact).

Из PowerShell:

```powershell
cd "E:\AI\Work\Anima-edit\Anima training framework"
python -m pip install -e ".[test]"
```

Для `optimizer: adamw8bit` дополнительно:

```powershell
python -m pip install -e ".[eightbit,test]"
```

Если `bitsandbytes` в конкретной Windows/CUDA-сборке не работает, установите `train.optimizer: adamw`.

## Manifest

Manifest — UTF-8 JSONL: один JSON-объект на изображение. Caption описывает содержание, а trigger добавляется trainer-ом. Не нужно превращать caption в `@name, ...` вручную.

```json
{"id":"cafe-01","image":"images/cafe.png","caption":"1girl, black hair, sitting in a cafe, night","trigger":"@kor_lili","concept_type":"style","weight":1.0,"hard_tags":[],"facets":{"content":"portrait","subject_count":"1","scene":"cafe"},"split":"train"}
{"id":"street-01","image":"images/street.png","caption":"2boys, city street, rain","trigger":"@kor_lili","concept_type":"style","weight":1.0,"hard_tags":[],"facets":{"content":"people","subject_count":"2","scene":"street"},"split":"validation"}
```

Поля:

- `image` — обязательный абсолютный путь или путь относительно manifest.
- `caption` — content-only описание; для `style`, `character` и `object` не может быть пустым.
- `trigger` — можно задать в записи или один раз через `concept.trigger` в YAML.
- `concept_type` — `style`, `character`, `object` или `general`; должен совпадать с режимом run.
- `weight` — положительный базовый вес примера, по умолчанию `1.0`.
- `hard_tags` — теги, чьи множители задаются в `concept.hard_tag_weights`.
- `facets` — диагностические признаки разнообразия; на loss напрямую не влияют.
- `split` — `train`, `validation` или `test`.
- `id` — уникальный стабильный идентификатор; если пропущен, строится из имени файла и строки.

Для папки с парами `image.png` + `image.txt` базовый manifest можно получить так:

```powershell
python scripts/make_manifest.py `
  --data-dir E:\datasets\kor-lili `
  --output E:\datasets\kor-lili\manifest.jsonl `
  --mode style `
  --trigger "@kor_lili" `
  --recursive
```

После генерации выделите часть записей в `validation` и заполните `facets`. Аудит считает покрытие и число уникальных значений. Ожидаемые facets:

| Режим | Facets для проверки декорреляции |
| --- | --- |
| `style` | `content`, `subject_count`, `scene` |
| `character` | `pose`, `camera`, `clothing`, `background`, `expression`, `lighting` |
| `object` | `view`, `scale`, `occlusion`, `scene`, `lighting` |
| `general` | `pose`, `camera`, `interaction`, `composition` |

Для anatomy/physics hard examples размечаются явно:

```json
{"id":"pose-017","image":"images/pose-017.png","caption":"two people carrying a heavy box, low camera, strong foreshortening","concept_type":"general","weight":1.0,"hard_tags":["foreshortening","hand_object_contact","multi_person_interaction"],"facets":{"pose":"dynamic","camera":"low three-quarter","interaction":"carry","composition":"two-person"},"split":"train"}
```

Вес такой записи равен `weight`, умноженному на каждый соответствующий множитель из YAML.

## Быстрый запуск

Готовые шаблоны находятся в `configs/` и уже показывают ожидаемые пути к локальной установке моделей:

- `style.example.yaml`
- `style-core.example.yaml` — **проверенный минимальный рецепт** (см. ниже)
- `character.example.yaml`
- `object.example.yaml`
- `general-hard.example.yaml`
- `selected-blocks-full.example.yaml`

### Проверенный рецепт: минимальный style-адаптер (`style-core.example.yaml`)

Полный пайплайн от папки картинок до экспорта, на реальном прогоне (74 изображения, 1024²):

```powershell
# 1. Манифест из пар image/.txt (триггер может уже лежать в подписи — он вычистится и вставится сам)
python scripts/make_manifest.py --data-dir dataset/my-style `
  --output data/my-style/manifest.jsonl --mode style --trigger "@my_style"

# 2. Кэш: VAE latents + conditioning (отдельная команда; train без неё упадёт с "run cache first")
anima-trainer cache --config configs/style-core.example.yaml

# 3. Тренинг
anima-trainer train --config configs/style-core.example.yaml

# 4. Экспорт в kohya-совместимый safetensors для ComfyUI
anima-trainer export --config configs/style-core.example.yaml `
  --checkpoint runs/style-core-example/checkpoints/training-step-000300.pt `
  --output my_style_core.safetensors
```

Что в этом рецепте особенного (обоснование — разделы ниже + результаты прогонов):

| Параметр | Значение | Зачем |
| --- | --- | --- |
| `blocks: [14..18]` | только ядро стиля | Каузальная абляция на Anima DiT: хвосты b19–27 несут ~0, b14–18 + modulation + CA-v держат манеру рендера |
| `components: [modulation, cross_attn, mlp]`, self_attn отсутствует | — | Self-attn не показал каузального вклада в стиль |
| `rank_overrides: mod 2 / mlp 6`, база ca 8 | 60 модулей ≈ **1.45M** параметров (~1.5 MB bf16) против 17.3M у широкого scope | Компактность без потери качества стиля |
| `anchor_no_trigger_weight: 8.0` | function-space якорь | Держит leak ratio (реакцию на no-trigger промпты) в пределах ~30% вместо 93% у голого LoRA |
| `checkpoint_mode: block` + `checkpoint_sac: sdpa` | exact execution | ~5.5 GiB peak VRAM, ~7.5 c/шаг на RTX 3060 12GB при 1024² |

Здоровые числа validation по ходу тренинга (`runs/<name>/events.jsonl`, event=validation): `preservation_drift_no_trigger` ≤ ~0.01, `target_drift` растёт до ~0.03–0.06 к шагу 300, `trigger_response` > 0. Если preservation догоняет target — стиль не учится; если preservation ползёт выше ~0.01 при живом trigger_response — началась утечка триггера в базу.

Скопируйте подходящий YAML, поменяйте manifest/output и сначала выполните аудит:

```powershell
anima-trainer audit --config configs/style.example.yaml
```

Затем закэшируйте target и regularization distribution:

```powershell
anima-trainer cache --config configs/style.example.yaml --include-prior
```

Кэширование разделено на VAE latents и conditioning. Это позволяет менять train scope, rank, learning rate и число steps без повторного запуска VAE/TE. `--force` пересобирает существующий кэш.

Запуск:

```powershell
anima-trainer train --config configs/style.example.yaml
```

Продолжение с сохранением optimizer, scheduler и RNG state:

```powershell
anima-trainer train `
  --config configs/style.example.yaml `
  --resume runs/kor-lili-style/checkpoints/training-step-000500.pt
```

Отдельная проверка checkpoint:

```powershell
anima-trainer validate `
  --config configs/style.example.yaml `
  --checkpoint runs/kor-lili-style/checkpoints/training-step-000500.pt
```

Повторный export LoRA:

```powershell
anima-trainer export `
  --config configs/style.example.yaml `
  --checkpoint runs/kor-lili-style/checkpoints/training-step-000500.pt `
  --output runs/kor-lili-style/artifacts/kor-lili.safetensors
```

Для full-weight scope CLI автоматически пишет merged модель в официальном `net.*` keyspace. Флаг `--merged` позволяет явно запросить merged export.

## Train scopes

```yaml
train:
  scope:
    kind: dit_lora
    blocks: all
    components: [self_attn, cross_attn, mlp]
    rank: 32
    alpha: 32
    dropout: 0.0
    trainable_dtype: float32
```

Поддерживаются:

- `dit_lora` — LoRA во всех выбранных компонентах DiT.
- `selected_blocks_lora` — LoRA только в заданных блоках `0..27`.
- `selected_blocks_full` — full weights выбранных блоков.
- `dit_full` — full DiT fine-tune.

Компоненты: `self_attn`, `cross_attn`, `mlp`, `modulation`, `x_embedder`, `timestep`, `final_layer`, `all`.

Пример scope-ablation:

```yaml
scope:
  kind: selected_blocks_lora
  blocks: [8, 9, 10, 11, 12, 13, 14, 15]
  components: [self_attn, cross_attn, mlp]
  rank: 32
  alpha: 32
```

Для честного A/B-теста сохраняйте одинаковые manifest, validation split, seed, sigmas и steps, меняя только scope и отдельный `project.output_dir`. Full-weight режимы требуют существенно больше VRAM и места под checkpoint, чем LoRA.

## Validation и anti-overfit

Перед первым шагом trainer сохраняет ответы base model на фиксированных validation latents, noise seeds и sigmas. После checkpoint считаются:

- `flow_loss` — held-out flow loss на triggered conditioning;
- `target_drift` — насколько изменился triggered ответ относительно base model;
- `preservation_drift_no_trigger` — collateral drift без trigger;
- `trigger_response` — различие triggered и content-only ответов;
- `trigger_response_change` — изменение этой разницы относительно base model.

Это диагностические latent/trajectory метрики, а не окончательная оценка качества изображений. Сравнивать checkpoint нужно ещё и на фиксированном behavioral prompt/seed bank. Меньший train loss сам по себе не является критерием выбора.

Выход run:

```text
output_dir/
├── run_config.yaml
├── events.jsonl
├── validation_baseline.pt
├── checkpoints/
│   └── training-step-XXXXXX.pt
└── artifacts/
    ├── adapter-step-XXXXXX.safetensors
    └── adapter-final.safetensors
```

LoRA export использует имена `lora_unet_blocks_...lora_down/up.weight` и `alpha`, совместимые с Anima LoRA loader в ComfyUI.

## Update geometry: rank и scope как результат измерения

Trainer не предполагает заранее, что знание живёт в `rank=16` и `q/k/v`. Первый вопрос — **какова реальная геометрия update**, который нужен модели. Для этого есть `analyze`:

```powershell
anima-trainer analyze --adapter path/to/lora.safetensors --output report.json
anima-trainer analyze --config configs/style.example.yaml --checkpoint runs/.../training-step-000500.pt --output report.json
```

- Адаптер читается в любом формате: kohya (`lora_unet_*` down/up), diffusers (`*.lora_A/B`) и LoKr (`lokr_w1/w2_a/w2_b`); ориентация матриц разрешается автоматически.
- Для низкоранговых адаптеров спектры считаются через маленькие SVD (без материализации `B@A`), для dense-чекпоинтов — прямой `svdvals` дельты `W_trained - W_base`.
- Отчёт: per-layer сингулярные спектры, stable/effective rank, ранги до 50/90/99% энергии, энергия по компонентам и блокам.

Дальше — **functional truncation**: усечь дельту до разных рангов (`truncated_lora_state`) и реально генерировать с `r=1,4,8,16...`. Может оказаться, что 20% математической энергии несут 95% perceptual эффекта. LoRA — это просто факторизация `ΔW ≈ AB`; выбор ранга и скоупа — вывод анализа, а не гиперпараметр вслепую.

### Что показал анализ 10 реальных LoRA (7 стилей, 3 персонажа)

Измерены подпространства update `ΔW = B@A` на 448 модулях DiT (all blocks, attn+mlp+modulation), выравнивание 4-мерных главных подпространств между независимыми LoRA (шанс ≈ 0.002):

| Компонент | стиль↔стиль | персонаж↔персонаж | стиль↔персонаж | Вывод |
| --- | --- | --- | --- | --- |
| `modulation` (adaln входы) | **0.131** | **0.115** | **0.125** | общий концепт-интерфейс (~65× шанс) |
| `cross_attn` | 0.017 | 0.030 | 0.022 | слабо общий; стиль-пик в `v_proj` блоков 23–27 |
| `mlp` | 0.005 | 0.009 | 0.007 | ≈ шанс: приватная ёмкость |
| `self_attn` | 0.008 | 0.014 | 0.010 | ≈ шанс: приватная ёмкость |

Дополнительно:

- **Энергия**: MLP 33–71%, self_attn 15–32%, modulation 10–22%. Полные LoRA концентрируются в поздних блоках 15–27 (60–70%); ранние 0–7 почти пустые. Attn-only LoRA — равномерно.
- **Эффективный ранг ~4 из 8** у полно-скоупных стилевых LoRA: стиль на слой действительно низкоранговый.
- **Консенсус-интерфейс** (топ модулей с общими подпространствами): `adaln_modulation_cross_attn.1` и `adaln_modulation_self_attn.1` в блоках 0–2, 16–20, 23–27 (ov4 0.39–0.57), затем `cross_attn.v_proj` блоков 23–27 (0.08–0.19).

Практический вывод для стиля (отражён в `configs/style.example.yaml`):

```yaml
scope:
  kind: dit_lora
  blocks: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
  components: [self_attn, cross_attn, mlp, modulation]   # modulation обязателен
  rank: 8
train:
  steps: 350
  lr_scheduler: cosine
  warmup_steps: 30
  timestep_sampling: weighted   # uniform σ∈[0,1] + bell-веса лосса (конвенция ai-toolkit)
  sigma_shift: 1.0              # shift инференса (3.0) в тренинге НЕ применяется
```

Критично: `sigma_shift: 1.0` при обучении. Inference-расписание Anima использует shift 3.0 (ModelSamplingDiscreteFlow), но тренинг должен покрывать **весь** диапазон σ равномерно — если зажать обучение в высоких σ (shift 3.0 + logit-normal даёт σ∈0.6–0.95), LoRA никогда не видит низкие шумы, а при генерации ΔW применяется на всех уровнях: детализация разрушается (текстурная каша). Проверено на практике: 12-картиночный style-датасет, ~300 шагов достаточно. `balanced`/`style_balanced` — псевдонимы uniform; `logit_normal` оставлен для контента/персонажей, `weighted` добавляет bell-веса лосса как в ai-toolkit.

MLP остаётся главной по энергии целью, но его направления приватны для каждого стиля — поэтому ранги в MLP могут понадобиться выше, а общий интерфейс (modulation, поздний cross_attn v_proj) переносится между концептами.

## Function-space anchor (no-trigger регуляризация)

Утечка стиля в no-trigger генерацию лечится якорем на функциональном пространстве вместо image-space prior:

```yaml
train:
  anchor_no_trigger_weight: 8.0   # 1.0 слишком слаб против target-градиента; 8.0 разделяет каналы
  anchor_every: 2                 # можно через шаг: функция связи не требует ежешагового форса
```

Механика на шаг: один широкий grad-forward на `cat([cond, cond_no_trigger])` с теми же x_t/σ + один no-grad forward замороженной базы (`lora_disabled`) → `anchor_loss = MSE(pred_notrig_θ, pred_notrig_base)`. В Anima текст входит в DiT **только** через cross-attn context (AdaLN читает чистый t_emb), поэтому сильный якорь выталкивает стиль именно в токен-читающий путь: на 300-м шаге ratio утечки preserve/target упал с ~93% до ~39%, триггер-ответ растёт.

Валидационные метрики для контроля: `preservation_drift_no_trigger` (должна застрять ≤ начального уровня), `target_drift` (продолжает расти), их отношение = доля утечки.

## Sequential-loss execution и вердикт по чекпоинтингу

∇(Σ wᵢLᵢ) = Σ wᵢ∇Lᵢ, поэтому target / anchor / prior исполняются **по очереди** (forward→backward→free на каждый), а не одним совместным графом: одновременно живёт максимум один autograd-граф. Явный якорный выигрыш: −1.4 ГиБ пика при той же скорости (6.72→5.32 GiB, 13.1 c/шаг) — `anima_trainer/config.py: checkpoint_mode`.

Гранулярность checkpoint'а измерена на 12 GiB (RTX 3060, сдвоенный якорный батч устранён sequential-исполнением):

| режим | c/шаг | torch peak | вердикт |
| --- | --- | --- | --- |
| block (whole) | 13.1 | 5.32 GiB | **оптимум, дефолт** |
| selective [self_attn, mlp] | 14.5 | 10.86 GiB (≈11.6 физически + WDDM spillover в shared) | не влезает в 12 GiB, медленнее |
| off / [mlp] | — | OOM / WDDM-fallback | мертво |

Вывод: whole-block checkpoint — Pareto-точка для этого графа (огромные spatial-активации 64×64×2048 при дешёвом recomputе); экономия FLOPs на cross-attn стоит больше, чем вся память, которую она требует. Экономия от sequential — реальна и бесплатна; дальнейшее дробление — тупик, не инвестировать.

## Gradient transport policy: измеренный вердикт (закрыто oracle-свипом)

`transport={"patterns"}` — detach входа ветки: параметрные градиенты внутри ветки точные, branch-Jacobian для upstream не строится. Forward побайтово идентичен; на одном блоке механизм чистый (diag: mlp-градиенты cos 1.00000).

Механизм деградации на глубине — не экспонента от per-block cosine (косинусы не мультиплицируются): небольшое систематическое изменение VJP каждого residual-пути композируется через глубокую Jacobian-цепь и декоррелирует adjoint. Правило безопасности тоже не «расстояние до выхода», а «сколько upstream-trainable параметров зависят от точного транспорта через разрез».

Horizon-свип (scripts/verify_policy.py --horizon, 4 батча, санити exact-vs-exact = 0.99991 = шумовой пол bf16):

| local policy | total grad cos | worst adjoint |
| --- | --- | --- |
| cross_attn blocks 23-27 | **0.991** | 0.85 |
| mlp block 27 only | 0.958 | 0.65 |
| mlp 26-27 | 0.925 | 0.51 |
| mlp 22-27 | 0.791 | 0.20 |
| self_attn 23-27 | 0.559 | 0.07 |
| mlp все блоки | 0.183 | −0.01 |

Вывод: **ни одна политика не проходит пол 0.995** — даже один последний MLP-блок (0.958) и даже «безопасный» хвост. R_f-карта цен Jacobian'ов: MLP-ветки несут основной adjoint-сигнал на ЛЮБОЙ глубине, SA тяжёлый, CA лёгкий (согласуется с каузальной таблицей: CA малозначим и для стиля, и для adjoint). Approximate backward для этой архитектуры закрыт; политик-механизм и oracle (границы-хуки + per-module cos) остаются как измерительный стенд. Память: экономии нет и там — сэкономленные активации веток теряются на хранение их входов для параметрных градиентов.

## Exact execution engine: что принято, что отвергнуто измерениями

Аудит блока (scripts/block_audit.py, D=2048, seq=1024, bf16): fwd SA 36% / MLP 32% / CA 23% / modulation 9%; recompute-overhead whole-block checkpoint = 15.7 ms; SA — 82 kernel launch против 22 у MLP; fused QKV −47% на проекциях.

Принято (всё exact, oracle-пол 0.9998):
1. **Sequential losses** — один autograd-граф в моменте, −1.4 GiB бесплатно.
2. **SAC (`checkpoint_sac: sdpa`)** — MUST_SAVE на `_scaled_dot_product_*` внутри whole-block регионов: −4.5% времени за +0.9 GiB. Структура критична: 28 границ сохранены, политика только внутри блока. `sac_mm` — ловушка (+8.6 GiB, ×6 время).
3. **Fused QKV self-attn** — один GEMM вместо трёх; cat-веса кэшируются (без кэша cat дороже сэкономленного GEMM!); LoRA-дельты — slice-add. −1.2% времени.
4. **Modulation hoist** — 84 AdaLN GEMM на пассаж вынесены из checkpoint-регионов: −0.3% времени, минус мусорный recompute.
Суммарно: 6.80 → 6.35 s/batch (−6.6%), peak 6.85 GiB, градиенты exact.

Отвергнуто измерениями (детали в git-истории):
- гранулярный/групповой чекпоинтинг, selective-режимы: 10.9+ GiB → WDDM-коллапс; whole-block — Pareto-точка;
- gradient transport (detach VJP): каскадная декорреляция adjoint до cos 0.07–0.24;
- вынос инвариантов (x_embedder+t_emb+rope = 2 ms, cross-attn K/V = 3–8 ms из 777): не стоит усложнения;
- **MLP exact VJP** (custom autograd.Function, точные градиенты cos 1.0 plain / 0.9998 lora): **медленнее** vanilla checkpoint (16.8 vs 14.0 ms) — сохранение h/a/x дороже избегнутого W2-replay (2.25 ms). Generic checkpoint снова оптимален.

Урок серии: на этом графе (гигантские spatial-активации, дешёвый recompute, LoRA-мелочь вокруг) универсальный PyTorch + точечные политики бьёт ручные специализации почти везде. Дополнительная скорость живёт только в op-level SAC и группировке мелких op — и уже взята.

## ConvRot-W8A8 квант замороженных оснований (`model.quantization: convrot8`)

Собственная реализация схемы ConvRot (arXiv:2512.03673): регулярный Hadamard-поворот R4-Kronecker (блок 256), симметричный per-channel int8 весов, per-token динамический int8 активаций, GEMM через `torch._int_mm`, аналитический STE-backward. Инъекция после загрузки весов и до inject_lora; экспорт не затрагивается.

Ключевые находки измерениями на GA106:
- **Лейаут B-операнда решает всё**: `_int_mm` выбирает быстрый IMMA-кернел только при column-major B → хранятся две транспонированные копии кодов (`cr_w_fwd/cr_w_bwd`); row-major даёт 0.92x от bf16, col-major — 2.7–3.3x;
- `torch.compile` (triton 3.2, Windows) работает и сжимает цепочку квантования 0.98→0.14 ms; эпилог и предмасштабирование dY компилируются так же;
- масштабы весов сворачиваются в dY **до** квантизации градиента — иначе ось contraction не позволяет корректный пост-эпилог;
- покомпонентный выигрыш растёт с размером GEMM: mlp up 1.56x fwd / 1.35x f+b; мелкие слои (1024-wide) в минус 0.98x/0.73x — квантовать их выгодно только целиком со стеком.

Режимы: `quantize_extent: below_trainable` (блоки строго ниже нижнего обучаемого — чистый конвейер признаков без риска для фиделити адаптера) | `all`. По умолчанию выключено.

### Измеренный вердикт (GA106 / RTX 3060, честные полный шаги @1280²)

| Метрика | float-turbo | qsafe (below_trainable) |
| --- | --- | --- |
| c/оптимайзер-шаг (s40–60) | 28.7–28.8 | 28.0–31.7 → **паритет ±3%** |
| validation @50 preserve | 0.0060 | 0.0099 (в пределах ≤0.01) |
| validation @50 target_drift | 0.0190 | 0.0182 ✓ совпадает |
| anchor_loss @50 | 4.9e-05 | 7.2e-05 ✓ держится |
| веса модели | 2.09B | **1.36B** (−0.73 GB кодов) |

Итог: как **ускорение** ConvRot-int8 на этом железе закрыт (критерий ≥1.2x не достигнут: мелкие слои и двойной проход anchor'а съедают выигрыш mlp-GEMM'ов; заявленные кем-то 1.5x требуют Blackwell-fp4 или глубокой ручной фузии кернелов). Оставлен как **инструмент памяти** (−0.7 GB весов при том же качестве обучения) для тесных конфигураций — включается только явно через конфиг.

### Профайлер шага (`scripts/profile_step.py`) — где живут секунды

`python scripts/profile_step.py --config <yaml> --steps 4` печатает фазовые тайминги (шаг целиком / без якоря), проекции `anchor_every=2|3|4` и топ ядер; JSON сохраняется в `E:\Temp\opencode\profiles`. На Windows kineto не отдаёт CUDA-тайминги (нет CUPTI) — скрипт автоматически переключается на host-inclusive время операций (валидный прокси для синхронного GPU-bound цикла).

Замеры на turbo-конфиге (1280², accum×2):

| Факт | Число | Следствие |
| --- | --- | --- |
| **Якорь стоит 18 с из 31.6** (второй полный train-проход: no_grad-base + STE-градиентный) | 57% шага | `anchor_every=4` → **~18 с/шаг ≈ 1.76x** бесплатно; every=2 → 1.40x |
| **GEMM — лишь ~10% времени шага** | потолок int8 = **1.04x** | Математически закрывает ConvRot-скорость на этом железе |
| `aten::item`/`is_nonzero` = 18% хост-времени | синхронизирующие `.item()` | per-шаг `isfinite`-проверки теперь под флагом `train.debug_sync_checks` (по умолчанию выкл — пайплайн не синхронизируется) |

Микробенч ai-toolkit (`scripts/test_quantizations.py`), откуда берутся таблицы вида «convrot8 train 1.53x», меряет **одиночный слой FLUX-класса** (4096×3072→12288) — без 28 блоков, якоря, SAC и лоадера. Наши слои мельче (2048/8192), поэтому переносной коэффициент ещё ниже.

### Подтверждённый рецепт: `anchor_every=4` — производственный турбо-режим

Прогон `red-lili-turbo-ae4` (lr 2e-4, якорь 8.0 раз в 4 шага, 1280², accum×2):

| шаг | c/шаг | preserve | target_drift |
| --- | --- | --- | --- |
| 50 | 17.9 | 0.0133 (прогрев) | 0.0223 |
| 100 | 18.4 | 0.0099 | 0.0295 |
| 150 | ~18 | **0.0098** плато | 0.0310 |

**~18 с/шаг против 28.8 у ежешагового якоря = 1.6–1.85x** при сохранении здоровья (порог ≤0.01 соблюдён со 100-го шага, target/preserve = 3.2, якорные лоссы без разгона).

Рекомендации по каденции якоря:
- `anchor_every: 4` — **дефолт** для боевых прогонов (1.7–1.85x);
- `anchor_every: 1` — отладка и прогоны с пристальным контролем утечки;
- `anchor_every: 2` — компромисс, если на длинных прогонах preserve поплывёт, либо при lr выше 2e-4;
- всплеск preserve на первых ~50 шагах — норма прогрева (нулевой LoRA + высокий LR), не триггерит остановку сама по себе.

## Гетерогенные ранги (rank_overrides)

Каузальный анализ (drop-one/keep-only) + энергетика спектров задают ранг per-module вместо униформного:

```yaml
scope:
  rank: 8
  alpha: 8
  rank_overrides:
    modulation: {rank: 2}        # reff 1.7-2.0 стабильно
    mlp: {rank: 6}
    self_attn: {rank: 0}         # rank 0 = модуль не инжектится
    cross_attn.v_proj: {rank: 4}
    blocks.3.mlp.layer2: {alpha: 6}   # точечный fqn; longest-pattern-wins
```

`resolve_rank` выбирает самый длинный совпавший паттерн; alpha по умолчанию = рангу; kohya-экспорт несёт per-module alpha (ComfyUI совместим); `deltas_from_lora_state` читает per-module alpha из стейта. Пайплайн: causal-metrics JSON → `E:\Temp\opencode\rank_spec_gen.py` → yaml-сниппет в конфиг компактного ретрейна.

## Production parity: обязательная проверка перед тренингом

Порт DiT обязан быть численно идентичен продакшену. Две ошибки в rope стоили нескольких разрушенных ранов (лосс падал, внутренняя validation выглядела здоровой, но ΔW не переносился в ComfyUI и уничтожал генерацию):

1. **NTK-факторы rope**: продакшен (comfy `model_detection` для anima) ставит `rope_h/w_extrapolation_ratio = 4.0`, `t = 1.0`. С дефолтными 1.0 пространственные частоты в ~4.3 раза выше.
2. **Парование rope**: чекпоинт обучен под split-half пары `(i, i + D/2)` (конвенция comfy-кернела `rms_rope_split_half`), а не смежные пары `(2i, 2i+1)`.

Метод проверки (поблочный бисект против живого ComfyUI): один euler-шаг из фиксированного шума при σ=1.0, cfg=1, `LoadLatent → KSampler(add_noise=disable, steps=1) → SaveLatent` против нашего `forward_latent` на тех же входах; далее хуки на все 28 блоков с обеих сторон — первый расходящийся блок локализует баг. Критерий: rel L2 финальной velocity ≲1% (bf16-шум). После фиксов: block_00 0.48 → все блоки ≤0.022, v 0.005.

## Геометрические маркеры здоровья тренинга

Траектория ΔW по чекпоинтам отличает здоровый ран от разрушительного раньше визуальных артефактов. Эталон — траектория успешного внешнего рана (tsuniya, 957 шагов, все блоки) против нашего разрушенного первого прогона:

| сигнатура | здоровый ран | разрушительный |
| --- | --- | --- |
| effective rank | стабилен (4.0–4.1 всё обучение) | коллапсирует (2.16→1.81) |
| energy@1 | стабильна (0.57–0.59) | растёт (0.82→0.87) |
| доли компонентов | квазистационарны | переворот (modulation 43→27%, self_attn 17→45%) |
| ‖ΔW‖(t) | линейный рост или выход на плато | рост без насыщения до самого конца |

Здоровый style-ран (kor-lili v2, 350 шагов, blocks 14–27 rank 8): eff rank 3.1 константа, energy@1/2/4 = 0.71/0.82/0.92, modulation 42% / mlp 25% / self_attn 17% / cross_attn 15% без дрейфа, validation drift выходит на плато 0.044 к шагу 200. Мониторинг: `analyze` по чекпоинтам после рана; в будущем — online-метрики eff_rank(t) в events.jsonl.

## Границы v0.1

- Нет связи с L2P.
- B1/attention mechanistic probe suite пока не встроен; B1 остаётся внешней диагностикой.
- Text encoder и LLMAdapter намеренно не обучаются.
- Нет CLIP/style/identity/anatomy perceptual losses и автоматической генерации preview grid.
- Нет multi-GPU/distributed train.
- Train требует CUDA; `float16` отклоняется, используйте `bfloat16`.
- Actual image quality надо реплицировать на нескольких prompts, seeds и sessions, а не на одном красивом примере.

## Проверка кода

```powershell
python -m pytest
python -m compileall -q anima_trainer scripts
python -m anima_trainer --help
```

