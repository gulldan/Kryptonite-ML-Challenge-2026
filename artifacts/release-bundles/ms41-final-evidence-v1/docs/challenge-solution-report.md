# Отчёт по решению

Этот документ сделан как самодостаточная версия для жюри: по нему можно понять,
почему финальным кандидатом выбран именно `MS41`, не читая весь `trails/`. Файл
[challenge-experiment-history.md](./challenge-experiment-history.md) остаётся полным
лабораторным журналом, а [trails/](./trails/) содержат команды, логи и расширенные записи.

## 1. Что решаем

Нужно для каждого аудиофайла из тестовой выборки найти 10 самых похожих аудиофайлов
из той же выборки. Качество считается по `Precision@10`: чем больше среди 10 соседей
файлов того же диктора, тем выше оценка.

Итоговый рейтинг считается по закрытой private-выборке. Открытый public leaderboard
использовался только как внешний индикатор, потому что public-лейблы скрыты и локально
public score пересчитать нельзя.

## 2. Текущий финальный кандидат

- run id: `MS41_ms32_classaware_c4_weak_20260415T0530Z`
- public LB: `0.7473`
- recorded submission path in the release workspace:
  `artifacts/submissions/MS41_ms32_classaware_c4_weak_20260415T0530Z_submission.csv`
- SHA-256:
  `8b58013c3a710ef7e4c9f2fc5466ee9b2918d2ee271b5eaaa095b4976e194e84`

Почему именно он:

- даёт лучший закрытый submitted result среди завершённых веток;
- улучшает `MS32` на `+0.0094` public при контролируемом изменении соседей;
- сильнее `MS38` weight soup (`0.7396`) и `MS40` row-wise router (`0.7441`);
- использует слабую class-aware поправку внутри уже безопасного `MS32` top-200 graph,
  а не жёсткое class-first назначение.

## 3. Коротко про финальный пайплайн

1. Читает CSV тестовой выборки и аудиофайлы.
2. Приводит звук к общему формату: `16 kHz`, mono.
3. Строит аудиопризнаки официальным CAM++/3D-Speaker frontend.
4. Делит длинные записи на несколько 6-секундных сегментов и усредняет результат.
5. Получает embedding для каждого файла через CAM++ encoder.
6. Ищет соседей по косинусной близости.
7. Применяет графовую постобработку C4: взаимные соседи, локальная плотность,
   контролируемое label propagation.
8. Добавляет слабую class-aware поправку только внутри уже найденных `top-200`
   соседей: posterior overlap `0.03`, same-top1 `0.01`, same-query-top3 `0.005`.
9. Записывает `submission.csv` и проверяет его валидатором.

Главная идея: сначала строится устойчивый voice graph, затем графовая постобработка
убирает часть случайных связей, а classifier posterior используется только как слабый
edge bonus, не как отдельный источник жёстких соседей.

## 4. Наблюдения по данным и как они изменили решение

| Наблюдение | Что увидели | Что изменили |
| --- | --- | --- |
| Локальная validation была слишком оптимистичной | После исправления baseline speaker-disjoint val давала `P@10 = 0.9174`, а public был только `0.1024`. | Перестали трактовать локальную метрику как самостоятельное доказательство и ввели отдельный protocol: local proxy для диагностики, public LB для финального выбора. |
| Public-домен отличался по тишине, длинам и каналам | EDA показала сдвиг по silent edges, duration, narrowband/channel distortions и peak limiting. | Зафиксировали `6s x 3` segment-mean inference, repeat padding, trim policy и channel/noise/reverb/codec/silence augmentations. |
| Локальный CAM++ frontend не воспроизводил сильный baseline | Конвертация весов без официального frontend давала overlap с сильным ModelScope baseline около `2.5/10`, а официальный путь воспроизводил `9.961/10`. | Зафиксировали официальный CAM++/3D-Speaker frontend как источник истины для сильной ветки. |
| В графе появлялись чрезмерно популярные соседи | Некоторые файлы слишком часто попадали в top-k. | Добавили reciprocal support, density penalty и C4 graph tail. |
| Жёсткое назначение класса было рискованным | Class-first и агрессивные logits-пробы резко повышали hubness. | Оставили только слабую class-aware поправку внутри безопасного graph cache. |

## 5. Единый протокол сравнения

Финальные решения принимались по трём уровням evidence:

| Уровень | Что фиксирует | Как использовался | Чего не доказывает |
| --- | --- | --- | --- |
| Format validator | правильность `submission.csv`: число строк, порядок `filepath`, отсутствие self-match и дубликатов | обязательный gate перед любым upload | не говорит о качестве retrieval |
| Локальный proxy | стандартный для финальной CAM++ ветки tuple: `top10_mean_score_mean`, `Gini@10`, `max in-degree`, иногда overlap vs parent | сравнение близких вариантов внутри одной семьи, поиск hubness и нестабильности | не выбирает победителя сам по себе |
| Public LB | скрытая внешняя оценка на public split | окончательное принятие или отклонение submitted run | не гарантирует private score |

Чтобы не смешивать несопоставимые сигналы, в финальной матрице ниже используются только:

- `Public LB`;
- одинаковый graph-proxy tuple там, где он реально посчитан;
- `n/a`, если для старой ветки этот proxy не был доступен.

Train loss, train accuracy и статусы запусков использовались только как индикаторы того,
что обучение/инференс завершились корректно, но не как доказательство улучшения retrieval.

### Где local proxy ломался

Это было не теоретическое опасение, а повторяющийся факт:

| Ветка | Локальный сигнал | Public LB | Вывод |
| --- | --- | --- | --- |
| `baseline_fixed_participants` | speaker-disjoint val `precision@10 = 0.9174` | `0.1024` | baseline-like val не похожа на hidden public/private distribution |
| `MS34` | C4 proxy `0.7225`, Gini `0.3333`, max in-degree `54` | `0.6791` | строгие pseudo-label refinement могут улучшать local graph и ухудшать hidden transfer |
| `MS35` | C4 proxy `0.7401`, Gini `0.3298`, max in-degree `55` | `0.6884` | weighted pseudo labels не спасают от этого разрыва |
| `MS36b` | лучший local proxy в этой семье: `0.7735`, Gini `0.3220`, max in-degree `42` | `0.6906` | даже очень красивый local graph после `MS32` не гарантирует public gain |

Главный вывод: после `MS32` решение нельзя было выбирать по local proxy alone. Поэтому
финальный выбор делался только среди закрытых submitted public runs, а local proxy
использовался как риск-сигнал, а не как hidden-score substitute.

## 6. Финальная матрица закрытых экспериментов

В эту матрицу включены только завершённые ветки, которые реально участвовали в
финальном выборе. Pending, running и local-only диагностические ветки вынесены из
аргумента выбора и остаются только в полном журнале.

Локальный proxy ниже всегда читается как:
`C4 top10_mean_score_mean / Gini@10 / max in-degree`.

| Run | Родитель | Главное изменение | Local proxy | Public LB | Решение |
| --- | --- | --- | --- | --- | --- |
| `MS1` | none | Готовый ModelScope CAM++ pretrained с официальным frontend | `n/a / 0.4917 / 214` | `0.5695` | Принят как первая по-настоящему сильная safe branch и как доказательство, что официальный frontend критичен. |
| [`MS30`](./trails/2026-04-13-ms30-official-cam-plus-plus-pretrained-low-lr-adaptation-launch.md) | `MS1` | Low-LR supervised adaptation на train участников | `0.6261 / 0.3363 / 68` | `0.6953` | Принят: supervised adaptation даёт большой переносимый прирост. |
| [`MS31`](./trails/2026-04-13-ms31-official-cam-plus-plus-voxblink2-like-augmentation-launch.md) | `MS1` | Та же ветка, но с более сильными VoxBlink2-like augmentations | `0.6207 / 0.3380 / 91` | `0.7018` | Принят: public лучше `MS30`, значит augmentation переносится, несмотря на чуть худший local proxy. |
| [`MS32`](./trails/2026-04-14-ms32-filtered-pseudo-label-self-training-from-ms31-launch.md) | `MS31` | Filtered pseudo-label self-training с кластерами `[8,80]` | `0.6564 / 0.3326 / 57` | `0.7379` | Принят как новая safe branch; это главный устойчивый скачок после supervised adaptation. |
| [`MS33c`](./trails/2026-04-14-ms33-tail-ablations-on-ms32.md) | `MS32` | Убрали hard label propagation, оставили reciprocal/local-density only | `0.6635 / 0.3096 / 54` | `0.6980` | Отклонён: локально hubness лучше, но hidden score падает на `-0.0399` против `MS32`. |
| [`MS34`](./trails/2026-04-14-ms34-strict-consensus-pseudo-plus-clean-finish-launch.md) | `MS32` | Strict-consensus pseudo + short clean finish | `0.7225 / 0.3333 / 54` | `0.6791` | Отклонён: самый наглядный пример local/public divergence. |
| [`MS35`](./trails/2026-04-14-ms35-gll-style-weighted-pseudo-labels-from-ms32-launch.md) | `MS32` | Weighted/GLL-style pseudo continuation | `0.7401 / 0.3298 / 55` | `0.6884` | Отклонён: локально красиво, скрытое качество хуже safe branch. |
| [`MS36b`](./trails/2026-04-14-ms36b-multi-teacher-soft-pseudo-fast-retry.md) | `MS32` | Multi-teacher soft pseudo labels | `0.7735 / 0.3220 / 42` | `0.6906` | Отклонён: лучший local proxy в семье не переносится на hidden score. |
| [`MS38`](./trails/2026-04-15-ms38-official-cam-plus-plus-weight-space-soup-launch.md) | `MS32` | Near-`MS32` weight-space interpolation/soup | `0.6535 / 0.3330 / 63` | `0.7396` | Принят как реальный, но маленький gain над `MS32` (`+0.0017`). |
| [`MS40`](./trails/2026-04-15-ms40-rowwise-tail-router.md) | `MS32/MS38/MS41` | Row-wise router между несколькими хвостами | `n/a / 0.3473 / 108` | `0.7441` | Отклонён как финальный: лучше `MS32`, но хуже `MS41`, при этом max in-degree заметно хуже. |
| [`MS41`](./trails/2026-04-15-ms41-ms32-weak-class-aware-c4-probe.md) | `MS32` | Weak class-aware edge bonus внутри безопасного `top-200`, затем обычный C4 | `0.6859 / 0.3437 / 56` | `0.7473` | Финальный выбор: лучший public score, контролируемая perturbation profile, без перехода в hard class assignment. |

Что доказывает эта матрица:

- главный надёжный прирост пришёл от последовательности `MS1 -> MS30/MS31 -> MS32`;
- после `MS32` почти все более агрессивные pseudo/refinement ветки были ложными
  локальными улучшениями;
- переносимыми оказались только маленькие, topology-preserving изменения:
  near-`MS32` soup (`MS38`) и weak class-aware bonus (`MS41`);
- более сложный router (`MS40`) не оправдал свою сложность и уступил простому `MS41`.

## 7. Почему выбран именно MS41

### Сравнение с ближайшими альтернативами

| Кандидат | Public LB | Что показывает сравнение с MS41 |
| --- | --- | --- |
| `MS32` | `0.7379` | `MS41` улучшает safe branch на `+0.0094` при умеренном вмешательстве: pre-C4 top1 changed share всего `0.0836`, overlap после C4 `8.63/10`, max in-degree `56` против `57`. |
| `MS38` | `0.7396` | Weight soup подтверждает, что около `MS32` ещё есть скрытый резерв, но слабая class-aware поправка даёт больший переносимый выигрыш. |
| `MS40` | `0.7441` | Row-wise router сильнее `MS32/MS38`, но проигрывает простому глобальному правилу `MS41` и ухудшает hubness (`max in-degree 108`). |
| `MS33c` | `0.6980` | Полное выключение hard label propagation оказалось ошибкой: MS41 удерживает базовый graph tail и лишь слегка корректирует edge weights. |

### Почему class-aware поправка слабая, а не жёсткая

Финальная слабая схема не возникла из воздуха. До неё были отдельные контроли на P1-ветке:

| Вариант | Как использовался classifier signal | Локальный итог | Вывод |
| --- | --- | --- | --- |
| [`H1`](./trails/2026-04-12-local-p1-classifier-first-diagnostic.md) | Hard class-first buckets | `Gini@10 = 0.5953`, max in-degree `474`, overlap с safe P1 только `2.35/10` | Жёсткое class-first назначение явно небезопасно. |
| [`H2`](./trails/2026-04-12-local-conservative-logits-and-class-aware-graph-diagnostics.md) | Conservative logits rerank | `Gini@10 = 0.5362`, overlap `4.88/10` | Всё ещё слишком агрессивно. |
| [`H3`](./trails/2026-04-12-local-conservative-logits-and-class-aware-graph-diagnostics.md) | Strong class-aware graph bonus `0.08/0.02/0.01` | `Gini@10 = 0.3830`, max in-degree `97`, overlap `6.96/10` | Уже лучше, но риск hubness ещё заметен. |
| [`H3b`](./trails/2026-04-12-local-conservative-logits-and-class-aware-graph-diagnostics.md) | Weak class-aware graph bonus `0.03/0.01/0.005` | `Gini@10 = 0.3619`, max in-degree `71`, overlap `7.89/10` | Лучший локальный режим для переноса идеи на более сильную ветку. |
| [`MS41`](./trails/2026-04-15-ms41-ms32-weak-class-aware-c4-probe.md) | Тот же слабый режим, но на safe branch `MS32` | `Gini@10 = 0.3437`, max in-degree `56`, public `0.7473` | Подтверждённый финальный вариант. |

Честное ограничение: полного weight sweep именно на `MS32`-ветке не было. На финальной
семье был проверен один слабый режим, потому что hard/strong варианты уже дали
отрицательные контроли на более ранней P1-ветке. Это не скрытая дыра, а сознательный
tradeoff по submission budget: сначала доказать знак эффекта безопасной слабой поправки,
а не перебирать агрессивные веса, которые уже выглядели рискованно.

## 8. Evidence bundle и воспроизводимость

Ниже перечислены доказательства, которыми подтверждён финальный выбор.

Repo-versioned bundle manifest:

- `artifacts/release-bundles/ms41-final-evidence-v1/submission_bundle.md`

### Версионируемые доказательства в git

| Что | Где | Зачем |
| --- | --- | --- |
| Финальная аргументация | этот файл | Самодостаточное объяснение, почему выбран `MS41`. |
| Полный submitted ledger | [challenge-experiment-history.md](./challenge-experiment-history.md) | Точная public chronology, delta и решения по закрытым веткам. |
| Подробный trail для финального run | [trails/2026-04-15-ms41-ms32-weak-class-aware-c4-probe.md](./trails/2026-04-15-ms41-ms32-weak-class-aware-c4-probe.md) | Команды, веса бонусов, completion time, paths, overlap diagnostics. |
| README chart | [assets/public-lb-score.svg](./assets/public-lb-score.svg) | Визуальная траектория submitted public runs. |
| Точный release path | [release-runbook.md](./release-runbook.md) | Пошаговый запуск, валидация, SHA и speed summary. |

### Recorded runtime artifacts для `MS41`

Ниже перечислены recorded runtime paths. Они зафиксированы в trail/history и в bundle
manifest, но не все из них материализованы в текущем локальном workspace.

| Артефакт | Path |
| --- | --- |
| Submission CSV | `artifacts/submissions/MS41_ms32_classaware_c4_weak_20260415T0530Z_submission.csv` |
| Summary JSON | `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/MS41_ms32_classaware_c4_weak_20260415T0530Z_summary.json` |
| Overlap vs `MS32` | `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/MS41_ms32_classaware_c4_weak_20260415T0530Z_vs_MS32_overlap.json` |
| Class posterior cache | `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/class_indices_*.npy` и `class_probs_*.npy` |
| Remote log | `artifacts/logs/MS41_ms32_classaware_c4_weak_20260415T0530Z.log` |
| Completed-at timestamp | `2026-04-15T05:32:11Z` |

### Release-time verification outputs

Финальный runbook дополнительно пишет:

- `artifacts/release/submission_validation.json` или `$OUT/submission_validation.json`;
- `artifacts/release/submission.sha256` или `$OUT/submission.sha256`;
- `artifacts/release/speed_summary.txt` или `$OUT/speed_summary.txt`.

Это не заменяет public leaderboard, но делает пакет проверяемым: есть exact command,
validator JSON, SHA-256 и speed summary для полного пересчёта.

Важно: крупные runtime artifacts в `artifacts/` не хранятся в git. Поэтому в репозиторий
вынесены path map, SHA-256, run ids, trail-файлы и воспроизводимые команды, а не сами
веса и кеши.

## 9. Ограничения и остаточный риск

- Public score не гарантирует private score. Это главный внешний риск задачи.
- Local proxy оказался ненадёжным именно в поздних `MS32`-производных ветках. Поэтому
  даже красивый graph diagnostic не трактуется как доказательство без public submission.
- Полная независимая перепроверка требует локальных `datasets/` и runtime artifacts из
  `artifacts/`, потому что они сознательно не версионируются в git.
- На финальной `MS32`-семье не делался полный sweep class-aware weights; подтверждён только
  слабый режим, выбранный после более ранних отрицательных контролей.
- Pending/running строки в полном лабораторном журнале существуют, но они не использовались
  для выбора финального кандидата и не входят в матрицу закрытых доказательств выше.

## 10. Что бы делали дальше

1. Построили бы более жёсткий local validation protocol, лучше похожий на hidden
   public/private split, чтобы меньше зависеть от leaderboard.
2. Доделали бы отдельный ablation sweep around `MS41`: no bonus, overlap-only,
   same-top1-only и несколько weak weight settings уже на `MS32`-ветке.
3. Подготовили бы полноценный release bundle для жюри: final submission, validator JSON,
   summary JSON, SHA-256, logs и checkpoints в одной поставке.
4. Проверили бы external supervised adaptation только после полной сборки внешних данных,
   а не в режиме незавершённых side branches.
5. Ускорили бы первый полный проход по аудио: после TensorRT bottleneck остался не в
   encoder, а в official audio frontend/fbank.
