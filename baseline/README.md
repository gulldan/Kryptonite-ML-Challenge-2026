# Дататон Криптонит.Тембр

## Общее описание задачи

Целью дататона является построение модели распознавания диктора по голосу, устойчивой к различным искажениям, которые встречаются в реальном мире при записывании аудиозаписей (окружающие шумы, реверберации, аудиокодеки, искажения каналов связи). 

## Данные

Одноканальные аудиозаписи в формате ".flac" с частотой дискретизации 16000 и глубиной 16 бит.

train.csv - таблицы с разметкой тренировочных данных.
Содержит следующие поля: 
```
speaker_id - метка объекта, уникальный идентификатор диктора (у разных людей разные id).
filepath - расположение аудиофайла относительно директории data (см. ниже структуру датасета).
```

test_public.csv - таблицы с разметкой тестовых данных
Содержит следующие поля: 
```
filepath - расположение аудиофайла относительно директории data (см. ниже структуру датасета).
```


Данные расположены в архиве data.tar.gz, который  имеет следующую структуру:
```
data
├── train.csv
├── train  # данные для обучения модели
│   ├── OhbVrpoiVg    # speaker_id - Идентификатор диктора
│   │   ├── 00001.flac
        ...
│   │   └── 0000n.flac
│   ├── RV5IfLBcbf
│   │   ├── 00001.flac
|   |   ...
│   │   └── 0000n.flac
...
├── test_public.csv
└── test_public   # данные для тестирования
    ├── 000000.flac
    ├── 000001.flac
    ├── 000002.flac
    ...
    └── 134696.flac
```

Данный архив разбит на несколько частей, для того, чтобы распаковать архив, используйте команду:
```
cat data.tar.gz.part-* | tar -xzf -
```

## Бейзлайн решение

### Окружение
Тестирование работы скриптов проводилось в `Python 3.13.12`.
  
Создание окружения:

```bash
VENV_DIR="../../venvs/datathon_kryptonite"
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install -r requirements.txt
```


### Запуск скриптов

#### Получение baseline-модели
Для обучения бейзлайн модели используется скрипт `train.py`, для конфигурирования которого используется конфиг в формате ".json". Результатом работы скрипта является — чекпоинт (в формате ".pt") с максимальным значением  метрики Precision@10.

Примеры запуска:

```bash
python train.py --config configs/baseline.json
```

Заметки по конфигурации `train.py`:
- `configs/baseline.json` содержит основные параметры аудио и обучения (sample_rate, длительность чанков, Mel‑параметры, batch_size, epochs, device и т. д.).
- `train_ratio` задаёт долю train-speakers; validation строится speaker-disjoint split-ом.
- `min_val_utts` ограничивает validation speakers теми, у кого есть минимум `K + 1` записей для P@K.
- `seed` фиксирует split, torch/numpy/python random и DataLoader workers.
- `device: "auto"` — выбирает CUDA при наличии, иначе CPU.
- Train использует random crop; validation/test используют детерминированный center crop.

#### Тестирование
Для оценки качества полученной модели используется скрипт `test_model.py`.

Описание флагов `test_model.py`:
- `--model_path`: путь к `.pt` чекпоинту модели.
- `--csv`: путь к CSV с колонками `filepath`, `speaker_id`.
- `--batch_size`: размер батча при инференсе.
- `--num_workers`: число воркеров DataLoader.
- `--device`: `auto` выбирает `cuda`, если доступна, иначе `cpu`; можно явно указать `cpu`/`cuda`.
- `--sample_rate`: частота дискретизации входного аудио.
- `--chunk_seconds`: длина аудиофрагмента для валидации.
- `--n_mels`: число mel-bins.
- `--n_fft`: окно FFT.
- `--hop_length`: шаг STFT.
- `--ks`: список K через запятую для Precision@K.

```
python test_model.py \
  --model_path experiments/baseline/model.pt \
  --csv experiments/baseline/val_split.csv \
  --batch_size 32 --device auto --sample_rate 16000 \
  --chunk_seconds 6.0 --n_mels 80 --n_fft 400 --hop_length 160 \
  --ks 10
```

#### Формирование сабмита

Для формирования сабмита используется скрипт `convert_to_onnx.py` и `inference_onnx.py`.

Назначение `convert_to_onnx.py`: экспортировать обученный `.pt` чекпоинт в ONNX-формат для последующего инференса и формирования сабмита.

Флаги `convert_to_onnx.py`:
- `--config`: путь к JSON-конфигу обучения; используется для чтения аудио-параметров и `save_path`, если `--pt` не задан.
- `--pt`: путь к `.pt` чекпоинту модели; если не указан, берётся `save_path` из конфига.
- `--out`: путь для сохранения `.onnx` файла; по умолчанию используется путь чекпоинта с заменой `.pt` на `.onnx`.
- `--chunk_seconds`: длина dummy-входа для ONNX export; по умолчанию берётся из конфига.
- `--opset`: ONNX opset version; по умолчанию `20`.
- `--include_logits`: дополнительно экспортировать classifier logits. По умолчанию экспортируется только embedding output.

```
python convert_to_onnx.py \
  --config configs/baseline.json \
  --pt experiments/baseline/model.pt
```

Назначение `inference_onnx.py`: извлечь эмбеддинги для набора данных по CSV и сохранить их в `emb.npy` (а также `labels.npy` для соответствующих меток), затем из `emb.npy` получить `submission.csv` (топ-K индексов ближайших соседей к каждой записи).

Флаги `inference_onnx.py`:
- `--onnx_path`: путь к `.onnx` модели.
- `--csv`: путь к CSV с колонками `filepath`, `speaker_id`.
- `--output_emb`: путь для сохранения эмбеддингов (shape: N × D).
- `--output_labels`: путь для сохранения меток (shape: N, int).
- `--output_indices`: Путь для сохранения индексов (shape: N, 2)
- `--batch_size`: размер батча при инференсе.
- `--num_workers`: число воркеров DataLoader.
- `--device`: `auto` выбирает `cuda`, если доступна, иначе `cpu`.
- `--sample_rate`: частота дискретизации входного аудио.
- `--chunk_seconds`: длина аудиофрагмента.
- `--num_crops`: число детерминированных eval-crops на файл. `1` означает center crop, `5` — 5-crop mean.
- `--topk`: размер выдачи для тестового поиска в FAISS (self‑match исключается).
- `--filepath_col`: имя колонки с путями к аудиофайлам; по умолчанию - filepath.
- `--speaker_id_col`: имя колонки с идентификатором диктора; по умолчанию - speaker_id.
- `--data_base_dir`: базовая директория для относительных путей; по умолчанию используется директория CSV.

```
python inference_onnx.py \
  --onnx_path experiments/baseline/model.onnx \
  --csv data/test_public.csv \
  --output_emb experiments/baseline/emb.npy \
  --output_labels experiments/baseline/labels.npy \
  --output_indices experiments/baseline/submission.csv \
  --batch_size 32 --num_workers 4 \
  --device auto --sample_rate 16000 --chunk_seconds 6.0 --num_crops 1 --topk 10
```

### Структура labels.npy

Файл `labels.npy` представляет собой одномерный массив NumPy длины `N`, где `N` - число строк во входном CSV и в `submission.csv`.
Каждый элемент массива содержит целочисленную метку диктора для соответствующей записи. Порядок элементов в `labels.npy` должен совпадать с порядком строк в `test_public.csv`.

Пример создания `labels.npy` с случайными метками:

```python
import numpy as np

N = 134697
labels = np.random.randint(0, 100, size=N, dtype=np.int64)
np.save("labels.npy", labels)
```

### Структура сабмита

Для тестирования своего решения, вам необходимо загрузить на платформу файл "submission.csv" - это файл формата ".csv".
Для каждой тестовой записи из `test_public.csv` (порядок записей в submission.csv должен быть сохранен) содержатся топ-10 ближайших соседей без дублей.

В "submission.csv" есть 2 колонки:
- `filepath` - содержит наименования тестовых записей в порядке тестового .csv
- `neighbours` - содержит топ-10 ближайших соседей записи из колонки "filepath"

Пример:
```python
import pandas as pd

N = 134697

filepaths = [f"test_public/{i:06d}.flac" for i in range(N)]
neighbours = [",".join(str((i + k) % N) for k in range(1, 11)) for i in range(N)]

df = pd.DataFrame({"filepath": filepaths, "neighbours": neighbours})
df.to_csv("submission.csv", index=False)
```

## Целевая метрика

Целевая метрика - [Precision@K](https://torchmetrics.readthedocs.io/en/v0.8.2/retrieval/precision.html). Для публичного теста K = 10. На приватном тесте K >= 10. Гарантируется, что для каждого диктора  количество записей как миниум K + 1. 

"Расстояние" между эмбеддингами — [косинусное сходство](https://ru.wikipedia.org/wiki/%D0%92%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BD%D0%B0%D1%8F_%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C) между эмбеддингами.


Назначение `calc_metrics.py` считать Precision@K из сохранённых индексов `submission.csv` и меток `labels.npy`.
Флаги:
- `--indices`: путь к `submission.csv`.
- `--labels`: путь к `labels.npy`.
- `--k`: значение K (по умолчанию 10).
- `--template_csv`: опциональный CSV-шаблон; если указан, `filepath` и порядок строк в submission должны совпасть точно.

Пример запуска:
```bash
python calc_metrics.py \
  --indices experiments/baseline/submission.csv \
  --labels experiments/baseline/labels.npy \
  --template_csv data/test_public.csv \
  --k 10
```

## Ссылки

- [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143) 
- [ASTP](https://arxiv.org/pdf/2307.09856)
- [Криптонит официальный сайт](https://kryptonite.ru/) 

## Заметки

Если при использовании скрипта `inference_onnx.py` возникает ошибка:

`Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn_adv.so.9: cannot open shared object file: No such file or directory`.

Попробуйте выполнить следующую команду в командной строке:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0] + '/lib')")
```
