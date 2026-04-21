# Deployment

Основная инструкция по запуску находится в [../README.md](../README.md).

## Runtime Contract

- `docker/submission.Dockerfile` собирает только code/runtime слой. Каталог
  `data/` не копируется в image.
- Веса не вшиваются в image. Их докачивает `./run.sh` рантаймом в
  host-mounted `data/models/...` через [`artifacts.toml`](./artifacts.toml).
- Build ставит только зависимости выбранного submit path:
  - `w2v-trt`: `submit_common` + `submit_w2v_trt`
  - `campp-pt`: `submit_common` + точный `torch`/`torchaudio` runtime для CAM++

## Manual Docker Smoke Check

Дефолтный organizer-facing путь `w2v-trt`:

```bash
docker build -f deployment/docker/submission.Dockerfile \
  -t kryptonite-submit:w2v-trt .

docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  kryptonite-submit:w2v-trt \
  ./run.sh --container-only \
    --test-csv "data/Для участников/test_public.csv" \
    --data-root "data/Для участников" \
    --dry-run
```

Альтернативный ручной прогон `campp-pt`:

```bash
docker build --build-arg SUBMIT_MODEL=campp-pt \
  -f deployment/docker/submission.Dockerfile \
  -t kryptonite-submit:campp-pt .

docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  kryptonite-submit:campp-pt \
  ./run.sh --container-only \
    --model campp-pt \
    --test-csv "data/Для участников/test_public.csv" \
    --data-root "data/Для участников" \
    --dry-run
```

Batch size и ширина ответа для обеих моделей переопределяются через
organizer-facing entrypoint, без редактирования YAML-конфигов:

```bash
CAMPP_BATCH_SIZE=384 ./run.sh --model campp-pt --test-csv "data/Для участников/test_public.csv" --data-root "data/Для участников"
W2V_BATCH_SIZE=1536 ./run.sh --test-csv "data/Для участников/test_public.csv" --data-root "data/Для участников"
TOP_K=100 ./run.sh --test-csv "data/Для участников/test_public.csv" --data-root "data/Для участников"
```

`TOP_K` соответствует внутреннему `--output-top-k` / `output_top_k` в tail-скриптах.
По умолчанию используется `10` соседей.

Первый cold build может быть заметно тяжёлым из-за базового образа
`nvcr.io/nvidia/pytorch`, но это не связано с весами модели: веса остаются
во внешнем `data/models/...` и materialize-ятся только рантаймом.
