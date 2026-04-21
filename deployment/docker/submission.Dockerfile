ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7
ARG SUBMIT_MODEL=w2v-trt
ARG CAMPP_TORCH_VERSION=2.11.0+cu128
ARG CAMPP_TORCH_INDEX=https://download.pytorch.org/whl/cu128

FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE} AS runtime

ARG SUBMIT_MODEL
ARG CAMPP_TORCH_VERSION
ARG CAMPP_TORCH_INDEX

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/kryptonite-venv \
    VIRTUAL_ENV=/opt/kryptonite-venv \
    PATH="/opt/kryptonite-venv/bin:${PATH}" \
    KRYPTONITE_ARTIFACTS_ROOT=/opt/kryptonite-artifacts \
    PYTHONPATH="/workspace/src:/workspace/code/campp/ms42_release:/workspace" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    USE_UV_RUN=0 \
    SUBMIT_MODEL="${SUBMIT_MODEL}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libsndfile1 \
        zstd \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock
COPY src ./src
COPY code/campp ./code/campp
COPY research/scripts/run_teacher_peft_c4_tail.py ./research/scripts/run_teacher_peft_c4_tail.py
COPY utils ./utils
COPY deployment/artifacts.toml ./deployment/artifacts.toml
COPY deployment/docker/.submit-artifacts/current/ "${KRYPTONITE_ARTIFACTS_ROOT}/"
COPY run.sh ./run.sh

# Install only the dependency surface needed for the selected submit path.
RUN uv venv --system-site-packages "${UV_PROJECT_ENVIRONMENT}" --python "$(command -v python3)" \
    && case "${SUBMIT_MODEL}" in \
         w2v-trt) \
           "${UV_PROJECT_ENVIRONMENT}/bin/python" -m pip install --no-cache-dir --disable-pip-version-check \
             "numpy>=1.26,<2" \
             "pandas>=2.2.0" \
             "polars>=1.39.3" \
             "pyarrow>=16.1.0" \
             "pyyaml>=6.0.1" \
             "requests>=2.32.0" \
             "scipy>=1.13.0" \
             "soundfile>=0.13.1" \
             "soxr>=1.0.0" \
             "peft>=0.18.1" \
             "transformers>=5.4.0" ;; \
         campp-pt) \
           uv sync --frozen --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
             --no-install-project --no-default-groups \
             --only-group submit_common \
           && uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
             --index-url "${CAMPP_TORCH_INDEX}" \
             "torch==${CAMPP_TORCH_VERSION}" \
             "torchaudio==${CAMPP_TORCH_VERSION}" ;; \
         *) \
           echo "Unsupported SUBMIT_MODEL=${SUBMIT_MODEL}" >&2; exit 2 ;; \
       esac

RUN chmod +x ./run.sh \
    && bash -n ./run.sh \
    && ./run.sh --help >/tmp/submission-entrypoint-help.txt \
    && ./run.sh --container-only --model "${SUBMIT_MODEL}" --test-csv /workspace/test.csv --data-root /workspace --dry-run >/tmp/submission-entrypoint-dry-run.txt \
    && python -c "import torch, requests, soundfile, polars; print('torch', torch.__version__); print('requests', requests.__version__); print('soundfile', soundfile.__libsndfile_version__); print('polars', polars.__version__)" \
    && case "${SUBMIT_MODEL}" in \
         w2v-trt) python -c "import tensorrt, torch, torchvision, transformers; from transformers import Wav2Vec2BertModel; from kryptonite.runtime.tensorrt_generic import MultiInputTensorRTEngineRunner; assert tensorrt.Logger is not None; assert torch.__version__; assert torchvision.__version__; assert transformers.AutoFeatureExtractor is not None; assert Wav2Vec2BertModel is not None; assert MultiInputTensorRTEngineRunner is not None; print('w2v-trt imports ok')" ;; \
         campp-pt) python -c "import torchaudio; from kryptonite.eda.official_campp_tail import main as official_campp_main; from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint; assert torchaudio.compliance.kaldi.fbank is not None; assert official_campp_main is not None; assert load_campp_encoder_from_checkpoint is not None; print('campp imports ok')" ;; \
       esac \
    && python utils/validate_submission.py --help >/tmp/validate-help.txt

CMD ["./run.sh", "--container-only"]
