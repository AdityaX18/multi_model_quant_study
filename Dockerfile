# Same job in a container (needs the NVIDIA container toolkit on the host):
#   docker build -t quant-toolcall .
#   docker run --gpus all -v $PWD/results:/app/results -v $PWD/models:/app/models quant-toolcall configs/l40_run_a.json
# Not yet tested on a GPU host: if the build fails, use scripts/setup_gpu.sh instead.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
RUN git clone https://github.com/ggml-org/llama.cpp .cache/llama.cpp && git -C .cache/llama.cpp checkout -q f3b407c \
    && cmake -S .cache/llama.cpp -B .cache/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    && cmake --build .cache/llama.cpp/build --target llama-server -j
COPY . .
ENV LLAMA_SERVER=/app/.cache/llama.cpp/build/bin/llama-server
ENTRYPOINT [".venv/bin/python", "-m", "harness.pipeline", "--config"]
CMD ["configs/l40_run_a.json"]
