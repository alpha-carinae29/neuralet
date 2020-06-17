ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git pkg-config gcc ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all
RUN git clone https://github.com/saic-vul/iterdet.git /iterdet
WORKDIR /iterdet
ENV FORCE_CUDA="1"
RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
RUN pip install --no-cache-dir -e .


VOLUME  /repo
WORKDIR /repo/applications/smart-distancing


RUN pip install --upgrade pip setuptools==41.0.0 && pip install opencv-python wget flask scipy image

EXPOSE 8000

ENTRYPOINT ["python", "neuralet-distancing.py"]
CMD ["--config", "config-x86-iterdet.ini"]
