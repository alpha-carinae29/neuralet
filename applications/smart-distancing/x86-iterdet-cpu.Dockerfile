FROM continuumio/miniconda3

RUN conda install pytorch torchvision cpuonly -c pytorch

RUN apt-get update && apt-get install -y git libglib2.0-0 libsm6 libxrender-dev libxext6 gcc g++

RUN git clone https://github.com/saic-vul/iterdet.git /iterdet
WORKDIR /iterdet
RUN pip install -r requirements/build.txt
RUN pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
RUN pip install -e .


VOLUME  /repo
WORKDIR /repo/applications/smart-distancing


RUN pip install --upgrade pip setuptools==41.0.0 && pip install opencv-python wget flask scipy image

EXPOSE 8000

ENTRYPOINT ["python", "neuralet-distancing.py"]
CMD ["--config", "config-x86-iterdet.ini"]
