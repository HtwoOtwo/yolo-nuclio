FROM ubuntu:22.04

RUN apt-get update && apt-get -y install curl git python3 python3-pip
RUN apt-get -y install openssh-client libgl1-mesa-glx libglib2.0-dev
RUN mkdir -p /root/.pip \
        && echo "[global]" > /root/.pip/pip.conf \
        && echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf
RUN pip3 install ultralytics opencv-python numpy timm loguru
RUN ln -s /usr/bin/pip3 /usr/local/bin/pip
RUN ln -s /usr/bin/python3 /usr/local/bin/python

COPY ai-tools /opt/nuclio/
COPY yolo11n-seg.pt /opt/nuclio/model.pt
COPY function.yaml /opt/nuclio/function.yaml