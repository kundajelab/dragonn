FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER Kundaje Lab <annashch@stanford.edu>
RUN apt-get update --fix-missing 
RUN apt-get install -y wget git libhdf5-dev g++ graphviz python3-pip
ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ENV NB_USER keras
ENV NB_UID 1000
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER 
RUN mkdir -p /src && \
    chown keras /src

RUN usermod -aG sudo keras

USER keras


RUN pip3 install numpy==1.12.1 && \
    pip3 install scikit-learn==0.18 && \
    pip3 install pandas==0.18.0 && \
    pip3 install notebook==4.1.0 && \
    pip3 install matplotlib==1.5.1 && \
    pip3 install mkl nose pyyaml six h5py && \
    pip3 install shapely==1.5.13 && \ 
    pip3 install pytest


WORKDIR /src 
RUN git clone https://github.com/kundajelab/dragonn.git
WORKDIR /src/dragonn
### switch to docker branch for changes in setup.py [to be removed before merging to master]
RUN git checkout -b docker origin/docker
###
RUN python setup.py install
RUN mkdir ~/.keras
WORKDIR ~/.keras
COPY keras.json ~/.keras/keras.json
RUN dragonn --help 
RUN py.test
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
RUN apt-get install -y emacs vim
ENV TINI_VERSION v0.14.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini

USER root 
RUN chmod +x /usr/bin/tini
RUN apt-get install -y emacs vim

USER keras 
RUN jupyter notebook --generate-config
COPY jupyter_config.py /home/keras/.jupyter/jupyter_notebook_config.py
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
EXPOSE 80
