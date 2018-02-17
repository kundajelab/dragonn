FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER Kundaje Lab <annashch@stanford.edu>
ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN apt-get update --fix-missing 
RUN apt-get install -y build-essential gfortran libatlas-base-dev python3-dev wget git libhdf5-dev g++ graphviz python3-pip libfreetype6-dev libgeos-dev sudo
RUN apt-get install pkg-config

RUN pip3 install numpy==1.12.1 && \
    pip3 install scikit-learn==0.18 && \
    pip3 install pandas==0.18.0 && \
    pip3 install notebook==4.1.0 && \
    pip3 install matplotlib==1.5.1 && \
    pip3 install mkl nose pyyaml six h5py && \
    pip3 install shapely==1.5.13 && \ 
    pip3 install pytest && \
    pip3 install tensorflow-gpu==1.0.1 \
    pip3 install tensorflow


WORKDIR /src 
RUN git clone https://github.com/kundajelab/dragonn.git
WORKDIR /src/dragonn
RUN git checkout -b docker origin/docker

RUN sudo python3 setup.py install
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

RUN chmod +x /usr/bin/tini
RUN apt-get install -y emacs vim

RUN jupyter notebook --generate-config
COPY jupyter_config.py /home/keras/.jupyter/jupyter_notebook_config.py
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
EXPOSE 80
