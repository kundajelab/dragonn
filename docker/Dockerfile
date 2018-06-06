FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER Kundaje Lab <annashch@stanford.edu>
ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN apt-get update --fix-missing 
RUN apt-get install -y build-essential gfortran libatlas-base-dev python3-dev wget git libhdf5-dev g++ graphviz python3-pip libfreetype6-dev libgeos-dev sudo pkg-config emacs vim cmake
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip3 install numpy==1.12.1 && \
    pip3 install scikit-learn==0.18 && \
    pip3 install pandas==0.18.0 && \
    pip3 install notebook==4.1.0 && \
    pip3 install matplotlib==1.5.1 && \
    pip3 install mkl nose pyyaml six h5py sphinx pydot-ng && \
    pip3 install shapely==1.5.13 && \ 
    pip3 install pytest && \
    pip3 install dlib && \
    pip3 install tornado==5 && \
    pip3 install --upgrade jupyter

ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
RUN ldconfig
RUN mkdir /src
RUN mkdir /src/dragonn

WORKDIR /src 
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.14.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

RUN mkdir /root/.jupyter
WORKDIR /root/.jupyter
ADD jupyter_config.py jupyter_notebook_config.py


EXPOSE 8888
EXPOSE 80
