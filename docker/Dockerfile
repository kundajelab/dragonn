FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER Kundaje Lab <annashch@stanford.edu>
# 
# Install Miniconda in /opt/conda 
# 

ENV CONDA_DIR /opt/conda 
ENV PATH $CONDA_DIR/bin:$PATH
RUN apt-get update --fix-missing 
RUN apt-get install -y wget git libhdf5-dev g++ graphviz 
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.1.11-Linux-x86_64.sh -O ~/miniconda.sh && \ 
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \ 
    rm ~/miniconda.sh 
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh 
ENV PATH /opt/conda/bin:$PATH 
ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ENV NB_USER keras
ENV NB_UID 1000
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

RUN usermod -aG sudo keras

USER keras
RUN echo "export PATH=$CONDA_DIR/bin:${PATH}" >> /home/keras/.bashrc

RUN conda install numpy=1.12.1 && \
    conda install scikit-learn=0.18 && \
    conda install pandas=0.18.0 && \
    conda install notebook=4.1.0 && \
    conda install matplotlib=1.5.1 && \
    conda install mkl nose pyyaml six h5py && \
    conda install pygpu -c conda-forge && \
    conda install shapely==1.5.13 && \ 
    conda install pytest

COPY theanorc /home/keras/.theanorc
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
