FROM kundajelab/dragonn_base:docker_base
MAINTAINER Kundaje Lab <annashch@stanford.edu>
RUN pip3 install --upgrade numpy
RUN pip3 install cython
RUN pip3 install theano==0.9.0

#copy all dragonn files from the current branch to the /src directory

WORKDIR /src/dragonn
COPY . /src/dragonn/
RUN python setup.py install
RUN mkdir /root/.keras
WORKDIR /root/.keras
COPY docker/keras.json keras.json

WORKDIR /root
COPY docker/theanorc .theanorc

WORKDIR /src/dragonn
RUN dragonn --help
RUN py.test

ENTRYPOINT ["/usr/local/bin/jupyter", "notebook", "--config", "/root/.jupyter/jupyterhub_notebook_config.py",  "--port", "80", "--ip=0.0.0.0", "--no-browser", "--allow-root"]