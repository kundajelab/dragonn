FROM kundajelab/dragonn_base:docker_base
MAINTAINER Kundaje Lab <annashch@stanford.edu>

#copy all dragonn files from the current branch to the /src directory

WORKDIR /src/dragonn
COPY ../ /src/dragonn/
RUN python setup.py install

RUN mkdir ~/.keras
WORKDIR ~/.keras
COPY keras.json ~/.keras/keras.json


WORKDIR /src/dragonn
RUN dragonn --help 
RUN py.test

RUN echo "root:dragonn" | chpasswd
ENTRYPOINT ["jupyterhub", "-f", "~/.jupyter/jupyterhub_notebook_config.py", "--no-ssl", "--port", "80"]