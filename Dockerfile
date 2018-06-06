FROM kundajelab/dragonn_base:docker_base
MAINTAINER Kundaje Lab <annashch@stanford.edu>

#copy all dragonn files from the current branch to the /src directory

WORKDIR /src/dragonn
COPY . /src/dragonn/
RUN python setup.py install

RUN mkdir /root/.keras
WORKDIR /root/.keras
COPY docker/keras.json keras.json


WORKDIR /src/dragonn
RUN dragonn --help 
RUN py.test


ENTRYPOINT ["jupyter","notebook","--allow-root","--config", "/root/.jupyter/jupyterhub_notebook_config.py", "--no-browser", "--port", "80","--ip","0.0.0.0"]