# Gitlab docker builder image
# contemporary version of skopeo needed
FROM docker:20.10

ENV DOCKER_TLS_CERTDIR "/certs"

ENV DOCKER_CLI_EXPERIMENTAL enabled

ENV BUILDX_URL https://github.com/docker/buildx/releases/download/v0.8.2/buildx-v0.8.2.linux-amd64

RUN apk add --no-cache wget git bash findutils curl g++ libmagic skopeo jq

RUN apk add python3 python3-dev --repository=http://dl-cdn.alpinelinux.org/alpine/edge/main

RUN mkdir -p $HOME/.docker/cli-plugins/

RUN wget -q -O $HOME/.docker/cli-plugins/docker-buildx $BUILDX_URL

RUN chmod a+x $HOME/.docker/cli-plugins/docker-buildx

RUN python3 -m ensurepip

RUN rm -r /usr/lib/python*/ensurepip

RUN pip3 install --upgrade pip setuptools

RUN if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi

RUN if [ ! -e /usr/bin/python ]; then ln -sf /usr/bin/python3 /usr/bin/python; fi

COPY pyproject.toml /root/
COPY poetry.lock /root/

WORKDIR /root

ENV PIP_DISABLE_PIP_VERSION_CHECK=true
ENV VIRTUAL_ENV_DISABLE_PROMPT 1
ENV PATH=$PATH:/root/.local/bin

RUN pip install poetry==1.2 && \
    poetry config virtualenvs.create true --local && \
    poetry config virtualenvs.in-project true --local && \
    poetry install --no-interaction -vv

# Provide a known path for the virtual environment by creating a symlink
RUN ln -s $(poetry env info --path) /root/cuda_manager_env

# Clean up project files. You can add them with a Docker mount later.
RUN rm pyproject.toml poetry.lock

# Start virtual env when bash starts
RUN echo 'source /root/cuda_manager_env/bin/activate' >> ~/.bashrc
