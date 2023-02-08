FROM nvidia/cuda:11.2.2-cudnn8-devel

# Environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PATH "$PATH:/bin/3.2/python/bin/"
ENV BLENDER_PATH "/bin/3.2"
ENV BLENDERPIP "/bin/3.2/python/bin/pip3"
ENV BLENDERPY "/bin/3.2/python/bin/python3.10"
ENV HW="CPU"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list



# Install dependencies
RUN apt-get update && apt-get install -y \
	wget \
	libopenexr-dev \
	bzip2 \
	build-essential \
	zlib1g-dev \
	libxmu-dev \
	libxi-dev \
	libxxf86vm-dev \
	libfontconfig1 \
	libxrender1 \
	libgl1-mesa-glx \
	xz-utils

# Download and install Blender
RUN wget https://mirror.clarkson.edu/blender/release/Blender3.2/blender-3.2.0-linux-x64.tar.xz \
	&& tar -xvf blender-3.2.0-linux-x64.tar.xz --strip-components=1 -C /bin \
	&& rm -rf blender-3.2.0-linux-x64.tar.xz \
	&& rm -rf blender-3.2.0-linux-x64

# Download the Python source since it is not bundled with Blender
RUN wget https://www.python.org/ftp/python/3.10.5/Python-3.10.5.tgz \
	&& tar -xzf Python-3.10.5.tgz \
	&& cp -r Python-3.10.5/Include/* $BLENDER_PATH/python/include/python3.10/ \
	&& rm -rf Python-3.10.5.tgz \
	&& rm -rf Python-3.10.5

# Blender comes with a super outdated version of numpy (which is needed for matplotlib / opencv) so override it with a modern one
RUN rm -rf ${BLENDER_PATH}/python/lib/python3.10/site-packages/numpy

# Must first ensurepip to install Blender pip3 and then new numpy
RUN ${BLENDERPY} -m ensurepip && ${BLENDERPIP} install --upgrade pip && ${BLENDERPIP} install numpy





ENV BLENDER_PYTHON_PATH /bin/3.2/python/bin/python3.10
ENV BLENDER_PYTHON_SITE_PACKAGES /bin/3.2/python/lib/python3.10/site-packages
COPY ./requirements.txt .

RUN apt-get update && apt-get install -y wget gnupg curl && apt-get install --no-install-recommends -y \
    libgl1 \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
#RUN add-apt-repository ppa:deadsnakes/ppa && apt update && apt install python3.8 && apt install python3.8-distutils
#RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py


RUN $BLENDER_PYTHON_PATH -m ensurepip
RUN $BLENDER_PYTHON_PATH -m pip install --upgrade pip
RUN $BLENDER_PYTHON_PATH -m pip install -r requirements.txt

WORKDIR /app/blender
COPY . /app/blender
