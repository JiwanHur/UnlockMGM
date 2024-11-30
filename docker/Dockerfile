# Builds GPU docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
# Use base conda image to reduce time
FROM continuumio/miniconda3:latest AS compile-image
# Specify py version
ENV PYTHON_VERSION=3.10
# Install apt libs
RUN apt-get update && \
    apt-get install -y curl git wget nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Create our conda env
RUN conda create --name accelerate python=${PYTHON_VERSION} ipython jupyter pip
# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
ENV PATH /opt/conda/envs/accelerate/bin:$PATH
# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
# Activate the conda env and install torch + accelerate
RUN source activate accelerate && \
    python3 -m pip install --no-cache-dir \
    # torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
    --index-url https://download.pytorch.org/whl/cu116 
    # python3 -m pip install --no-cache-dir accelerate 
    # python3 -m pip install --no-cache-dir \
    # git+https://github.com/huggingface/accelerate#egg=accelerate[testing,test_trackers] \
    # --extra-index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install --no-cache-dir bitsandbytes

# Stage 2
# FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 AS build-image
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

# Install apt libs
RUN apt-get update && \
    apt-get install -y curl git wget tmux htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN echo "source activate accelerate" >> ~/.profile

#RUN /bin/bash -c "source activate accelerate && pip install xformers"
#RUN echo "Installing Apex..."
#WORKDIR /tmp/unique_for_apex
#RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
#WORKDIR /tmp/unique_for_apex/apex

#RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
#WORKDIR /muse # move to git page
#RUN pip install -e ".[extra]"
#WORKDIR /
# Activate the virtualenv
CMD ["/bin/bash"]
