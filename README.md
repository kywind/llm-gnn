# LLM-GNN

```
## installation
conda create -n llm-gnn python=3.9
conda activate llm-gnn
pip install -r requirements.txt

## install grounding DINO
echo $CUDA_HOME  # make sure it is set
mkdir third-party
cd third-party
git clone git@github.com:IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ../..

## install PyFleX
# this follows the instructions in https://github.com/WangYixuan12/dyn-res-pile-manip/
# ensure docker is installed
cd third-party
git clone git@github.com:kywind/PyFleX.git
conda install pybind11 -c conda-forge

CURR_CONDA=$CONDA_DEFAULT_ENV
CONDA_BASE=$(conda info --base)
docker pull xingyu/softgym
docker run \
    -v ${PWD}/PyFleX:/workspace/PyFleX \
    -v ${CONDA_PREFIX}:/workspace/anaconda \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it xingyu/softgym:latest bash \
    -c "export PATH=/workspace/anaconda/bin:$PATH; cd /workspace/PyFleX; export PYFLEXROOT=/workspace/PyFleX; export PYTHONPATH=/workspace/PyFleX/bindings/build:$PYTHONPATH; export LD_LIBRARY_PATH=$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH; cd bindings; mkdir build; cd build; /usr/bin/cmake ..; make -j"

echo '' >> ~/.bashrc
echo '# PyFleX' >> ~/.bashrc
echo "export PYFLEXROOT=${PWD}/PyFleX" >> ~/.bashrc
echo 'export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo '' >> ~/.bashrc

source ~/.bashrc
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $CURR_CONDA
cd ..

## download weights
cd src
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..

## download data
mkdir data
cd data
gdown 1RriJbYKIVR60HZDpMWB0F8sL9iWpZl7n
unzip 2023-09-13-15-19-50-765863.zip
cd ..

## run
# store openai API key in ./api_key.txt
cd src/
python rollout.py
```
