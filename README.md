# LLM-GNN

```
## installation
conda create -n llm-gnn python=3.9
conda activate llm-gnn
pip install -r requirements.txt

echo $CUDA_HOME  # make sure it is set
mkdir third-party
cd third-party
git clone git@github.com:IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ../..

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
