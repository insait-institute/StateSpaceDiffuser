if [ -z "${CUDA_HOME:-}" ]; then
  echo "CUDA_HOME is not set. Ensure your CUDA module/environment is loaded."
else
  export CUDA_HOME
fi

# pip install -r requirements.txt
conda env create -f universe_lab_env.yaml
conda activate universe_lab
# pip install mamba-ssm==2.2.4 --no-build-isolation --no-cache-dir
pip install git+https://github.com/NSavov/dev_utils.git
mkdir -p external
git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git external/Cosmos-Tokenizer
cd external/Cosmos-Tokenizer
pip install -e .
cd -
echo "Done."
