
# CUDA

查看 GPU 狀態、CUDA 版本、正在使用 GPU 的程式等：`nvidia-smi`

各映像檔的版本與環境可以參考：
- https://man.twcc.ai/@twccdocs/ccs-concept-image-main-zh/%2F%40twccdocs%2Fccs-concept-image-pytorch-zh
- https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/overview.html

# 虛擬環境
## Pyenv

__國網環境使用 Docker，pyenv 可能無法正常運作__

安裝 pyenv：
1. `curl -fsSL https://pyenv.run | bash`
2. 將以下程式碼複製到 `$HOME/.bashrc` 最後面
    ```bash
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"
    eval "$(pyenv virtualenv-init -)"
    ```
3. 重新啟動終端使其生效


環境架設：

1. 安裝 Python 版本：`pyenv install 3.8.13`
3. 設定專案 Python 版本：`pyenv local 3.8.13`
4. 建立虛擬環境：`python -m venv s2c `
5. 進入環境：`source ./s2c/bin/activate`
6. 安裝 Python 套件
    - `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113`
    - `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.21.1+cu113.html --no-index`
    - `pip list`
    - `pip check`
    - 若出現 `ERROR: Can not perform a '--user' install. User site-packages are not visible in this virtualenv.` 錯誤，將虛擬環境資料夾 `./s2c/pyenv.cfg` 中的 `include-system-site-packages` 設為 true
7. 修改 SAM 檔案，將 `mask_decoder.py`, `sam.py` 複製到 SAM Module 中：
    - `cp ./modeling/mask_decoder.py ./s2c/lib/python3.8/site-packages/segment_anything/modeling/`
    - `cp ./modeling/sam.py ./s2c/lib/python3.8/site-packages/segment_anything/modeling/`

套件安裝過程若出現 `ERROR: Can not perform a '--user' install. User site-packages are not visible in this virtualenv.` 錯誤，將虛擬環境資料夾 `./s2c/pyenv.cfg` 中的 `include-system-site-packages` 設為 true

Pyenv 使用方式：
- 啟動環境：`./s2c/bin/activate`
- 離開環境：`deactivate`
- 刪除環境：`rm -rf ./s2c`

環境內可直接使用 `python`, `pip` 等指令

## Conda

安裝 miniconda：

1. 執行以下腳本：
    ```bash
    mkdir -p ~/miniconda3
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "~/miniconda3/miniconda.sh"
    bash "~/miniconda3/miniconda.sh" -b -u -p "~/miniconda3"
    rm "~/miniconda3/miniconda.sh"
    ```
2. 初始化 conda：
    ```bash
    source "~/miniconda3/bin/activate"
    conda init --all
    ```
3. 重新啟動終端使其生效

環境架設：

1. 從 yaml 檔安裝環境：`conda env create -f s2c.yml`
2. 安裝 SAM：`pip install git+https://github.com/facebookresearch/segment-anything.git`
3. 安裝 pip 套件：
    - `pip install opencv-python`
    - `pip install pillow-avif-plugin`
4. 安裝 torch-scatter：`pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html`，支援版本詳見：https://data.pyg.org/whl/
5. 啟動環境：`conda activate s2c`

Conda 使用方式：
- 環境列表：`conda env list`
- 啟動環境：`conda activate s2c`
- 查看套件：`conda list`
- 安裝套件：`conda install <套件名稱>`
- 離開環境：`deactivate`
- 刪除環境：`conda env remove -n <環境名稱>`

## TWCC

TWCC 環境中 Pyenv 無法正常運作，請參考 __Conda__ 章節建立環境