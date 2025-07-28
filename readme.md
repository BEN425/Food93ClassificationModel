# Introduction

#### 前置作業

1. 安裝 library
    - `sudo apt update`
    - `sudo apt-get install build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libncurses-dev tk-dev`
    - `sudo apt-get install libsqlite3-dev`

#### 架設環境

1. 詳細安裝過程可以查看 `environment.md`
7. 修改 SAM 檔案，將 `mask_decoder.py`, `sam.py` 複製到 SAM Module 中：
    - `cp ./modeling/mask_decoder.py ./s2c/lib/python3.8/site-packages/segment_anything/modeling/`
    - `cp ./modeling/sam.py ./s2c/lib/python3.8/site-packages/segment_anything/modeling/`

#### 訓練前準備

1. 下載 SAM model weights 放到 `./pretrained`：https://github.com/facebookresearch/segment-anything
2. 將資料集放到 `Database` 資料夾, 詳細訊息可以查看章節 **Database**

3. 啟動環境
4. 檢查資料集中是否有問題：`cd utility`, `python check_labels.py`, `python ckeck_image.py`
5. 調整 `FoodImageCode/cfg/Setting.yml` 的設定，包含路徑、訓練參數等，詳細資訊參考 __Config__ 章節
6. `cd ../FoodImageCode`
5. 清除 CUDA 記憶體：`nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9`
7. 開始訓練：`python main.py`
8. 訓練結果會存放在 `FoodImageCode/Results` 中，包含 checkpoints 與 tensorboard logs

# Database

資料夾：Database

圖片格式：**JPG, PNG, WEBP, AVIF**, AVIF 格式需要 `pillow-avif-plugin` 套件

類別：93類, multi-class

包含 _AI Food Database_ 與 _Single Food Database_

若一張圖片是 multi-label, 該圖片會同時出現在不同類別的資料夾中. Example：若 `fruit1.jpg` 同時包含 Apple, Orange 類別, 則會同時出現在 Apple 資料夾與 Orange 資料夾中

### 資料夾結構

總共有 4 層資料夾結構

-   First Level：6 Categories Food (六大類食物) 不包含 _乳品類_ 和 _其他_， Index 為 1, 2, 3, 5, 6
    -   `1_CerealsGrainsTubersAndRoots`：全榖雜糧類
    -   `2_OilFatNutAndSeed`：油脂與堅果種子類
    -   `3_FishMeatAndEgg`：豆魚蛋肉類
    -   `5_Vegetable`：蔬菜類
    -   `6_Fruit`：水果類
-   Second Level：13 Categories，Index 從 A 到 I
    -   `A_CeralsGrainsTubersAndRoots`
    -   `B_FatsAndOils`
    -   `C_Poultry`
    -   `D_Meat`
    -   `E_Seafood`
    -   `F_OtherProteinrichFoods`
    -   `G_Vegetables`
    -   `H_Fruits`
    -   `I_RefreshmentAndSnacks`
-   Third Level：將第二層進行細分，Index 為英文+數字。Example：`A1_RiceAndProducts` 是 `A_CeralsGrainsTubersAndRoots` 的子類別
-   Fourth Level：食物名稱本身，總共有 **93 categories**。Example：`Congee`, `Rice` 是 `A1_RiceAndProducts` 的子類別

### 其他檔案

-   `class.txt`：93 類別食物的 ID 與名稱
-   `class_freq.txt`：每個食物類別的比例
-   `class_entropy.txt`：每個食物類別的 entropy (self-information)
-   `FoodSeg_cate_mapping.csv`：FoodSeg103 類別名稱對應到 93 食物類別
-   `id_mapping.csv`：FoodSeg103 類別 ID 對應到 93 食物類別 ID
-   `class_chinese.txt`：食物類別的中文名稱

# Code

資料夾：FoodImageCode

### 資料夾結構

-   `cfg`
    -   `Setting.yml`：關於 training, inference 的各種設定
-   `model`
    -   `ResNet_modified.py`：原本為嘉宏的 ResNet50，品潔重寫 Pytorch。輸入圖片大小為 3x224x224 (CxWxH). 輸出類別設定在 `Setting.yml`: `MODEL`, `CATEGORY_NUM`.
-   `Results`
    -   `checkpoints`：訓練過程儲存的模型權重，每個 epoch 儲存一次
    -   `logs`：訓練過程產生的 tensorboard 的記錄檔
    -   子資料夾的名稱設定在 `Setting.yml`: `SAVE_SUB_NAME`.
-   `csv`
    -   CSV 檔由 `make_image_csv.py` 產生，包含 training, validation, test set 三個 CSV 檔。每個 CSV 檔包含圖片的路徑與 multi-hot labels
    -   訓練時讀取的 CSV 檔路徑以及 `make_image_csv.py` 產生的 CSV 檔路徑設定在 (兩者共用) `Setting.yml`: `ALL_CSV_DIR`, `TRAIN_CSV_DIR`, `VALID_CSV_DIR`, `TEST_CSV_DIR`
    -   詳細訊息參考 **Creating CSV files** 章節
-   `main.py`：讀取 dataset 與 model 進行訓練
-   `dataset.py`：定義如何讀取資料與資料的內容。`FoodDataset` 包含圖片的 Tensor 與 label 的 Tensor。`FoodDatasetWithMasks` 包含圖片的 Tensor、label 的 Tensor 與 SAM mask 的 Tensor。
-   `training_loop.py`：定義訓練的過程。每個 epoch 後會計算 validation set 的指標，將 training, validation 的指標印出來，並將該 epoch 的紀錄儲存在 `Results` 資料夾中。
-   `metrics.py`：定義準確率的指標，包含 F1 score, zero accuracy, hamming loss 等。`evaluate_dataset` 函數讓模型衡量 datase 的指標並回傳結果
-   `loss.py`：定義 loss function，包含 classification loss, SSC loss, CPM loss
-   `cfgparser.py`：定義 `CfgParser` class 讀取 yaml 設定檔
-   `make_image_csv.py`：產生 training, validation, test set 的 CSV 檔
    -   詳細訊息參考 **Creating CSV files** 章節
-   `test_and_confusion_matrix.py`：__目前沒有使用__

### Configs

Training, inference 相關設定在 `./FoodImageCode/cfg/Setting.yml`。Yaml 檔透過 `./FoodImageCode/cfgparser.py` 的 `CfgParser` class 讀取，以 dict 的形式儲存在 `self.cfg_dict` 變數中

-   `PARSING_SETTING`
    -   `SEED`：隨機種子
    -   `GPU_ID`：Pytorch 使用的 GPU
    -   `WORKERS`：`Dataloader` 的 worker 數量，使用多少個 subprocess 來載入資料集
    -   `EPOCHS`：訓練的 epoch 數
    -   `BATCH_SIZE`：Training batch size
    -   `EVAL_BATCH_SIZE`：Validation batch size
-   `TRAIN_SETTING`
    #### 基本設定
    -   `ROOT`: 專案的 root directory。下面的路徑都會以 `ROOT` 作為 parent
    -   `RESUME`：是否從 checkpoint 繼續訓練。若為 False，則會從 ImageNet 的 pretrained weight 開始訓練
    -   `CHECKPOINT_PATH`：若 `RESUME` 為 True，會從此路徑載入 checkpoint
    -   `TEST_METRICS`: 訓練完一個 epoch 後，是否要計算 validation set 的指標
    #### 路徑設定
    -   `DATA_BASE_DIR`：Database 的路徑，在 `make_image_csv.py` 中會使用，設為 `Database/<名稱>`
    -   `FOODSEG_DIR`：FoodSeg103 database 的路徑，__目前沒有使用__
    -   `ALL_CSV_DIR`：包含 training, validation, test 所有圖片的 CSV 檔，不會用在 training
    -   `TRAIN_CSV_DIR`, `VALID_CSV_DIR`, `TEST_CSV_DIR`：分別包含 training, validation, test 所有圖片的 CSV 檔，會用在 training 與 inference，設為 `FoodImageCode/csv/<子資料夾名>/<CSV檔名>`。詳細訊息參考 **Creating CSV files** 章節
    -   `FOODSEG_CSV_DIR`：包含 FoodSeg103 圖片的 CSV 檔，__目前沒有使用__
    -   `SAVE_DIR`：Training 結果的儲存位置，包含 checkpoints 與 tensorboard logs。設為 `FoodImageCode/Results/`。
    -   `SAVE_SUB_NAME`: `SAVE_DIR` 的子資料夾名稱，完整路徑會是 `SAVE_DIR + SAVE_SUB_NAME + checkpoints/logs`
    #### S2C 設定
    -   `SAM_DIR`：SAM model 的路徑
    -   `SAM_MASK_DIR`：SAM mask 的路徑，用在 SSC 中
    -   `USE_SSC`：是否加入 SSC loss
    -   `USE_CPM`：是否加入 CPM loss
    #### 模型設定
    -   `MODEL`
        -   `TYPE`：CNN network，預設為 ResNet50，可以設為 ResNet38
        -   `NAME`：儲存 checkpoint 的模型名稱
        -   `INCHANNELS`: 輸入資料的 channel 數，通常為 RGB 三個 channel
        -   `OUTCHANNELS`: 64 for ResNet50.
        -   `CATEGORY_NUM`: 類別數量，AIFood, SingleFood 為 93 類
        -   `LR`：Learning rate
        -   `MOMENTUM`：Momentum for SGD optimizer
        -   `WEIGHT_DECAY`：Weight decay for SGD and Adam optimizer
        -   `CBAM`, `SENET`: 是否加入 CBAM 與 SENet layers，array 中四個職表示是否要加在 ResNet 中間的四層 CNN layers
    -   `LOSS`: Focal loss 中的 `ALPHA`,  `GAMMA`

### Creating CSV files

Training 時會從 CSV 檔讀進資料的路徑。CSV 檔由 `make_image_csv.py` 產生，包含 training, validation, test set 三個 CSV 檔。每個 CSV 檔包含圖片的路徑與 multi-hot labels。
CSV 檔的路徑設定在 `Setting.yml`: `TRAIN_CSV_DIR`, `VALID_CSV_DIR`, `TEST_CSV_DIR`

CSV 檔中，每個 row 為一筆資料。Column 1 為資料的路徑，以 `Setting.yml`: `ROOT` 為 parent。其餘路徑為 multi-hot labels，總共有 93 個值，0 為 negative、1 為 positive。

# Utility

資料夾：utility

用於 debug、產生 metadata、檢查 dataset 等工具的 script

# Reference

-   <a href=https://openaccess.thecvf.com/content/CVPR2024/papers/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.pdf>  From SAMtoCAMs: Exploring Segment Anything Model for Weakly Supervised Semantic Segmentation
-   <a href=https://segment-anything.com/>Segment Anything</a>
-   <a href=https://arxiv.org/abs/1803.10464>Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation</a>
-   <a href=https://arxiv.org/abs/1512.03385>Deep Residual Learning for Image Recognition</a>
-   <a href=https://arxiv.org/abs/1412.6980>Adam: A Method for Stochastic Optimization</a>
-   <a href=https://arxiv.org/abs/1708.02002>Focal Loss for Dense Object Detection</a>
-   <a href=https://arxiv.org/abs/1807.06521>CBAM: Convolutional Block Attention Module</a>
-   <a href=https://arxiv.org/abs/1709.01507>Squeeze-and-Excitation Networks</a>
-   <a href=https://xiongweiwu.github.io/foodseg103.html>A Large-Scale Benchmark for Food Image Segmentation</a>
-   <a href=https://pytorch.org/>PyTorch</a>
-   <a href=https://github.com/sangrockEG/S2C>S2C Github</a>
-   <a href=https://github.com/facebookresearch/segment-anything>SAM Github</a>
