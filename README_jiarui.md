## Installation

### Install VLABench
1. Prepare conda environment
```sh
conda create -n vlabench python=3.10
conda activate vlabench

git clone https://github.com/1mather/vlabench.git
cd VLABench
pip install -r requirements.txt
pip install -e .
```
2. Download the assets
```sh
python script/download_assetes.py
```
The script will automatically download the necessary assets and unzip them into the correct directory.




## Trajectory Generation

### Generate Episode

#### 1. generate the  .hdf5 episodes

---

#### simple mode (fixed_pos)

**select_apple**
```bash
python /VLABench/scripts/trajectory_generation.py --task-name select_fruit --start-id 0 \
  --save-dir=/mnt/data/310_jiarui/VLABench/media/select_fruit \
  --config-path=/mnt/data/310_jiarui/VLABench/VLABench/configs/task_related/task_specific_config/select_apple/task_config_1_pos_200_table_0.json \
  --n-sample=50
```

**add_condiment**
```bash
python /VLABench/scripts/trajectory_generation.py --task-name add_condiment --start-id 0 \
  --save-dir=/mnt/data/310_jiarui/VLABench/media/add_condiment \
  --config-path=/mnt/data/310_jiarui/VLABench/VLABench/configs/task_related/task_specific_config/add_condiment/task_config_1_pos_200.json \
  --n-sample=50
```

**select_chemistry_tube**
```bash
python /VLABench/scripts/trajectory_generation.py --task-name select_chemistry_tube --start-id 0 \
  --save-dir=/mnt/data/310_jiarui/VLABench/media/select_chemistry_tube \
  --config-path=/mnt/data/310_jiarui/VLABench/VLABench/configs/task_related/task_specific_config/select_chemistry_tube/task_config_1_pos_200.json \
  --n-sample=50
```

---

#### difficult mode (difficulty/random_pos)

**select_apple**
```bash
python /VLABench/scripts/trajectory_generation.py --task-name select_fruit --start-id 0 \
  --save-dir=/mnt/data/310_jiarui/VLABench/media/select_fruit \
  --config-path=/mnt/data/310_jiarui/VLABench/VLABench/configs/task_related/task_specific_config/select_apple_difficult/task_config_1_pos_200_table_0.json \
  --n-sample=50
```

**add_condiment**
```bash
python /VLABench/scripts/trajectory_generation.py --task-name add_condiment --start-id 0 \
  --save-dir=/mnt/data/310_jiarui/VLABench/media/add_condiment \
  --config-path=/mnt/data/310_jiarui/VLABench/VLABench/configs/task_related/task_specific_config/add_condiment_difficult/task_config_1_pos_200.json \
  --n-sample=50
```

**select_chemistry_tube**
```bash
python /VLABench/scripts/trajectory_generation.py --task-name select_chemistry_tube --start-id 0 \
  --save-dir=/mnt/data/310_jiarui/VLABench/media/select_chemistry_tube \
  --config-path=/mnt/data/310_jiarui/VLABench/VLABench/configs/task_related/task_specific_config/select_chemistry_tube_difficult/task_config_1_pos_200.json \
  --n-sample=50

```
---
#### 2.convert to the lerobot dataset format
**add_condiment**
```bash
python /VLABench/scripts/convert_to_lerobot.py --dataset-name add_condiment_1_50_diff \
  --dataset-path /mnt/data/310_jiarui/VLABench/media/difficulty/add_condiment \
  --max-files 50 --max-episodes 50
```

**select_chemistry_tube**
```bash
python /VLABench/scripts/convert_to_lerobot.py --dataset-name select_chemistry_tube_1_50_diff \
  --dataset-path /mnt/data/310_jiarui/VLABench/media/difficulty/select_chemistry_tube \
  --max-files 50 --max-episodes 50
```

**select_fruit**
```bash
python /VLABench/scripts/convert_to_lerobot.py --dataset-name select_fruit_1_50_diff \
  --dataset-path /mnt/data/310_jiarui/VLABench/media/difficulty/select_fruit\
  --max-files 50 --max-episodes 50
```

---

#### 3. Evaluate your policy in a different thread or machine via WebSocket

**select_fruit_difficult**
```bash
python /VLABench/tutorials/evaluation.py --task select_fruit_difficult --n_episodes 50 --max_substeps 10 \
  --save_dir /mnt/data/310_jiarui/VLABench/logs/select_fruit_difficult
```

> **Note:**  
> Before running the above command, please make sure the policy server has been launched and is listening on `127.0.0.1:8000`.
