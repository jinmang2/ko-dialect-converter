# KoBART-dialect

**WIP**
- 만족할만한 성능 나오면 배포

## 실행 순서

```
# 1. 데이터 빌드 (raw Arrow → SFT/GRPO/분류기 Arrow)
python scripts/stage0_build_datasets.py outputs/dialect_raw_new

# 2. SFT
python scripts/stage1_sft.py outputs/datasets/sft

# 3. 분류기 학습
python scripts/stage2_train_classifier.py outputs/datasets/classifier

# 4. GRPO
python scripts/stage3_grpo.py outputs/datasets/grpo outputs/classifier

# 5. 평가
python scripts/evaluate.py outputs/grpo outputs/dialect_raw_new outputs/classifier

# 6. GGUF 변환 & 서빙
python scripts/export.. --lora outputs/grpo--out outputs/merged
python scripts/exportuts/merged
python scripts/export_gguf.py quantize --input outputs/gguf/model_f16.gguf
python scripts/exportoutputs/gguf/model_Q4_K_M.gguf
```

```
# 기본 (tracking 없음)
python scripts/stage1_sft.py data.sft_dataset_path=outputs/datasets/sft

# Wandb로 실험 preset 사용
python scripts/stage1_sft.py +experiment=sft_baseline data.sft_dataset_path=outputs/datasets/sft

# CLI로 직접 MLflow 선택 + 실험명 지정
python scripts/stage1_sft.py logger=mlflow experiment.name=my-exp experiment.run_name=lora-r32 training.lora_r=32 data.sft_dataset_path=...

# 하이퍼파라미터 즉석 오버라이드 (Hydra sweeper와 연동 가능)
python scripts/stage1_sft.py logger=wandb training.learning_rate=1e-4 training.lora_r=32 data.sft_dataset_path=...

● Ran 3 stop hooks
  ⎿  Stop hook error: [AUTOPILOT - Phase: unspecified] Autopilot not complete. Continue working. When all phases are complete, run /oh-my-claudecode:cancel to cleanly exit and clean up this session's autopilot state files. If cancel fails, retry with /oh-my-claudecode:cancel --force.

Phase 3 QA는 완료 (19 tests passed, ruff clean). Phase 4 Validation을 병렬로 실행합니다.

● 2 background agents launched (↓ to manage)
oh-my-claudecode:architect (Architect validation of Hydra+MLflow+Wandb integration)
oh-my-claudecode:code-reviewer (Code quality review of tracking integration)


# 기본 (style + content, 논문 그대로)
python scripts/stage3_grpo.py \
  data.grpo_dataset_path=outputs/datasets/grpo \
  data.classifier_path=outputs/classifier \
  logger=wandb

# r_edit 활성화 (모델이 복사만 할 때)
python scripts/stage3_grpo.py \
  training.use_edit_reward=true \
  training.reward_weights=[1.0,0.5,0.3] \
  data.grpo_dataset_path=... data.classifier_path=...
```
```

python scripts/stage1_sft.py \
  logger=wandb \
  experiment.run_name=sft-qwen-v1 \
  data.sft_dataset_path=outputs/datasets/sft

python scripts/stage2_train_classifier.py \
  data.cls_dataset_path=outputs/datasets/classifier \
  logger=wandb \
  experiment.run_name=cls-textcnn-v1 \
  training.per_device_train_batch_size=128 \
  training.per_device_eval_batch_size=256 \
  training.eval_strategy=steps \
  training.eval_steps=200 \
  training.logging_steps=100

python scripts/stage1_sft.py \
  data.sft_dataset_path=outputs/datasets/sft \
  training.num_train_samples=300 \
  training.num_eval_samples=200 \
  training.num_train_epochs=1 \
  training.eval_strategy=steps \
  training.eval_steps=50 \
  training.save_steps=50 \
  training.logging_steps=10 \
  'training.output_dir=outputs/sft_speedrun'

```
```
# rebuild — combine downsample + (later) weighted loss
python scripts/stage0_build_datasets.py --raw_dataset_path outputs/dialect_raw_new \
    --cls_standard_cap_ratio 2.0
# train with weighted loss
python scripts/stage2_train_classifier.py data.cls_dataset_path=outputs/datasets/classifier \
    model.class_weighting=balanced
python scripts/stage2_train_classifier.py \
  data.cls_dataset_path=outputs/datasets/classifier \
  logger=wandb \
  experiment.run_name=cls-textcnn-balanced-test \
  training.per_device_train_batch_size=128 \
  training.per_device_eval_batch_size=256 \
  training.eval_strategy=steps \
  training.eval_steps=200 \
  training.logging_steps=100 \
  model.class_weighting=balanced
python scripts/stage3_grpo.py \
  data.grpo_dataset_path=outputs/datasets/grpo \
  data.classifier_path=outputs/classifier \
  logger=wandb \
  data.classifier_path=outputs/classifier_clean/checkpoint-8500 \
  data.base_model_path=outputs/sft_merged
```

## Data preparing

- First, download the file below from [aihub](https://aihub.or.kr/aihub-data/natural-language/about) and set it up as follows.

```
.
└── data/
│   ├── 한국어 방언 발화 데이터(강원도)
│   │   ├── Training/[라벨]강원도_학습데이터_1.zip
│   │   └── Validation/[라벨]강원도_학습데이터_2.zip
│   ├── 한국어 방언 발화 데이터(경상도)
│   │   ├── Training/[라벨]경상도_학습데이터_1.zip
│   │   └── Validation/[라벨]경상도_학습데이터_2.zip
│   ├── 한국어 방언 발화 데이터(전라도)
│   │   ├── Training/[라벨]전라도_학습데이터_1.zip
│   │   └── Validation/[라벨]전라도_학습데이터_2.zip
│   ├── 한국어 방언 발화 데이터(제주도)
│   │   ├── Training/[라벨]제주도_학습데이터_1.zip
│   │   └── Validation/[라벨]제주도_학습데이터_3.zip
│   └── 한국어 방언 발화 데이터(충청도)
│       ├── Training/[라벨]충청도_학습데이터_1.zip
│       └── Validation/[라벨]충청도_학습데이터_2.zip
├── kodialect/..
├── .gitignore
├── LICENSE
└── README.md
```

- Second, unzip files

```shell
$ sh unzip.sh
```

- Third, run `prepare_data.py`
    - There may be errors in the json data itself provided by aihub. Please refer to the [issue](https://github.com/jinmang2/KoBART-dialect/issues/1) and edit the file directly and run the above python script.

```shell
$ python prepare_data.py
```

- Final data folder

```
.
└── data/
│   ├── chungcheongdo/..
│   ├── gangwondo/..
│   ├── gyeongsangdo/..
│   ├── jejudo/..
│   ├── jeollado/..
│   ├── style_classification/..
│   ├── style_transfer/..
│   ├── train_dialect.json
│   └── valid_dialect.json
├── kodialect/..
├── .gitignore
├── LICENSE
└── README.md
```


## Citations

```
@inproceedings{lai-etal-2021-thank,
    title = "Thank you {BART}! Rewarding Pre-Trained Models Improves Formality Style Transfer",
    author = "Lai, Huiyuan and Toral, Antonio and Nissim, Malvina",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.62",
    doi = "10.18653/v1/2021.acl-short.62",
    pages = "484--494",
}
```
```
@article{park-etal-2025-steering,
    title = "Steering {LLM}s toward {K}orean Local Speech: Iterative Refinement Framework for Faithful Dialect Translation",
    author = "Park, Keunhyeung and Yu, Seunguk and Kim, Youngbin",
    journal = "arXiv preprint arXiv:2511.06680",
    year = "2025",
    url = "https://arxiv.org/abs/2511.06680",
}
```
