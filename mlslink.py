from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    # Keep behavior close to the original BTLink code when available.
    from transformers import AdamW
except Exception:
    from torch.optim import AdamW

from model import BTModel
from preprocessor import textProcess


DATA_DIR = r"D:\tyt\BTLink-main\BTLink-main\dataset_aug"
TEXT_MODEL_PATH = r"D:\tyt\EALink-main\pretrained_models\roberta-large"
CODE_MODEL_PATH = r"D:\tyt\EALink-main\pretrained_models\codebert-base"
OUTPUT_ROOT = r"D:\tyt\BTLink-main\BTLink-main\WithinCode\maml_outputs"

RUN_NAME = None


SOURCE_PROJECTS = ["OODT"]
TARGET_PROJECTS = ["logging-log4net", "Buildr", "Giraph", 'Nutch', ]
TARGET_PROJECT = TARGET_PROJECTS[0]

PROJECT_KEYS = {
    "Flink": "FLINK",
    "Avro": "AVRO",
    "Buildr": "BUILDR",
    "Beam": "BEAM",
    "Giraph": "GIRAPH",
    "ant-ivy": "IVY",
    "logging-log4net": "LOG4NET",
    "Nutch": "NUTCH",
    "OODT": "OODT",
    "Tez": "TEZ",
    "Tika": "TIKA",
    "Isis": "ISIS",
    "keras": "#",
}

ALLOW_SOURCE_TARGET_OVERLAP = True

# -----------------------------
# Meta-task construction
# -----------------------------
# "split_project":
#   each SOURCE project's TRAIN set is stratified and divided into
#   NUM_TASKS_PER_PROJECT fixed task pools.
#
# "project_as_task":
#   each SOURCE project itself is one task.
TASK_MODE = "project_as_task"
NUM_TASKS_PER_PROJECT = 5

# -----------------------------
# Episode construction
# -----------------------------
# K_SHOT is PER CLASS:
# support = K_SHOT positive + K_SHOT negative
K_SHOT = 8

# Q_QUERY is PER CLASS:
# query = Q_QUERY positive + Q_QUERY negative
Q_QUERY = 8

# Number of tasks used in one outer/meta update.
META_BATCH_SIZE = 2

# Number of meta updates per meta epoch.
EPISODES_PER_EPOCH = 100

META_EPOCHS = 10

INNER_LR = 1e-5
INNER_STEPS = 1
META_LR = 1e-5

FIRST_ORDER = True

META_TRAIN_STRATEGY = "all"
META_LAST_K_LAYERS = 1

META_WEIGHT_DECAY = 0.0
META_ADAM_EPSILON = 1e-8
META_MAX_GRAD_NORM = 1.0
META_WARMUP_RATIO = 0.10

# Paired original/LLM-enhanced prediction-ranking loss.
RANK_MARGIN = 0
RANK_WEIGHT = 0.2

ORIGINAL_TRAIN_RATIO_TOTAL = 0.8
TARGET_TRAIN_RATIO_TOTAL_LIST = [0.10, ]
TARGET_TRAIN_RATIO_TOTAL = TARGET_TRAIN_RATIO_TOTAL_LIST[0]

RUN_K_SAMPLE_COMPARE = False
TARGET_K_LIST = [50, ]

TARGET_EPOCHS = 10
TARGET_LR = 1e-5
TARGET_WEIGHT_DECAY = 0.0
TARGET_ADAM_EPSILON = 1e-8
TARGET_MAX_GRAD_NORM = 1.0
TARGET_WARMUP_RATIO = 0.10

TARGET_TRAIN_STRATEGY = "all"
TARGET_LAST_K_LAYERS = 1

FEWSHOT_K_SHOT = 8
FEWSHOT_INNER_LR = 1e-5
FEWSHOT_INNER_STEPS = 1
FEWSHOT_BATCH_SIZE = 8

# 少样本适配更新哪些参数
FEWSHOT_TRAIN_STRATEGY = "all"
FEWSHOT_LAST_K_LAYERS = 1

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

# -----------------------------
# General
# -----------------------------
MAX_SEQ_LENGTH = 512
# SEED = 42
SEEDS = [2014, 2015, 2016, 2017, 2018,]
SEED = SEEDS[0]


RUN_META_TRAIN = False
RUN_TARGET_COMPARE = True
RUN_FEWSHOT_COMPARE = False


EXISTING_META_CHECKPOINT = r"D:\tyt\BTLink-main\BTLink-main\WithinCode\maml_outputs\maml_btlink_Nutch_to_OODT_20261102_220500\checkpoints\meta_best.pt"

RUN_RAW_BASELINE = False
RUN_MAML_INIT = True
RUN_SUPERVISED_INIT = False

NUM_WORKERS = 0

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_gpu_memory(stage):
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)

    print(
        f"\n[GPU MEMORY] {stage}\n"
        f"  Allocated: {allocated:.2f} GB\n"
        f"  Reserved:  {reserved:.2f} GB\n"
        f"  Peak:      {peak:.2f} GB\n"
        f"  Free:      {free_gb:.2f} GB\n"
        f"  Total:     {total_gb:.2f} GB\n",
        flush=True
    )

def make_run_dir() -> Path:
    run_name = RUN_NAME

    if not run_name:
        source_tag = "_".join(SOURCE_PROJECTS)

        run_name = (
            f"maml_btlink_"
            f"{source_tag}_to_{TARGET_PROJECT}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )

    out = Path(OUTPUT_ROOT) / run_name
    out.mkdir(parents=True, exist_ok=False)

    (out / "checkpoints").mkdir(
        parents=True,
        exist_ok=True
    )

    (out / "results").mkdir(
        parents=True,
        exist_ok=True
    )

    return out


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def current_config_dict() -> dict:
    return {
        "DATA_DIR": DATA_DIR,
        "TEXT_MODEL_PATH": TEXT_MODEL_PATH,
        "CODE_MODEL_PATH": CODE_MODEL_PATH,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "SOURCE_PROJECTS": SOURCE_PROJECTS,
        "TARGET_PROJECT": TARGET_PROJECT,
        "TASK_MODE": TASK_MODE,
        "NUM_TASKS_PER_PROJECT": NUM_TASKS_PER_PROJECT,
        "K_SHOT": K_SHOT,
        "Q_QUERY": Q_QUERY,
        "META_BATCH_SIZE": META_BATCH_SIZE,
        "EPISODES_PER_EPOCH": EPISODES_PER_EPOCH,
        "META_EPOCHS": META_EPOCHS,
        "INNER_LR": INNER_LR,
        "INNER_STEPS": INNER_STEPS,
        "META_LR": META_LR,
        "FIRST_ORDER": FIRST_ORDER,
        "META_TRAIN_STRATEGY": META_TRAIN_STRATEGY,
        "META_LAST_K_LAYERS": META_LAST_K_LAYERS,
        "RANK_MARGIN": RANK_MARGIN,
        "RANK_WEIGHT": RANK_WEIGHT,
        "TARGET_TRAIN_RATIO_TOTAL": TARGET_TRAIN_RATIO_TOTAL,
        "ORIGINAL_TRAIN_RATIO_TOTAL": ORIGINAL_TRAIN_RATIO_TOTAL,
        "TARGET_TRAIN_RATIO_TOTAL_LIST": TARGET_TRAIN_RATIO_TOTAL_LIST,
        "RUN_K_SAMPLE_COMPARE":RUN_K_SAMPLE_COMPARE,
        "TARGET_K_LIST":TARGET_K_LIST,
        "TARGET_EPOCHS": TARGET_EPOCHS,
        "TARGET_LR": TARGET_LR,
        "TARGET_TRAIN_STRATEGY": TARGET_TRAIN_STRATEGY,
        "TARGET_LAST_K_LAYERS": TARGET_LAST_K_LAYERS,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "EVAL_BATCH_SIZE": EVAL_BATCH_SIZE,
        "MAX_SEQ_LENGTH": MAX_SEQ_LENGTH,
        "SEED": SEED,
    }


def validate_config() -> None:
    if TASK_MODE not in {"split_project", "project_as_task"}:
        raise ValueError("TASK_MODE must be 'split_project' or 'project_as_task'.")

    for strategy in [META_TRAIN_STRATEGY, TARGET_TRAIN_STRATEGY]:
        if strategy not in {"all", "last_k", "classifier_only"}:
            raise ValueError(
                f"Unknown train strategy: {strategy}. "
                "Use 'all', 'last_k', or 'classifier_only'."
            )

    if not SOURCE_PROJECTS:
        raise ValueError("SOURCE_PROJECTS cannot be empty.")

    unknown = [p for p in SOURCE_PROJECTS + [TARGET_PROJECT] if p not in PROJECT_KEYS]
    if unknown:
        raise ValueError(
            f"Missing project key(s) for {unknown}. Add them to PROJECT_KEYS."
        )

    if (
        TARGET_PROJECT in SOURCE_PROJECTS
        and not ALLOW_SOURCE_TARGET_OVERLAP
    ):
        raise ValueError(
            f"TARGET_PROJECT={TARGET_PROJECT} is also in SOURCE_PROJECTS. "
            "This can leak target information into meta-training. "
            "Use a different source project or explicitly set "
            "ALLOW_SOURCE_TARGET_OVERLAP=True for a diagnostic-only experiment."
        )

    if TARGET_TRAIN_RATIO_TOTAL <= 0:
        raise ValueError("TARGET_TRAIN_RATIO_TOTAL must be > 0.")

    if TARGET_TRAIN_RATIO_TOTAL > ORIGINAL_TRAIN_RATIO_TOTAL:
        raise ValueError(
            "TARGET_TRAIN_RATIO_TOTAL cannot be larger than "
            "ORIGINAL_TRAIN_RATIO_TOTAL when using the existing *_TRAIN.csv."
        )

    if K_SHOT <= 0 or Q_QUERY <= 0:
        raise ValueError("K_SHOT and Q_QUERY must both be positive.")

    if INNER_STEPS <= 0:
        raise ValueError("INNER_STEPS must be >= 1.")

def convert_row_to_ids(
    row: pd.Series,
    tokenizer: RobertaTokenizer,
    commit_type: str,
    project_key: str,
    max_seq_length: int,
) -> List[int]:
    """
    Replicates WithinCode/utils.py::convert_examples_to_features.
    """
    issue_text = textProcess(row["Issue_Text"], project_key)
    commit_text = textProcess(row[commit_type], project_key)

    issue_token = tokenizer.tokenize(issue_text)
    commit_token = tokenizer.tokenize(commit_text)

    if len(issue_token) + len(commit_token) > max_seq_length - 3:
        half = (max_seq_length - 3) / 2

        if len(issue_token) > half and len(commit_token) > half:
            issue_token = issue_token[: int(half)]
            commit_token = commit_token[
                : max_seq_length - 3 - len(issue_token)
            ]
        elif len(issue_token) > half:
            issue_token = issue_token[
                : max_seq_length - 3 - len(commit_token)
            ]
        elif len(commit_token) > half:
            commit_token = commit_token[
                : max_seq_length - 3 - len(issue_token)
            ]

    combined_token = (
        [tokenizer.cls_token]
        + issue_token
        + [tokenizer.sep_token]
        + commit_token
        + [tokenizer.sep_token]
    )

    combined_ids = tokenizer.convert_tokens_to_ids(combined_token)

    if len(combined_ids) < max_seq_length:
        padding_length = max_seq_length - len(combined_ids)
        combined_ids += [tokenizer.pad_token_id] * padding_length

    return combined_ids


REQUIRED_COLUMNS = [
    "label",
    "Issue_KEY",
    "Commit_SHA",
    "Issue_Text",
    "Commit_Text",
    "Commit_Text_Enhanced",
    "Commit_Code",
]


# def read_project_csv(project: str, split: str) -> pd.DataFrame:
#     path = Path(DATA_DIR) / f"{project}_{split}_LLM_DIFF_AUG.csv"
def read_project_csv(project: str, split: str) -> pd.DataFrame:
    data_dir = Path(PROJECT_DATA_DIRS.get(project, DATA_DIR))
    path = data_dir / f"{project}_{split}_LLM_DIFF_AUG.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    base_columns = [
        column for column in REQUIRED_COLUMNS
        if column != "Commit_Text_Enhanced"
    ]
    df = df.dropna(subset=base_columns).reset_index(drop=True)

    enhanced = df["Commit_Text_Enhanced"]
    valid_enhanced = (
        enhanced.notna()
        & enhanced.astype(str).str.strip().ne("")
    )
    df.loc[~valid_enhanced, "Commit_Text_Enhanced"] = df.loc[
        ~valid_enhanced, "Commit_Text"
    ]
    df["label"] = df["label"].astype(int)

    bad_labels = sorted(set(df["label"].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"{path.name} has non-binary labels: {bad_labels}")

    return df


@dataclass
class EncodedData:
    project: str
    project_key: str
    text_ids: torch.Tensor
    original_text_ids: torch.Tensor
    code_ids: torch.Tensor
    labels: torch.Tensor
    paired_mask: torch.Tensor
    issue_keys: List[str]
    commit_shas: List[str]

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def encode_dataframe(
    df: pd.DataFrame,
    project: str,
    project_key: str,
    tokenizer: RobertaTokenizer,
    desc: Optional[str] = None,
) -> EncodedData:
    text_ids = []
    original_text_ids = []
    code_ids = []
    labels = []
    paired_mask = []
    issue_keys = []
    commit_shas = []

    iterator = df.iterrows()
    iterator = tqdm(
        iterator,
        total=len(df),
        desc=desc or f"Encode {project}",
        leave=False,
    )

    for _, row in iterator:
        original_ids = convert_row_to_ids(
            row,
            tokenizer,
            "Commit_Text",
            project_key,
            MAX_SEQ_LENGTH,
        )
        augmented_ids = convert_row_to_ids(
            row,
            tokenizer,
            "Commit_Text_Enhanced",
            project_key,
            MAX_SEQ_LENGTH,
        )
        original_text_ids.append(original_ids)
        text_ids.append(
            augmented_ids
        )
        code_ids.append(
            convert_row_to_ids(
                row,
                tokenizer,
                "Commit_Code",
                project_key,
                MAX_SEQ_LENGTH,
            )
        )
        labels.append(int(row["label"]))
        paired_mask.append(original_ids != augmented_ids)
        issue_keys.append(str(row["Issue_KEY"]))
        commit_shas.append(str(row["Commit_SHA"]))

    return EncodedData(
        project=project,
        project_key=project_key,
        text_ids=torch.tensor(text_ids, dtype=torch.long),
        original_text_ids=torch.tensor(original_text_ids, dtype=torch.long),
        code_ids=torch.tensor(code_ids, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        paired_mask=torch.tensor(paired_mask, dtype=torch.bool),
        issue_keys=issue_keys,
        commit_shas=commit_shas,
    )

def build_btlink_model(device: torch.device) -> BTModel:
    text_config = RobertaConfig.from_pretrained(TEXT_MODEL_PATH)
    text_config.num_labels = 2

    code_config = RobertaConfig.from_pretrained(CODE_MODEL_PATH)
    code_config.num_labels = 2

    text_encoder = AutoModel.from_pretrained(
        TEXT_MODEL_PATH,
        config=text_config,
    )
    code_encoder = AutoModel.from_pretrained(
        CODE_MODEL_PATH,
        config=code_config,
    )

    text_encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    code_encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )


    model = BTModel(
        text_encoder,
        code_encoder,
        text_config.hidden_size,
        code_config.hidden_size,
        2,
    )

    return model.to(device)


def _enable_module(module: Optional[torch.nn.Module]) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = True


def apply_train_strategy(
    model: BTModel,
    strategy: str,
    last_k_layers: int,
) -> None:
    """
    all:
        all parameters trainable

    classifier_only:
        fc, fc1, fc2 trainable

    last_k:
        classifier + both encoder poolers + last K Transformer layers
    """
    if strategy == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    for p in model.parameters():
        p.requires_grad = False

    _enable_module(model.fc)
    _enable_module(model.fc1)
    _enable_module(model.fc2)

    if strategy == "classifier_only":
        return

    if strategy != "last_k":
        raise ValueError(f"Unknown strategy: {strategy}")

    if last_k_layers <= 0:
        raise ValueError("last_k_layers must be >= 1 for strategy='last_k'.")

    for encoder_name in ["textEncoder", "codeEncoder"]:
        encoder = getattr(model, encoder_name)

        # RoBERTa / CodeBERT (RobertaModel)
        layers = encoder.encoder.layer
        if last_k_layers > len(layers):
            raise ValueError(
                f"{encoder_name} has only {len(layers)} layers, "
                f"but last_k_layers={last_k_layers}."
            )

        for layer in layers[-last_k_layers:]:
            _enable_module(layer)

        # Keep the encoder's pooler trainable because BTLink uses output[1].
        if hasattr(encoder, "pooler"):
            _enable_module(encoder.pooler)


def trainable_parameter_names(model: BTModel) -> List[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def print_trainable_summary(model: BTModel, title: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / total if total else 0.0

    print("\n" + "=" * 90)
    print(title)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,} ({ratio:.2f}%)")
    print("=" * 90)


def make_optimizer(
    model: BTModel,
    lr: float,
    weight_decay: float,
    eps: float,
):
    no_decay = ["bias", "LayerNorm.weight"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(groups, lr=lr, eps=eps)

@dataclass
class TaskPool:
    name: str
    project: str
    data: EncodedData
    indices: np.ndarray


def build_task_pools(
    encoded_sources: Dict[str, EncodedData],
    seed: int,
) -> List[TaskPool]:
    rng = np.random.default_rng(seed)
    tasks: List[TaskPool] = []

    for project, data in encoded_sources.items():
        labels = data.labels.numpy()

        if TASK_MODE == "project_as_task":
            indices = np.arange(len(data), dtype=np.int64)
            tasks.append(
                TaskPool( name=f"{project}__project_task", project=project, data=data, indices=indices,)
            )
            continue

        # TASK_MODE == "split_project"
        per_class_chunks: Dict[int, List[np.ndarray]] = {}

        for label in [0, 1]:
            cls_indices = np.where(labels == label)[0].astype(np.int64)
            rng.shuffle(cls_indices)
            per_class_chunks[label] = list(
                np.array_split(cls_indices, NUM_TASKS_PER_PROJECT)
            )

        for task_idx in range(NUM_TASKS_PER_PROJECT):
            pool_indices = np.concatenate( [per_class_chunks[0][task_idx],per_class_chunks[1][task_idx], ] ).astype(np.int64)
            rng.shuffle(pool_indices)

            tasks.append(
                TaskPool( name=f"{project}__task_{task_idx}", project=project, data=data, indices=pool_indices, )
            )

    validate_task_pools(tasks)
    return tasks


def validate_task_pools(tasks: Sequence[TaskPool]) -> None:
    required_per_class = K_SHOT + Q_QUERY

    print("\nMeta task pools:")
    for task in tasks:
        labels = task.data.labels[torch.as_tensor(task.indices)].numpy()
        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())

        print(
            f"  {task.name:<28} "
            f"size={len(task.indices):5d} label0={n0:5d} label1={n1:5d}"
        )

        if n0 < required_per_class or n1 < required_per_class:
            raise ValueError(
                f"Task {task.name} does not have enough examples. "
                f"Need at least K_SHOT + Q_QUERY = {required_per_class} "
                f"examples PER CLASS, but got label0={n0}, label1={n1}. "
                "Reduce K_SHOT/Q_QUERY or NUM_TASKS_PER_PROJECT."
            )


def sample_episode_indices(
    task: TaskPool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    labels_all = task.data.labels.numpy()

    support = []
    query = []

    for label in [0, 1]:
        cls_candidates = task.indices[
            labels_all[task.indices] == label
        ]

        picked = rng.choice(
            cls_candidates,
            size=K_SHOT + Q_QUERY,
            replace=False,
        )

        support.extend(picked[:K_SHOT].tolist())
        query.extend(picked[K_SHOT:].tolist())

    support = np.asarray(support, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)

    rng.shuffle(support)
    rng.shuffle(query)

    # Explicit safety check: no leakage inside an episode.
    if set(support.tolist()) & set(query.tolist()):
        raise RuntimeError("Support and Query overlap in the same episode.")

    return support, query


def batch_from_indices(
    data: EncodedData,
    indices: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return (
        data.text_ids[idx].to(device, non_blocking=True),
        data.code_ids[idx].to(device, non_blocking=True),
        data.labels[idx].to(device, non_blocking=True),
        data.original_text_ids[idx].to(device, non_blocking=True),
        data.paired_mask[idx].to(device, non_blocking=True),
    )


def functional_btlink_forward(
    model: BTModel,
    params: OrderedDict,
    buffers: OrderedDict,
    text_ids: torch.Tensor,
    code_ids: torch.Tensor,
    labels: torch.Tensor,
    return_logits: bool = False,
):
    return functional_call(
        model,
        (params, buffers),
        args=(text_ids, code_ids, labels),
        kwargs={"return_logits": return_logits},
        strict=False,
    )


def combine_paired_losses(
    original_loss: torch.Tensor,
    augmented_loss: torch.Tensor,
    original_logits: torch.Tensor,
    augmented_logits: torch.Tensor,
    labels: torch.Tensor,
    paired_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sign = labels.float() * 2.0 - 1.0
    original_score = sign * (original_logits[:, 1] - original_logits[:, 0])
    augmented_score = sign * (augmented_logits[:, 1] - augmented_logits[:, 0])
    rank_terms = torch.relu(
        RANK_MARGIN + original_score - augmented_score
    )

    if bool(paired_mask.any().item()):
        rank_loss = rank_terms[paired_mask].mean()
    else:
        rank_loss = rank_terms.sum() * 0.0

    total_loss = (
        original_loss
        + augmented_loss
        + RANK_WEIGHT * rank_loss
    )
    return total_loss, rank_loss


def functional_paired_loss(
    model: BTModel,
    params: OrderedDict,
    buffers: OrderedDict,
    augmented_text_ids: torch.Tensor,
    code_ids: torch.Tensor,
    labels: torch.Tensor,
    original_text_ids: torch.Tensor,
    paired_mask: torch.Tensor,
) -> torch.Tensor:
    augmented_loss, augmented_logits = functional_btlink_forward(
        model, params, buffers,
        augmented_text_ids, code_ids, labels,
        return_logits=True,
    )
    original_loss, original_logits = functional_btlink_forward(
        model, params, buffers,
        original_text_ids, code_ids, labels,
        return_logits=True,
    )
    total_loss, _ = combine_paired_losses(
        original_loss,
        augmented_loss,
        original_logits,
        augmented_logits,
        labels,
        paired_mask,
    )
    return total_loss


def model_paired_loss(
    model: BTModel,
    augmented_text_ids: torch.Tensor,
    code_ids: torch.Tensor,
    labels: torch.Tensor,
    original_text_ids: torch.Tensor,
    paired_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    augmented_loss, augmented_logits = model(
        augmented_text_ids, code_ids, labels, return_logits=True
    )
    original_loss, original_logits = model(
        original_text_ids, code_ids, labels, return_logits=True
    )
    total_loss, rank_loss = combine_paired_losses(
        original_loss,
        augmented_loss,
        original_logits,
        augmented_logits,
        labels,
        paired_mask,
    )
    return total_loss, original_loss, augmented_loss, rank_loss


def adapt_on_support(
    model: BTModel,
    support_batch: Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
) -> Tuple[OrderedDict, OrderedDict, float]:
    text_ids, code_ids, labels, original_text_ids, paired_mask = support_batch

    fast_params = OrderedDict(
        (name, param) for name, param in model.named_parameters()
    )
    buffers = OrderedDict(
        (name, buf) for name, buf in model.named_buffers()
    )

    train_names = [ name for name, param in model.named_parameters() if param.requires_grad]

    if not train_names:
        raise RuntimeError("No trainable parameters for MAML.")

    last_support_loss = None

    for _ in range(INNER_STEPS):
        support_loss = functional_paired_loss(
            model,
            fast_params,
            buffers,
            text_ids,
            code_ids,
            labels,
            original_text_ids,
            paired_mask,
        )

        current_train_tensors = [fast_params[name] for name in train_names]

        grads = torch.autograd.grad(
            support_loss,
            current_train_tensors,
            create_graph=not FIRST_ORDER,
            retain_graph=not FIRST_ORDER,
            allow_unused=False,
        )

        updated = OrderedDict(fast_params)
        for name, grad in zip(train_names, grads):
            updated[name] = fast_params[name] - INNER_LR * grad

        fast_params = updated
        last_support_loss = float(support_loss.detach().cpu())

    return fast_params, buffers, last_support_loss


def save_meta_checkpoint(
    path: Path,
    model: BTModel,
    epoch: int,
    meta_query_loss: float,
) -> None:

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "meta_query_loss": meta_query_loss,
        "config": current_config_dict(),
    }

    torch.save(
        checkpoint,
        path,
    )


def meta_train(
    tokenizer: RobertaTokenizer,
    device: torch.device,
    run_dir: Path,
) -> Path:
    print("\n" + "#" * 100)
    print("META TRAIN")
    print("#" * 100)

    # Encode each source project ONCE.
    encoded_sources: Dict[str, EncodedData] = {}

    for project in SOURCE_PROJECTS:
        df = read_project_csv(project, "TRAIN")
        key = PROJECT_KEYS[project]

        print(
            f"\nSource {project}: rows={len(df)}, "
            f"labels={df['label'].value_counts().to_dict()}"
        )

        encoded_sources[project] = encode_dataframe( df,project,key,tokenizer, desc=f"Encode source {project}",)

    tasks = build_task_pools(encoded_sources, SEED)

    set_global_seed(SEED)
    model = build_btlink_model(device)
    apply_train_strategy(model,META_TRAIN_STRATEGY,META_LAST_K_LAYERS,)
    print_trainable_summary( model, f"MAML strategy = {META_TRAIN_STRATEGY}, " f"last_k = {META_LAST_K_LAYERS}",)

    if META_TRAIN_STRATEGY == "all":
        print(
            "\n[Warning] Full-model MAML on RoBERTa-large + CodeBERT "
            "is very memory intensive. If CUDA OOM occurs, switch to "
            "META_TRAIN_STRATEGY='last_k'."
        )

    meta_optimizer = make_optimizer(model, lr=META_LR,weight_decay=META_WEIGHT_DECAY,eps=META_ADAM_EPSILON,)

    total_outer_steps = META_EPOCHS * EPISODES_PER_EPOCH
    warmup_steps = int(total_outer_steps * META_WARMUP_RATIO)

    meta_scheduler = get_linear_schedule_with_warmup( meta_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_outer_steps,)

    rng = np.random.default_rng(SEED + 1000)

    history = []
    best_meta_query_loss = float("inf")

    # best_path = run_dir / "checkpoints" / "meta_best.pt"
    # last_path = run_dir / "checkpoints" / "meta_last.pt"
    best_path = (run_dir / "checkpoints" / "meta_best.pt")

    for epoch in range(META_EPOCHS):
        model.train()

        epoch_query_losses = []
        epoch_support_losses = []

        bar = tqdm( range(EPISODES_PER_EPOCH), desc=f"Meta epoch {epoch + 1}/{META_EPOCHS}", )

        for outer_step in bar:
            meta_optimizer.zero_grad(set_to_none=True)

            trace_mem = (epoch == 0 and outer_step < 3)

            if trace_mem:
                torch.cuda.reset_peak_memory_stats()
                show_gpu_memory(
                    f"Epoch {epoch + 1}, Outer Step {outer_step + 1}: START"
                )


            task_query_losses = []
            task_support_losses = []

            selected_tasks = [tasks[int(rng.integers(0, len(tasks)))] for _ in range(META_BATCH_SIZE) ]

            for task in selected_tasks:
                support_idx, query_idx = sample_episode_indices(task, rng)

                support_batch = batch_from_indices( task.data,support_idx,device, )
                query_batch = batch_from_indices( task.data, query_idx, device, )

                if trace_mem:
                    show_gpu_memory(
                        f"Step {outer_step + 1}: BEFORE SUPPORT"
                    )

                fast_params, buffers, support_loss_value = adapt_on_support(
                    model, support_batch,
                )

                if trace_mem:
                    show_gpu_memory(
                        f"Step {outer_step + 1}: AFTER SUPPORT"
                    )

                q_text, q_code, q_labels, q_original_text, q_paired_mask = query_batch
                query_loss = functional_paired_loss(
                    model,
                    fast_params,
                    buffers,
                    q_text,
                    q_code,
                    q_labels,
                    q_original_text,
                    q_paired_mask,
                )
                if trace_mem:
                    show_gpu_memory(
                        f"Step {outer_step + 1}: AFTER QUERY"
                    )

                (query_loss / len(selected_tasks)).backward()

                task_query_losses.append(
                    float(query_loss.detach().cpu())
                )
                task_support_losses.append(support_loss_value)


                del query_loss
                del fast_params, buffers
                del support_batch, query_batch
                del q_text, q_code, q_labels
                del q_original_text, q_paired_mask

            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],META_MAX_GRAD_NORM, )

            meta_optimizer.step()
            meta_scheduler.step()

            q_loss_value = float(np.mean(task_query_losses))
            s_loss_value = float(np.mean(task_support_losses))

            epoch_query_losses.append(q_loss_value)
            epoch_support_losses.append(s_loss_value)

            bar.set_postfix(
                support=f"{s_loss_value:.4f}",
                query=f"{q_loss_value:.4f}",
                lr=f"{meta_scheduler.get_last_lr()[0]:.2e}",
            )

            # Free references early; helpful for the large dual encoder.
            del task_query_losses #, meta_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if trace_mem:
                show_gpu_memory(
                    f"Step {outer_step + 1}: END"
                )

        mean_support = float(np.mean(epoch_support_losses))
        mean_query = float(np.mean(epoch_query_losses))

        history.append(
            {
                "epoch": epoch + 1,
                "mean_support_loss": mean_support,
                "mean_query_loss": mean_query,
                "k_shot_per_class": K_SHOT,
                "q_query_per_class": Q_QUERY,
                "meta_batch_size": META_BATCH_SIZE,
                "episodes_per_epoch": EPISODES_PER_EPOCH,
                "inner_steps": INNER_STEPS,
                "inner_lr": INNER_LR,
                "meta_lr": META_LR,
                "first_order": FIRST_ORDER,
                "task_mode": TASK_MODE,
                "meta_train_strategy": META_TRAIN_STRATEGY,
                "meta_last_k_layers": META_LAST_K_LAYERS,
            }
        )

        pd.DataFrame(history).to_csv(
            run_dir / "results" / "meta_history.csv",
            index=False,
        )

        print(
            f"\nMeta epoch {epoch + 1}: "
            f"support_loss={mean_support:.6f}, "
            f"query_loss={mean_query:.6f}"
        )

        if mean_query <= best_meta_query_loss:
            best_meta_query_loss = mean_query
            save_meta_checkpoint(
                best_path,
                model,
                epoch + 1,
                mean_query,
            )
            print(
                f"Saved best meta checkpoint: {best_path} "
                f"(query loss={best_meta_query_loss:.6f})"
            )

    print(f"\nMeta training complete. Best checkpoint: {best_path}")
    return best_path


def stratified_take_target_train(
    train_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    frac = TARGET_TRAIN_RATIO_TOTAL / ORIGINAL_TRAIN_RATIO_TOTAL

    if frac > 1.0 + 1e-12:
        raise ValueError("Requested target train ratio exceeds existing TRAIN pool.")

    if abs(frac - 1.0) < 1e-12:
        sampled = train_df.copy()
    else:
        sampled = (
            train_df.groupby("label", group_keys=False)
            .sample(frac=frac, random_state=seed)
            .reset_index(drop=True)
        )

    return sampled


def balanced_under_sample(
    df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    counts = df["label"].value_counts().to_dict()

    if 0 not in counts or 1 not in counts:
        raise ValueError(
            f"Target train pool must contain both classes, got {counts}."
        )

    n = min(counts[0], counts[1])

    part0 = df[df["label"] == 0].sample(n=n, random_state=seed)
    part1 = df[df["label"] == 1].sample(n=n, random_state=seed)

    out = pd.concat([part0, part1], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


class EncodedDataset(Dataset):
    def __init__(self, data: EncodedData):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data.text_ids[idx],
            self.data.code_ids[idx],
            self.data.labels[idx],
            idx,
        )


class PairedEncodedDataset(Dataset):
    def __init__(self, data: EncodedData):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data.text_ids[idx],
            self.data.code_ids[idx],
            self.data.labels[idx],
            idx,
            self.data.original_text_ids[idx],
            self.data.paired_mask[idx],
        )


def prepare_target_data(
    tokenizer: RobertaTokenizer,
) -> Tuple[EncodedData, EncodedData, EncodedData, dict]:
    train_full = read_project_csv(TARGET_PROJECT, "TRAIN")
    dev_df = read_project_csv(TARGET_PROJECT, "DEV")
    test_df = read_project_csv(TARGET_PROJECT, "TEST")

    target_subset = stratified_take_target_train(train_full, SEED)
    target_balanced = balanced_under_sample(target_subset, SEED)

    print("\n" + "=" * 100)
    print("TARGET DATA")
    print("=" * 100)
    print(
        f"Original {SOURCE_PROJECTS}_{TARGET_PROJECT}_TRAIN.csv: {len(train_full)} | "
        f"{train_full['label'].value_counts().to_dict()}"
    )
    print(
        f"Target train pool before balancing "
        f"(total ratio={TARGET_TRAIN_RATIO_TOTAL}): "
        f"{len(target_subset)} | "
        f"{target_subset['label'].value_counts().to_dict()}"
    )
    print(
        f"Target train after balancing: {len(target_balanced)} | "
        f"{target_balanced['label'].value_counts().to_dict()}"
    )
    print(
        f"DEV full: {len(dev_df)} | "
        f"{dev_df['label'].value_counts().to_dict()}"
    )
    print(
        f"TEST full: {len(test_df)} | "
        f"{test_df['label'].value_counts().to_dict()}"
    )

    key = PROJECT_KEYS[TARGET_PROJECT]

    train_encoded = encode_dataframe(
        target_balanced,
        TARGET_PROJECT,
        key,
        tokenizer,
        desc=f"Encode target train {TARGET_PROJECT}",
    )
    dev_encoded = encode_dataframe(
        dev_df,
        TARGET_PROJECT,
        key,
        tokenizer,
        desc=f"Encode target dev {TARGET_PROJECT}",
    )
    test_encoded = encode_dataframe(
        test_df,
        TARGET_PROJECT,
        key,
        tokenizer,
        desc=f"Encode target test {TARGET_PROJECT}",
    )

    info = {
        "original_train_size": len(train_full),
        "original_train_label_counts": train_full["label"].value_counts().to_dict(),
        "selected_train_pool_size": len(target_subset),
        "selected_train_pool_label_counts": target_subset["label"].value_counts().to_dict(),
        "balanced_train_size": len(target_balanced),
        "balanced_train_label_counts": target_balanced["label"].value_counts().to_dict(),
        "dev_size": len(dev_df),
        "dev_label_counts": dev_df["label"].value_counts().to_dict(),
        "test_size": len(test_df),
        "test_label_counts": test_df["label"].value_counts().to_dict(),
    }

    return train_encoded, dev_encoded, test_encoded, info

@torch.no_grad()
def evaluate_encoded(
    model: BTModel,
    data: EncodedData,
    device: torch.device,
    batch_size: int,
) -> Tuple[dict, pd.DataFrame]:
    model.eval()

    loader = DataLoader(
        EncodedDataset(data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    losses = []
    probs_list = []
    labels_list = []
    row_indices = []

    for text_ids, code_ids, labels, idx in loader:
        text_ids = text_ids.to(device, non_blocking=True)
        code_ids = code_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss, probs = model(text_ids, code_ids, labels)

        losses.append(float(loss.detach().cpu()))
        probs_list.append(probs.detach().cpu())
        labels_list.append(labels.detach().cpu())
        row_indices.extend(idx.tolist())

    probs = torch.cat(probs_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    preds = probs.argmax(axis=1)

    accuracy = float(np.mean(labels == preds))
    precision = float(
        precision_score(labels, preds, zero_division=0)
    )
    recall = float(
        recall_score(labels, preds, zero_division=0)
    )
    f1 = float(
        f1_score(labels, preds, zero_division=0)
    )
    mcc = float(matthews_corrcoef(labels, preds))

    # Keep BTLink's original metric behavior for comparability:
    # original run_pre.py computes AUC and Brier from HARD predictions.
    try:
        auc_hard = float(roc_auc_score(labels, preds))
    except ValueError:
        auc_hard = float("nan")

    brier_hard = float(brier_score_loss(labels, preds))

    tn, fp, fn, tp = confusion_matrix(
        labels,
        preds,
        labels=[0, 1],
    ).ravel()

    pf = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")

    metrics = {
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
        "eval_acc": accuracy,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_auc": auc_hard,
        "eval_mcc": mcc,
        "eval_brier": brier_hard,
        "eval_pf": pf,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    pred_df = pd.DataFrame(
        {
            "Issue_Key": [data.issue_keys[i] for i in row_indices],
            "Commit_SHA": [data.commit_shas[i] for i in row_indices],
            "prob_0": probs[:, 0],
            "prob_1": probs[:, 1],
            "preds": preds,
            "labels": labels,
        }
    )

    return metrics, pred_df

def load_checkpoint_initialization(
    model: BTModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

        checkpoint_type = "wrapped_checkpoint"

    elif (
        isinstance(checkpoint, dict)
        and checkpoint
        and all(
            torch.is_tensor(v)
            for v in checkpoint.values()
        )
    ):

        state_dict = checkpoint

        checkpoint_type = "state_dict"

    elif isinstance(
        checkpoint,
        torch.nn.Module,
    ):

        state_dict = checkpoint.state_dict()

        checkpoint_type = "full_model"

    else:

        raise ValueError(
            f"Unsupported checkpoint format: "
            f"{checkpoint_path}"
        )

    if any(
        key.startswith("module.")
        for key in state_dict.keys()
    ):

        state_dict = {
            key.removeprefix("module."): value
            for key, value
            in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        f"Loaded checkpoint "
        f"type={checkpoint_type}: "
        f"{checkpoint_path}"
    )

    return {
        "checkpoint_type":
            checkpoint_type,
        "checkpoint_path":
            str(checkpoint_path),
    }

def sample_exact_k_target(
    train_df: pd.DataFrame,
    k: int,
    seed: int,
) -> pd.DataFrame:
    pos_df = (
        train_df[
            train_df["label"] == 1
        ]
        .reset_index(drop=True)
    )

    neg_df = (
        train_df[
            train_df["label"] == 0
        ]
        .reset_index(drop=True)
    )

    n_pos = (k + 1) // 2
    n_neg = k // 2

    if len(pos_df) < n_pos:
        raise ValueError(
            f"Not enough positive samples: "
            f"need={n_pos}, have={len(pos_df)}"
        )

    if len(neg_df) < n_neg:
        raise ValueError(
            f"Not enough negative samples: "
            f"need={n_neg}, have={len(neg_df)}"
        )

    pos_rng = np.random.default_rng(
        seed + 101
    )

    neg_rng = np.random.default_rng(
        seed + 202
    )

    pos_order = np.arange(
        len(pos_df)
    )

    neg_order = np.arange(
        len(neg_df)
    )

    pos_rng.shuffle(
        pos_order
    )

    neg_rng.shuffle(
        neg_order
    )

    selected_pos = pos_df.iloc[
        pos_order[:n_pos]
    ]

    selected_neg = neg_df.iloc[
        neg_order[:n_neg]
    ]

    sampled = pd.concat(
        [
            selected_pos,
            selected_neg,
        ],
        ignore_index=True,
    )

    sampled = sampled.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return sampled

def prepare_target_k_data(
    tokenizer: RobertaTokenizer,
    k: int,
    seed: int,
):
    train_full = read_project_csv(
        TARGET_PROJECT,
        "TRAIN",
    )

    dev_df = read_project_csv(
        TARGET_PROJECT,
        "DEV",
    )

    test_df = read_project_csv(
        TARGET_PROJECT,
        "TEST",
    )

    target_k_df = sample_exact_k_target(
        train_full,
        k,
        seed,
    )

    print("\n" + "=" * 100)

    print(
        f"TARGET K-SAMPLE DATA | "
        f"project={TARGET_PROJECT} | "
        f"K={k} | "
        f"seed={seed}"
    )

    print("=" * 100)

    print(
        f"Original TRAIN: "
        f"{len(train_full)} | "
        f"{train_full['label'].value_counts().to_dict()}"
    )

    print(
        f"K samples: "
        f"{len(target_k_df)} | "
        f"{target_k_df['label'].value_counts().to_dict()}"
    )

    print(
        f"DEV: "
        f"{len(dev_df)} | "
        f"{dev_df['label'].value_counts().to_dict()}"
    )

    print(
        f"TEST: "
        f"{len(test_df)} | "
        f"{test_df['label'].value_counts().to_dict()}"
    )

    key = PROJECT_KEYS[
        TARGET_PROJECT
    ]

    train_encoded = encode_dataframe(
        target_k_df,
        TARGET_PROJECT,
        key,
        tokenizer,
        desc=(
            f"Encode target "
            f"{TARGET_PROJECT} K={k}"
        ),
    )

    dev_encoded = encode_dataframe(
        dev_df,
        TARGET_PROJECT,
        key,
        tokenizer,
        desc=f"Encode DEV {TARGET_PROJECT}",
    )

    test_encoded = encode_dataframe(
        test_df,
        TARGET_PROJECT,
        key,
        tokenizer,
        desc=f"Encode TEST {TARGET_PROJECT}",
    )

    info = {
        "adaptation_mode":
            "k_sample_supervised",

        "target_project":
            TARGET_PROJECT,

        "seed":
            seed,

        "target_k":
            k,

        "actual_target_train_size":
            len(target_k_df),

        "target_train_label_counts":
            target_k_df[
                "label"
            ].value_counts().to_dict(),

        "dev_size":
            len(dev_df),

        "test_size":
            len(test_df),
    }

    return (
        train_encoded,
        dev_encoded,
        test_encoded,
        info,
    )


def target_finetune(
    init_name: str,
    train_data: EncodedData,
    dev_data: EncodedData,
    test_data: EncodedData,
    device: torch.device,
    run_dir: Path,
    meta_checkpoint_path=None,
    supervised_checkpoint_path=None,

    adaptation_mode: str = "ratio",
    target_k: Optional[int] = None,
):

    print("\n" + "#" * 100)
    print(f"TARGET FINETUNE | initialization = {init_name}")
    print("#" * 100)

    # Reset seed before EACH comparison condition.
    set_global_seed(SEED)

    model = build_btlink_model(device)

    loaded_meta_info = None
    if init_name == "maml":

        if meta_checkpoint_path is None:
            raise ValueError(
                "MAML initialization requires a checkpoint."
            )

        load_checkpoint_initialization(
            model,
            meta_checkpoint_path,
            device,
        )

        print(
            f"Loaded MAML initialization: "
            f"{meta_checkpoint_path}"
        )


    elif init_name == "supervised":

        if supervised_checkpoint_path is None:
            raise ValueError(
                "Supervised initialization requires a checkpoint."
            )

        load_checkpoint_initialization(
            model,
            supervised_checkpoint_path,
            device,
        )

        print(
            f"Loaded supervised initialization: "
            f"{supervised_checkpoint_path}"
        )


    elif init_name == "raw":

        print(
            "Using raw pretrained "
            "RoBERTa + CodeBERT initialization."
        )


    else:

        raise ValueError(
            f"Unknown init_name: {init_name}"
        )

    apply_train_strategy(
        model,
        TARGET_TRAIN_STRATEGY,
        TARGET_LAST_K_LAYERS,
    )

    print_trainable_summary(
        model,
        f"Target strategy = {TARGET_TRAIN_STRATEGY}, "
        f"last_k = {TARGET_LAST_K_LAYERS}",
    )

    train_dataset = PairedEncodedDataset(train_data)

    # Explicit generator ensures the same shuffle order in raw vs MAML.
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = make_optimizer(
        model,
        lr=TARGET_LR,
        weight_decay=TARGET_WEIGHT_DECAY,
        eps=TARGET_ADAM_EPSILON,
    )

    total_steps = len(train_loader) * TARGET_EPOCHS
    warmup_steps = int(total_steps * TARGET_WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_dev_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    history = []

    best_ckpt = (
        run_dir
        / "checkpoints"
        / f"{TARGET_PROJECT}_{init_name}_target_best.pt"
    )

    for epoch in range(TARGET_EPOCHS):
        model.train()
        epoch_losses = []

        bar = tqdm(
            train_loader,
            desc=f"{init_name} target epoch {epoch + 1}/{TARGET_EPOCHS}",
        )

        for text_ids, code_ids, labels, _, original_text_ids, paired_mask in bar:
            text_ids = text_ids.to(device, non_blocking=True)
            code_ids = code_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            original_text_ids = original_text_ids.to(device, non_blocking=True)
            paired_mask = paired_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            loss, _, _, _ = model_paired_loss(
                model,
                text_ids,
                code_ids,
                labels,
                original_text_ids,
                paired_mask,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                TARGET_MAX_GRAD_NORM,
            )

            optimizer.step()
            scheduler.step()

            epoch_losses.append(float(loss.detach().cpu()))

            bar.set_postfix(
                loss=f"{np.mean(epoch_losses):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        dev_metrics, _ = evaluate_encoded(
            model,
            dev_data,
            device,
            EVAL_BATCH_SIZE,
        )

        row = {
            "initialization": init_name,
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
        }
        row.update({f"dev_{k}": v for k, v in dev_metrics.items()})
        history.append(row)

        print(
            f"\n{init_name} epoch {epoch + 1}: "
            f"train_loss={np.mean(epoch_losses):.6f}, "
            f"DEV F1={dev_metrics['eval_f1']:.4f}, "
            f"Precision={dev_metrics['eval_precision']:.4f}, "
            f"Recall={dev_metrics['eval_recall']:.4f}"
        )

        # Same selection principle as BTLink: DEV F1.
        if dev_metrics["eval_f1"] >= best_dev_f1:
            best_dev_f1 = dev_metrics["eval_f1"]
            best_epoch = epoch + 1

            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

            print(
                f"Updated best target model in RAM "
                f"(epoch={best_epoch}, "
                f"DEV F1={best_dev_f1:.4f})"
            )

        pd.DataFrame(history).to_csv(
            run_dir
            / "results"
            / f"{SOURCE_PROJECTS}_{TARGET_PROJECT}_{init_name}_target_history.csv",
            index=False,
        )

    # Load BEST target checkpoint, then test once.
    if best_state_dict is None:
        raise RuntimeError("No best target model was recorded.")
    model.load_state_dict(best_state_dict,strict=True,)
    del best_state_dict

    test_metrics, pred_df = evaluate_encoded(
        model,
        test_data,
        device,
        EVAL_BATCH_SIZE,
    )

    pred_path = (
        run_dir
        / "results"
        / f"{SOURCE_PROJECTS}_{TARGET_PROJECT}_{init_name}_TEST_predictions.csv"
    )
    pred_df.to_csv(pred_path, index=False)

    result = {
        "experiment_type": "supervised",
        "initialization": init_name,
        "target_project": TARGET_PROJECT,
        "source_projects": "|".join(SOURCE_PROJECTS),
        "task_mode": TASK_MODE,
        "num_tasks_per_project": NUM_TASKS_PER_PROJECT
        if TASK_MODE == "split_project"
        else 1,
        "seed": SEED,
        "target_train_ratio_total":
            TARGET_TRAIN_RATIO_TOTAL
            if adaptation_mode == "ratio"
            else None,
        "actual_target_train_size": len(train_data),
        "adaptation_mode": adaptation_mode,
        "target_k": target_k,
        "dev_size": len(dev_data),
        "test_size": len(test_data),
        "target_train_strategy": TARGET_TRAIN_STRATEGY,
        "target_last_k_layers": TARGET_LAST_K_LAYERS,
        "target_epochs": TARGET_EPOCHS,
        "target_lr": TARGET_LR,
        "best_target_epoch": best_epoch,
        "best_dev_f1": best_dev_f1,
        "meta_checkpoint": str(meta_checkpoint_path)
            if meta_checkpoint_path
            else "",
        "supervised_checkpoint":
            str(supervised_checkpoint_path)
            if supervised_checkpoint_path
            else "",

    }
    result.update(test_metrics)

    print("\nFinal TEST:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def sample_target_fewshot_support(
    train_data: EncodedData,
    seed: int,
) -> EncodedData:

    labels = train_data.labels.numpy()

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    rng = np.random.default_rng(seed)

    pos_pick = rng.choice(
        pos_indices,
        size=FEWSHOT_K_SHOT,
        replace=False,
    )

    neg_pick = rng.choice(
        neg_indices,
        size=FEWSHOT_K_SHOT,
        replace=False,
    )

    support_indices = np.concatenate([
        pos_pick,
        neg_pick,
    ])

    rng.shuffle(support_indices)

    idx = torch.as_tensor(
        support_indices,
        dtype=torch.long,
    )

    return EncodedData(
        project=train_data.project,
        project_key=train_data.project_key,

        text_ids=train_data.text_ids[idx],
        original_text_ids=train_data.original_text_ids[idx],
        code_ids=train_data.code_ids[idx],
        labels=train_data.labels[idx],
        paired_mask=train_data.paired_mask[idx],

        issue_keys=[
            train_data.issue_keys[i]
            for i in support_indices
        ],

        commit_shas=[
            train_data.commit_shas[i]
            for i in support_indices
        ],
    )

def target_fewshot_adapt(
    init_name: str,
    support_data: EncodedData,
    test_data: EncodedData,
    device: torch.device,
    meta_checkpoint_path: Optional[Path] = None,
    supervised_checkpoint_path: Optional[Path] = None,
) -> dict:

    print("\n" + "=" * 100)
    print(
        f"TARGET FEW-SHOT | "
        f"initialization={init_name}"
    )
    print("=" * 100)

    set_global_seed(SEED)

    model = build_btlink_model(device)

    if init_name == "maml":

        if meta_checkpoint_path is None:
            raise ValueError(
                "MAML initialization requires checkpoint."
            )

        load_checkpoint_initialization(
            model,
            meta_checkpoint_path,
            device,
        )

        print(
            f"Loaded MAML initialization: "
            f"{meta_checkpoint_path}"
        )

    elif init_name == "raw":

        print(
            "Using raw pretrained "
            "RoBERTa + CodeBERT initialization."
        )

    else:
        raise ValueError(init_name)

    apply_train_strategy(
        model,
        FEWSHOT_TRAIN_STRATEGY,
        FEWSHOT_LAST_K_LAYERS,
    )

    model.train()

    support_dataset = PairedEncodedDataset(support_data)

    support_loader = DataLoader(
        support_dataset,
        batch_size=FEWSHOT_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    support_size = len(support_data)
    support_paired_size = int(support_data.paired_mask.sum().item())

    for step in range(FEWSHOT_INNER_STEPS):

        model.zero_grad(set_to_none=True)

        total_classification_loss = 0.0
        total_rank_loss = 0.0

        for text_ids, code_ids, labels, _, original_text_ids, paired_mask in support_loader:
            text_ids = text_ids.to(device, non_blocking=True, )

            code_ids = code_ids.to(device, non_blocking=True, )

            labels = labels.to(device,non_blocking=True,)
            original_text_ids = original_text_ids.to(device, non_blocking=True)
            paired_mask = paired_mask.to(device, non_blocking=True)

            _, original_loss, augmented_loss, rank_loss = model_paired_loss(
                model,
                text_ids,
                code_ids,
                labels,
                original_text_ids,
                paired_mask,
            )

            batch_size = labels.size(0)
            batch_paired_size = int(paired_mask.sum().item())

            scaled_loss = (
                (original_loss + augmented_loss)
                * batch_size
                / support_size
            )
            if support_paired_size > 0 and batch_paired_size > 0:
                scaled_loss = scaled_loss + (
                    RANK_WEIGHT
                    * rank_loss
                    * batch_paired_size
                    / support_paired_size
                )

            scaled_loss.backward()

            total_classification_loss += (
                (original_loss.item() + augmented_loss.item())
                * batch_size
            )
            total_rank_loss += rank_loss.item() * batch_paired_size

        with torch.no_grad():
            for param in trainable_params:
                if param.grad is not None:
                    param -= ( FEWSHOT_INNER_LR * param.grad )

        mean_support_loss = total_classification_loss / support_size
        if support_paired_size > 0:
            mean_support_loss += (
                RANK_WEIGHT * total_rank_loss / support_paired_size
            )

        print(
            f"Few-shot step {step + 1}: "
            f"support_loss="
            f"{mean_support_loss:.6f}"
        )

    test_metrics, _ = evaluate_encoded(model,test_data,device, EVAL_BATCH_SIZE, )

    result = {
        "experiment_type": "fewshot",
        "initialization": init_name,
        "source_projects":"|".join(SOURCE_PROJECTS),
        "target_project":TARGET_PROJECT,
        "seed":SEED,
        "target_train_ratio_total":TARGET_TRAIN_RATIO_TOTAL,
        "support_size":len(support_data.labels),
        "k_shot_per_class":FEWSHOT_K_SHOT,
        "fewshot_inner_steps":FEWSHOT_INNER_STEPS,
        "fewshot_inner_lr":FEWSHOT_INNER_LR,
        "fewshot_batch_size":FEWSHOT_BATCH_SIZE,
        "fewshot_train_strategy":FEWSHOT_TRAIN_STRATEGY,
        "fewshot_last_k_layers": FEWSHOT_LAST_K_LAYERS,
        "test_size": len(test_data),
    }

    result.update(test_metrics)

    print("\nFew-shot TEST:")
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def append_to_master_results(
    results: List[dict],
    run_dir: Path,
) -> Path:
    if not results:
        return Path(OUTPUT_ROOT) / "all_experiments_summary.csv"

    master_path = (
        Path(OUTPUT_ROOT)
        / "all_experiments_summary.csv"
    )

    new_df = pd.DataFrame(results)

    new_df.insert(
        0,
        "run_name",
        run_dir.name,
    )

    new_df.insert(
        1,
        "run_time",
        time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    if master_path.exists():

        old_df = pd.read_csv(
            master_path
        )

        combined_df = pd.concat(
            [
                old_df,
                new_df,
            ],
            ignore_index=True,
            sort=False,
        )

    else:

        combined_df = new_df

    important_cols = [
        "run_name",
        "run_time",

        "experiment_type",
        "adaptation_mode",

        "source_projects",
        "target_project",

        "seed",

        "target_train_ratio_total",
        "target_k",
        "actual_target_train_size",

        "initialization",

        "support_size",
        "k_shot_per_class",

        "target_train_strategy",
        "target_last_k_layers",

        "target_epochs",
        "target_lr",

        "best_target_epoch",
        "best_dev_f1",

        "eval_acc",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_mcc",
        "eval_auc",
        "eval_brier",
        "eval_pf",

        "tn",
        "fp",
        "fn",
        "tp",
    ]

    first_cols = [
        c for c in important_cols
        if c in combined_df.columns
    ]

    other_cols = [
        c for c in combined_df.columns
        if c not in first_cols
    ]

    combined_df = combined_df[
        first_cols + other_cols
    ]

    combined_df.to_csv(
        master_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nUpdated master result table:"f"\n{master_path}")
    print( f"New rows added: {len(new_df)}")
    print(f"Total rows: {len(combined_df)}")

    return master_path


def main():

    global TARGET_PROJECT
    global SEED
    global TARGET_TRAIN_RATIO_TOTAL

    device = get_device()

    print("=" * 100)
    print("BTLink Multi-Target / Multi-Seed Experiment")
    print("=" * 100)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    run_dir = make_run_dir()

    print(
        f"\nMain experiment directory:\n"
        f"{run_dir}"
    )

    tokenizer = RobertaTokenizer.from_pretrained(
        TEXT_MODEL_PATH
    )

    meta_checkpoint_path: Optional[Path] = None

    if RUN_META_TRAIN:

        SEED = SEEDS[0]

        set_global_seed(SEED)

        print("\n" + "=" * 100)

        print(
            f"Meta-training once | "
            f"source={SOURCE_PROJECTS} | "
            f"seed={SEED}"
        )

        print("=" * 100)

        meta_checkpoint_path = meta_train(
            tokenizer,
            device,
            run_dir,
        )

    elif RUN_MAML_INIT:

        if not EXISTING_META_CHECKPOINT:

            raise ValueError(
                "RUN_MAML_INIT=True, "
                "but EXISTING_META_CHECKPOINT is empty."
            )

        meta_checkpoint_path = Path(
            EXISTING_META_CHECKPOINT
        )

        if not meta_checkpoint_path.exists():

            raise FileNotFoundError(
                meta_checkpoint_path
            )

        print(
            f"\nUse existing MAML checkpoint:\n"
            f"{meta_checkpoint_path}"
        )

    supervised_checkpoint_path: Optional[Path] = None

    if RUN_SUPERVISED_INIT:

        if not EXISTING_SUPERVISED_CHECKPOINT:

            raise ValueError(
                "RUN_SUPERVISED_INIT=True, "
                "but EXISTING_SUPERVISED_CHECKPOINT is empty."
            )

        supervised_checkpoint_path = Path(
            EXISTING_SUPERVISED_CHECKPOINT
        )

        if not supervised_checkpoint_path.exists():

            raise FileNotFoundError(
                supervised_checkpoint_path
            )

        print(
            f"\nUse existing supervised checkpoint:\n"
            f"{supervised_checkpoint_path}"
        )

    master_supervised_results = []
    master_k_results = []
    master_fewshot_results = []
    if RUN_TARGET_COMPARE or RUN_FEWSHOT_COMPARE:

        total_ratio_conditions = (
            len(TARGET_PROJECTS)
            * len(SEEDS)
            * len(TARGET_TRAIN_RATIO_TOTAL_LIST)
        )

        ratio_condition_index = 0

        for target_project in TARGET_PROJECTS:

            for seed in SEEDS:

                for train_ratio in TARGET_TRAIN_RATIO_TOTAL_LIST:

                    ratio_condition_index += 1
                    TARGET_PROJECT = target_project
                    SEED = seed
                    TARGET_TRAIN_RATIO_TOTAL = train_ratio

                    set_global_seed(SEED)

                    validate_config()

                    print("\n\n")
                    print("#" * 100)

                    print(
                        f"RATIO EXPERIMENT "
                        f"{ratio_condition_index}/"
                        f"{total_ratio_conditions}"
                    )

                    print(
                        f"TARGET = {TARGET_PROJECT} | "
                        f"SEED = {SEED} | "
                        f"TRAIN RATIO = "
                        f"{TARGET_TRAIN_RATIO_TOTAL}"
                    )

                    print("#" * 100)

                    ratio_tag = (
                        f"{TARGET_TRAIN_RATIO_TOTAL:.4f}"
                        .rstrip("0")
                        .rstrip(".")
                        .replace(".", "p")
                    )

                    condition_dir = (
                        run_dir
                        / TARGET_PROJECT
                        / f"seed_{SEED}"
                        / f"ratio_{ratio_tag}"
                    )

                    condition_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    (
                        condition_dir
                        / "checkpoints"
                    ).mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    (
                        condition_dir
                        / "results"
                    ).mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    print(
                        f"\nCondition directory:\n"
                        f"{condition_dir}"
                    )

                    save_json(
                        current_config_dict(),
                        condition_dir
                        / "config.json",
                    )

                    (
                        train_data,
                        dev_data,
                        test_data,
                        target_info,
                    ) = prepare_target_data(
                        tokenizer
                    )

                    save_json(
                        target_info,
                        condition_dir
                        / "results"
                        / "target_data_info.json",
                    )

                    if RUN_TARGET_COMPARE:

                        condition_results = []

                        # -----------------------------------------
                        # Raw
                        # -----------------------------------------

                        if RUN_RAW_BASELINE:

                            raw_result = target_finetune(
                                "raw",
                                train_data,
                                dev_data,
                                test_data,
                                device,
                                condition_dir,

                                meta_checkpoint_path=None,
                                supervised_checkpoint_path=None,

                                adaptation_mode="ratio",
                                target_k=None,
                            )

                            condition_results.append(
                                raw_result
                            )

                            master_supervised_results.append(
                                raw_result
                            )

                            append_to_master_results(
                                [raw_result],
                                run_dir,
                            )

                        # -----------------------------------------
                        # Supervised initialization
                        # -----------------------------------------

                        if RUN_SUPERVISED_INIT:

                            supervised_result = target_finetune(
                                "supervised",
                                train_data,
                                dev_data,
                                test_data,
                                device,
                                condition_dir,

                                meta_checkpoint_path=None,

                                supervised_checkpoint_path=(
                                    supervised_checkpoint_path
                                ),

                                adaptation_mode="ratio",
                                target_k=None,
                            )

                            condition_results.append(
                                supervised_result
                            )

                            master_supervised_results.append(
                                supervised_result
                            )

                            append_to_master_results(
                                [supervised_result],
                                run_dir,
                            )

                        # -----------------------------------------
                        # MAML initialization
                        # -----------------------------------------

                        if RUN_MAML_INIT:

                            maml_result = target_finetune(
                                "maml",
                                train_data,
                                dev_data,
                                test_data,
                                device,
                                condition_dir,

                                meta_checkpoint_path=(
                                    meta_checkpoint_path
                                ),

                                supervised_checkpoint_path=None,

                                adaptation_mode="ratio",
                                target_k=None,
                            )

                            condition_results.append(
                                maml_result
                            )

                            master_supervised_results.append(
                                maml_result
                            )

                            append_to_master_results(
                                [maml_result],
                                run_dir,
                            )

                        comparison_path = (
                            condition_dir
                            / "results"
                            / (
                                f"{TARGET_PROJECT}_"
                                f"seed_{SEED}_"
                                f"ratio_{ratio_tag}_"
                                f"initialization_comparison.csv"
                            )
                        )

                        condition_df = pd.DataFrame(
                            condition_results
                        )

                        condition_df.to_csv(
                            comparison_path,
                            index=False,
                        )

                        print("\n" + "=" * 100)

                        print(
                            f"RATIO RESULT | "
                            f"target={TARGET_PROJECT} | "
                            f"seed={SEED} | "
                            f"ratio={TARGET_TRAIN_RATIO_TOTAL}"
                        )

                        print("=" * 100)

                        if len(condition_df) > 0:

                            display_cols = [
                                "target_project",
                                "initialization",
                                "seed",
                                "adaptation_mode",
                                "target_train_ratio_total",
                                "actual_target_train_size",
                                "eval_acc",
                                "eval_precision",
                                "eval_recall",
                                "eval_f1",
                                "eval_mcc",
                                "eval_auc",
                                "eval_pf",
                                "best_dev_f1",
                                "best_target_epoch",
                            ]

                            print(
                                condition_df[
                                    display_cols
                                ].to_string(
                                    index=False
                                )
                            )
                        master_ratio_path = (
                            run_dir
                            / "results"
                            / "all_supervised_results.csv"
                        )

                        (
                            run_dir
                            / "results"
                        ).mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        pd.DataFrame(
                            master_supervised_results
                        ).to_csv(
                            master_ratio_path,
                            index=False,
                        )

                        print(
                            f"\nSaved Ratio master results:\n"
                            f"{master_ratio_path}"
                        )
                    if RUN_FEWSHOT_COMPARE:

                        support_data = (
                            sample_target_fewshot_support(
                                train_data,
                                SEED,
                            )
                        )

                        fewshot_results = []
                        if RUN_RAW_BASELINE:

                            result = target_fewshot_adapt(
                                "raw",
                                support_data,
                                test_data,
                                device,
                                meta_checkpoint_path=None,
                            )

                            result[
                                "target_project"
                            ] = TARGET_PROJECT

                            result[
                                "seed"
                            ] = SEED

                            result[
                                "target_train_ratio_total"
                            ] = TARGET_TRAIN_RATIO_TOTAL

                            fewshot_results.append(
                                result
                            )

                            master_fewshot_results.append(
                                result
                            )

                            append_to_master_results(
                                [result],
                                run_dir,
                            )
                        if RUN_MAML_INIT:

                            result = target_fewshot_adapt(
                                "maml",
                                support_data,
                                test_data,
                                device,

                                meta_checkpoint_path=(
                                    meta_checkpoint_path
                                ),
                            )

                            result[
                                "target_project"
                            ] = TARGET_PROJECT

                            result[
                                "seed"
                            ] = SEED

                            result[
                                "target_train_ratio_total"
                            ] = TARGET_TRAIN_RATIO_TOTAL

                            fewshot_results.append(
                                result
                            )

                            master_fewshot_results.append(
                                result
                            )

                            append_to_master_results(
                                [result],
                                run_dir,
                            )
                        fewshot_path = (
                            condition_dir
                            / "results"
                            / (
                                f"{TARGET_PROJECT}_"
                                f"seed_{SEED}_"
                                f"ratio_{ratio_tag}_"
                                f"fewshot.csv"
                            )
                        )

                        pd.DataFrame(
                            fewshot_results
                        ).to_csv(
                            fewshot_path,
                            index=False,
                        )

                        master_fewshot_path = (
                            run_dir
                            / "results"
                            / "all_fewshot_results.csv"
                        )

                        pd.DataFrame(
                            master_fewshot_results
                        ).to_csv(
                            master_fewshot_path,
                            index=False,
                        )

                        del support_data
                    del train_data
                    del dev_data
                    del test_data

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if RUN_K_SAMPLE_COMPARE:

        total_k_conditions = (
            len(TARGET_PROJECTS)
            * len(SEEDS)
            * len(TARGET_K_LIST)
        )

        k_condition_index = 0

        for target_project in TARGET_PROJECTS:

            for seed in SEEDS:
                TARGET_PROJECT = target_project
                SEED = seed

                set_global_seed(SEED)
                TARGET_TRAIN_RATIO_TOTAL = (
                    TARGET_TRAIN_RATIO_TOTAL_LIST[0]
                )

                validate_config()

                for k in TARGET_K_LIST:

                    k_condition_index += 1

                    print("\n\n")
                    print("#" * 100)

                    print(
                        f"K-SAMPLE EXPERIMENT "
                        f"{k_condition_index}/"
                        f"{total_k_conditions}"
                    )

                    print(
                        f"TARGET = {TARGET_PROJECT} | "
                        f"SEED = {SEED} | "
                        f"K = {k}"
                    )

                    print("#" * 100)

                    # ==========================================
                    # K experiment directory
                    # ==========================================

                    condition_dir = (
                        run_dir
                        / TARGET_PROJECT
                        / f"seed_{SEED}"
                        / f"k_{k}"
                    )

                    condition_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    (
                        condition_dir
                        / "checkpoints"
                    ).mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    (
                        condition_dir
                        / "results"
                    ).mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    print(
                        f"\nCondition directory:\n"
                        f"{condition_dir}"
                    )

                    (
                        train_data,
                        dev_data,
                        test_data,
                        target_info,
                    ) = prepare_target_k_data(
                        tokenizer,
                        k,
                        SEED,
                    )

                    save_json(
                        target_info,
                        condition_dir
                        / "results"
                        / "target_k_info.json",
                    )

                    k_results = []
                    if RUN_RAW_BASELINE:

                        raw_result = target_finetune(
                            "raw",
                            train_data,
                            dev_data,
                            test_data,
                            device,
                            condition_dir,

                            meta_checkpoint_path=None,
                            supervised_checkpoint_path=None,

                            adaptation_mode="k_sample",
                            target_k=k,
                        )

                        k_results.append(
                            raw_result
                        )

                        master_k_results.append(
                            raw_result
                        )

                        append_to_master_results(
                            [raw_result],
                            run_dir,
                        )

                    if RUN_SUPERVISED_INIT:

                        supervised_result = target_finetune(
                            "supervised",
                            train_data,
                            dev_data,
                            test_data,
                            device,
                            condition_dir,

                            meta_checkpoint_path=None,

                            supervised_checkpoint_path=(
                                supervised_checkpoint_path
                            ),

                            adaptation_mode="k_sample",
                            target_k=k,
                        )

                        k_results.append(
                            supervised_result
                        )

                        master_k_results.append(
                            supervised_result
                        )

                        append_to_master_results(
                            [supervised_result],
                            run_dir,
                        )

                    if RUN_MAML_INIT:

                        maml_result = target_finetune(
                            "maml",
                            train_data,
                            dev_data,
                            test_data,
                            device,
                            condition_dir,

                            meta_checkpoint_path=(
                                meta_checkpoint_path
                            ),

                            supervised_checkpoint_path=None,

                            adaptation_mode="k_sample",
                            target_k=k,
                        )

                        k_results.append(
                            maml_result
                        )

                        master_k_results.append(
                            maml_result
                        )

                        append_to_master_results(
                            [maml_result],
                            run_dir,
                        )

                    k_comparison_path = (
                        condition_dir
                        / "results"
                        / (
                            f"{TARGET_PROJECT}_"
                            f"seed_{SEED}_"
                            f"k_{k}_"
                            f"initialization_comparison.csv"
                        )
                    )

                    k_df = pd.DataFrame(
                        k_results
                    )

                    k_df.to_csv(
                        k_comparison_path,
                        index=False,
                    )

                    print("\n" + "=" * 100)

                    print(
                        f"K-SAMPLE RESULT | "
                        f"target={TARGET_PROJECT} | "
                        f"seed={SEED} | "
                        f"K={k}"
                    )

                    print("=" * 100)

                    if len(k_df) > 0:

                        display_cols = [
                            "target_project",
                            "initialization",
                            "seed",
                            "adaptation_mode",
                            "target_k",
                            "actual_target_train_size",
                            "eval_acc",
                            "eval_precision",
                            "eval_recall",
                            "eval_f1",
                            "eval_mcc",
                            "eval_auc",
                            "eval_pf",
                            "best_dev_f1",
                            "best_target_epoch",
                        ]

                        print(
                            k_df[
                                display_cols
                            ].to_string(
                                index=False
                            )
                        )

                    master_k_path = (
                        run_dir
                        / "results"
                        / "all_k_sample_results.csv"
                    )

                    pd.DataFrame(
                        master_k_results
                    ).to_csv(
                        master_k_path,
                        index=False,
                    )

                    print(
                        f"\nSaved K-sample master results:\n"
                        f"{master_k_path}"
                    )

                    del train_data
                    del dev_data
                    del test_data

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print("\n\n")
    print("=" * 100)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 100)

    if master_supervised_results:

        master_ratio_df = pd.DataFrame(
            master_supervised_results
        )

        ratio_display_cols = [
            "target_project",
            "seed",
            "adaptation_mode",
            "target_train_ratio_total",
            "initialization",
            "actual_target_train_size",
            "eval_acc",
            "eval_precision",
            "eval_recall",
            "eval_f1",
            "eval_mcc",
        ]

        print("\nRATIO RESULTS:")

        print(
            master_ratio_df[
                ratio_display_cols
            ].to_string(
                index=False
            )
        )

        print(
            f"\nRatio master results:\n"
            f"{run_dir / 'results' / 'all_supervised_results.csv'}"
        )

    if master_k_results:

        master_k_df = pd.DataFrame(
            master_k_results
        )

        k_display_cols = [
            "target_project",
            "seed",
            "adaptation_mode",
            "target_k",
            "initialization",
            "actual_target_train_size",
            "eval_acc",
            "eval_precision",
            "eval_recall",
            "eval_f1",
            "eval_mcc",
        ]

        print("\nK-SAMPLE RESULTS:")

        print(
            master_k_df[
                k_display_cols
            ].to_string(
                index=False
            )
        )

        print(
            f"\nK-sample master results:\n"
            f"{run_dir / 'results' / 'all_k_sample_results.csv'}"
        )

    print(
        f"\nAll outputs:\n"
        f"{run_dir}"
    )


if __name__ == "__main__":
    main()
