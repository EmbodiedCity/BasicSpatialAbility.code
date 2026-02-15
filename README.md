# [ACL'25 Main] Defining and Evaluating Visual Language Models’ Basic Spatial Abilities: A Perspective from Psychometrics
[![Published Paper](https://img.shields.io/badge/Published-ACL_Main-red)](https://aclanthology.org/2025.acl-long.567/)
[![Arxiv](https://img.shields.io/badge/arXiv-2502.11859-darkred?logo=arxiv)](https://arxiv.org/abs/2502.11859)
[![Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/EmbodiedCity/BasicSpatialAbility)
[![Code](https://img.shields.io/badge/Github-Code-blue?logo=github)](https://github.com/EmbodiedCity/BasicSpatialAbility.code)

This dataset is a benchmark designed for evaluating Multimodal Large Language Models' Basic Spatial Abilities based on authentic Psychometric theories. It is structured specifically to support both **Zero-shot** and **Few-shot** evaluation protocols.

## 📂 Dataset Structure (Important)
The dataset is organized into two distinct splits. **Please read this carefully to ensure valid evaluation results.**
| Split Name | Role | Description |
| :--- | :--- | :--- |
| **`test`** | **Query Set** | Contains the actual benchmark questions (images & queries) to be evaluated. <br>⚠️ **Evaluation Only.** Do not use for training or as few-shot examples. |
| **`validation`** | **Support Set** | Contains high-quality examples intended to be used as **Few-shot Prompts (In-Context Learning)**. <br>These samples should be prepended to the test queries to demonstrate the task to the model. |

## ⚙️ Usage & Evaluation Protocol
You can load the dataset using the Hugging Face `datasets` library.

### 1. Zero-Shot Evaluation
**Logic:** Directly evaluate the model on the `test` split without any prior examples.

```python
from datasets import load_dataset

# Load the evaluation queries
test_dataset = load_dataset("EmbodiedCity/BasicSpatialAbility", split="test")

for sample in test_dataset:
    image = sample['image']
    question = sample['question']
    # Model inference...
```

### 2. Few-Shot Evaluation
**Logic:** Use examples from the validation split as the context (demonstrations), followed by the query from the test split.
1. Load the validation split.
2. Format them into the prompt history.
3. Append the target question from the test split.

```python
from datasets import load_dataset

# 1. Load the support set (demonstrations)
support_set = load_dataset("EmbodiedCity/BasicSpatialAbility", split="validation")

# 2. Load the query set (evaluation)
test_set = load_dataset("EmbodiedCity/BasicSpatialAbility", split="test")

# Pseudo-code for prompt construction
prompt_context = []
for ex in support_set:
    prompt_context.append(f"User: {ex['question']}\nAssistant: {ex['answer']}")

# 3. Evaluate on Test Set
for sample in test_set:
    # Combine context + current test question
    final_prompt = prompt_context + [f"User: {sample['question']}"]
    
    # Model inference...
```

## 🔬 Underlying Theory
The Theory of Multiple Intelligences underscores the hierarchical nature of cognitive capabilities. To advance Spatial Artificial Intelligence, we pioneer a psychometric framework defining five Basic Spatial Abilities (BSAs) in Visual Language Models (VLMs): Spatial Perception, Spatial Relation, Spatial Orientation, Mental Rotation, and Spatial Visualization. Benchmarking 13 mainstream VLMs through nine validated psychometric experiments reveals significant gaps versus humans, with three key findings: 1) VLMs mirror human hierarchies (strongest in 2D orientation, weakest in 3D rotation) with independent BSAs; 2) Many smaller models surpass larger counterparts, with Qwen leading and InternVL2 lagging; 3) Interventions like CoT and few-shot training show limits from architectural constraints, while ToT demonstrates the most effective enhancement. Identified barriers include weak geometry encoding and missing dynamic simulation. By linking Psychometrics to VLMs, we provide a comprehensive BSA evaluation benchmark, a methodological perspective for embodied AI development, and a cognitive science-informed roadmap for achieving human-like spatial intelligence.
|          Type          |                                                       Definition                                                      |          Tests          |
|:----------------------:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------:|
|   Spatial  Perception  | The ability to perceive  horizontal and vertical  orientations without  interference from  miscellaneous information. |           SVT           |
|    Spatial  Relation   |                         The ability of recognizing  relationships between  parts of an entity.                        | NCIT  DAT:SR  R-Cube-SR |
|  Spatial  Orientation  |                               The ability to navigate  or enter a given  spatial state.                               |           MRMT          |
|    Mental  Rotation    |                                      The ability to mentally  rotate 3D objects.                                      |       MRT  PSVT:R       |
| Spatial  Visualization |                         The ability to mentally  manipulate and transform  2D and 3D objects.                         |     SBST  R-Cube-Vis    |****

<p align="center">
  <img width="600" src="https://github.com/EmbodiedCity/BasicSpatialAbility.code/raw/main/framework.jpg">
</p>

<p align="center">
  The Framework of Basic Spatial Abilities (Image sources are cited in the paper)
</p>

## Citation
If you use this project in your research, please cite the following paper:
```bibtex
@inproceedings{xu-etal-2025-defining,
    title = "Defining and Evaluating Visual Language Models' Basic Spatial Abilities: A Perspective from Psychometrics",
    author = "Xu, Wenrui  and
      Lyu, Dalin  and
      Wang, Weihang  and
      Feng, Jie  and
      Gao, Chen  and
      Li, Yong",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.567/",
    doi = "10.18653/v1/2025.acl-long.567",
    pages = "11571--11590",
    ISBN = "979-8-89176-251-0"
}
