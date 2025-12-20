# [ACL'25 Main] Defining and Evaluating Visual Language Models’ Basic Spatial Abilities: A Perspective from Psychometrics
[![Published Paper](https://img.shields.io/badge/Published-ACL_Paper-red)](https://aclanthology.org/2025.acl-long.567/)
[![Arxiv](https://img.shields.io/badge/arXiv-2502.11859-darkred?logo=arxiv)](https://arxiv.org/abs/2502.11859)
[![Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/EmbodiedCity/BasicSpatialAbility)
[![Code](https://img.shields.io/badge/Github-Code-blue?logo=github)](https://github.com/EmbodiedCity/BasicSpatialAbility.code)

The Theory of Multiple Intelligences underscores the hierarchical nature of cognitive capabilities. To advance Spatial Artificial Intelligence, we pioneer a psychometric framework defining five Basic Spatial Abilities (BSAs) in Visual Language Models (VLMs): Spatial Perception, Spatial Relation, Spatial Orientation, Mental Rotation, and Spatial Visualization. Benchmarking 13 mainstream VLMs through nine validated psychometric experiments reveals significant gaps versus humans, with three key findings: 1) VLMs mirror human hierarchies (strongest in 2D orientation, weakest in 3D rotation) with independent BSAs; 2) Many smaller models surpass larger counterparts, with Qwen leading and InternVL2 lagging; 3) Interventions like CoT and few-shot training show limits from architectural constraints, while ToT demonstrates the most effective enhancement. Identified barriers include weak geometry encoding and missing dynamic simulation. By linking Psychometrics to VLMs, we provide a comprehensive BSA evaluation benchmark, a methodological perspective for embodied AI development, and a cognitive science-informed roadmap for achieving human-like spatial intelligence.

|          Type          |                                                       Definition                                                      |          Tests          |
|:----------------------:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------:|
|   Spatial  Perception  | The ability to perceive  horizontal and vertical  orientations without  interference from  miscellaneous information. |           SVT           |
|    Spatial  Relation   |                         The ability of recognizing  relationships between  parts of an entity.                        | NCIT  DAT:SR  R-Cube-SR |
|  Spatial  Orientation  |                               The ability to navigate  or enter a given  spatial state.                               |           MRMT          |
|    Mental  Rotation    |                                      The ability to mentally  rotate 3D objects.                                      |       MRT  PSVT:R       |
| Spatial  Visualization |                         The ability to mentally  manipulate and transform  2D and 3D objects.                         |     SBST  R-Cube-Vis    |****

<p align="center">
  <img width="600" src="framework.jpg">
</p>

<p align="center">
  The Framework of Basic Spatial Abilities (Image sources are cited in the paper)
</p>

# Citation
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
