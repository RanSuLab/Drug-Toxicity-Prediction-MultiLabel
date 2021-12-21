# multilabel-learn: Multilabel-Classification Algorithms


## Implemented Algorithms
* [RethinkNet](mlearn/models/rethinknet/rethinkNet.py): mlearn.models.RethinkNet

# Installation
Install packages
```bash
pip install numpy Cython
```

Compile and install the C-extensions
Change Directory in CMD: change to project working directory using the "cd" command
```bash
python ./setup.py install
python ./setup.py build_ext -i
```

Run example in project directory
```bash
python ./examples/classification.py
```

# Citations

If you use some of my works in a scientific publication, we would appreciate citations to the following papers:

For RethinkNet, please cite
```bib
@article{yang2018deep,
  title={Deep learning with a rethinking structure for multi-label classification},
  author={Yang, Yao-Yuan and Lin, Yi-An and Chu, Hong-Min and Lin, Hsuan-Tien},
  journal={arXiv preprint arXiv:1802.01697},
  year={2018}
}
```

For Cost-Sensitive Reference Pair Encoding (CSRPE), please cite
```bib
@inproceedings{YY2018csrpe,
  title = {Cost-Sensitive Reference Pair Encoding for Multi-Label Learning},
  author = {Yao-Yuan Yang and Kuan-Hao Huang and Chih-Wei Chang and Hsuan-Tien Lin},
  booktitle = {Proceedings of the Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  year = 2018,
  arxiv = {https://arxiv.org/abs/1611.09461},
  software = {https://github.com/yangarbiter/multilabel-learn/blob/master/mlearn/models/csrpe.py},
}
```
