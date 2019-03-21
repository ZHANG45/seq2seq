# Coverage model

This code reproduces model *[Modeling Coverage for Neural Machine Translation](https://arxiv.org/abs/1601.04811)* by pytorch with multi-gpu function.

## Requirements

[torchtext](https://github.com/pytorch/text)==0.2.3

python==3.5.0

pytorch==0.4.0

## Preprocessing

```
python preprocessing.py
``` 

Remember to set parameters *train_src_file, train_trg_file, valid_src_file, valid_trg_file, test_src_file, test_trg_file* in file ```DefinedParameters.py```

## Training 

```
python start.py
```

Remember to set parameter *SET* in file ```DefinedParameters.py```

