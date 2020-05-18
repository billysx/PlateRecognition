# PlateRecognition

 Python Datamining final project

models can be downloaded at
https://disk.pku.edu.cn:443/link/7CEB1A5D1C20D6F75F582E5D36D13442

## Part 1 Character recognizing

### Training

To train a model for character recognizing, first run

```bash
python gen_my_chardata.py
```

To generate real car plate character dataset.

Then, run

```bash
python train_char.py --data_path "../data/mychar_data"  --save "model_path" --lr 1e-4 --batchsize 64 --epoch 20
```

To train your character recognition model.



### Test

To test your model at “model path”

```bash
python run_char.py --classifier_path "model_path"
```







## Part 2 Car plate localization

### Training

To train a model for plate localization, run

```bash
python train_plate.py --data_path "../data/Plate_dataset"  --save "model_path" --lr 2e-4 --batchsize 16 --epoch 1000
```

To train your model.



### Test

To test your model at “model path”

```bash
python run_plate.py --model_path "model_path"
```



