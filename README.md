# metal-artifact-reduction


## metal-simulation
```
     cd data
     cd deep_lesion
     ./download_deep_lesion.sh
```

use matlab run file prepare_deep_lesion.m


## Radon-inverse-layer

```
    ./train.py
    ./test.py
```


## Denoise and Inpainting

##### You should prepare Data First
```sh
cd  data
chmod a+x download_deep_lesion.sh
./download_deep_lesion.sh
```
When Download finished
##### run the matlab script prepare_deep_lesion.m

##### train the model
```sh
python train.py
```
the result will be in runs

##### test the model
```sh
python test.py
```
the result willbe in directory testt


## Residual_learning
```sh
cd  data
chmod a+x download_deep_lesion.sh
./download_deep_lesion.sh
```

```sh
./train.py
./test.py
```

