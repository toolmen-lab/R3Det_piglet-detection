# Piglets detection using R3Det
This tensorflow implementation code is based on [R3Det_Tensorflow](https://github.com/Thinklab-SJTU/R3Det_Tensorflow).


## Environment
python3.6  
cuda 10.0  
opencv(cv2)  
tensorflow-gpu 1.13  



## Compile
```
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace (or make)

cd $PATH_ROOT/libs/box_utils/
python setup.py build_ext --inplace
```


## Test
```
cd $PATH_ROOT/tools
python test_r3det.py 
```
test data path: $PATH_ROOT/data/io/PIGLET/test/images  
output path: $PATH_ROOT/output/predictions


## Reference
https://github.com/Thinklab-SJTU/R3Det_Tensorflow
