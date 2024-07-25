# RobustUnrolling

To train constrained LISTA (or use the RunMe.sh file)
```
python3 LISTA.py --nLayers 10 --lr 1e-5 --lr_dual 1e-3 --eps 0.05 --noisyOuts --constrained --Trial exp_name
```

To train standard LISTA
```
python3 LISTA.py --nLayers 10 --lr 1e-6 --eps 0.05 --Trial exp_name
```

To train standard LISTA with noise
```
python3 LISTA.py --nLayers 10 --lr 1e-6 --eps 0.05 --noisyOuts --Trial exp_name
```

For testing and generating the figures
```
python3 testing.py 
```