
## Examples:

+ LINE

With label file and auto stop:

```shell
python src/main.py --method line --input data/blogCatalog/bc_adjlist.txt --graph-format adjlist --output vec_all.txt --epochs 10 --order 3 --lr 0.001 --label-file data/blogCatalog/bc_labels.txt --no-auto-stop
```

Without label file:

```shell
python src/main.py --method line --input data/blogCatalog/bc_adjlist.txt --graph-format adjlist --output vec_all.txt --epochs 10 --order 3 --lr 0.001
```

+ SNE

```shell
python src/sne_main.py --method sne --input1 data/blogCatalog/bc_adjlist.txt --input2 data/blogCatalog/bc_adjlist.txt --graph-format adjlist --output res/vec_all --epochs 10 --label-file data/blogCatalog/align_labels --rho0 0.001 --rho1 0.001 --mode-size 200
```
