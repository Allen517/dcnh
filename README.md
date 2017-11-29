
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