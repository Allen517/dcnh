
## Examples:

+ LINE

```shell
python src/ne_main.py --input data/blogCatalog/network/source_net --output blog --table-size 10000000 --rep-size 3 --epoch 10 --neg-ratio 5 --order 3 --method line
```

+ LINE with anchor link pretrain

```shell
python src/main.py --input data/test.src.net --output toy --table-size 100 --rep-size 3 --epoch 3 --neg-ratio 2 --order 3 --embed-file toy.epoch3.node_embeddings_all --anchor-file data/test.align --method a_line_pretrain
```

+ DCNH

```shell
python src/dcnh_main.py --embedding1 tmp_res.node_embeddings_src --embedding2 tmp_res.node_embeddings_obj --identity-linkage data/test/test.align --output t_res --method dcnh-sp --batch-size 4 --learning-rate 0.1 --negative-ratio 3 --input-size 4 --output-size 2 --hidden-size 3 --layers 1 --saving-step 1 --max-epochs 10 --log-file DCNH
```

```shell
python src/dcnh_main.py --embedding1 64/source_all.txt --embedding2 64/target_all.txt --identity-linkage 64/blogcatalog_labels.p0dot5.train --output bc.dcn-sp --method dcn-sp --batch-size 128 --learning-rate 0.1 --negative-ratio 5 --input-size 64 --output-size 14--hidden-size 8 --layers 1 --saving-step 1 --max-epochs 100 --log-file DCN_SP --device :/cpu:0
```

+ DCNH

```shell
python src/ne_main.py --input data/test/test.src.net --output test --lr .1 --batch-size 2 --table-size 10 --rep-size 5 --epochs 10 --method line --graph-format adjlist --neg-ratio 5
```

```shell
python src/ne_main.py --input data/douban_proc.net --output douban.emb --lr .001 --batch-size 1024 --table-size 5000000000 --rep-size 32 --epochs 50 --method line --graph-format adjlist --neg-ratio 5 --last-emb-file douban.emb.epoch8
```