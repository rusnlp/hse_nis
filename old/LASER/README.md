# LASER

[*LASER installation*](https://github.com/facebookresearch/LASER/blob/master/README.md)

[*ru data*](https://github.com/facebookresearch/LASER/blob/master/data/tatoeba/v1/tatoeba.rus-eng.rus)

[*en data*](https://github.com/facebookresearch/LASER/blob/master/data/tatoeba/v1/tatoeba.rus-eng.eng)

## Calculation of sentence embeddings

```
./embed.sh ${LASER}/data/tatoeba/v1/tatoeba.rus-eng.rus ru ru_embd.raw
./embed.sh ${LASER}/data/tatoeba/v1/tatoeba.rus-eng.eng en en_embd.raw
```

## Output format

The embeddings are stored in float32 matrices in raw binary format.
They can be read in Python by raw_to_bin.py