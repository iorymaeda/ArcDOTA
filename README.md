# ArcDOTA 😏

Project contains tools, utils and models to predict dota outcomes (win, game stats, e.t.c.)

## Content

| Source               | Description                                  |
| -------------------- | -------------------------------------------- |
| [configs](configs/)     | configuration files and utils                |
| [inference](inference/) | inference scripts (rest api, odds scarper)   |
| [parse](parse/)         | parse pipelines: games, BD, odds             |
| [scarpe](scarpe/)       | scarpe pipelines: odds, games, dependencies  |
| [train](train/)         | train pipelines: prematch model, draft model |

## Results

#### Validation set

| metric            | This project | Bookmaker |
| ----------------- | ------------ | --------- |
| Accuracy          | **0.660**   | 0.635     |
| Balanced Accuracy | **0.660**   | 0.639     |
| AUC               | **0.747**   | 0.709     |
| LogLoss           | **0.620**   | 0.633     |

#### Test set

| metric            | This project | Bookmaker  |
| ----------------- | ------------ | ---------- |
| Accuracy          | **0.611**   | **0.611** |
| Balanced Accuracy | 0.607        | **0.609** |
| AUC               | **0.663**   | 0.625      |
| LogLoss           | **0.651**   | 0.670      |

## Notes

1. data stores as `.json`, `.csv`, `.npy` (numpy arrays)
2. tmp files stores as `.pkl` (python pickle)
3. model checkpoint stores as `.torch`
4. model checkpoint contains (key-value):
   - "model" - model state dict
   - "optimizer" - optimizer state dict
   - "configs" - dict with configs
   - "model_tag" - string model tag
   - "kwargs"
5. jupyter noteeboks (`.ipynb`) is just for research

## TODO

* [ ] REST API

  * [X] Prematch
  * [ ] With drafts
* [ ] Pretrain:

  * [ ] Teams
  * [ ] Players
  * [ ] Heroes
* [ ] Build evaluate platform
