# ArcDOTA 😏

Project contains tools, utils and models to predict dota outcomes (win, game stats, e.t.c.)

## Content

| Source               | Description                                  |
| -------------------- | -------------------------------------------- |
| [configs](configs/)     | configuration files and utils                |
| [inference](inference/) | inference scripts (rest api)                 |
| [parse](parse/)         | parse pipelines: games, BD                   |
| [scarpe](scarpe/)       | scarpe pipelines: odds, teams information    |
| [train](train/)         | train pipelines: prematch model, draft model |

## Architecture

//TODO: fill this out

---

## Notes

1. all configs stores as `.yalm`
2. data stores as `.json`, `.csv`, `.npy` (numpy arrays)
3. tmp files stores as `.pkl` (python pickle)
4. model checkpoint stores as `.torch`
5. model checkpoint contains (key-value):
   - "model" - model state dict
   - "optimizer" - optimizer state dict
   - "configs" - configs
6. jupyter noteeboks (`.ipynb`) is just for research

---

## TODO

* [ ] REST API
  * [X] Prematch
  * [ ] With drafts
* [ ] Pretrain:
  * [ ] Teams
  * [ ] Players
  * [ ] Heroes
* [ ] Build evaluate platform

---
