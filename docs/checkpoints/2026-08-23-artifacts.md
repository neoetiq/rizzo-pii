# Checkpoint artefatti — 2026-08-23

Questo inventario identifica gli artefatti autorevoli necessari per riprendere
il progetto e preservarne la tracciabilita'. I path sono relativi alla root del
repository.

Per i tre dataset sigillati sono riportati esclusivamente i metadati presenti
nei rispettivi manifest. Il loro contenuto non e' stato aperto o valutato e lo
stato resta `consumed=false`.

## Modelli autorevoli

| Modello | Path autorevole | SHA-256 bundle modello | SHA-256 `model.safetensors` | Byte pesi | Stato |
|---|---|---|---|---:|---|
| Teacher FP32 | `artifacts/quantization/model` | `f10d112dffdc69da1e4d23a698e67176d08ac182cb686b4288b8a0ab7670fc79` | `b1e046e3aec52c3cd74eaf166b034644a4dac0c91128bf06803e1478657fcd9f` | 1.230.273.700 | Baseline originale |
| C | `artifacts/training/runs/full-clean-1024-kd-type-boundary-a010-t2-20260823/final` | `7c3221b3387db2285d28e975145b9884f77fd4077b8264c2e7cee54a49b2e63a` | `86d8f249b1091db53bee8024d32c80b2460f41f3618df8d2f2793495372cdbfe` | 562.649.620 | Completo; baseline student di ricerca |
| D0 | `artifacts/training/runs/id-doc-d0-control-ft64-20260823/checkpoints/checkpoint-64` | `7e6086846730499bac9a8af3ebe908832cea01d3c0191ee1a1ad21937f353a24` | `a041cae39a47d451c1d1ef6ce4a645639cf206db74cfdbf58c6f382592dc217f` | 562.649.620 | Completo al passo 64; manifest incoerente, vedere sotto |
| D1 LR `5e-6` | `artifacts/training/runs/id-doc-d1-targeted-ft64-20260823/final` | `2561e757f8aa73483efdac01740dfca5b0e12cefd7aa7fa77e65c58af24bdd7f` | `f9ff430d39c3913c7d93da51fd42ed0b945c97b25f7f4aefc9d2c89127e28986` | 562.649.620 | Completo; miglior candidato privacy sul dev osservato, non release-ready |
| D1 LR `2e-6` | `artifacts/training/runs/id-doc-d1-targeted-ft64-lr2e6-20260823/checkpoints/checkpoint-64` | `8838cac4276ccd23352ac7d1e4c242f8486d4ed62606235df17d7b09d273b4cf` | `60b3a890e94bec52ae9d818ef986a71ff8e8161e11653309d9762199dd0ae84f` | 562.649.620 | Completo al passo 64; non promosso; manifest incoerente, vedere sotto |
| D2 | `artifacts/training/runs/id-doc-d2-synthetic-ft64-v2-20260823/checkpoints/checkpoint-64` | `778e0d8662da3e0c640d01d8215872a21fd9744bff10c2a084156f25dbd3b1b1` | `4c88c697f2895b52af5d7490e333ae13c651b880a08f295402ec1c66f0b0106a` | 562.649.620 | Completo; gate development non superato |

Per C e D1 LR `5e-6` esiste anche il checkpoint Trainer completo con gli
stessi pesi del rispettivo `final/`. Per una nuova inizializzazione
weights-only e' sufficiente conservare `final/`. D0, D1 LR `2e-6` e D2 non
hanno una copia `final/`: per essi il path autorevole e' `checkpoint-64`.

Il tokenizer bundle di D0, D1 e D2 ha SHA-256
`938614fc47c20c580007a99a9c192193f559fc6d665f1e459fb4a6eb104f8381`.

La directory `artifacts/training/runs/id-doc-d2-synthetic-ft64-20260823`
contiene soltanto il manifest del preflight fallito e non e' un checkpoint
modello.

## Incoerenza D0 e D1 LR `2e-6`

Per D0 e D1 LR `2e-6`, `training-summary.json`, il checkpoint al passo 64 e i
report di valutazione attestano il completamento del run. I rispettivi
`run-manifest.json` sono pero' rimasti nello stato precedente al resume:

- `status="complete"`;
- `result_status="partial_guard_stop"`;
- campo `authoritative_model` assente.

Ai fini di questo checkpoint progettuale, i rispettivi `checkpoint-64` sono i
modelli autorevoli. Questa eccezione deve rimanere esplicita finche' i manifest
non verranno normalizzati con una procedura auditabile. D2 possiede invece il
bookkeeping corretto del resume e del modello autorevole.

## Dataset e manifest

| Ruolo | Path | Byte | SHA-256 | Stato di consumo |
|---|---|---:|---|---|
| Train clean | `artifacts/training/data/clean-train-1024.jsonl` | 2.754.645 | `95e179775ae388c271bc5fb058614fe09961f99c4fddd0175f2beb875bd451b6` | Consumato |
| Development clean | `artifacts/training/data/clean-validation-512.jsonl` | 1.382.180 | `4001d5fce099f75cdcd03101f228187b03c1390774801bc818fd586b2ee9bafb` | Consumato ripetutamente; non e' un test cieco |
| D0 train | `artifacts/training/data/id-doc-targeted-ft-v1/d0-control-train-1024.jsonl` | 2.773.339 | `ace4708abca7f819f8e03741c4fa4aa9f9a0da83f9839cbbd33a45ad0d7304d8` | Consumato |
| D1 train | `artifacts/training/data/id-doc-targeted-ft-v1/d1-targeted-train-1024.jsonl` | 2.709.043 | `ea159d8b26aa4132e2feda42a7d3389e7c82d69ae0fe4ab6afb824ff418b3e6f` | Consumato |
| Holdout globale v1 | `artifacts/training/data/id-doc-targeted-ft-v1/sealed-global-2048.jsonl` | 5.559.171 | `c929c65bc979ddb3d25c9944b65799379f528a7246000f275944871f0a9e5b19` *(dal manifest)* | **Sealed; `consumed=false`; contenuto non letto** |
| Challenge ID_DOC v1 | `artifacts/training/data/id-doc-targeted-ft-v1/sealed-id-doc-2048.jsonl` | 5.329.761 | `44dba24d082004d48de1e32cc979fe58184bac1f9b8dde645c210d5e34cc4c1f` *(dal manifest)* | **Sealed; `consumed=false`; contenuto non letto** |
| D2 train | `artifacts/training/data/id-doc-targeted-ft-v2/d2-synthetic-targeted-train-1024.jsonl` | 2.357.119 | `026e5c1ad2cde5be38a2623f65a38a27121d7b0bb26b53c02d74f11d563a0e7e` | Consumato |
| Challenge sintetico D2 | `artifacts/training/data/id-doc-targeted-ft-v2/sealed-synthetic-id-doc-challenge-384.jsonl` | 407.833 | `1b0dd788a65788ddc97a7386f48ad39ae0a6d3bba9b4cb22f52ebc171c8417cf` *(dal manifest)* | **Sealed; `consumed=false`; contenuto non letto** |

| Manifest | Path | Byte | SHA-256 |
|---|---|---:|---|
| Selezione clean | `artifacts/training/data/clean-pilot-manifest.json` | 3.319 | `dddec96fa5ab51011650f296d1288af341cc8650662fe54c6b463e88e2035400` |
| Targeted v1 | `artifacts/training/data/id-doc-targeted-ft-v1/manifest.json` | 5.136.878 | `6290c25e1ea050f3d21d3023d8b2ba83d0327980c49007b9baad713d38787ed1` |
| Targeted v2 | `artifacts/training/data/id-doc-targeted-ft-v2/manifest.json` | 950.806 | `ffc0edeadea0f96911aec3ae5924ee709030c2465c3ce49ff7f41d6dc48e7fbb` |

Non risultano riferimenti ai tre file sigillati nei manifest o report dei run
completati.

### Sorgenti per rigenerare le selezioni

| Sorgente | Path | Byte | SHA-256 |
|---|---|---:|---|
| Train parquet clean | `artifacts/training/sources/clean/data/train-00000-of-00016.parquet` | 111.594.118 | `bc1467a66485b621d3a5077c7593a86fb92d05572f5da79c634eab2611e9b7e2` |
| Validation parquet clean | `artifacts/training/sources/clean/data/validation-00000-of-00001.parquet` | 29.236.862 | `30960e940acb44fd7217d2ba6e6c3df0999c04cdc07778b500ba36f40bb44041` |

## Report e predizioni chiave

Tutti i report seguenti sono relativi a
`evaluation/clean-validation-512-v2/` nel run indicato.

| Modello e run | Exact report: file, byte, SHA-256 | Predizioni: file, byte, SHA-256 | Privacy report: file, byte, SHA-256 |
|---|---|---|---|
| Teacher — `pilot-clean-256-full-adafactor-20260822` | `teacher-report.json`, 8.283, `b4a36dc67f1287c2425dcaacebd2d2a66788a1908645c0c84fe190258b0d204d` | `teacher-predictions.jsonl`, 750.607, `68646d3c96a0ab63692da701de151dbf531542f4dbb0e0163661b0b2e735aed5` | `privacy-utility-report.json`, 37.099, `4791decb83d0d424c20c9b5dfd869f55a1c7489e7aba33323f49ffa98b10a389` |
| C — `full-clean-1024-kd-type-boundary-a010-t2-20260823` | `student-report.json`, 8.625, `92457ac7bac6bc22743ebc78b8e2e650449b37880fe6bd85e71e994aad9bc3fd` | `student-predictions.jsonl`, 750.645, `2ae969e22a763893ee0f9a4ba01189a02869bedb6633561ea9c709b1c5a988a5` | `privacy-utility-report.json`, 37.407, `a1afac12a7d0e34a3abbd84eb176ea40ecf73decb99a5c2493418f08c7abd2b8` |
| D0 — `id-doc-d0-control-ft64-20260823` | `student-report.json`, 8.546, `676f2130c323a2647e95d7735037c3dcd1479ec046bd6811e56eddc423302a25` | `student-predictions.jsonl`, 750.713, `7173589d6d1bd447205adacd51f265a604ab8ffccdeea393740c53a722e48915` | `privacy-utility-report.json`, 37.404, `3da9c487a528923848c90061ad607c19784383aa071f45df7a86c953e703c5dd` |
| D1 LR `5e-6` — `id-doc-d1-targeted-ft64-20260823` | `student-report.json`, 8.593, `5e3b0e3cc64eb8e405626f38c4ebd2029ddce532edd689081db6b0bb26173cca` | `student-predictions.jsonl`, 750.798, `94ad6c6bf652949c44d05f1e67e0b7637f11027969476b53a90f18aaef1d0cc0` | `privacy-utility-report.json`, 37.276, `6360e119d8c600b633877728e8a1c2b3c1e3dcf5d8bcbc6b2f72ef9414561c8e` |
| D1 LR `2e-6` — `id-doc-d1-targeted-ft64-lr2e6-20260823` | `student-report.json`, 8.718, `ce39295badbb480fab7ea2aef376821ac8a70b5bbfb3c9d7dfb2d2d28d576c85` | `student-predictions.jsonl`, 750.878, `6e47e6ff6db9afa339646c978a9b4a59bc273f1254c2daa312cd5416d53276c1` | `privacy-utility-report.json`, 37.454, `30b194dc88139d87683745519573c140cd6bc7935f0f293dd1a56e02eecb728a` |
| D2 — `id-doc-d2-synthetic-ft64-v2-20260823` | `student-report.json`, 15.325, `f863a832d343b6bdb9c42b296d316fde28837f1c873412543deadc367f170e67` | `student-predictions.jsonl`, 750.552, `65b040f1d83aad80eb14a5b75cc4425c2b4a880ff67d1bd62e5deebed7b0c64b` | `privacy-utility-report.json`, 37.328, `7e1d32d154929db7273bd76854bb9ce70600447646487533c27d5583535224b1` |

Il confronto diretto D1 LR `5e-6` vs teacher e':

- path:
  `artifacts/training/runs/id-doc-d1-targeted-ft64-20260823/evaluation/clean-validation-512-v2/vs-teacher.json`;
- byte: 18.580;
- SHA-256:
  `27c151e315e9016abff2a617c3345f9b9e33e8ea83520d6d1ef434326ca3d480`.

Le predizioni devono essere conservate: permettono di ricalcolare metriche e
bootstrap senza eseguire nuovamente l'inferenza.

## Cache KD

| Artefatto | Path | Byte | SHA-256 | Uso |
|---|---|---:|---|---|
| Logits teacher FP16 | `artifacts/training/distillation-cache/full-clean-1024-teacher-fp16-20260822/cache.safetensors` | 57.387.168 | `fc8f4601ec41af21bccfe52c8be798071129b910635f7eee13fd2cb5e8188df9` | Ripetizione della KD sul clean-train-1024 corrente |
| Manifest cache | `artifacts/training/distillation-cache/full-clean-1024-teacher-fp16-20260822/manifest.json` | 904.056 | `c10bf8311757f606402dc5d9fa91a7c15f6d8b0f88ea0ba8c5819dff15a9756d` | Identita', ordine finestre e provenance della cache |

Questa cache e' legata al dataset clean-train-1024 e al tokenizer correnti. Per
un futuro D3 con dati differenti i logits teacher dovranno essere ricalcolati.

## Minimo operativo per ripartire

1. Codice, test e documentazione del repository.
2. Teacher completo, se sono previste nuova distillazione o valutazione di
   regressione.
3. C `final/` come baseline student causale.
4. D1 LR `5e-6` `final/` come candidato privacy sul development osservato.
5. Clean train/dev, parquet sorgenti e manifest di selezione.
6. Dataset e manifest targeted v1/v2, mantenendo separati e sigillati i tre
   holdout/challenge con `consumed=false`.
7. Report e predizioni di teacher, C, D0, D1 LR `5e-6`, D1 LR `2e-6` e D2.
8. `run-manifest.json` e `training-summary.json` di ogni run autorevole.

Per una conservazione scientifica completa vanno inclusi anche i checkpoint
autorevoli di D0, D1 LR `2e-6` e D2. Gli stati optimizer, scheduler e RNG non
sono necessari per iniziare un nuovo fine-tuning weights-only, ma servono per
un resume esatto o una ricostruzione forense.

## Dimensione del bundle

L'insieme composto dai sei bundle modello totali (teacher e cinque student),
dataset, manifest, sorgenti parquet e cache KD occupa circa **4.378.976 KiB**,
pari a circa **4,18 GiB**, escludendo checkpoint duplicati e run storici non
autorevoli.

## Stato del backup esterno

Al momento del checkpoint non e' configurata una destinazione esterna dedicata
allo student. `PII_HF_REPO` non e' impostata e l'unico repository modello
presente nella configurazione locale e' quello del teacher: non deve essere
usato implicitamente per pubblicare questi artefatti.

| Controllo | Stato |
|---|---|
| Destinazione/URI esterna | **Non configurata** |
| Bundle da replicare | Minimo operativo sopra; circa **4,18 GiB** per il set scientifico indicato |
| SHA-256 archivio esterno | **Non disponibile** |
| Round-trip download + verifica hash | **Non eseguito** |

Fino a quando una destinazione privata o altrimenti autorizzata non viene
scelta, i byte ignorati da Git restano confermati soltanto sul disco locale.
Il commit Git conserva codice, test, protocollo e questo inventario, ma non e'
da solo un backup dei modelli o dei dataset.

Il bootstrap paired di E-014 e' documentato in `docs/EXPERIMENT_LOG.md`, ma non
ha ancora un artefatto machine-readable dedicato. Prima di usarlo come evidenza
pubblicabile va materializzato, hashato e aggiunto a questo inventario.
