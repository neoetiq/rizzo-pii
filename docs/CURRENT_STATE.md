# Stato corrente — checkpoint 2026-08-23

Questo file e' il punto di ripartenza operativo del progetto PII edge. Descrive
la working copy locale al 2026-08-23; non certifica una release e non autorizza
l'apertura dei set sigillati.

L'inventory corrente di file, byte e hash del checkpoint e' in
`docs/checkpoints/2026-08-23-artifacts.md`; in caso di divergenza fra una copia
archiviata e questa narrativa, fermarsi e verificare l'inventory prima di usare
l'artefatto.

## Sintesi

- Il teacher FP32 e' un ModernBERT 22x768 da 307.564.845 parametri.
- Lo student corrente e' `mmBERT-small` 22x384 da 140.658.861 parametri
  (`0,141B`), con la stessa tassonomia BIO a 45 logits.
- D1 LR `5e-6` e' il miglior candidato osservato per copertura privacy su
  clean-512; D0 e' il miglior controllo student exact-span. Nessuno e' ancora
  un modello di release.
- D1 riduce parametri e file pesi FP32 del `54,27%`; sul dev exact perde
  `0,0006065` micro-F1 dal teacher, ma lascia scoperti 26 caratteri sensibili
  contro 104 del teacher, oscurando 58 caratteri innocui contro 2.
- Il collo di bottiglia `ID_DOC`/`DOCID` osservato in D2 e' attribuito prima di
  tutto al disegno non fattoriale dei dati e alla tassonomia vicina, non a una
  dimostrata insufficienza dell'hidden size 384.
- Il supporto GPU non e' stato rimosso: training/evaluation student supportano
  Apple MPS e CPU; i checkpoint restano device-agnostic. ONNX e i test low-bit
  correnti sono intenzionalmente CPU-first. MLX resta un backend opzionale dopo
  freeze e quantizzazione di qualita'.

Il razionale e la cronologia completa sono in `docs/EXPERIMENT_LOG.md`, in
particolare E-009 fino a E-014.

## Contratto riproducibile bloccato

- **Seed unico:** `20260822` per selezione dati, training e bootstrap.
- **Tokenizer:** policy `preserve-modernbert-metaspace-v1`, con
  `fix_mistral_regex=false`. Non attivare il fix Mistral su ModernBERT.
- **Backbone small pinned:** `jhu-clsp/mmBERT-small` revisione
  `abc32620dd4f6ab06f5fbe905dc25f310618e09f`.
- **Label:** 45 label BIO nello stesso ordine del teacher; non rigenerare la
  tassonomia dagli split.
- **Input di qualita':** raw text + span carattere. Il proxy legacy token/BIO e'
  ammesso soltanto per smoke meccanici esplicitamente dichiarati.
- **Evaluator clean-512:** tutti i subword, finestre max 256, stride 32, logits
  mediati per offset identico e aggregazione globale
  `mean_logits_by_exact_offset_then_global_simple_bio`.
- **Metrica primaria da prospettivizzare:**
  `production_privacy_utility_unicode_alnum_coverage_v1`, separando copertura
  type-agnostic, type-correct e collateral masking. Exact-span resta metrica di
  boundary/tassonomia e confronto benchmark.
- **Policy applicativa da dichiarare:** i risultati privacy correnti assumono
  che tutte le label PII siano oscurate. Se l'API consente label selettive, gli
  errori di tipo devono essere trattati come potenziale leakage.

## Checkpoint e dati autorevoli locali

| Ruolo | Path locale autorevole | SHA-256 modello/group | Stato |
|---|---|---|---|
| Teacher FP32 | `artifacts/quantization/model` | `f10d112dffdc69da1e4d23a698e67176d08ac182cb686b4288b8a0ab7670fc79` | riferimento |
| C, parent distillato | `artifacts/training/runs/full-clean-1024-kd-type-boundary-a010-t2-20260823/final` | `7c3221b3387db2285d28e975145b9884f77fd4077b8264c2e7cee54a49b2e63a` | baseline parent |
| D0 controllo | `artifacts/training/runs/id-doc-d0-control-ft64-20260823/checkpoints/checkpoint-64` | `7e6086846730499bac9a8af3ebe908832cea01d3c0191ee1a1ad21937f353a24` | miglior exact student osservato |
| D1 LR `5e-6` | `artifacts/training/runs/id-doc-d1-targeted-ft64-20260823/final` | `2561e757f8aa73483efdac01740dfca5b0e12cefd7aa7fa77e65c58af24bdd7f` | candidato privacy corrente |
| D1 LR `2e-6` | `artifacts/training/runs/id-doc-d1-targeted-ft64-lr2e6-20260823/checkpoints/checkpoint-64` | `8838cac4276ccd23352ac7d1e4c242f8486d4ed62606235df17d7b09d273b4cf` | ablation respinta |
| D2 value-only | `artifacts/training/runs/id-doc-d2-synthetic-ft64-v2-20260823/checkpoints/checkpoint-64` | `778e0d8662da3e0c640d01d8215872a21fd9744bff10c2a084156f25dbd3b1b1` | gate fallito |

Le directory D0 e D1 LR `2e-6` appartengono a run completati via resume esatto,
ma i loro `training-summary.json` storici hanno `final_model=null`; i path e gli
hash autorevoli sono quelli registrati in E-011. I relativi
`run-manifest.json` hanno inoltre `result_status="partial_guard_stop"` e nessun
`authoritative_model`, pur avendo `status="complete"`: l'eccezione e'
inventariata esplicitamente nel checkpoint artefatti. Non copiare un checkpoint
in `final/` per "normalizzarlo": una copia cambierebbe la provenienza senza
aggiungere evidenza.

Dataset development osservati:

- `artifacts/training/data/clean-train-1024.jsonl`, SHA-256
  `95e179775ae388c271bc5fb058614fe09961f99c4fddd0175f2beb875bd451b6`;
- `artifacts/training/data/clean-validation-512.jsonl`, SHA-256
  `4001d5fce099f75cdcd03101f228187b03c1390774801bc818fd586b2ee9bafb`;
- D1 train, SHA-256
  `ea159d8b26aa4132e2feda42a7d3389e7c82d69ae0fe4ab6afb824ff418b3e6f`;
- D0 train, SHA-256
  `ace4708abca7f819f8e03741c4fa4aa9f9a0da83f9839cbbd33a45ad0d7304d8`;
- D2 train, SHA-256
  `026e5c1ad2cde5be38a2623f65a38a27121d7b0bb26b53c02d74f11d563a0e7e`.

## Risultati da cui ripartire

| Modello | Micro/Macro F1 exact | TP/FP/FN | Leak caratteri | Documenti privacy completi | Collateral |
|---|---:|---:|---:|---:|---:|
| Teacher | `0,9963527 / 0,9960684` | `17.210/67/59` | 104 | `500/512` | 2 |
| C | `0,9956003 / 0,9953513` | `17.198/81/71` | 80 | `498/512` | 28 |
| D0 | `0,9957745 / 0,9954137` | `17.203/80/66` | 57 | `502/512` | 37 |
| D1 LR `5e-6` | `0,9957462 / 0,9951381` | `17.205/83/64` | **26** | **`507/512`** | 58 |
| D1 LR `2e-6` | `0,9953994 / 0,9949231` | `17.201/91/68` | 49 | `504/512` | 58 |
| D2 | `0,9953391 / 0,9949371` | `17.191/83/78` | 104 | `496/512` | **14** |

I valori privacy sono un ricalcolo post-hoc sul dev sintetico gia' osservato.
Per D1 contro teacher, il bootstrap paired 20k/seed `20260822` non rende
conclusivo al 95% bilaterale il miglioramento di character o document coverage;
l'aumento di collateral masking di D1 e' invece netto. Vedere E-014 per gli
intervalli completi.

## Set sigillati: non aprire

I manifest riportano ancora `consumed=false`:

| Set | Path | SHA-256 |
|---|---|---|
| Holdout globale v1, 2.048 | `artifacts/training/data/id-doc-targeted-ft-v1/sealed-global-2048.jsonl` | `c929c65bc979ddb3d25c9944b65799379f528a7246000f275944871f0a9e5b19` |
| Challenge `ID_DOC` v1, 2.048 | `artifacts/training/data/id-doc-targeted-ft-v1/sealed-id-doc-2048.jsonl` | `44dba24d082004d48de1e32cc979fe58184bac1f9b8dde645c210d5e34cc4c1f` |
| Challenge sintetico D2, 384 | `artifacts/training/data/id-doc-targeted-ft-v2/sealed-synthetic-id-doc-challenge-384.jsonl` | `1b0dd788a65788ddc97a7386f48ad39ae0a6d3bba9b4cb22f52ebc171c8417cf` |

Non usare questi file per scegliere dataset, architettura, learning rate,
soglie o quantizzazione. Verranno aperti una volta sola dopo freeze del
protocollo, confrontando teacher FP32, student FP32 congelato e un solo candidato
quantizzato preregistrato. Il challenge D2 resta chiuso anche se D2 e' stato
respinto.

## Roadmap immediata, in ordine

1. **Gate v2 prospettico.** Fissare prima di D3 soglie per character/entity/
   document coverage, typed coverage, label protette, collateral ratio e
   exact-span diagnostico. Dichiarare quali label l'API oscura e il costo
   relativo di leak e overmasking.
2. **`PIIEngine` unico.** Rendere evaluator, REST e backend originale
   intercambiabili dietro un contratto `label/start/end/score`; verificare
   chunk, merge, regex/checksum, selezione label, placeholder e mapping
   reversibile end-to-end. UI e REST non devono dipendere da PyTorch, ONNX o
   MLX.
3. **D3 + probe 384/768.** Costruire minimal pair fattoriali
   `formato x contesto x superficie`, circa 1:1 `ID_DOC/DOCID`, con famiglie di
   template disgiunte. In parallelo, confrontare encoder congelati 384 e 768
   con lo stesso piccolo classificatore per capire se l'informazione e'
   separabile senza altro fine-tuning.
4. **Depth reduction.** Solo dopo il gate D3, provare `22 -> 18 -> 16` layer con
   retraining/distillazione e un controllo causale per ogni riduzione. Non
   cancellare layer zero-shot.
5. **Freeze FP32.** Congelare pesi, tokenizer, label, aggregatore, policy REST,
   manifest, hash e soglie. Nessuna quantizzazione prima di questo punto.
6. **Quantizzazione CPU-first.** ONNX FP32 per parita', poi INT8 completo
   inclusi embedding, W6 e W4 reali. Un kernel W6 custom si giustifica soltanto
   se la qualita' e il footprint lo rendono un punto di Pareto migliore di W8 e
   W4. W2/FP8 restano esperimenti successivi, non scorciatoie.
7. **Edge deployment.** Misurare l'intero processo REST, non solo i pesi: VPS
   low-cost e Raspberry Pi 5 prima, Zero 2 W poi, Zero prima versione come
   stress test funzionale ARMv6. Target ambizioso: picco processo `<=256 MiB`;
   riportare PSS/USS/RSS, cold start, latenza e throughput su hardware reale.
8. **MLX nice-to-have.** Dopo aver selezionato il formato quantizzato, portare
   lo stesso checkpoint su MLX e verificare parita' logits/span prima di
   confrontare FP16/W8/W6/W4 su Apple Silicon. Il port MLX non sostituisce il
   percorso CPU/VPS/Raspberry.

## Verifica locale e ripartenza sicura

L'interprete usato per gli ultimi run e test su questa macchina e'
`/tmp/rizzo-pii-quant-venv/bin/python`, con NumPy 2.5.2. E' una directory
temporanea e non fa parte del checkpoint; se non esiste, ricreare un virtualenv
dal file `requirements-training.txt` prima di eseguire training o evaluator.

Verifica suite e CLI senza aprire holdout:

```bash
PII_PY=/tmp/rizzo-pii-quant-venv/bin/python
"$PII_PY" -m unittest discover -s tests
"$PII_PY" -m src.training.evaluate_student --help
"$PII_PY" -m src.training.train_student --help
```

Verifica gli hash dei due modelli principali:

```bash
shasum -a 256 \
  artifacts/quantization/model/model.safetensors \
  artifacts/training/runs/id-doc-d1-targeted-ft64-20260823/final/model.safetensors
```

Valori attesi per i file pesi: teacher
`b1e046e3aec52c3cd74eaf166b034644a4dac0c91128bf06803e1478657fcd9f`,
D1 `f9ff430d39c3913c7d93da51fd42ed0b945c97b25f7f4aefc9d2c89127e28986`.

Riproduci il rescore privacy in `/tmp`, senza inferenza e senza sovrascrivere i
report di checkpoint:

```bash
"$PII_PY" -m src.training.evaluate_student rescore-privacy \
  --dataset-path artifacts/training/data/clean-validation-512.jsonl \
  --predictions-path artifacts/training/runs/id-doc-d1-targeted-ft64-20260823/evaluation/clean-validation-512-v2/student-predictions.jsonl \
  --expected-predictions-sha256 94ad6c6bf652949c44d05f1e67e0b7637f11027969476b53a90f18aaef1d0cc0 \
  --output-report /tmp/rizzo-pii-d1-privacy-rescore.json
```

Riproduci il confronto exact/disagreement dalle predizioni salvate:

```bash
"$PII_PY" -m src.training.evaluate_student compare \
  --teacher-report artifacts/training/runs/pilot-clean-256-full-adafactor-20260822/evaluation/clean-validation-512-v2/teacher-report.json \
  --student-report artifacts/training/runs/id-doc-d1-targeted-ft64-20260823/evaluation/clean-validation-512-v2/student-report.json \
  --teacher-predictions artifacts/training/runs/pilot-clean-256-full-adafactor-20260822/evaluation/clean-validation-512-v2/teacher-predictions.jsonl \
  --student-predictions artifacts/training/runs/id-doc-d1-targeted-ft64-20260823/evaluation/clean-validation-512-v2/student-predictions.jsonl \
  --output-report /tmp/rizzo-pii-d1-vs-teacher.json
```

Non esiste ancora un config D3 materializzato: non improvvisare un nuovo run da
questa pagina e non fare resume di D2, che e' gia' completo e respinto. Il
prossimo comando di training va scritto soltanto dopo il lock del gate v2 e deve
puntare a un nuovo output directory, con `--parent-run` fail-closed e dati D3
hashati. `--resume-exact` e' riservato alla continuazione della stessa run dopo
uno stop della memory guard; non e' un metodo per inizializzare una nuova
ablation.

## Persistenza e backup

`.gitignore` esclude esplicitamente `artifacts/training/` e
`artifacts/quantization/`. Di conseguenza checkpoint, dataset materializzati,
predizioni e report sopra elencati esistono localmente ma non entrano in un
normale commit Git. `docs/CURRENT_STATE.md` e `docs/EXPERIMENT_LOG.md` conservano
la provenienza narrativa, non i byte dei modelli.

Prima di considerare il checkpoint trasferibile a un clone pulito servono:

- la verifica dell'inventory `docs/checkpoints/2026-08-23-artifacts.md` contro
  i byte locali prima dell'upload;
- una destinazione esterna verificata per modelli, dataset e report;
- il round-trip di almeno un artefatto dall'archivio e la verifica del suo hash.

Destinazione/URI esterna e hash dell'inventory: **da compilare in
`docs/checkpoints/2026-08-23-artifacts.md`**. Finche' questi campi non esistono,
questa working copy e il relativo storage locale restano l'unica copia
confermata degli artefatti ignorati da Git.
