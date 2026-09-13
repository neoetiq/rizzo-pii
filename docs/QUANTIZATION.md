# Quantizzazione CPU: INT8, INT4, INT2 e regressione

Questo percorso è uno **spike riproducibile**: confronta il checkpoint Hugging Face
`rizzoaiacademy/rizzo-pii-0.3B` **v1.5.0** FP32 con candidati ONNX CPU. Non modifica il
modello pubblicato, non modifica la desktop app e non introduce ancora la REST API
minimale. Produce artefatti e misure con cui scegliere il runtime del futuro servizio.

## Obiettivo e perimetro

Il target è una VPS **CPU-only**. I candidati dichiarano separatamente bit dei
pesi e percorso delle attivazioni:

| Candidato | Operatori che cambiano | Scopo |
|---|---|---|
| ONNX FP32 | nessuno | controlla che l'export non introduca regressioni |
| INT8 dinamico | `MatMul` lineari; pesi INT8, attivazioni quantizzate a runtime | percorso ORT tradizionale |
| W8A8 MatMulNBits | `MatMul`; pesi INT8 e input quantizzato dinamicamente INT8, embedding FP32 | controllo qualità dei lineari |
| INT8 completo | `MatMul` W8A8 + embedding INT8 per-riga, dequantizzato dopo `Gather` | compromesso qualità/footprint |
| W4A32 | `MatMul` con pesi INT4 e input FP32; embedding W4 | separa costo del kernel FP32 |
| W4A8 | stessi pesi INT4, input MatMul quantizzato dinamicamente INT8; embedding W4 | candidato footprint minimo |
| W2A8 + embedding W4 | MatMul W2, embedding W4 | esperimento esplicito, già bocciato dalla qualità |

Le due INT8 standard di ONNX Runtime non comprimono la tabella di embedding. La variante
`int8-full` usa invece un initializer INT8 e una scala FP32 per riga, con soli operatori
ONNX standard (`Gather`, `Cast`, `Mul`): si dequantizzano soltanto i token richiesti dal
documento. Le varianti INT4 devono includere anche
`Gather`/embedding **solo** quando l'operatore quantizzato è eseguibile sulla CPU-target;
altrimenti il run viene marcato non idoneo, non promosso come INT4.

`accuracy_level=1` di `MatMulNBits` identifica W4A32;
`accuracy_level=4` identifica il percorso W4A8 con quantizzazione dinamica
dell'input. La vecchia descrizione generica “INT4 weight-only” non era
sufficiente a interpretare le prestazioni.

ONNX Runtime 1.29 supporta MatMul a 2 bit, ma il suo
`GatherBlockQuantized` CPU è soltanto W4. Per questo l'esperimento `int2` è
onestamente mixed-bit: W2A8 sui 90 MatMul e W4 sull'embedding dominante.

FP8 è volutamente fuori da questo spike. Su CPU general-purpose non è il formato di
inferenza standard di ONNX Runtime, spesso porta ricadute a FP16/FP32 e non riduce in
modo affidabile RAM o latenza. È pertinente a GPU con hardware FP8, non al target
CPU-only.

## Ambiente e sorgenti bloccate

Creare un ambiente dedicato e installare le dipendenze opzionali:

```bash
python -m venv .venv-quant
. .venv-quant/bin/activate
pip install -r requirements-quantization.txt
```

`constraints-quantization.txt` fissa le versioni dirette effettivamente validate; viene
applicato automaticamente dal requirements. Il run di riferimento usa Python 3.13.13,
ONNX Runtime 1.29.0, Torch 2.13.0 e Transformers 4.57.6.

La configurazione in [`src/quantization/config.json`](../src/quantization/config.json)
blocca commit immutabili, non il ramo mobile `main`:

- modello `v1.5.0`: `a1c3c83827eca22e9675e30c1111c4641caf5901`;
- dataset: `3a61195c0ab01d1db40f799493e60ce3dc490291`;
- regressione: `validation/validation_real.jsonl`, esattamente 7.000 record;
- calibrazione: `subsets/train_subset_10k.jsonl`, esattamente 10.000 record.

Il comando è:

```bash
python -m src.quantization.cli sources
```

Scarica in `artifacts/quantization/`, valida per ogni riga `tokens` e `bio_labels`
(liste non vuote e di uguale lunghezza) e produce
`artifacts/quantization/sources.lock.json`. Il lock conserva revisioni risolte, SHA-256 e
dimensione di tutti i file, conteggi JSONL e versioni di Python/pacchetti. Export,
quantizzazione, regressione e profilo rileggono il lock e ricontrollano i file: un hash
diverso interrompe il comando. La regressione verifica le sorgenti sia prima sia
dopo l'intera suite.

## Comandi disponibili

L'interfaccia unica sarà sempre il modulo, per evitare script eseguiti da directory
diverse:

```bash
python -m src.quantization.cli sources
python -m src.quantization.cli export
python -m src.quantization.cli quantize --variant all
python -m src.quantization.cli regress --threads 4 --batch-size 16
```

`benchmark` è un alias di `regress`. Per eseguire l'intera pipeline da zero si può usare:

```bash
python -m src.quantization.cli all --threads 4 --batch-size 16
```

Le varianti accettate singolarmente sono `int8` (dinamico), `int8-weight`,
`int8-full`, `int4-a32`, `int4-a8`/`int4` e l'esperimento esplicito `int2`.
Senza `--variants`, la regressione esegue in processi isolati `torch-fp32`, `onnx-fp32`,
`onnx-int8`, `onnx-int8-weight`, `onnx-int8-full`, `onnx-int4-a32` e
`onnx-int4`. INT2 non fa parte del default perché il suo smoke ha già fallito
nettamente i gate. `--limit N` serve solo
per smoke test: il report lo marca `SMOKE - NOT FOR RELEASE`. Una decisione di rilascio
deve usare tutti i 7.000 documenti. `--enforce-gates` restituisce un codice non zero se
una variante fallisce o il run non è promuovibile.

Per rigenerare confronti e gate senza ripetere l'inferenza, e per simulare richieste
REST a batch 1:

```bash
python -m src.quantization.cli report
python -m src.quantization.cli profile --documents 256 --threads 4 \
  --repeats 6 --warmup-batches 8 --order balanced --memory-sample-ms 20
```

`train_subset_10k` è materiale di calibrazione, mai di regressione. Il confronto di
qualità usa soltanto i 7.000 esempi held-out. Le quantizzazioni correnti sono
post-training dinamiche o groupwise/DEFAULT affine e quindi **non usano** calibrazione; il subset è bloccato ora per rendere
riproducibile un eventuale esperimento statico INT8 successivo. Se una variante non può
essere esportata o eseguita dal provider CPU, il comando fallisce esplicitamente invece
di degradare silenziosamente a FP32.

La regressione qualità acquisisce prima dell'inferenza snapshot economici di
modello, validation e preprocessor, esegue il modello e calcola gli hash completi
dopo la regione misurata; in questo modo l'hash del modello non scalda la page
cache del solo candidato. Tutte le varianti usano tokenizer e `config.json`/`id2label`
canonici del checkpoint bloccato, non le copie accanto agli ONNX. Il report lega
byte per byte artefatto, validation, tokenizer/config e prediction JSONL; inoltre
richiede la verifica delle sorgenti prima e dopo la suite. Il profilo prestazionale
verifica invece tutti gli hash prima della schedulazione e di nuovo dopo l'intera
suite. `report` rifiuta contenuti cambiati. Solo per leggere risultati legacy già prodotti è disponibile
`report --allow-integrity-backfill`: registra esplicitamente
`backfilled-after-evaluation`; non equivale a un hash acquisito durante l'inferenza e
rimane **non promuovibile**.

Il profilo usa un processo nuovo per ogni variante e repeat, warm-up escluso da
latenza/throughput, ordine bilanciato deterministico e file distinti per ogni
run. Riporta mediana, p05/p95, massimo e variabilità. Su Linux legge anche
`smaps_rollup` per PSS, USS e memoria anonima; su macOS tali campi restano
esplicitamente `n/d`. Un singolo RSS non è più considerato un benchmark valido.

Anche il riuso intermedio è verificato: `quantize --variant int8-full` confronta il
modello INT8 weight-only con il relativo report e con l'ONNX FP32 corrente. Se uno dei
due hash non coincide, rigenera la base prima di quantizzare l'embedding.

## Artefatti e gate di promozione

La pipeline crea directory autonome sotto `artifacts/quantization/`:

- `onnx-fp32/`, `onnx-int8/`, `onnx-int8-weight/`, `onnx-int8-full/`,
  `onnx-int4-a32/`, `onnx-int4/` e l'esplorativo `onnx-int2/`:
  modello, eventuale external data, tokenizer e report di export/quantizzazione;
- `sources.lock.json`: revisioni, hash, dimensioni e versioni software;
- `regression/metrics/*.json`: qualità, p50/p95 per batch, throughput, RSS e dimensione;
- `regression/predictions/*.jsonl`: label, span word-level e character-span
  model-only, senza testo dei documenti, con SHA-256 legato alle metriche;
- `regression/report.json` e `report.md`: confronto e gate rispetto a PyTorch FP32.
- `service-profile/suites/<suite_id>/`: run e prediction batch-1 process-isolati,
  più una copia immutabile del report della suite; `service-profile/report.*`
  punta all'ultima suite verificata senza confondere file residui di campagne
  precedenti. Il subset non è promuovibile come regressione qualità.

Un candidato viene considerato per la VPS solo se completa l'inferenza CPU e rispetta i
gate di **qualità** dichiarati in `config.json`: nessuna regressione oltre soglia sui tag
critici e delta macro/micro F1 entro il limite, sia word-level sia sugli span
character-level ricostruiti da tutti i subword. Servono inoltre gli hash di
predizioni, validation, tokenizer/config e modello acquisiti con i relativi guard,
oltre al pre/postflight delle sorgenti. Dimensione, RSS e latenza sono misurati
ma non hanno una soglia universale: il gate infrastrutturale va fissato sulla VPS target.
I valori correnti sono criteri iniziali espliciti e versionati; non vanno modificati per
far passare retroattivamente un candidato. Una variante assente dalla sezione
`gates`, o priva di almeno una soglia qualità esplicita, fallisce in modalità
fail-closed e non può essere promossa.

## Risultati storici word-level del run completo

Run del 22 agosto 2026: 7.000 documenti, 255.154 token e 15.511 span prodotti dalla
baseline. Batch 16, quattro thread, macOS ARM. Questi numeri sono una regressione
storica first-subword: precedono il salvataggio dei character-span, i digest delle
predizioni e l'integrità evaluation-time della validation/preprocessor. I vecchi
`PASS` indicano soltanto l'esito del gate legacy e **non sono più promuovibili**.
Vanno rigenerati col gate attuale prima di qualsiasi decisione di qualità.

| Variante | Artefatto MiB | micro-F1 | Delta FP32 | Span nuovi/rimossi | Gate |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 1173,3 | 0,988672 | riferimento | - | golden |
| ONNX FP32 | 1173,8 | 0,988672 | 0,000000 | 0 / 0 | legacy PASS; rivalutare |
| INT8 dinamico | 857,4 | 0,985018 | -0,003654 | 135 / 150 | legacy **FAIL** |
| W8A8 MatMul, embedding FP32 | 859,8 | 0,988770 | +0,000098 | 15 / 12 | legacy PASS; rivalutare |
| W8A8 MatMul + embedding W8 | 298,3 | 0,988577 | -0,000095 | 18 / 13 | legacy PASS; rivalutare |
| W4A8 MatMul + embedding W4 | 156,5 | 0,986974 | -0,001698 | 107 / 78 | legacy PASS; rivalutare |

Nel gate legacy, INT8 dinamico viene escluso perché supera il limite di perdita
micro-F1 `0,002`. W8A8
MatMulNBits conserva molto bene la qualità ma non risolve il footprint perché lascia
l'embedding FP32. **INT8 completo è il candidato più conservativo
dell'architettura attuale**: riduce
l'artefatto del 74,6% rispetto a FP32 e cambia la F1 di meno di un decimillesimo. INT4
riduce ulteriormente il disco dell'86,7%, ma modifica più span e perde circa 0,17 punti
percentuali di micro-F1. Nessuno dei due viene ancora scelto come formato finale:
prima si ridurranno hidden size e layer e si selezionerà lo student FP32 sulla
sola qualità. Le soglie release INT4 sono ora allineate a INT8 (`0,002` micro,
`0,003` macro e `0,005` sui recall critici); INT2 resta esplicitamente exploration-only e
non può essere promosso anche qualora un futuro smoke superasse le soglie lasche.

Lo smoke mixed W2A8/W4 su 512 documenti è stato interrotto come candidato:
micro-F1 `0,368421`, macro-F1 `0,373188`, soltanto 78/512 documenti con gli
stessi span del teacher e perdite di recall fino al 100% su `DOCID`. L'artefatto
è 130,1 MiB: appena 26,4 MiB meno del W4 completo. È una chiara bocciatura della
post-training quantization W2; non dimostra l'impossibilità di QAT/distillazione
a 2 bit, che sarebbe un esperimento di training diverso.

Il vecchio profilo batch-1 a singolo run è **ritirato**: hashava l'artefatto
prima del campionamento, non faceva warm-up e confrontava RSS macOS in ordine
fisso. Ripetizioni alternate hanno invertito più volte il rapporto RAM fra INT4
e INT8, dimostrando che il dato “INT4 usa più RAM” non era stabile. Anche il
successivo report multi-run locale precede il namespace di suite e la verifica
dei prediction file introdotti dallo schema 2: resta diagnostico e il rebuild lo
rifiuta esplicitamente. Non esiste quindi ancora una graduatoria prestazionale
promossa; la nuova suite va eseguita sulla VPS Linux target.

Il profilo diagnostico ha usato cinque processi freschi per variante, warm-up e ordine
bilanciato. Il MacBook Air è però fanless: il thermal throttling rende i numeri
di velocità inadatti a scegliere il runtime. Il report resta utile per
verificare isolamento, memoria e code path, per esempio la forte differenza fra
W4A32 e W4A8, ma la graduatoria prestazionale è rinviata alla VPS target. La
qualità non dipende da questi tempi.

Transformers 4.57.6 segnala erroneamente questo tokenizer ModernBERT/Gemma come possibile
tokenizer Mistral. La pipeline passa esplicitamente `fix_mistral_regex=False`: conserva
il pre-tokenizer Metaspace pubblicato e usato dalla baseline, senza applicare una patch
Mistral incompatibile.

## Probe PyTorch INT4 e INT6

`src/quantization/lowbit_probe.py` separa storage, attivazioni e kernel. Sul
checkpoint reale ha convertito 90/90 `nn.Linear` al kernel privato ATen
`_weight_int4pack_mm_for_cpu` W4A32 e l'unico embedding a nibble packed. Lo
storage persistente dei pesi target è circa 165,0 MiB, ma l'embedding non ha un
kernel fused: le sole righe selezionate vengono unpacked/dequantizzate con
normali operazioni PyTorch.

Smoke sintetico su Mac ARM, sequenza 64, quattro thread, due campioni dopo
warm-up:

| Percorso PyTorch | Mediana modello |
|---|---:|
| FP32 | 40,15 ms |
| ATen W4A32 | 579,56 ms |

Quindi il percorso ATen disponibile su questo host è circa 14,4× più lento del
FP32 in questo smoke. È una prova end-to-end del kernel, non una regressione
dataset né un benchmark affidabile: throttling e stato termico del Mac possono
cambiare i tempi, e il dato non è trasferibile a x86. Per il deployment servirebbe inoltre serializzare e caricare
direttamente i buffer packed senza materializzare prima il teacher FP32.

INT6 resta estimate-only: PyTorch espone un dtype shell `int6`, ma non prova
packing o kernel CPU; ONNX/ORT non hanno un percorso W6 general-purpose. La
stima groupwise simmetrica, group 128 e scale FP32, è circa 229,1 MiB per i
pesi target. Un contenitore INT8 con valori a 6 bit avrebbe qualità W6 ma costo
di storage/compute INT8 e non viene accettato come risultato.

Prima di scrivere un kernel W6 custom conviene provare una precisione mista W4/W8
sui layer sensibili: usa kernel maturi e può ottenere un budget medio vicino a
6 bit senza il packing irregolare di W6.

## Caveat VPS

La compatibilità va verificata sulla **stessa famiglia CPU** della VPS prevista
(architettura x86_64/ARM, AVX2/AVX-512 o assenza di tali istruzioni, RAM e vCPU). Una
misura su Mac o workstation non dimostra né picco RSS né latenza della VPS. INT4 può
diminuire il peso su disco ma essere più lento di INT8 su CPU senza kernel ottimizzati:
la decisione usa memoria *e* p95, non soltanto la dimensione del file.
