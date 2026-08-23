# Ottimizzazione architetturale per una REST PII CPU-only

## Verdetto iniziale

Per ridurre davvero RAM e latenza, il primo esperimento non dovrebbe essere un
Mixture-of-Experts. Il checkpoint attuale spreca la maggior parte dei parametri
nella combinazione di vocabolario multilingue molto ampio e hidden size 768. La
sequenza con il miglior rapporto rischio/beneficio è:

1. student ModernBERT con hidden size 384 e tokenizer invariato;
2. distillazione dal teacher PII corrente;
3. selezione fra 22, 18 e 16 layer sulla sola qualità FP32;
4. INT8, INT6 e INT4 del migliore student;
5. solo dopo, tokenizer italiano più piccolo o embedding fattorizzato;
6. kernel W6 custom soltanto per l'ISA della VPS realmente scelta.

Il MacBook Air usato per lo spike è fanless e va in thermal throttling. I tempi
locali servono a verificare che un kernel venga davvero eseguito, non a scegliere
architettura o runtime. La selezione dello student usa qualità e robustezza;
latenza, throughput e memoria verranno misurati sulla VPS target con processi
isolati e condizioni termiche controllate.

La recipe, l'audit dei dataset e le stime di memoria del retraining sono in
[`STUDENT_TRAINING.md`](STUDENT_TRAINING.md).

## Dove sono i parametri

Il conteggio viene dal `config.json` e dal checkpoint Safetensors bloccato in
`artifacts/quantization/model/`.

| Componente | Parametri | Quota del modello |
|---|---:|---:|
| Embedding `256000 × 768` | 196.608.000 | 63,92% |
| Attention, 22 layer | 51.904.512 | 16,88% |
| MLP, 22 layer | 58.392.576 | 18,99% |
| Norm e testa PII | circa 660.000 | 0,21% |
| **Totale** | **307.564.845** | **100%** |

Togliere layer riduce bene il calcolo, ma poco il footprint finché l'embedding
resta invariato. Ridurre hidden size agisce contemporaneamente sull'embedding e
sui blocchi Transformer.

## Larghezza e numero di layer

`768` non è solo la dimensione dei vettori di input: è la larghezza nascosta di
tutta la rete. Uno student largo 384 dimezza l'embedding e riduce in modo
quadratico molte matrici dei layer.

| Architettura | Parametri stimati | Pesi INT8 ideali | MAC lineari relative |
|---|---:|---:|---:|
| Teacher attuale, 22×768 | 307,56M | 293,3 MiB | 100% |
| Teacher a 16 layer | 277,47M | 264,6 MiB | 72,7% |
| `mmBERT-small`, 22×384 | 140,66M | 134,1 MiB | 38,2% |
| Student 12×384, vocab 256k | 121,48M | 115,9 MiB | 20,9% |
| Student 22×384, vocab circa 51k | 62,02M | 59,1 MiB | 38,2% |
| Student 12×384, vocab circa 51k | 42,84M | 40,9 MiB | 20,9% |

I valori di memoria sono il solo payload teorico dei pesi: non includono scale,
grafo, tokenizer, allocator, prepacking e workspace del runtime.

Il target edge è **256 MiB per il processo completo**, non per il solo file del
modello. Per lasciare spazio a runtime, tokenizer, prepacking, workspace,
attivazioni e API, l'artefatto dovrebbe idealmente stare fra 60 e 120 MiB. Il
teacher a 12 layer non raggiunge questo obiettivo: il suo embedding invariato da
196,61M parametri domina ancora la memoria. Uno student largo 384, soprattutto
se poi quantizzato, è invece nella fascia in cui il target diventa plausibile.
Il rispetto dei 256 MiB va validato come PSS/USS su Linux e su Raspberry Pi
reale; la dimensione dell'ONNX non basta.

[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) è il
punto di partenza preferito: usa la stessa famiglia ModernBERT e lo stesso
tokenizer/vocabolario, ma hidden size 384 e 6 head. Consente quindi fine-tuning e
distillazione token-aligned senza introdurre subito la variabile tokenizer.

L'ablation zero-shot è un **diagnostico storico word-level/first-subword**, non
una selezione definitiva dei layer: non include ancora il nuovo gate character-span
su tutti i subword. Su 512 documenti, togliere un layer alla volta produce delta micro-F1 compresi fra
`+0,000902` per il layer 5 e `-0,091960` per il layer 0. I meno sensibili sono
`5, 16, 15, 18, 3, 7`; il layer 0 è imprescindibile e i layer finali 19–21
pesano soprattutto sulla macro-F1.

Togliere insieme i sei layer apparentemente meno sensibili non conserva però la
qualità. Su un secondo subset di 256 documenti, la migliore variante 16-layer
senza retraining (`3,5,7,15,16,18` rimossi) scende da micro-F1 `0,986474` a
`0,849574` e da macro-F1 `0,983335` a `0,769204`; aggiunge 50 span e ne rimuove
102. Tutte le varianti 12-layer scendono sotto `0,10` di micro-F1. Preservare
tutti i layer globali è persino peggiore perché crea buchi consecutivi nella
trasformazione delle rappresentazioni.

Quindi non useremo un checkpoint ottenuto cancellando layer. Per il pruning
servono rimozione greedy con nuova valutazione dopo ogni passo, retraining e
distillazione. Il primo gradino sensato è 18 layer, poi 16; un eventuale 12-layer
va distillato dal 16-layer, non direttamente dal teacher 22-layer.

Dettaglio delle combinazioni zero-shot, su subset deterministico ma distinto da
quello leave-one-out:

| Variante | Layer rimossi | micro-F1 | Delta micro | macro-F1 | Delta macro | Doc. identici |
|---|---|---:|---:|---:|---:|---:|
| 16L uniforme | `2,5,9,12,16,19` | 0,637363 | -0,349112 | 0,541443 | -0,441892 | 109/256 |
| 16L ranking LOO | `3,5,7,15,16,18` | **0,849574** | **-0,136900** | **0,769204** | **-0,214131** | **170/256** |
| 16L preserva globali | `5,7,10,14,16,17` | 0,507368 | -0,479106 | 0,340087 | -0,643248 | 73/256 |
| 12L uniforme | `1,3,5,7,9,12,14,16,18,20` | 0,091884 | -0,894591 | 0,084467 | -0,898868 | 29/256 |
| 12L ranking LOO | `3,5,7,10,13,14,15,16,17,18` | 0,099125 | -0,887349 | 0,081100 | -0,902235 | 26/256 |
| 12L preserva globali | `4,5,7,8,10,13,14,16,17,19` | 0,003591 | -0,982884 | 0,006061 | -0,977275 | 23/256 |

Sono esperimenti diagnostici word-level su 256/512 documenti, non risultati di
rilascio. Servono a escludere il pruning zero-shot e a decidere il metodo di
training; il ranking LOO non guiderà la rimozione greedy finché ogni candidato
non sarà rivalutato anche character-level e per tag.

Il gate non si riduce alla sola micro-F1. Per ogni architettura vanno controllati
micro e macro-F1 entity-level, precision/recall per tipo, span aggiunti e rimossi
rispetto al teacher, documenti identici, bucket per lunghezza e casi ambigui come
nomi che sono anche parole comuni. CF, IBAN, email, telefono e identificativi
documento hanno recall protetta. Gli span vanno ricostruiti come nella pipeline
di produzione, includendo tutti i subword e gli offset: guardare soltanto il
primo subword può nascondere un codice spezzato.

Il confronto finale ha quattro assi distinti:

1. teacher FP32 contro gold umano;
2. student FP32 contro lo stesso gold umano, che è il confronto primario;
3. disagreement exact-span student/teacher, senza chiamare il teacher ground truth;
4. student quantizzato contro gold, contro il proprio FP32 e contro il teacher.

La validation da 7.000 record è già stata usata ripetutamente per quantizzazione
e ablation, quindi non è più un test completamente cieco. Prima del training va
creato o recuperato un test finale intatto. Il `metrics.json` del modello dichiara
36.297 record di validation, ma localmente ne sono presenti solo 7.000: le altre
29.297 righe vanno recuperate e ne va verificata l'indipendenza dal train.

Il gate character-span appena aggiunto è model-only: il dataset locale contiene
token e BIO label, non testo originale e offset umani. Il secondo gate realmente
end-to-end richiede un corpus raw-text annotato e confronta l'output dopo modello,
regex/checksum e merge di produzione. Entrambe le viste sono necessarie: la prima
non nasconde debolezze della rete, la seconda misura ciò che riceverà il client REST.

## Embedding

Se si vuole mantenere il tokenizer esistente, una seconda strada è la
fattorizzazione in stile ALBERT:

```text
Embedding(256000, 128) -> Linear(128, hidden_size)
```

Le stime diventano circa 143,82M parametri sul body corrente o 75,17M sul body
largo 384. Richiede però un modello custom, inizializzazione SVD e distillazione;
non è una conversione lossless.

Per lo student largo 384, il candidato prioritario è una dimensione embedding
`E=192` proiettata nella hidden size `H=384`:

```text
token ID -> Embedding(256000, 192) -> Linear(192, 384, bias=False) -> Transformer
```

Il modello risultante è stimato in **91.580.589 parametri**: si sostituiscono i
98.304.000 parametri dell'embedding 256k x 384 con 49.152.000 parametri di lookup
e 73.728 della proiezione bias-free. Il solo payload teorico dei pesi sarebbe:

| Formato | Payload teorico E192 -> H384 |
|---|---:|
| FP32 | 349,35 MiB |
| BF16 | 174,68 MiB |
| INT8 | 87,34 MiB |
| INT4 | 43,67 MiB |

Questi valori non includono scale, allineamento, prepacking, grafo, workspace,
attivazioni, tokenizer o processo REST. Non è necessario inventare un nuovo
"motore di embedding": il forward usa operazioni standard, un `Gather` seguito
da un `MatMul`. Serve comunque una definizione/configurazione di modello che
inserisca la proiezione e che i backend di esportazione la rappresentino
correttamente; un kernel custom si valuta solo dopo misure sul runtime target.

La trasformazione non conserva esattamente i pesi. Va inizializzata con
SVD/randomized SVD della tabella E384 e poi addestrata, preferibilmente con
distillazione. L'esperimento E192 -> H384 viene quindi dopo il consolidamento
della baseline 22x384 e del fine-tuning distillato: in questo modo la perdita
dovuta alla fattorizzazione resta isolabile.

Un tokenizer italiano da 50–64k ridurrebbe ancora di più il footprint, ma non si
può costruire conservando soltanto gli ID osservati nel subset locale. Fra
validation e calibrazione sono presenti 27.327 ID distinti, e 5.122 ID della
validation non compaiono nella calibrazione. Servono un corpus italiano molto
più ampio, byte fallback, continued pretraining e regressione specifica su CF,
IBAN, email e documenti.

## Perché un MoE classico non aiuta

Un MoE classico replica gli FFN e mantiene residenti tutti gli expert. Con
quattro expert completi top-1, questo modello passerebbe da 307,56M a circa
482,81M parametri. I parametri attivi per token resterebbero quasi uguali al
modello denso, con in più il routing. È coerente con lo scopo di
[Switch Transformer](https://www.jmlr.org/beta/papers/v23/21-0998.html): aumentare
la capacità a compute per token quasi costante, non diminuire la RAM residente.

Una MoEfication che partiziona l'FFN esistente conserva quasi invariata la
memoria e può saltare una parte delle MAC. Con quattro partizioni, top-1 ha un
limite teorico di circa 1,66× sulle operazioni lineari. Sui documenti mediani da
circa 35 token, però, vengono toccate quasi tutte le partizioni e le piccole GEMM
più dispatch/scatter sono sfavorevoli alla CPU. È un esperimento successivo, non
il primo intervento. Riferimento: [MoEfication](https://aclanthology.org/2022.findings-acl.71/).

## Cosa significa davvero INT4

Il formato dei pesi e il percorso di calcolo vanno dichiarati separatamente:

| Variante | Pesi | Input MatMul | Interpretazione |
|---|---:|---:|---|
| ORT `accuracy_level=1` | INT4 packed | FP32 | W4A32 |
| ORT `accuracy_level=4` | INT4 packed | quantizzazione dinamica INT8 | W4A8 |
| ATen CPU int4pack | INT4 packed affine | FP32 | W4A32 |

L'INT4 ONNX originariamente misurato in questo repository usa
`accuracy_level=4`: il suo vantaggio non può essere attribuito al solo storage
INT4. Il benchmark corretto include anche W4A32. Il probe PyTorch/ATen W4A32 ha
convertito end-to-end tutti i 90 layer lineari e l'embedding, ma resta uno smoke
sintetico locale: dimostra il percorso di esecuzione, non sostituisce la
regressione del modello e non è un confronto prestazionale affidabile sulla VPS.

La post-training quantization W2A8 dei lineari, con embedding W4, è stata
provata su 512 documenti ed è fallita nettamente (`micro-F1 0,368421`). Il task
non è una classificazione binaria: il modello assegna 45 label BIO e deve
ricostruire tipo e confini esatti di circa 22 categorie. INT2 non è quindi un
candidato PTQ per lo student; QAT o distillazione a 2 bit sarebbero un progetto
di training separato, da considerare soltanto dopo uno student FP32 robusto.

## INT6

ONNX Runtime 1.29 e `MatMulNBits` offrono kernel packed per 2, 4 e 8 bit, non un
percorso CPU INT6 general-purpose. Salvare valori a 6 bit in un contenitore INT8
misurerebbe la qualità W6, ma non il footprint né il compute W6.

Un candidato reale sarebbe **W6A8 signed groupwise**, group size 128, con scale
per blocco e packing denso. L'asimmetrico e group size 64 sono ablation da usare
solo se la qualità del simmetrico non supera il gate. Il payload teorico del
modello corrente starebbe fra W4 e W8, circa 229 MiB incluse stime conservative
per scale, ma il vantaggio esiste soltanto se il kernel consuma direttamente il
packing a 6 bit.

Scrivere il kernel è possibile, ma va fatto sulla CPU target:

- x86_64: almeno AVX2; VNNI/AVX-512 cambiano radicalmente la strategia W4/W6A8;
- ARM64: NEON/DotProd o I8MM, con packing diverso;
- embedding: lookup packed e dequantizzazione delle sole righe richieste;
- GEMM: fused unpack/dequant/dot-product, senza materializzare tutti i pesi FP32.

Prima dell'integrazione serve la regressione qualità W8/W6/W4 sullo student
congelato, poi microbenchmark sulle sue shape reali. Il kernel W6 ha senso solo
se recupera una parte materiale degli errori W4 mantenendo un footprint
inferiore a W8; latenza e PSS verranno fissati e misurati sulla macchina target,
non sul portatile.

## Backend MLX futuro

MLX è applicabile su Apple Silicon, ma l'[elenco modelli di
`mlx-lm`](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models) non
include oggi un `ModernBertForTokenClassification` pronto all'uso. Il port va
quindi scritto e validato esplicitamente: embedding, blocchi pre-norm, QKV fusa, RoPE, alternanza
di attenzione locale/globale, MLP GeGLU e testa a 45 label. L'esempio BERT di
MLX non è intercambiabile con ModernBERT.

Il vantaggio è che la [quantizzazione
MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
offre già 2/3/4/5/6/8 bit, mentre i layer includono
[`QuantizedLinear` e `QuantizedEmbedding`](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py).
In particolare W6 è un formato affine packed realmente
eseguito dai kernel Metal, non un INT8 etichettato come 6 bit. Non serve quindi
un kernel W6 custom per il Mac. Il tokenizer Hugging Face e la ricostruzione
degli offset possono restare condivisi fra PyTorch, ONNX e MLX.

Il port verrà fatto soltanto dopo aver congelato lo student. La parità si
verifica progressivamente su embedding, layer locali/globali, testa, logits,
argmax e infine span sull'intera validation. Solo dopo si confrontano MLX
FP16/W8/W6/W4. Per misure future occorre sincronizzare le operazioni lazy con
`mx.eval()`; sul MacBook Air il soak termico va separato dalla regressione
qualità.

## Piano sperimentale

1. Rendere production-like la regressione: tutti i subword, offset/span,
   metriche per tag e bucket critici.
2. Fine-tuning diretto di `mmBERT-small`.
3. Se necessario, knowledge distillation con loss supervisionata BIO più KL sui
   logits del teacher; opzionale proiezione hidden 768→384.
4. Valutare student 22/18/16 layer in FP32, poi quantizzare soltanto quelli che
   superano il gate.
5. Provare embedding fattorizzato o tokenizer italiano solo dopo aver fissato
   una baseline student affidabile.
6. Generare W8/W6/W4 dallo student e misurare prima la qualità.
7. Portare benchmark e l'eventuale kernel W6 sulla VPS target prima di scegliere
   il formato di produzione; in parallelo portare lo stesso student in MLX.

Per ogni student vanno separati tre confronti: student FP32 contro teacher FP32,
student quantizzato contro student FP32 e student quantizzato contro teacher
FP32. La regressione deve includere span a offset carattere equivalenti alla
pipeline reale, non soltanto la label del primo subword.
