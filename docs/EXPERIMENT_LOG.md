# Registro sperimentale — PII edge

Questo documento e' la sorgente narrativa del futuro paper. Ogni esperimento
deve distinguere chiaramente **misure**, **stime**, **ipotesi** e **decisioni**.
I report machine-readable, i manifest e i checkpoint restano negli artefatti del
run; qui vengono registrati solo risultati verificabili e riferimenti ai loro
hash.

## Regole del registro

Per ogni esperimento annotare:

1. ID e data;
2. domanda/ipotesi;
3. variabile modificata e baseline;
4. revisioni di modello, tokenizer, dati e codice;
5. hardware, runtime e configurazione;
6. metriche di qualita', memoria e solo successivamente velocita';
7. risultato, limiti e decisione successiva;
8. path e SHA-256 degli artefatti che provano il risultato.

Un risultato diagnostico non diventa automaticamente una conclusione di
release. Le metriche legacy first-subword, i run singoli di velocita' e le stime
teoriche devono restare etichettati come tali.

## Decisioni e osservazioni iniziali

### D-001 — Ottimizzare l'architettura prima della quantizzazione

- **Stato:** accettata.
- **Motivo:** quantizzare il teacher riduce i byte dei pesi, ma non elimina
  parametri e calcolo non necessari. Prima si cerca il piu' piccolo modello FP32
  che conserva la qualita'; W8/W6/W4/W2 vengono applicati dopo.
- **Ordine:** student 22x384 -> eventuale riduzione depth/embedding -> quant.

### D-002 — Baseline student `mmBERT-small`

- **Stato:** compatibilita' verificata, training da eseguire.
- **Backbone:** `jhu-clsp/mmBERT-small`.
- **Revisione:** `abc32620dd4f6ab06f5fbe905dc25f310618e09f`.
- **Architettura:** ModernBERT, 22 layer, hidden 384, intermediate 1152,
  6 attention head, vocab 256.000, contesto 8.192.
- **Parametri stimati con testa PII:** circa 140,66M, di cui 98,30M embedding.
- **Confronto teacher:** stessa profondita' e stesso head dimension 64; hidden
  768->384 e attention head 12->6.

### D-003 — Contratto tokenizer e label

- **Stato:** verificato sul corpus locale.
- Il vocabolario small e teacher contiene gli stessi 256.000 token con gli
  stessi ID, ma i file tokenizer non sono byte-identici.
- Su tutti i 7.000 record locali non sono emerse differenze in input ID,
  word ID o offset; questo non sostituisce una verifica sul corpus completo
  prima della distillazione offline.
- Il fine-tuning diretto usa il tokenizer pinned dello small.
- La testa usa esattamente le 45 label e lo stesso ordine del teacher v1.5;
  non ricava la tassonomia dagli split.

### D-004 — Training sul MacBook Air

- **Hardware verificato:** MacBook Air M2, 8 GB di memoria unificata, 8 core.
- **Runtime:** PyTorch 2.13.0; MPS e BF16 funzionano fuori dalla sandbox.
- **Nota:** dentro la sandbox MPS appare indisponibile; era un vincolo di
  isolamento, non del Mac.
- **Prima configurazione da sondare:** BF16, microbatch 1, gradient
  checkpointing, SDPA, niente compile e niente fallback CPU silenzioso.
- **Baseline qualita':** full fine-tuning. Embedding frozen e LoRA sono fallback
  controllati se memoria o tempo rendono il full fine-tuning impraticabile.
- **Termica:** i benchmark di velocita' sul Mac fanless non sono una conclusione
  prestazionale; il pilot serve soprattutto per picco memoria, stabilita' ed ETA.

### D-005 — Dati raw prima dei token legacy

- **Stato:** decisione di pipeline.
- Ai4Privacy conserva `source_text` e `privacy_mask` con offset, ma lo script
  legacy usa `mbert_tokens` gia' frammentati (`##...`) come se fossero parole e
  li tokenizza nuovamente.
- Clean conserva `source_text` ed `entities` con offset.
- La pipeline student usa quindi raw text + char span quando disponibili.
- Il subset locale 10k, che conserva solo tokens/BIO, e' ammesso come proxy per
  smoke meccanici, non come evidenza finale di qualita'.

### D-006 — Riproducibilita' del corpus

- **Stato:** gap identificato.
- Il run v1.5 non e' ricostruibile esattamente: il numero dichiarato implica
  744.912 righe storiche + 350.000 clean, ma manca il manifest delle 350.000.
- Il nuovo run deve fissare revisioni immutabili, file, hash, righe, schema,
  split, seed e trasformazioni.
- Revisioni iniziali:
  - teacher v1.5: `a1c3c83827eca22e9675e30c1111c4641caf5901`;
  - storico: `3a61195c0ab01d1db40f799493e60ce3dc490291`;
  - clean: `50163bcda973efe818004d053cfa159f0927663f`;
  - Ai4Privacy: `506996d625ed970a0063432daf6007cf4a3a48e3`.

### D-007 — Target edge e criterio di successo

- **Stato:** accettata.
- **Obiettivo del progetto:** eseguire la stessa task PII su macchine sulle
  quali il teacher FP32, con un artefatto pesi di circa 1,23 GB, non puo'
  essere caricato. Il successo non consiste quindi soltanto nel comprimere un
  file, ma nel conservare la qualita' entro il gate e far entrare l'intero
  processo REST nel budget del dispositivo.
- **Target primario:** Raspberry Pi 5, per validare una REST API edge realmente
  utilizzabile e sviluppare/misurare i kernel ARM64.
- **Target ultra-low-resource:** Raspberry Pi Zero 2 W, 512 MB, per verificare
  un processo preferibilmente entro 256 MiB e un servizio minimale ARM64.
- **Stress test:** Raspberry Pi Zero prima versione, 512 MB e ARMv6
  single-core. Qui il primo obiettivo e' la fattibilita' funzionale; la latenza
  puo' essere elevata.
- **Emulazione:** QEMU `raspi0` sul Mac viene usato per compatibilita' ARMv6,
  boot, packaging e memoria guest. Non viene usato come benchmark di velocita'.
  Per Zero 2 si puo' usare una macchina QEMU Cortex-A53/512 MiB come
  preflight, mentre Pi 5 viene misurato sull'hardware reale.
- **Ordine delle prove:** qualita' su gold -> dimensione artefatto -> picco
  PSS/USS/RSS del processo -> avvio/API -> latenza e throughput sul dispositivo
  reale. I risultati del Mac non vengono trasferiti per estrapolazione.

### D-008 — Prossimo training: completare prima Adafactor full

- **Stato:** accettata per il pilot corrente.
- AdamW e Adafactor sono stati fermati rispettivamente a 30 e 20 update su 146;
  le F1 intermedie non sono confrontabili e non dimostrano che un optimizer
  produca qualita' migliore.
- Il checkpoint Adafactor al passo 20 conserva modello, optimizer, scheduler,
  RNG e stato Trainer; il suo optimizer occupa circa 1,8 MB contro circa 1,1 GB
  del checkpoint AdamW.
- Il prossimo run mantiene dati, ordine, seed, max length 256, stride 32,
  BF16, batch `1 x accumulo 4`, learning rate e scheduler. Cambiare anche
  contesto o freeze renderebbe impossibile attribuire il risultato.
- Prima del run il resume deve essere fail-closed, usare la stessa directory
  senza duplicare checkpoint e registrare hash/stato del parent. Il run va
  eseguito a Mac appena riavviato e con applicazioni non necessarie chiuse,
  mantenendo le guard di memoria.
- Lo schedule full -> token-embedding frozen resta piano B se anche questo
  resume non puo' concludersi. In quel caso sara' un nuovo stadio weights-only
  con optimizer e scheduler esplicitamente reinizializzati, non un resume.

### H-001 — Embedding fattorizzato

- **Stato:** ipotesi, da provare dopo la baseline 22x384.
- **Idea:** sostituire `Embedding(256k, 384)` con
  `Embedding(256k, E) + Linear(E, 384)`, con E in `{256, 192, 128}`.
- **E=128:** embedding da 98,30M a circa 32,82M parametri; modello complessivo
  da circa 140,66M a circa 75,17M.
- **Inizializzazione proposta:** SVD/randomized SVD della matrice pretrained,
  poi supervised retraining ed eventuale distillazione.
- **Rischio:** perdita di informazione lessicale multilingue, nomi rari e codici.
- **Gate:** gold character-span, tag critici e challenge set con nomi ambigui.

### H-002 — Quantizzazione dell'embedding

- **Stato:** tecnicamente applicabile; qualita' da rivalidare col gate corrente.
- Teacher embedding 256k x 768: circa 750 MiB FP32 o 187,5 MiB INT8 teorici.
- Small embedding 256k x 384: circa 375 MiB FP32 o 93,8 MiB INT8 teorici.
- In training si mantengono pesi continui; l'embedding packed INT8/W6/W4 e'
  una trasformazione di inferenza, eventualmente preceduta da QAT se necessario.

### H-003 — Throughput atteso dello student su Apple Silicon

- **Stato:** stima analitica, non benchmark dello student.
- **Baseline osservata:** teacher PyTorch FP32, CPU 4 thread, batch 1,
  documenti corti, 5 processi freschi: mediana `18,62 doc/s` e p05
  `17,86 doc/s` nel profilo locale warm-cache.
- **Rapporto architetturale:** `mmBERT-small` passa da 307.564.845 a
  140.658.861 parametri e da 110.921.472 a 42.337.152 MAC lineari per token.
  Il limite ideale e' quindi circa `2,62x`, ma 22 layer, kernel piccoli,
  normalizzazioni, RoPE, softmax e overhead fissi restano.
- **Range prudente iniziale:** PyTorch CPU FP32 `32-45 doc/s` sui documenti
  corti dello stesso profilo; valore centrale circa `39 doc/s`. Per chunk da
  circa 180 subword l'ordine di grandezza atteso scende a circa `10-14
  chunk/s`.
- **Nota bandwidth:** l'intera tabella embedding domina il footprint residente,
  ma per una richiesta il Gather legge soltanto le righe dei token presenti.
  La riduzione embedding migliora RAM, cold start e comportamento cache; non
  trasforma automaticamente la latenza in proporzione ai byte totali del file.
- **Decisione:** nessun numero viene presentato come risultato finche' lo
  student non e' completo. Le misure Mac saranno diagnostiche e separate dai
  benchmark sostenuti su Raspberry Pi/VPS a causa del throttling del portatile.

### H-004 — Embedding demand-paged con `mmap`

- **Stato:** ipotesi architetturale/runtime; nessuna implementazione ancora.
- Nei runtime convenzionali la matrice embedding completa fa parte del modello
  residente o mappato, ma il `Gather` di una richiesta legge soltanto le righe
  corrispondenti agli ID realmente presenti.
- Lo student ha righe da 384 valori: 1.536 byte FP32, 384 byte INT8 o 192 byte
  INT4 prima di scale e allineamento. Un documento con 60 token distinti usa
  rispettivamente circa 90, 23 o 12 KiB di payload embedding utile.
- **Esperimento lossless:** mantenere la tabella FP32/BF16 originale in un file
  read-only memory-mapped e materializzare soltanto le righe richieste. Se i
  valori e l'ordine restano identici, questa trasformazione di storage non
  altera logits o qualita'; cambiano page fault, cold start e working set.
- **Esperimento lossy separato:** usare lo stesso lookup demand-paged su righe
  W8/W6/W4 packed e dequantizzarle on demand. Questo richiede nuovamente il
  gate di qualita'.
- **Rischi:** un initializer PyTorch/ORT standard puo' essere copiato o toccato
  integralmente; la garanzia di RSS richiede un custom op/runtime oppure un
  grafo che riceva gli embedding gia' raccolti. Le pagine OS sono tipicamente
  piu' grandi di una riga, il working set cresce nel tempo e l'accesso casuale
  da microSD va misurato sui Raspberry reali.
- **Misure:** RSS/PSS cold e warm, major/minor page fault, righe/pagine uniche,
  hit rate e parita' bitwise/logit rispetto all'embedding residente.

## Esperimenti

### E-001 — Sensibilita' zero-shot alla rimozione dei layer

- **Tipo:** diagnostico legacy word-level; non release gate.
- **Baseline subset 256:** micro-F1 `0,986474`, macro-F1 `0,983335`.
- **Migliore 16 layer senza retraining:** micro-F1 `0,849574`, macro-F1
  `0,769204`.
- **12 layer senza retraining:** fallimento netto.
- **Conclusione:** la cancellazione multipla dei blocchi non e' additiva e non
  produce un modello distribuibile. Ogni riduzione stabile della depth richiede
  retraining/distillazione.
- **Limite:** la misura usava la vista first-subword; il ranking dei layer va
  rifatto con span carattere prima di guidare pruning strutturale.

### E-002 — Disponibilita' MPS reale

- **Tipo:** preflight funzionale.
- **Esito:** `torch.backends.mps.is_available() == True`, un tensore su `mps:0`
  e un piccolo backward BF16 ModernBERT sono riusciti fuori sandbox.
- **Decisione:** PyTorch/MPS e' il backend del primo training locale; MLX resta
  un port futuro dopo il congelamento dell'architettura.

### E-003 — Probe full fine-tuning vs embedding frozen

- **Stato:** completato su un singolo step; prova funzionale/memoria, non
  benchmark di velocita'.
- **Obiettivo:** misurare picco MPS/RSS, finitezza di loss/gradienti e ETA con
  identico batch; non confrontare ancora la qualita'.
- **Configurazione:** BF16, MPS, SDPA, seq max 64, microbatch 1, un solo step,
  seed `20260822`, proxy legacy da 4 record. Il proxy ha prodotto 27 finestre.
- **Full:** 140.658.861 parametri trainable; loss finita `9,51159`; MPS current
  dopo lo step 1.696.314.624 byte; driver 3.316.318.208 byte.
- **Embedding frozen:** 42.354.861 parametri trainable; stessa loss iniziale;
  MPS current 909.882.368 byte; driver 1.168.867.328 byte.
- **Artefatti:**
  - full: `artifacts/training/runs/20260822T190444Z-b1bc5dd959`;
  - frozen: `artifacts/training/runs/20260822T190508Z-ab15973ad3`.
- **Decisione:** il full fine-tuning entra sotto il limite MPS prudenziale ed e'
  la baseline di qualita'. Frozen resta fallback. I tempi non sono confrontabili:
  run singoli, cache e temperatura non controllate.

### E-004 — Pilot raw-span full fine-tuning

- **Stato:** interrotto correttamente dalla guard al passo 30/146.
- **Sorgente:** uno shard train clean pinned + validation clean nativa.
- **Selezione materializzata:** lowest salted SHA-256 su contenuto canonico;
  train 1.024 e validation 512 disponibili, zero overlap di contenuto e zero
  overlap dello skeleton ottenuto sostituendo le entita' con il loro tipo.
- **Hash train 1.024:** `95e179775ae388c271bc5fb058614fe09961f99c4fddd0175f2beb875bd451b6`.
- **Hash validation 512:** `4001d5fce099f75cdcd03101f228187b03c1390774801bc818fd586b2ee9bafb`.
- **Primo stadio:** 256 train / 256 validation, max 256 token, stride 32,
  microbatch 1, accumulo 4, BF16, full fine-tuning, una epoca.
- **Scopo:** validare convergenza e qualita' iniziale; non e' ancora il confronto
  finale col teacher ne' una misura prestazionale.
- **Guard:** MPS max 75% della memoria raccomandata; stop se swap cresce oltre
  2 GiB o la memoria disponibile scende sotto 512 MiB.
- **Risultato AdamW parziale:** loss per blocchi di 10 update `6,4221 -> 1,1895
  -> 0,5052`; validation micro-F1 `0,624654`, macro-F1 `0,556424`, recall
  micro `0,712182`. Sono metriche dopo solo il 20,6% dell'epoca.
- **Memoria:** MPS driver massimo osservato 3.448.029.184 byte; swap
  2.153.054.208 -> 3.730.571.264 byte; memoria disponibile minima
  533.643.264 byte. Stop causato dalla soglia di memoria disponibile.
- **Decisione:** non alzare la guard e non continuare AdamW full sul Mac.
- **Artefatto:** `artifacts/training/runs/pilot-clean-256-full-20260822`.

### E-005 — Full fine-tuning con Adafactor

- **Stato:** probe completato; pilot interrotto correttamente dalla guard al
  passo 20/146.
- **Ipotesi:** mantenere tutti i pesi aggiornabili riducendo gli stati
  dell'optimizer rispetto ad AdamW.
- **Soak 10 step:** MPS current circa 618 MB, driver stabilizzato circa 3,358 GB,
  nessuna crescita swap, memoria disponibile circa 677-732 MB, loss finita e in
  discesa.
- **Pilot raw 256/256:** max 256, accumulo 4, tutti i 140.658.861 parametri
  aggiornabili. Loss per il primo blocco di dieci update `6,4069` e secondo
  blocco `1,2530`; validation al passo 20 micro-F1 `0,280913`, macro-F1
  `0,196728`, recall micro `0,3684`. Il risultato e' troppo precoce per
  confrontare la qualita' dell'optimizer.
- **Memoria pilot:** MPS current circa 636 MB, driver circa 3,47 GB; swap
  sostanzialmente stabile rispetto ad AdamW. Stop per memoria disponibile
  `524.353.536` byte, appena sotto la guard da 512 MiB.
- **Decisione:** Adafactor dimostra che il full training puo' ridurre molto gli
  stati dell'optimizer, ma non viene promosso sulla sola memoria. Va confrontato
  a parita' di update e su un run completato.
- **Artefatto:**
  `artifacts/training/runs/pilot-clean-256-full-adafactor-20260822`.

### E-006 — Svuotamento della cache MPS con AdamW

- **Stato:** probe completato; ipotesi non confermata.
- **Ipotesi:** chiamare periodicamente `torch.mps.empty_cache()` avrebbe potuto
  conservare AdamW full e recuperare abbastanza memoria di sistema.
- **Configurazione:** raw span, max 256, tutti i parametri aggiornabili,
  svuotamento cache ogni update.
- **Risultato:** la guard e' scattata gia' al passo 1 con `527.843.328` byte
  disponibili; MPS current circa 1,69 GB e driver circa 3,32 GB. Lo swap non e'
  cresciuto durante il singolo passo.
- **Decisione:** lo svuotamento cache non risolve il limite strutturale degli
  stati AdamW. Non continuare il full AdamW max-256 nelle condizioni correnti;
  valutare uno schedule a stadi con embedding frozen e un eventuale breve
  refinishing full soltanto se la regressione di qualita' lo richiede.
- **Artefatto:** `artifacts/training/runs/20260822T192135Z-b5240d8741`.

### E-007 — Baseline fresh 1.024 documenti con Adafactor

- **Data:** 2026-08-22.
- **Stato:** completato `578/578`; `memory_guard.triggered=false`. Il run non e'
  ancora eleggibile per il rilascio.
- **Domanda:** aumentare da 256 a 1.024 documenti e' sufficiente a colmare il
  divario dello student 22x384 senza modificare architettura o quantizzazione?
- **Configurazione controllata:** inizializzazione fresh da
  `jhu-clsp/mmBERT-small` revisione
  `abc32620dd4f6ab06f5fbe905dc25f310618e09f`; 1 epoca, Adafactor, learning
  rate `5e-5`, BF16/MPS, max length 256, stride 32, microbatch 1, accumulo 4,
  tutti i 140.658.861 parametri aggiornabili. Il seed fissato e registrato e'
  **`20260822`**.
- **Dati:** train raw-span 1.024 documenti / 2.311 finestre, SHA-256
  `95e179775ae388c271bc5fb058614fe09961f99c4fddd0175f2beb875bd451b6`;
  validation raw-span 512 documenti / 1.140 finestre, SHA-256
  `4001d5fce099f75cdcd03101f228187b03c1390774801bc818fd586b2ee9bafb`.
- **Metriche interne del Trainer:** train loss `0,259563`; validation loss
  `0,0105192`, micro-F1 `0,990662`, macro-F1 `0,988566`, recall micro
  `0,992857` e recall critica minima `0,966154`. Sono diagnostiche della
  rappresentazione token/window interna e non sostituiscono il gate raw
  character-span.
- **Gate raw character-span su clean-512:** student micro-F1 `0,9954568`,
  macro-F1 `0,9950827`, precision/recall micro `0,9949098/0,9960044` (TP
  17.200, FP 88, FN 69). Il teacher ottiene micro-F1 `0,9963527` e macro-F1
  `0,9960684`; i delta student-teacher sono quindi `-0,0008959` e
  `-0,0009858`.
- **Confronto col pilot 256:** il precedente student otteneva micro-F1
  `0,9845301` e macro-F1 `0,9801805`; i 1.024 documenti portano rispettivamente
  `+0,0109266` e `+0,0149022`. Questo supporta l'ipotesi che il gap principale
  del pilot fosse la scarsita' di fine-tuning, non la capacita' dello student.
- **Esito del gate stretto:** i limiti globali (`0,002` micro e `0,003` macro)
  sono rispettati. Il rilascio e' pero' bocciato per recall protetta: `DOCID`
  ha delta `-0,005256` e `ID_DOC` `-0,006826`, entrambi oltre la soglia
  `-0,005`. Non si puo' promuovere il modello guardando soltanto micro-F1.
- **Memoria diagnostica:** minimo disponibile `541.294.592` byte, massimo MPS
  driver `3.409.625.088` byte; nessuna guard attivata. I tempi del MacBook Air
  fanless non vengono usati come benchmark prestazionale.
- **Limiti:** clean-512 e' una validation sintetica gia' osservata e non un test
  finale cieco o una prova real-world. Il risultato va confermato sul test
  finale indipendente e sul challenge set dei casi ambigui.
- **Artefatti:** run
  `artifacts/training/runs/full-clean-1024-adafactor-20260822`; pesi finali
  SHA-256 `81913a40c23ce5a1c336ceffb0be948519cb046b078207ce828189f4d583f052`;
  `training-summary.json` SHA-256
  `0d74a30c12c90155b73f972dede9ad1323c358b12dd064c6abb33ad712e50a50`;
  report raw student SHA-256
  `edfc82571706ee4cee9b9089197266ccfa099e0b1f18ad27cbf0cb6b8a33b6f6`;
  confronto teacher SHA-256
  `7442ae27cbcd529124f662b902dbbc08e7d95d82987b8f5fe3220bc197e60b13`.

### E-008 — Fine-tuning distillato breve

- **Data:** 2026-08-23.
- **Stato:** esperimento A/B completato. Nessuno dei due checkpoint e' ancora
  eleggibile per il rilascio e il KD provato e' un'ablation negativa.
- **Ipotesi:** un breve fine-tuning dal checkpoint E-007 avrebbe potuto
  recuperare i pochi errori residui, in particolare la recall di `DOCID` e
  `ID_DOC`, usando la distribuzione del teacher senza aumentare la capacita'
  dello student.
- **Controllo causale:** A e B partono entrambi weights-only dallo stesso
  `final/` E-007; usano optimizer e scheduler Adafactor nuovi, ordine dati e
  seed `20260822`, learning rate `1e-5`, `0,25` epoca e 145 update. A usa solo
  CE gold (`alpha_KD=0`); B usa `0,75 * L_gold + 0,25 * T^2 * KL`, con `T=2`.
  B non parte da A e nessuno dei due carica stati optimizer del parent.
- **Cache teacher:** teacher originale FP32 22x768, logits calcolati su CPU e
  memorizzati FP16 per 1.024 documenti / 2.311 finestre, tensore
  `[2311, 256, 45]` interamente finito. La cache pesa 57.387.168 byte; SHA-256
  safetensors `fc8f4601ec41af21bccfe52c8be798071129b910635f7eee13fd2cb5e8188df9`,
  manifest `c10bf8311757f606402dc5d9fa91a7c15f6d8b0f88ea0ba8c5819dff15a9756d`.
  Identita' semantica dei tokenizer teacher/student uguale; input, codice,
  versioni e artefatti vengono ricontrollati prima della pubblicazione atomica.
- **Correzione deterministica:** il primo tentativo A si e' fermato prima dello
  step 1 perche' il backward MPS non implementa deterministicamente
  l'advanced indexing usato dalla prima loss. La loss e' stata riscritta senza
  indicizzazione booleana: CE con `ignore_index=-100` e KL per token con
  maschera moltiplicativa. Valore e gradienti sono equivalenti nei test; il
  backward A/KD passa con determinismo PyTorch stretto. Non e' stato degradato
  a `warn_only`.
- **Correzione memory guard:** il secondo tentativo A si e' fermato al passo 10
  per un singolo campione a 529.891.328 byte disponibili, 6.979.584 byte sotto
  la soglia 512 MiB, mentre i campioni immediatamente successivi erano tornati
  a 1,67-2,02 GB. La guard richiede ora due osservazioni consecutive sotto
  soglia e si azzera al recupero; la crescita swap oltre 2 GiB resta uno stop
  immediato. A v3 e B hanno completato 145/145 senza guard. I tentativi falliti
  restano conservati e non sono candidati.
- **Diagnostica token vs produzione:** nella vista interna a finestre A ottiene
  micro-F1 `0,991542`, B `0,943460`; gli argmax teacher della cache ottengono
  solo `0,654700` contro le label token. Non e' una perdita FP16 o un errore
  SDPA: FP32/eager riproduce gli stessi argmax. Il teacher include spesso il
  token di spazio `▁` che precede date, importi e numeri nell'entita', mentre le
  label gold lo marcano `O`. L'aggregatore raw-span normalizza questi confini:
  il teacher raggiunge `0,998571` sui primi 50 documenti train. Per questo il
  gate resta esclusivamente raw character-span.
- **Quantificazione del conflitto KD:** su 478.724 token supervisionati gli
  hard mismatch teacher/gold sono 28.249 (`5,901%`); 12.654 sono token standalone
  `▁`. Le coppie `gold O / teacher B-X` sullo spazio seguite da
  `gold B-X / teacher I-X` sul token lessicale spiegano 24.952 mismatch
  (`88,33%`). Deduplicando le finestre sovrapposte per offset carattere, 22.854
  mismatch su 23.034 (`99,22%`) appartengono a queste coppie di boundary. Il
  teacher e' anche molto sicuro nel conflitto: sui token `▁` la probabilita'
  media della classe gold e' `0,00190`, mentre la top probability media e'
  `0,988`. Il KL non sta quindi trasferendo semplice rumore, ma una convenzione
  incompatibile con le label token dello student.
- **Risultati raw-span clean-512:** teacher micro/macro
  `0,9963527/0,9960684`; parent E-007 `0,9954568/0,9950827`; A gold-only
  `0,9954274/0,9950680`; B KD `0,9947326/0,9942586`. A-parent vale
  `-0,0000293/-0,0000147`; B-A vale `-0,0006949/-0,0008094`; A-teacher vale
  `-0,0009252/-0,0010005`; B-teacher vale `-0,0016201/-0,0018099`.
- **Errori raw:** teacher TP/FP/FN `17.210/67/59`; parent `17.200/88/69`;
  A `17.198/87/71`; B `17.185/98/84`. Gli ulteriori update gold-only non
  spiegano un miglioramento globale e il KD `alpha=0,25` peggiora A.
- **Incertezza accoppiata:** bootstrap paired sui 512 documenti, 20.000
  repliche con `numpy.random.default_rng(20260822)`, aggregazione entity-level
  exact-span di TP/FP/FN e CI percentile 2,5/97,5. Per il delta micro-F1 B-A il
  CI 95% e' `[-0,001491; +0,000056]` e solo il `3,1%` delle repliche ha B>A;
  il CI B-teacher e' interamente negativo. Il singolo dev set non dimostra da
  solo una differenza generalizzabile A-vs-B, ma rende B un candidato molto
  poco plausibile e non ne giustifica la promozione.
- **Gate protetto:** A recupera un `DOCID` rispetto al parent
  (`755 -> 756` TP), portando il delta recall dal teacher da `-0,005256` a
  `-0,003942`, quindi dentro soglia. `ID_DOC` resta `291/293`, delta
  `-0,006826`, e continua a bocciare A. B non migliora ulteriormente
  `DOCID/ID_DOC` e porta anche IBAN a `639/643`, delta recall `-0,006221`:
  fallisce quindi `ID_DOC` e `IBAN`.
- **Decisione:** non promuovere il KD `alpha=0,25, T=2`. A e' piu' vicino al
  gate protetto del parent ma non batte il teacher e non e' release-ready. Il
  solo prossimo KD giustificato e' C: stesso parent/seed/LR/145 step, `T=2` e
  `alpha=0,10`, ma KL type-selective e BIO-boundary-normalized. Il KD va
  disattivato quando il tipo hard teacher differisce dal gold dopo aver rimosso
  `B-/I-`; quando il tipo coincide ma cambia soltanto il prefisso BIO, la massa
  `B-X + I-X` va riproiettata sul prefisso gold. Il preflight deve garantire
  zero token KD attivi con type mismatch, zero `▁` gold-O trasformati in
  entita', probabilita' finite e coverage 96-98%. C deve battere A su micro e
  macro senza aumentare FP+FN; altrimenti si chiude il ramo KD e si passa a
  hard-example fine-tuning mirato a `ID_DOC` e agli errori residui. Clean-512
  resta development osservata: nessun test finale cieco e' stato consumato.
- **Artefatti A:**
  `artifacts/training/runs/full-clean-1024-gold-only-ft-v3-20260822`;
  pesi `423345cc2acf02ad941e513e224b44936b6ae783591f766d9fe841a7daba0faf`;
  summary `39961693d4f7546f7eb56b84db2d4c4473b68c1e09d6f1c14b62980d87fafb96`;
  report raw `095a078f71c668f7b818707107482a8e5e377242b2b21051db4b8f64705733d7`.
- **Artefatti B:**
  `artifacts/training/runs/full-clean-1024-kd-a025-t2-20260822`;
  pesi `f792dbd5ae20e019c5c38682651400f3103099df4e991fd2b1d1798c18c9d0e4`;
  summary `e712cf7522fbd160ba06f66223c7f89629b0b585ee7715c7b62da8e296a1c6a0`;
  report raw `884ad0cc92866ecbda8e21fc6657368b29818a776e46e4a54dac0dd869da5bfd`.

### E-009 — KD type-selective con normalizzazione BIO

- **Data:** 2026-08-23.
- **Stato:** completato; C diventa la migliore baseline student osservata, ma
  resta `release_eligible=false` per il gate protetto `ID_DOC` e perche'
  clean-512 non e' un test cieco.
- **Ipotesi:** il risultato negativo di B dipendeva in parte dal trasferimento
  di una convenzione BIO incompatibile e da un peso KD troppo alto. C mantiene
  parent, dati, seed `20260822`, Adafactor, LR `1e-5`, `0,25` epoca e 145
  update, ma usa `alpha_KD=0,10`, `T=2` e target teacher selettivi.
- **Loss:** la CE gold resta su tutti i token supervisionati. Il KL e' attivo
  solo quando il tipo hard del teacher coincide con il tipo gold dopo aver
  rimosso `B-/I-`. Nei soli mismatch BIO dello stesso tipo la massa
  `p(B-X)+p(I-X)` viene posta sul prefisso gold e il prefisso opposto viene
  azzerato; le altre 43 classi restano invariate. Exact-match e `O/O` non
  vengono modificati. Il KL e' mediato sui soli token attivi e moltiplicato per
  `T^2`; un batch senza token attivi fallisce esplicitamente.
- **Preflight fail-closed:** 478.724 token supervisionati, 463.249 attivi
  (`96,767448%`), 450.475 exact invariati, 12.774 BIO-only riproiettati e
  15.475 type mismatch esclusi. Le 2.311 finestre hanno tutte segnale KD:
  zero finestre vuote, minimo 4 e massimo 253 token attivi. I 12.546 casi
  `▁` gold-O/teacher-entita' sono tutti inattivi; errore massimo sulla somma
  delle probabilita' `7,152557e-7`.
- **Verifica implementazione:** modalita' `raw` mantenuta come default e
  compatibile con B; 59 test integrati verdi, inclusi valore e gradiente della
  loss, conservazione della massa, contratto label, gate e zero-active. Il
  backward deterministico della nuova loss e' stato provato anche su MPS con
  loss e gradienti finiti. Un audit indipendente non ha rilevato P0/P1 prima
  del training.
- **Run:** MPS BF16, 1.024 documenti / 2.311 finestre train, 512 / 1.140
  validation, 145/145 step, train loss `0,0178097`, tempo totale diagnostico
  `361,0 s`. La validation token/window interna ottiene micro/macro
  `0,9906318/0,9887957`; non sostituisce il gate raw-span.
- **Risultati raw-span clean-512:** C micro/macro
  `0,9956003/0,9953513`, TP/FP/FN `17.198/81/71`. A ottiene
  `0,9954274/0,9950680`, `17.198/87/71`; parent
  `0,9954568/0,9950827`, `17.200/88/69`; teacher
  `0,9963527/0,9960684`, `17.210/67/59`.
- **Criterio C predefinito:** `PASS`. C-A vale
  `+0,0001729/+0,0002833` su micro/macro e FP+FN scende da 158 a 152. C
  migliora anche il parent di `+0,0001435/+0,0002686`, ma resta sotto il
  teacher di `-0,0007524/-0,0007172`.
- **Incertezza accoppiata:** bootstrap paired exact-span sui medesimi 512
  documenti, 20.000 repliche e `numpy.random.default_rng(20260822)`. Il delta
  micro C-A ha CI 95% `[-0,0001470; +0,0005472]`, con `P(C>A)=81,60%`;
  C-parent `[-0,0002886; +0,0005797]`, `P=72,01%`; C-teacher
  `[-0,0019256; +0,0003900]`, `P=9,53%`. Il vantaggio puntuale di C non e'
  quindi una differenza statisticamente conclusiva sul solo dev set.
- **Gate protetto:** rispetto al teacher C rispetta il limite massimo di calo
  recall `0,005` per `CF`, `PIVA`, `IBAN`, `DOCID`, `EMAIL` e
  `TELEPHONENUM`. `DOCID` e' a `756/761`, delta `-0,003942`. `ID_DOC` resta
  a `291/293`, delta `-0,006826`, ed e' l'unico fallimento. Rispetto ad A i
  sette recall protetti sono invariati.
- **Memoria training:** guard mai attivata; una sola osservazione sotto 512 MiB
  disponibili, massimo consecutivo uno, poi recupero. Crescita swap massima
  509.542.400 byte contro il limite 2 GiB; massimo MPS driver osservato
  3.409.625.088 byte. Sono dati diagnostici del training, non il footprint di
  inferenza target da 256 MB.
- **Decisione:** promuovere C a baseline student di ricerca al posto di A/B,
  ma non a release. Il prossimo intervento di qualita' deve essere mirato agli
  hard example `ID_DOC` e agli errori residui, con un challenge set separato;
  continuare a ottimizzare su clean-512 aumenterebbe il rischio di leakage.
  Per attribuire causalmente il guadagno tra `alpha=0,10` e normalizzazione
  BIO servirebbe in futuro un controllo raw-KD a `alpha=0,10`; non e'
  necessario per scegliere ora il checkpoint migliore.
- **Artefatti C:** run
  `artifacts/training/runs/full-clean-1024-kd-type-boundary-a010-t2-20260823`;
  modello finale `7c3221b3387db2285d28e975145b9884f77fd4077b8264c2e7cee54a49b2e63a`;
  `model.safetensors`
  `86d8f249b1091db53bee8024d32c80b2460f41f3618df8d2f2793495372cdbfe`;
  manifest `b0749c90864bcd34a012d438b18e9d3a74ebf3c7ab6c47cf0b16beff5f735e9e`;
  summary `078c7ef01eea38a31553df5cd39aeefbd2b23f154f87ec36eaa1f3a91a09310f`;
  report raw `92457ac7bac6bc22743ebc78b8e2e650449b37880fe6bd85e71e994aad9bc3fd`;
  predictions `2ae969e22a763893ee0f9a4ba01189a02869bedb6633561ea9c709b1c5a988a5`.

### E-010 — Policy tokenizer ModernBERT e `fix_mistral_regex`

- **Data:** 2026-08-23.
- **Stato:** completato; nessun cambio ai token o ai pesi.
- **Problema:** Transformers 4.57.6 emetteva un warning che suggeriva
  `fix_mistral_regex=True`. Applicarlo senza verifica avrebbe potuto cambiare
  la segmentazione tra training, cache teacher ed evaluation.
- **Audit:** teacher, parent C e mmBERT-small usano lo stesso tokenizer fast
  con pre-tokenizer standalone Metaspace e vocabolario da 256.000 token. Sui
  corpus train-1024 e validation-512 il comportamento default e quello con
  `fix_mistral_regex=False` sono identici. Il valore `True` non e' applicabile:
  tenta di modificare Metaspace come fosse una configurazione Mistral e fallisce
  con `Metaspace object does not support item assignment`.
- **Decisione:** policy unica `preserve-modernbert-metaspace-v1`, con
  `fix_mistral_regex=false`, centralizzata in `student_utils.py` e usata
  esplicitamente da training, cache logits, distillazione ed evaluation. La
  policy viene registrata nei nuovi manifest/summary/report.
- **Interpretazione:** il warning era un falso positivo per questa famiglia di
  tokenizer. Questo intervento migliora riproducibilita' e fail-closed, non la
  qualita' predittiva.

### E-011 — Fine-tuning mirato `ID_DOC`: D0, D1 e tentativo LR ridotto

- **Data:** 2026-08-23.
- **Stato:** completato; nessun D1 supera il gate. I due holdout finali restano
  sigillati e non sono stati valutati.
- **Ipotesi:** i due falsi negativi `ID_DOC` residui di C potevano essere
  recuperati con hard-example fine-tuning, senza degradare `DOCID` o le altre
  label protette.
- **Dati v1:** D1 contiene 1.024 documenti: 226 target, 226 hard-negative e
  572 replay; D0 e' il controllo disgiunto da 1.024 documenti. Il manifest
  pubblica anche un holdout globale da 2.048 e un challenge `ID_DOC` da 2.048,
  entrambi non consumati. Zero overlap di record o skeleton con i dati gia'
  osservati. Seed di selezione e training `20260822`.
- **Audit del corpus:** lo shard clean contiene 51.965 entita' `ID_DOC`; D1 ne
  contiene 808. Tutti i 237 esempi della forma mirata `n.` piu' sette cifre
  provengono pero' dallo stesso `template_id=2100`, prodotto dal contributor
  `workingfm` con generatore `workingfm-community-1.0.0`, famiglia
  `long_thread`. Il generatore ha incluso il cue `n.` nel valore e quindi nello
  span gold; la pipeline clean lo ha conservato. D1 ha percio' un rischio reale
  di memorizzare sia il contesto sia una convenzione annotativa. Nel corpus
  sono assenti, con ricerca conservativa, cue come `C.I./CI` e `passaporto`,
  mentre esistono migliaia di near-negative `DOCID` numerici e alfanumerici.
- **Protocollo comune:** inizializzazione weights-only da C, tutti i 140,66 M
  parametri addestrabili, MPS BF16, Adafactor, batch 1, accumulo gradienti 4,
  64 update, seed `20260822`. D0 e D1 usano LR `5e-6`; un solo tentativo
  correttivo D1 riparte da C con LR `2e-6`.
- **Risultati clean-512 exact-span:** C micro/macro
  `0,9956003/0,9953513`, 152 errori, `ID_DOC` TP/FP/FN `291/0/2`;
  D0 `0,9957745/0,9954137`, 146 errori, `291/0/2`; D1 LR `5e-6`
  `0,9957462/0,9951381`, 147 errori, `293/4/0`; D1 LR `2e-6`
  `0,9953994/0,9949231`, 159 errori, `291/4/2`.
- **Analisi errori:** D1 `5e-6` recupera entrambi i target con tipo e confine
  exact, ma introduce quattro falsi positivi `ID_DOC`: due `DOCID` numerici,
  un IBAN numerico e un frammento di `DOCID` alfanumerico. Inoltre `DOCID`
  scende da 756 TP di D0 a 753. Con LR `2e-6` il modello assegna correttamente
  `ID_DOC` alle sette cifre dei due target, ma lascia fuori il cue innocuo
  `n.`; l'exact-span lo conta come due FP parziali e due FN completi. Questo non
  costituisce una fuga di PII se il payload numerico viene oscurato, ma i
  quattro falsi positivi e la regressione globale restano errori reali.
- **Correzione di un errore metodologico:** la convenzione del dataset era
  nota, ma e' stato scorretto trasformarla in un requisito di prodotto e in un
  gate esclusivo. Exact-span resta per confrontabilita' col benchmark. Si
  aggiunge, senza sostituirla, una vista production-oriented sul payload
  sensibile: differenze limitate al prefisso `n.`/`n°`/`nr.`/`numero` non sono
  perdita di privacy; tipo errato, payload parziale e invasione di testo o di
  un'altra entita' restano errori. Il gate di rilascio deve considerare entrambe
  le viste e dare priorita' alla mancata copertura di caratteri sensibili.
- **Ricalcolo payload senza nuova inferenza:** C e D0 restano invariati. D1
  `2e-6` passa da `ID_DOC` TP/FP/FN `291/4/2` exact a `293/2/0` payload e da
  micro/macro `0,9953994/0,9949231` a `0,9955152/0,9952324`; i due errori di
  confine spariscono, mentre restano due falsi positivi reali. D1 `5e-6` resta
  `293/4/0`. Anche con la metrica corretta D0 rimane migliore globalmente e
  nessun D1 supera il gate.
- **Decisione:** D0 e' il miglior checkpoint osservato globale, ma non risolve
  i due casi target. Nessun D1 viene promosso e abbassare soltanto il learning
  rate e' escluso come soluzione. D2 riparte da C con 128 positivi diversificati
  (32 per `n.`+7, CIE, passaporto e patente), 256 hard-negative surface-matched
  e 640 replay; gli span sintetici coprono soltanto il valore identificativo e
  non obbligano il modello a oscurare il cue.
- **Robustezza training:** il tentativo LR `2e-6` ha completato 64/64 con resume
  esatti dai checkpoint 10 e 60 dopo due stop della memory guard. Optimizer,
  scheduler, RNG e ordine dati sono stati ripristinati. E' stato inoltre
  corretto il bookkeeping dei resume futuri: il checkpoint finale viene ora
  registrato come modello autorevole senza creare una copia `final/`, e la
  provenienza parent resta nella catena audit. I pesi esistenti non sono stati
  modificati.
- **Artefatti dati v1:** D1
  `ea159d8b26aa4132e2feda42a7d3389e7c82d69ae0fe4ab6afb824ff418b3e6f`;
  D0 `ace4708abca7f819f8e03741c4fa4aa9f9a0da83f9839cbbd33a45ad0d7304d8`;
  holdout globale
  `c929c65bc979ddb3d25c9944b65799379f528a7246000f275944871f0a9e5b19`;
  challenge
  `44dba24d082004d48de1e32cc979fe58184bac1f9b8dde645c210d5e34cc4c1f`;
  manifest
  `6290c25e1ea050f3d21d3023d8b2ba83d0327980c49007b9baad713d38787ed1`.
- **Artefatti run:** D0 modello
  `7e6086846730499bac9a8af3ebe908832cea01d3c0191ee1a1ad21937f353a24`,
  report `676f2130c323a2647e95d7735037c3dcd1479ec046bd6811e56eddc423302a25`;
  D1 `5e-6` modello
  `2561e757f8aa73483efdac01740dfca5b0e12cefd7aa7fa77e65c58af24bdd7f`,
  report `5e3b0e3cc64eb8e405626f38c4ebd2029ddce532edd689081db6b0bb26173cca`;
  D1 `2e-6` modello
  `8838cac4276ccd23352ac7d1e4c242f8486d4ed62606235df17d7b09d273b4cf`,
  report `ce39295badbb480fab7ea2aef376821ac8a70b5bbfb3c9d7dfb2d2d28d576c85`.

### E-012 — D2 value-only, protocollo preregistrato

- **Data:** 2026-08-23.
- **Stato:** training completato `64/64` con resume esatto; il gate development
  preregistrato non e' superato. Il challenge D2 e i due holdout v1 non sono
  stati aperti.
- **Stato al lock del protocollo:** dataset materializzato e verificato; nessun
  peso D2 era ancora stato addestrato e nessun challenge/holdout era stato
  consumato.
- **Train D2:** 1.024 righe = 128 positivi sintetici (`32` per numerico a sette
  cifre, CIE, passaporto e patente), 128 `DOCID` con superficie esattamente
  matched, 128 boundary trap e 640 replay reale. Il replay contiene 172 righe
  con IBAN numerici e 445 con identificatori alfanumerici. Gli span sintetici
  etichettano soltanto il valore identificativo; cue e punteggiatura restano
  `O`.
- **Challenge D2:** 384 righe sintetiche sigillate, 128 positive e 256 negative,
  con famiglie di template completamente disgiunte dal train. Overlap di record,
  skeleton normalizzati e famiglie template: zero.
- **Protocollo training bloccato prima dei risultati:** inizializzazione
  weights-only da C, tutti i layer addestrabili, MPS BF16, Adafactor, batch 1,
  accumulo 4, LR `5e-6`, 64 update, seed `20260822`, nessuna KD. E' stato
  eseguito un solo tentativo, senza scegliere adattivamente il learning rate
  usando clean-512.
- **Gate development:** la vista payload deve ottenere `ID_DOC` TP/FP/FN
  `293/0/0`; `DOCID` non deve essere peggiore di D0 (`756/8/5`); micro e macro
  payload devono essere almeno `0,9957745/0,9954137` e FP+FN non oltre 146;
  nessuna label protetta puo' introdurre una regressione materiale. Exact-span
  viene riportata come benchmark, non come proxy esclusivo di privacy.
- **Ordine di apertura:** il challenge D2 e i due holdout reali v1 restano
  sigillati finche' il gate development non passa. Se il gate fallisce, non si
  usa un holdout per scegliere il prossimo iperparametro.
- **Verifica builder:** build byte-identica, 11/11 test D2 e 184/184 suite
  completa verdi, con due skip previsti.
- **Preflight schema e correzione loader:** il primo avvio si e' fermato prima
  del training perche' il train D2 conteneva colonne di provenance aggiuntive
  rispetto alla validation clean. La directory
  `artifacts/training/runs/id-doc-d2-synthetic-ft64-20260823` contiene soltanto
  il manifest del preflight fallito e non un modello parzialmente addestrato.
  Il loader e' stato corretto per caricare train e validation separatamente,
  validarne il formato e selezionare per il Trainer soltanto `source_text` ed
  `entities`, mantenendo comunque gli hash di identita' sui file originali. Il
  preflight corretto ha verificato `1.024/512` record prima del run effettivo.
- **Esecuzione e resume:** il run effettivo e'
  `artifacts/training/runs/id-doc-d2-synthetic-ft64-v2-20260823`. La memory
  guard lo ha fermato al passo 60 quando la crescita swap ha raggiunto
  `2.187.788.288` byte, oltre il limite di 2 GiB. Il resume esatto dal
  checkpoint 60 ha ripristinato modello, optimizer, scheduler, RNG e ordine dei
  dati e ha completato i passi 61-64; il checkpoint 64 e' registrato come
  autorevole.
- **Risultati exact-span su clean-512:** micro-F1 `0,9953391`, macro-F1
  `0,9949371`, TP/FP/FN globali `17.191/83/78`, cioe' 161 errori. `ID_DOC`
  ottiene `289/0/4`, `DOCID` `755/13/6` e `IBAN` `641/0/2`. D2 non raggiunge
  quindi il gate preregistrato ne' globalmente ne' sulle label target; il
  challenge e gli holdout non vengono aperti.
- **Incertezza exact-span:** bootstrap paired sugli stessi 512 documenti,
  20.000 repliche e seed `20260822`. Il delta micro D2-D0 e'
  `-0,000435342`, CI 95% `[-0,000879471; -0,000029794]`, con
  `P(D2>D0)=1,515%`; D2-C e' `-0,000261182`, CI 95%
  `[-0,000712292; +0,000142538]`, con `P(D2>C)=9,965%`. D2 e' dunque peggiore
  di D0 sul dev exact-span; il confronto con C non e' conclusivo al 95%.
- **Audit combinatorio post-run:** i 128 positivi coprono soltanto 5 delle 20
  combinazioni formato x famiglia di frase: numerico a sette cifre compare in
  due famiglie da 16 esempi, mentre CIE, passaporto e patente compaiono ciascuno
  in una sola famiglia da 32. Le coppie cue coprono 16/64 celle (25%) e le
  triple formato x cue x superficie soltanto 16/256 (6,25%); inoltre ci sono
  due hard-negative per ogni positivo. Il corpus non e' quindi fattoriale e
  permette al modello di apprendere correlazioni spurie tra formato, cue e
  label.
- **Analisi causale prudente:** i quattro `ID_DOC` mancati sono tutti coperti
  come `DOCID`; due casi CIE-like che C e D0 classificavano correttamente sono
  regrediti dopo D2, mentre D1 a LR `5e-6` aveva gia' appreso i due casi
  `n.`+sette cifre. Questo e' evidenza contro l'ipotesi che hidden size 384 sia
  il collo di bottiglia corrente: la capacita' discriminativa e' presente, ma
  il disegno D2 e la vicinanza tassonomica `ID_DOC`/`DOCID` l'hanno orientata
  male. Non dimostra che 384 sia sufficiente per ogni distribuzione futura; il
  prossimo test deve prima rendere formato e contesto indipendenti con coppie
  minime bilanciate, senza ridurre ulteriormente l'architettura.
- **Limite del development set:** clean-512 e' sintetico, e' gia' stato
  osservato in piu' iterazioni ed e' usato soltanto per sviluppo. Ne' il valore
  puntuale ne' il bootstrap dimostrano il comportamento su documenti reali o
  in produzione.
- **Artefatti dati:** train
  `026e5c1ad2cde5be38a2623f65a38a27121d7b0bb26b53c02d74f11d563a0e7e`;
  challenge
  `1b0dd788a65788ddc97a7386f48ad39ae0a6d3bba9b4cb22f52ebc171c8417cf`;
  recipe
  `d8addc68d6daf9acc4abea72e934fc15d3d6e093ece3b2a0b46ca95f604dac60`;
  manifest
  `ffc0edeadea0f96911aec3ae5924ee709030c2465c3ce49ff7f41d6dc48e7fbb`.
- **Artefatti run:** model group
  `778e0d8662da3e0c640d01d8215872a21fd9744bff10c2a084156f25dbd3b1b1`;
  `model.safetensors`
  `4c88c697f2895b52af5d7490e333ae13c651b880a08f295402ec1c66f0b0106a`;
  manifest
  `4196e618d2be21bc2e306bdc720dc90c565ed925aba0a11b60286d9920f7fd6a`;
  summary
  `537d83ca53276120d147d804d2f339e3ca461d601d98f10b14794c532f400f37`;
  report exact-span
  `f863a832d343b6bdb9c42b296d316fde28837f1c873412543deadc367f170e67`;
  predictions
  `65b040f1d83aad80eb14a5b75cc4425c2b4a880ff67d1bd62e5deebed7b0c64b`.

### E-013 — Metrica production-oriented privacy/utility

- **Data:** 2026-08-23.
- **Stato:** ricalcolo esplorativo completato sulle predizioni clean-512 gia'
  salvate; nessuna nuova inferenza e nessun challenge/holdout aperto.
- **Motivazione:** exact-span attribuisce lo stesso peso a una fuga di caratteri
  sensibili, a un tipo PII errato ma comunque oscurato e a una differenza di
  confine che coinvolge soltanto un cue innocuo come `n.`. E' utile per il
  benchmark e per diagnosticare tassonomia e boundary, ma non misura da solo
  il rischio privacy o il costo dell'oscuramento eccessivo.
- **Definizione:** la metrica
  `production_privacy_utility_unicode_alnum_coverage_v1` canonicalizza per
  `ID_DOC` i prefissi numerici innocui previsti dalla policy e considera
  sensibili i caratteri Unicode alfanumerici del payload gold. Riporta copertura
  type-agnostic e type-correct a livello di carattere, entita' e documento,
  caratteri coperti con tipo errato, caratteri non-PII oscurati, collateral
  ratio, mask precision e breakdown per label. Il report contiene soltanto
  aggregati, senza testo o identificatori dei documenti.
- **Condizione di lettura:** nella vista privacy seguente sono attive tutte le
  label PII. Un carattere `ID_DOC` predetto come `DOCID` e' quindi coperto per
  privacy ma resta errato nella vista typed. Se un'applicazione scegliesse di
  oscurare soltanto alcune label, la vista type-agnostic non sarebbe una
  garanzia sufficiente.

| Modello | Character coverage / leak | Entity coverage | Document coverage | Collateral / ratio | Mask precision | Typed char coverage |
|---|---:|---:|---:|---:|---:|---:|
| C | `0,9995834114` / 80 | `17.253/17.269` | `498/512` | 28 / `0,0001057326` | `0,9998541545` | `0,9991095420` |
| D0 | `0,9997031807` / 57 | `17.257/17.269` | `502/512` | 37 / `0,0001397181` | `0,9998073077` | `0,9991199567` |
| D1 LR `5e-6` | `0,9998646087` / 26 | `17.264/17.269` | `507/512` | 58 / `0,0002190175` | `0,9996980236` | `0,9993386657` |
| D1 LR `2e-6` | `0,9997448395` / 49 | `17.260/17.269` | `504/512` | 58 / `0,0002190175` | `0,9996979875` | `0,9993386657` |
| D2 | `0,9994584349` / 104 | `17.248/17.269` | `496/512` | 14 / `0,0000528663` | `0,9999270628` | `0,9987606490` |

- **Risultato privacy/utility:** sul clean-512 osservato D1 LR `5e-6` ha la
  migliore copertura privacy: 192.010/192.036 caratteri sensibili, 17.264
  entita' completamente coperte e 507 documenti senza leak. Oscura pero' 58
  caratteri innocui su 264.819, contro 37 di D0, 28 di C e 14 di D2. D2 e'
  quindi il meno invasivo ma anche il peggiore per leakage. Non esiste una
  dominanza assoluta: la selezione e' un trade-off esplicito tra recall privacy
  e collateral masking.
- **Separazione privacy/tassonomia:** D1 LR `5e-6` copre e tipizza correttamente
  tutti i 293 `ID_DOC`. D2 copre per privacy tutti i 293 payload quando tutte
  le label sono attive, ma ne tipizza correttamente soltanto 289; i quattro
  rimanenti sono `DOCID`. D1 conta inoltre 101 caratteri sensibili coperti con
  un tipo diverso dal gold nell'intera tassonomia. Una futura architettura puo'
  quindi separare una decisione coarse `PII/non-PII`, usata come fallback di
  oscuramento, dalla classificazione fine del tipo, senza confondere le due
  metriche.
- **Bootstrap paired production-oriented:** 20.000 repliche document-level,
  conteggi riaggregati a ogni replica e
  `numpy.random.default_rng(20260822)`. Rispetto a D0, D1 LR `5e-6` migliora la
  character coverage di `+0,000161428`, CI 95%
  `[+0,000036307; +0,000318130]`, `P(D1>D0)=99,755%`, e la document coverage di
  `+0,009765625`, CI `[+0,001953125; +0,019531250]`, `P=99,31%`. Il collateral
  ratio aumenta di `+0,000079299`, CI
  `[-0,000026649; +0,000186505]`; `P(D1<D0)=6,555%`, dove un valore minore e'
  migliore.
- **Bootstrap contro C e D2:** contro C, D1 migliora character coverage di
  `+0,000281197`, CI `[+0,000108416; +0,000490132]`, `P=99,995%`, e document
  coverage di `+0,017578125`, CI `[+0,007812500; +0,029296875]`, `P=99,99%`,
  ma aumenta il collateral ratio di `+0,000113285`, CI
  `[+0,000034536; +0,000208358]`, con `P(D1<C)=0%`. Contro D2, i delta sono
  `+0,000406174`, CI `[+0,000173992; +0,000678413]`, `P=100%`, per character
  coverage; `+0,021484375`, CI `[+0,009765625; +0,035156250]`, `P=100%`, per
  document coverage; e `+0,000166151`, CI
  `[+0,000063451; +0,000287779]`, con `P(D1<D2)=0%`, per collateral ratio.
- **Natura post-hoc:** questa metrica e' stata definita dopo aver osservato gli
  errori D1/D2 e dopo aver riconosciuto il limite concettuale dell'exact-span.
  E' una correzione metodologica necessaria, ma il suo ranking non e'
  confermatorio e non retro-promuove D1 ne' cambia il verdetto preregistrato di
  E-012. I threshold privacy/utility dovranno essere bloccati prima del prossimo
  esperimento e applicati prospetticamente.
- **Decisione:** D1 LR `5e-6` diventa il candidato di ricerca migliore per
  privacy sul dev osservato, non un modello di release. Exact-span resta una
  metrica secondaria per boundary e tipo. Prima di qualunque claim di
  produzione servono un holdout realistico, annotato e revisionato, e una prova
  end-to-end della policy di oscuramento. Clean-512 e' sintetico, e' stato
  consultato ripetutamente e non prova la qualita' su documenti reali; il
  challenge D2 e i due holdout v1 restano sigillati.

### E-014 — Confronto D1 LR `5e-6` contro teacher FP32

- **Data:** 2026-08-23.
- **Stato:** confronto di development completato sulle 512 predizioni gia'
  salvate. D1 e' il candidato di ricerca corrente per privacy, ma non e'
  congelato ne' eleggibile per il rilascio. Nessun challenge o holdout e' stato
  aperto.
- **Modelli confrontati:** teacher FP32 `ModernBERT` 22x768, 307.564.845
  parametri, contro D1 FP32 `mmBERT-small` 22x384, 140.658.861 parametri. D1
  riduce quindi i parametri del `54,27%`, senza ridurre la profondita'. Il file
  `model.safetensors` passa da 1.230.273.700 byte (`1.173,28 MiB`) a
  562.649.620 byte (`536,58 MiB`), riduzione del `54,27%`.

| Vista exact-span clean-512 | Teacher FP32 | D1 FP32 | Delta D1-teacher |
|---|---:|---:|---:|
| Micro-F1 | `0,9963526892` | `0,9957461585` | `-0,0006065306` |
| Macro-F1 | `0,9960684388` | `0,9951381364` | `-0,0009303024` |
| TP / FP / FN | `17.210 / 67 / 59` | `17.205 / 83 / 64` | 21 errori FP+FN in piu' |
| `ID_DOC` TP / FP / FN | `293 / 0 / 0` | `293 / 4 / 0` | stesso recall, 4 FP D1 |
| `DOCID` TP / FP / FN | `759 / 5 / 2` | `753 / 9 / 8` | regressione D1 |
| `IBAN` TP / FP / FN | `643 / 0 / 0` | `641 / 2 / 2` | regressione D1 |

- **Incertezza exact-span:** paired bootstrap document-level, 20.000 repliche,
  conteggi TP/FP/FN riaggregati e seed `20260822`. Il delta micro-F1
  D1-teacher e' `-0,000606531`, CI 95%
  `[-0,001603563; +0,000413449]`, con `P(D1>teacher)=12,51%`. Il teacher e'
  migliore puntualmente, ma l'intervallo bilaterale include zero su questo
  campione.

| Privacy/utility, tutte le label PII attive | Teacher FP32 | D1 FP32 | Lettura |
|---|---:|---:|---|
| Character coverage / leak | `0,9994584349` / 104 | `0,9998646087` / 26 | D1 copre 78 caratteri in piu' |
| Entity coverage | `17.257/17.269` | `17.264/17.269` | D1 +7 entita' complete |
| Document coverage | `500/512` | `507/512` | D1 +7 documenti completi |
| Collateral / ratio | 2 / `0,0000075523` | 58 / `0,0002190175` | teacher molto meno invasivo |
| Mask precision | `0,9999895798` | `0,9996980236` | teacher migliore |
| Typed character coverage | `0,9993178362` | `0,9993386657` | quasi pari, D1 +0,00002083 |

- **Incertezza privacy/utility:** con lo stesso bootstrap, D1-teacher sulla
  character coverage e' `+0,000406174`, CI 95%
  `[-0,000031565; +0,000993979]`, `P(D1>teacher)=95,805%`; sulla document
  coverage e' `+0,013671875`, CI
  `[-0,001953125; +0,029296875]`, `P=94,74%`. Il vantaggio privacy osservato
  non supera quindi un criterio bilaterale al 95%. Il collateral ratio aumenta
  invece di `+0,000211465`, CI
  `[+0,000081209; +0,000368649]`, e nessuna replica favorisce D1 nel verso
  `D1<teacher`: la maggiore invasivita' di D1 e' un effetto netto sul dev.
- **Inferenza CPU indicativa:** con lo stesso evaluator PyTorch eager, CPU,
  batch finestra 1, max length 256 e stride 32, il teacher ha elaborato i 512
  documenti in `118,7445 s` (`4,31 doc/s`) e D1 in `64,3678 s`
  (`7,95 doc/s`), rapporto osservato `1,84x`. E' una singola esecuzione su
  MacBook Air fanless, senza controllo di temperatura, cache o throttling: non
  e' un benchmark di release e non va estrapolata su VPS o Raspberry Pi.
- **Interpretazione:** D1 conserva quasi tutta la qualita' exact del teacher
  con meno della meta' dei parametri e mostra un profilo piu' orientato alla
  copertura. Il teacher resta superiore su exact-span, tipizzazione di
  `ID_DOC`/`DOCID`/`IBAN` e collateral masking. Se l'app permette di oscurare
  soltanto label selezionate, la copertura type-agnostic non basta: una
  confusione di tipo puo' trasformarsi in leakage e deve restare nel gate.
- **Caveat metodologico:** clean-512 e' sintetico ed e' stato osservato in piu'
  iterazioni. La metrica privacy/utility e questo ranking sono post-hoc rispetto
  agli errori D1/D2. I bootstrap misurano soltanto l'incertezza di campionamento
  su questo dev e non correggono selection bias, leakage di template o domain
  shift. Il prossimo gate deve essere scritto prima di D3 e applicato a dati
  non usati per scegliere modello e soglie.
- **Decisione:** non dichiarare D1 superiore al teacher e non congelarlo ancora.
  D1 resta la baseline student privacy-oriented per il gate v2; il teacher
  resta il riferimento FP32 e D0 il controllo exact student. Prima della
  quantizzazione si devono uniformare inferenza applicativa ed evaluator,
  correggere il disegno dati `ID_DOC`/`DOCID` e validare prospetticamente.
- **Artefatti:** confronto exact/disagreement
  `27c151e315e9016abff2a617c3345f9b9e33e8ea83520d6d1ef434326ca3d480`;
  report privacy D1
  `6360e119d8c600b633877728e8a1c2b3c1e3dcf5d8bcbc6b2f72ef9414561c8e`;
  report privacy teacher
  `4791decb83d0d424c20c9b5dfd869f55a1c7489e7aba33323f49ffa98b10a389`;
  report exact D1
  `5e3b0e3cc64eb8e405626f38c4ebd2029ddce532edd689081db6b0bb26173cca`;
  report exact teacher
  `b4a36dc67f1287c2425dcaacebd2d2a66788a1908645c0c84fe190258b0d204d`.
  Il risultato bootstrap E-014 e' stato ricalcolato dalle predizioni salvate ma
  non ha ancora un artefatto machine-readable dedicato: il relativo hash e' da
  compilare nel manifest inventory
  `docs/checkpoints/2026-08-23-artifacts.md` prima di usarlo come prova
  pubblicabile.

## Traccia prevista del paper

1. task, modello originale e vincoli edge;
2. anatomia dei parametri e del picco memoria;
3. protocollo di qualita' production-like;
4. riproducibilita' e correzione della pipeline dati;
5. width reduction 768->384;
6. depth reduction con retraining;
7. embedding factorization;
8. quantizzazione W8/W6/W4/W2 e kernel;
9. runtime PyTorch, ONNX e MLX;
10. confronto finale teacher/student su Mac, VPS, Raspberry Pi 5, Zero 2 W e
    Zero prima versione, distinguendo emulazione funzionale e hardware reale.
