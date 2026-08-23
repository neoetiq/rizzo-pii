# Training dello student PII edge

## Decisione

Cambiare hidden size da 768 a 384 o rimuovere stabilmente dei layer richiede
training. Non serve però un pretraining da zero: il punto di partenza è
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small), già
preaddestrato con la stessa famiglia ModernBERT e lo stesso tokenizer del teacher.

La sequenza qualità-first è:

1. fine-tuning supervisionato del modello 22-layer, hidden 384;
2. distillazione dal teacher solo se il fine-tuning diretto non supera i gate;
3. student 18-layer e poi 16-layer, con retraining/distillazione dopo il pruning;
4. quantizzazione W8/W6/W4 soltanto dopo aver congelato l'architettura FP32.

Le ablation zero-shot hanno già escluso la scorciatoia di cancellare blocchi e
distribuire il checkpoint: la migliore variante 16-layer è scesa da micro-F1
`0,986474` a `0,849574` sul subset diagnostico.

## Riproducibilità dei dati

Gli ingredienti del training sono pubblici, ma il run v1.5.0 non è riproducibile
esattamente dagli artefatti disponibili oggi.

- [`rizzo-pii-it-dataset`](https://huggingface.co/datasets/rizzoaiacademy/rizzo-pii-it-dataset)
  ricostruisce 744.912 righe storiche da dati sintetici, augment, DeepMount e
  Ai4Privacy;
- [`anonimizzazione-testi-italiano-clean`](https://huggingface.co/datasets/rizzoaiacademy/anonimizzazione-testi-italiano-clean)
  contiene 1.431.762 righe train e 29.297 validation;
- [`open-pii-masking-500k-ai4privacy`](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy)
  è la sorgente Ai4Privacy;
- [`pii-masking-ita`](https://huggingface.co/datasets/DeepMount00/pii-masking-ita)
  è gated e va acquisito separatamente.

Il branch v1.3 dichiara 2.176.674 righe train, cioè l'unione completa
`744.912 + 1.431.762`. Il modello v1.5/main dichiara invece 1.094.912 righe train
e 36.297 validation. La differenza implica aritmeticamente 350.000 righe clean
nel train, ma non esiste un manifest pubblico che identifichi quali righe siano
state selezionate.

Anche la recipe è contraddittoria: `training_args.bin` registra un'epoca, batch
128, learning rate `1e-4`, BF16 e fused AdamW; altre note parlano di due epoche e
batch effettivo 32; lo script locale usa batch 14, accumulo 2 e learning rate
`5e-5`, senza il clean train. Non tenteremo quindi di clonare v1.5.

Il nuovo run deve produrre:

- revisioni Hugging Face immutabili;
- manifest con ID, fonte, split e hash di ogni riga;
- seed e trasformazioni materializzate;
- configurazione completa, trainer state e hash dei checkpoint;
- split per fonte e famiglia di template, per evitare leakage.

La validation locale da 7.000 record è già stata usata per ablation e
quantizzazione. Serve un test finale cieco, tenuto fuori da tuning ed early
stopping.

## Licenze da verificare prima della distribuzione

`mmBERT-small`, il checkpoint Rizzo, il dataset clean e i sintetici Rizzo sono
dichiarati MIT. Il dataset storico complessivo è invece marcato come licenza
mista; Ai4Privacy combina metadata CC-BY-4.0 con condizioni aggiuntive descritte
nella card, mentre DeepMount è gated e non espone una licenza chiara nella card
pubblica. La licenza MIT del checkpoint non cancella automaticamente gli
obblighi dei dati a monte. Prima di distribuire commercialmente lo student
occorre una verifica formale di Ai4Privacy e DeepMount.

## Memoria di training

Lo student 22×384 con testa a 45 label ha circa 140,66 milioni di parametri:
98,30M sono ancora nell'embedding e 42,34M nell'encoder e nella testa.

Stima conservativa per mixed-precision AdamW, secondo la [memory anatomy di
Transformers](https://huggingface.co/docs/transformers/model_memory_anatomy):

| Componente persistente | Memoria stimata |
|---|---:|
| Parametri principali FP32 | 536,6 MiB |
| Copia BF16 forward/backward | 268,3 MiB |
| Gradienti FP32 | 536,6 MiB |
| Momentum e variance Adam FP32 | 1.073,1 MiB |
| **Totale statico prudenziale** | **circa 2,36 GiB** |

Attivazioni, contesto CUDA, kernel, allocator, batch e dataloader si aggiungono.
In distillazione online, il teacher congelato pesa altri 586,6 MiB se caricato
esplicitamente BF16; va mantenuto in `eval()` e senza gradienti.

| Memoria GPU | Fine-tuning diretto / KD offline | KD online |
|---:|---|---|
| 8 GB | possibile con microbatch 1–2 sui bucket lunghi, checkpointing e accumulo | troppo al limite, sconsigliato |
| 16 GB | configurazione raccomandata | fattibile con teacher BF16 |
| 24 GB | comoda | fattibile senza compromessi importanti |

Il MacBook Air M2 da 8 GB condivide RAM fra CPU e GPU. Inoltre lo script attuale
materializza fino a milioni di righe in liste Python e, nell'ambiente verificato,
PyTorch risulta compilato con MPS ma il backend non è disponibile. Anche dopo il
passaggio ad Arrow/Parquet memory-mapped, il portatile resta adatto a smoke test e
regressioni, non al run finale: swap e throttling renderebbero lento e poco
controllabile il training. Il target pratico è una GPU CUDA da 16 GB.

## Baseline supervisionata

Configurazione iniziale proposta:

- backbone `jhu-clsp/mmBERT-small`, tokenizer invariato e testa a 45 label;
- supervisione di tutti i subword, inclusi gli interni di CF, IBAN ed email;
- dataset Arrow/Parquet memory-mapped o streaming, non liste Python complete;
- dynamic padding e bucket per lunghezza; budget di token per update, non solo
  numero fisso di documenti;
- batch più piccolo per i documenti lunghi, gradient accumulation fino a circa
  16–32k subword per update;
- AdamW fused, BF16, weight decay `0,01`, warmup `0,05`, linear decay, gradient
  clipping `1,0`;
- sweep iniziale `3e-5 / 5e-5 / 1e-4` sul solo development interno;
- massimo tre epoche ed early stopping su macro-F1 e masking recall;
- activation checkpointing sulle GPU da 8/16 GB.

Il limite di sequenza va deciso dai percentili del nuovo manifest. Si possono
usare bucket 128/256/512 e un bucket lungo fino a 768–1152, invece di gonfiare
ogni batch al massimo. La REST attuale fa chunk corti, ma i casi lunghi devono
restare in un set di regressione separato.

## Distillazione eventuale

Se il direct student non supera i gate:

- teacher bloccato a `rizzo-pii-0.3B@v1.5.0`;
- loss iniziale `0,7 × CE_gold + 0,3 × T² × KL`, temperatura `T=2`;
- primo esperimento sui logits, senza hidden-state loss;
- maschera su padding e reweight di PII, boundary e hard negative, per evitare
  che la classe `O` domini la loss;
- preferenza per logits teacher offline FP16, sharded e legate allo stesso hash
  dell'input.

La cache full-logits costa 90 byte per subword: la stima è circa 21 GiB per un
mix v1.5-like e circa 59 GiB per l'unione completa. Una cache top-k è più piccola
ma altera la KL, quindi viene dopo il riferimento full-logits. Le augmentazioni
devono essere materializzate prima del caching; augmentation dinamica e logits
offline non sono allineabili.

Per il depth pruning si procede `22 → 18 → 16`, ricalcolando la sensibilità dopo
ogni rimozione. Il layer 0 non si rimuove; va conservata una quota dei layer
finali 19–21. Un 12-layer viene considerato soltanto se il 16-layer ha già
recuperato la qualità e può agire da teacher intermedio.

## Confronto originale contro student

Il teacher non è ground truth. Il protocollo finale misura:

1. teacher FP32 contro label umane;
2. student FP32 contro le stesse label;
3. disagreement paired exact-span fra teacher e student;
4. student quantizzato contro gold, contro student FP32 e contro teacher;
5. pipeline end-to-end dopo regex/checksum e merge, su raw text con offset umani.

Gate iniziali:

- delta micro-F1 non peggiore di `-0,002`;
- delta macro-F1 non peggiore di `-0,003`;
- nessun deterioramento statisticamente significativo della masking recall;
- recall protetta per CF, PIVA, IBAN, DOCID, ID_DOC, email e telefono;
- nessun crollo sui tag rari, bucket lunghi, OCR, casing e Markdown;
- paired bootstrap sugli stessi documenti, non solo differenza puntuale di F1.

Serve inoltre un blind challenge set con nomi ambigui (`Asia Argento`, `Rosa`,
`Viola`), hard negative, errori OCR e formattazione Markdown. Il set non deve
entrare in training o scelta degli iperparametri.

## Obiettivo 256 MiB

Il target è il picco del processo completo, non il solo artefatto. I soli pesi
teorici dello student 22×384 sono circa 134 MiB in W8, circa 105 MiB in W6 e
circa 67 MiB in W4. Questo rende 256 MiB plausibile, ma non ancora dimostrato:
runtime, tokenizer, prepacking, workspace, attivazioni e API devono entrare nello
stesso budget.

La verifica finale userà PSS/USS e picco del cgroup su Linux/Raspberry Pi, con
checkpoint già packed e senza materializzare FP32 all'avvio. Se Python e una
build ONNX Runtime generica superano il budget, il passo successivo sarà un
servizio nativo e una build runtime minimale limitata agli operatori del grafo.
