## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Ilies Eduard Cosmin |
| **Grupa / Specializare** | 632AB / Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/CosminIlies/RN_FastestPath |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python |
| **Domeniul Industrial de Interes (DII)** | [ex: Robotică / Producție / Medical / Energie / Automotive] |
| **Tip Rețea Neuronală** | GNN |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

Am decis sa folosesc R2 deoarece este mult mai potrivit pentru proiectul meu.

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 90% | 90% | - | ✓ |
| R2-Score | ≥0.65 | 0.93 | 0.93 | - | ✓ |
| Latență Inferență | [target student] | 48 ms | 48 ms | - | ✓ |
| Contribuție Date Originale | ≥40% | 100% | 100% | - | ✓ |
| Nr. Experimente Optimizare | ≥4 | 4 | 4 | - | ✓ |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | DA |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) |  DA  |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie |  DA  |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) |  DA  |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | DA  |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

*[Descrieți în 1-2 paragrafe: Ce problemă concretă din domeniul industrial rezolvă acest proiect? Care este contextul și situația actuală? De ce este importantă rezolvarea acestei probleme?]*

Prin utilizarea unei retele neuronale de tip Graph Neural Network (GNN) pentru modelarea densitatii urbane putem anticipa cu precizie punctele de expansiune si zonele cu potential ridicat de dezvoltare pe care un anteprenor le poate folosi sa fie cu un pas in fata competitorilor sai.

### 2.2 Beneficii Măsurabile Urmărite

*[Listați 3-5 beneficii concrete cu metrici țintă]*

1. Reducerea cu 50% a timpului de cercetare pentru identificarea unui loc potrivit pentru o locatie noua
2. Scaderea costului de achizitie cu macar 15% pentru campaniile care au la baza panouri publicitare 
3. Scaderea cu 30% a ratei de inchidere a locatiilor noi in primele 24 de luni


### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Gasirea unei locatii noi la un pret convenabil  | Predictie aglomeratie in functie de anumite modificari ale unor parametrii | RN + Web Service | acuratete > 75% |


---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Generate |
| **Sursa concretă** | [ex: Kaggle - dataset X / Senzori Arduino / Simulare Gazebo] |
| **Număr total observații finale (N)** | 302 |
| **Număr features** | 5 |
| **Tipuri de date** | Numerice |
| **Format fișiere** | JSON |
| **Perioada colectării/generării** | Noiembrie 2025 - Ianuarie 2026 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 302 |
| **Observații originale (M)** | 302 |
| **Procent contribuție originală** | 100% |
| **Tip contribuție** | Date sintetice |
| **Locație cod generare** | `src/data_acquisition/generate_data.py` |
| **Locație date originale** | `data/generated/` |

**Descriere metodă generare/achiziție:**

*[Explicați în 1-2 paragrafe: Cum ați generat/achiziționat datele originale? Ce parametri ați folosit? De ce sunt relevante pentru problema voastră?]*

Am incercat un algoritm care sa mi genereze date cat mai aprope de realitate.

Pasul 1: Generez, pentru fiecare nod, valori pentru x, y, populatie, gdp_per_capita, education_score, infrastructure_score, location_score. Fac asta tinand cont de city_coef, o valoare care denota cat de dezvoltat este orasul, astfel se elimina posibilitatea de generare a unui oras care nu ar avea sens sa existe (ex: un oras cu o economie foarte buna dar o educatie, infrastructura si populatie foarte proasta)

Pasul 2: In functie de coordonatele x si y generez muchiile dintre noduri tinand cont si de economia oraselor. Astfel daca 2 orase sunt mai departate dar au o economie similara se va genera o muchie.

Pasul 3: Aplicam influenta vecinilor unui nod, astfel orasele vor fi mai asemanatoare cu vecinii lor

Pasul 4: Simulam fenomenul de economic clustering, prin care orasele vecine asemanatoare economic au un scor de educatie si de infrastructura similar (https://www.c2es.org/wp-content/uploads/2025/08/Economic-Clustering-101_C2ES-Fact-Sheet.pdf)


Pasul 5: Calculez aglomeratie dupa o ponere a fiecarui parametru


### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 80% | 241 |
| Validation | 10% | 30 |
| Test | 10% | 31 |

**Preprocesări aplicate:**
- Din moment ce datele au fost generate fix cum aveam nevoie, nu a fost nevoie de preprocesare

**Referințe fișiere:** `data/README.md`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | Python | Generare date prin agloritm complex | `src/data_acquisition/` |
| **Neural Network** | PyTorch | GNN cu layere de GCN | `src/neural_network/` |
| **Web Service / UI** | Flask | Interfata unde utilizatorul poate sa faca modificari la reteaua de orase si sa genereze predictii | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine_v2.drawio` 

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `Init` | Initializere program | Start aplicație | Termina de executat |
| `GENERATE_DATA` | Genereaza datele | Nu exista deja un model pe care sa l incarcam | Date generate |
| `SAVE_DATA` | Imparte datele in train, validate si test si le salveaza in JSON | Datele sunt generate | Datele sunt salvate |
| `CREATE_MODEL` | Cream modelul | Datele sunt salvate | modelul este creat |
| `TRAIN` | Antrenam reteaua neuronala | Avem un model creat | modelul este antrenat |

| `PREDICT` | Generam o predictie| Primim cerere de la user | generam predictia |
| `IDLE` | Asteptam comenzi de efectuat | Avem modelul incarcat | Primim input de la user prin web |
| `ERROR` | Gestionare erori și logging | Excepție detectată| Recovery/Stop |

**Justificare alegere arhitectură State Machine:**

*[1 paragraf: De ce această structură pentru problema voastră specifică?]*

Aceasta structura contine toate state urile de care avem nevoie si tranzitii pe care le folosim sau le am putea folosi pe viitor.

### 4.3 Actualizări State Machine în Etapa 6 (dacă este cazul)

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| [ex: Threshold alertă] | [0.5] | [0.35] | [Minimizare False Negatives] |
| [ex: Stare nouă adăugată] | N/A | `CONFIDENCE_CHECK` | [Filtrare predicții incerte] |
| [Completați dacă e cazul] | | | |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
Input (shape: [5]) 
  → GCNConv(5, 128, leaky_relu)  → Dropout(0.3)
  → GCNConv(128, 128, leaky_relu) → Dropout(0.3)
  → GCNConv(128, 64, leaky_relu) → Dropout(0.3)
  → Linear(64, 1)
Output: agglomeratie
```

**Justificare alegere arhitectură:**

*[1-2 propoziții: De ce această arhitectură? Ce alternative ați considerat și de ce le-ați respins?]*

Prima idee era sa folosesc o retea neuronala simpla, insa asa nu tinea cont de vecinii unui oras.
In urma mai multor iteratii aceasta arhitectura a dar cele mai bune rezultate. Imbudatatiri mari a adus implementarea residual connection.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.001 | Valoare standard pentru Adam optimizer, asigură convergență stabilă. Ea este modificata automat in functie de loss |
| Batch size | 302 | Deoarece folosesc un GNN este greu de impartit in sub grafuri pentru a antrena pe batch uri.|
| Number of epochs | 3000 |  |
| Optimizer | Adam | Adaptive learning rate|
| Loss function | smooth_l1_loss (Huber loss) |  Combina avantajele MSE si MAE |
| Activation functions | leaky_relu (hidden), Sigmoid (output) |  |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | R2-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
|  **Baseline** | Configurația din Etapa 5 | 0.72 | 0.07 |  | Referință |
| Exp 1 | Imbunatatire alg generare date | 0.80 | 0.617 | 23.43s | in viata reala un oras are estimativ 3 - 6 vecini, reducerea numarului de vecini a ajutat |
| Exp 2 | neuroni hidden layer 64 - 128  | 0.82 | 0.65 | 29.79s | Îmbunătățire semnificativă |
| Exp 3 | silu -> leaky_relu | 0.86 | 0.81 | 31.2s | imbunatatire masiva. leaky_relu este mai folosit pentru GNN uri iar silu pentru CNN uri. Posibil din cauza faptului ca silu converge la 0 spre -x, iar leaky_relu contine si valori negative |
| Exp 4 | residual connections | 0.90 | 0.93 | 28.92s |  |
| **FINAL** | Configurația aleasă | 0.90 | 0.93 | 28.92s |  |

**Justificare alegere model final:**

*[1 paragraf: De ce această configurație? Ce compromisuri ați făcut între accuracy/timp/complexitate?]*

Am ales Exp 4 ca model final pentru că:
1. Oferă cel mai bun r2-score (0.93)
2. Timpul de antrenare este mic in oricare dintre cazuri
3. Cea mai mica diferenta dinre aglomeratia reala si cea prezisa

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/city_agglomeration_gnn.pt`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | [X.XX%] | ≥70% | [✓/✗] |
| **F1-Score (Macro)** | [X.XX] | ≥0.65 | [✓/✗] |
| **Precision (Macro)** | [X.XX] | - | - |
| **Recall (Macro)** | [X.XX] | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | [X.XX%] | [X.XX%] | [+X.XX%] |
| F1-Score | [X.XX] | [X.XX] | [+X.XX] |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| **Clasa cu cea mai bună performanță** | [Nume clasă] - Precision [X%], Recall [Y%] |
| **Clasa cu cea mai slabă performanță** | [Nume clasă] - Precision [X%], Recall [Y%] |
| **Confuzii frecvente** | [ex: Clasa A confundată frecvent cu Clasa B - posibil din cauza similarității vizuale] |
| **Dezechilibru clase** | [ex: Clasa C are doar 5% din date - recall scăzut explicabil] |

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
| 1 | [ex: Imagine sudură cu iluminare slabă] | [Clasa X] | [Clasa Y] | [ex: Contrast insuficient în zona defectului] | [ex: Defect nedetectat → produs defect la client] |
| 2 | [Completați] | [Completați] | [Completați] | [Completați] | [Completați] |
| 3 | [Completați] | [Completați] | [Completați] | [Completați] | [Completați] |
| 4 | [Completați] | [Completați] | [Completați] | [Completați] | [Completați] |
| 5 | [Completați] | [Completați] | [Completați] | [Completați] | [Completați] |

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

*[1 paragraf: Traduceți metricile în impact real în domeniul vostru industrial]*

[ex: Din 100 de piese cu defecte reale, modelul detectează corect 78 (Recall=78%). 22 de piese defecte ajung la client - cost estimat: 22 × 50 RON = 1100 RON/lot. În același timp, din 100 piese bune, 8 sunt clasificate greșit ca defecte (FP=8%) - cost reinspecție: 8 × 5 RON = 40 RON/lot.]

**Pragul de acceptabilitate pentru domeniu:** [ex: Recall ≥ 85% pentru defecte critice]  
**Status:** [Atins / Neatins - cu diferența]  
**Plan de îmbunătățire (dacă neatins):** [ex: Augmentare date pentru clasa subreprezentată, ajustare threshold]

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.h5` | `optimized_model.h5` | [ex: +8% accuracy, -12% FN] |
| **Threshold decizie** | [ex: 0.5 default] | [ex: 0.35 pentru clasa 'defect'] | [ex: Minimizare FN în context producție] |
| **UI - feedback vizual** | [ex: Da/Nu text] | [ex: Bară confidence + valoare %] | [ex: Informare operator pentru decizii] |
| **Logging** | [ex: Doar predicție] | [ex: Predicție + confidence + timestamp] | [ex: Audit trail pentru QA] |
| [Alte modificări] | [Completați] | [Completați] | [Completați] |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

*[Descriere scurtă: Ce se vede în screenshot? Ce demonstrează?]*

[Completați aici]

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/` *(GIF / Video / Secvență screenshots)*

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Input | [ex: Upload imagine nouă (NU din train/test)] |
| 2 | Procesare | [ex: Bară de progres + preprocesare vizibilă] |
| 3 | Inferență | [ex: Predicție afișată: "Clasa: Defect, Confidence: 87%"] |
| 4 | Decizie | [ex: Alertă roșie + sunet pentru operator] |

**Latență măsurată end-to-end:** [X] ms  
**Data și ora demonstrației:** [DD.MM.YYYY, HH:MM]

---

## 8. Structura Repository-ului Final

```
proiect-rn-[nume-prenume]/
│
├── README.md                               # ← ACEST FIȘIER (Overview Final Proiect - Pe moodle la Evaluare Finala RN > Upload Livrabil 1 - Proiect RN (Aplicatie Sofware) - trebuie incarcat cu numele: NUME_Prenume_Grupa_README_Proiect_RN.md)
│
├── docs/
│   ├── etapa3_analiza_date.md              # Documentație Etapa 3
│   ├── etapa4_arhitectura_SIA.md           # Documentație Etapa 4
│   ├── etapa5_antrenare_model.md           # Documentație Etapa 5
│   ├── etapa6_optimizare_concluzii.md      # Documentație Etapa 6
│   │
│   ├── state_machine.png                   # Diagrama State Machine inițială
│   ├── state_machine_v2.png                # (opțional) Versiune actualizată Etapa 6
│   ├── confusion_matrix_optimized.png      # Confusion matrix model final
│   │
│   ├── screenshots/
│   │   ├── ui_demo.png                     # Screenshot UI schelet (Etapa 4)
│   │   ├── inference_real.png              # Inferență model antrenat (Etapa 5)
│   │   └── inference_optimized.png         # Inferență model optimizat (Etapa 6)
│   │
│   ├── demo/                               # Demonstrație funcțională end-to-end
│   │   └── demo_end_to_end.gif             # (sau .mp4 / secvență screenshots)
│   │
│   ├── results/                            # Vizualizări finale
│   │   ├── loss_curve.png                  # Grafic loss/val_loss (Etapa 5)
│   │   ├── metrics_evolution.png           # Evoluție metrici (Etapa 6)
│   │   └── learning_curves_final.png       # Curbe învățare finale
│   │
│   └── optimization/                       # Grafice comparative optimizare
│       ├── accuracy_comparison.png         # Comparație accuracy experimente
│       └── f1_comparison.png               # Comparație F1 experimente
│
├── data/
│   ├── README.md                           # Descriere detaliată dataset
│   ├── raw/                                # Date brute originale
│   ├── processed/                          # Date curățate și transformate
│   ├── generated/                          # Date originale (contribuția ≥40%)
│   ├── train/                              # Set antrenare (70%)
│   ├── validation/                         # Set validare (15%)
│   └── test/                               # Set testare (15%)
│
├── src/
│   ├── data_acquisition/                   # MODUL 1: Generare/Achiziție date
│   │   ├── README.md                       # Documentație modul
│   │   ├── generate.py                     # Script generare date originale
│   │   └── [alte scripturi achiziție]
│   │
│   ├── preprocessing/                      # Preprocesare date (Etapa 3+)
│   │   ├── data_cleaner.py                 # Curățare date
│   │   ├── feature_engineering.py          # Extragere/transformare features
│   │   ├── data_splitter.py                # Împărțire train/val/test
│   │   └── combine_datasets.py             # Combinare date originale + externe
│   │
│   ├── neural_network/                     # MODUL 2: Model RN
│   │   ├── README.md                       # Documentație arhitectură RN
│   │   ├── model.py                        # Definire arhitectură (Etapa 4)
│   │   ├── train.py                        # Script antrenare (Etapa 5)
│   │   ├── evaluate.py                     # Script evaluare metrici (Etapa 5)
│   │   ├── optimize.py                     # Script experimente optimizare (Etapa 6)
│   │   └── visualize.py                    # Generare grafice și vizualizări
│   │
│   └── app/                                # MODUL 3: UI/Web Service
│       ├── README.md                       # Instrucțiuni lansare aplicație
│       └── main.py                         # Aplicație principală
│
├── models/
│   ├── untrained_model.h5                  # Model schelet neantrenat (Etapa 4)
│   ├── trained_model.h5                    # Model antrenat baseline (Etapa 5)
│   ├── optimized_model.h5                  # Model FINAL optimizat (Etapa 6) ← FOLOSIT
│   └── final_model.onnx                    # (opțional) Export ONNX pentru deployment
│
├── results/
│   ├── training_history.csv                # Istoric antrenare - toate epocile (Etapa 5)
│   ├── test_metrics.json                   # Metrici baseline test set (Etapa 5)
│   ├── optimization_experiments.csv        # Toate experimentele optimizare (Etapa 6)
│   ├── final_metrics.json                  # Metrici finale model optimizat (Etapa 6)
│   └── error_analysis.json                 # Analiza detaliată erori (Etapa 6)
│
├── config/
│   ├── preprocessing_params.pkl            # Parametri preprocesare salvați (Etapa 3)
│   └── optimized_config.yaml               # Configurație finală model (Etapa 6)
│
├── requirements.txt                        # Dependențe Python (actualizat la fiecare etapă)
└── .gitignore                              # Fișiere excluse din versionare
```

### Legendă Progresie pe Etape

| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| `data/raw/`, `processed/`, `train/`, `val/`, `test/` | ✓ Creat | - | Actualizat* | - |
| `data/generated/` | - | ✓ Creat | - | - |
| `src/preprocessing/` | ✓ Creat | - | Actualizat* | - |
| `src/data_acquisition/` | - | ✓ Creat | - | - |
| `src/neural_network/model.py` | - | ✓ Creat | - | - |
| `src/neural_network/train.py`, `evaluate.py` | - | - | ✓ Creat | - |
| `src/neural_network/optimize.py`, `visualize.py` | - | - | - | ✓ Creat |
| `src/app/` | - | ✓ Creat | Actualizat | Actualizat |
| `models/untrained_model.*` | - | ✓ Creat | - | - |
| `models/trained_model.*` | - | - | ✓ Creat | - |
| `models/optimized_model.*` | - | - | - | ✓ Creat |
| `docs/state_machine.*` | - | ✓ Creat | - | (v2 opțional) |
| `docs/etapa3_analiza_date.md` | ✓ Creat | - | - | - |
| `docs/etapa4_arhitectura_SIA.md` | - | ✓ Creat | - | - |
| `docs/etapa5_antrenare_model.md` | - | - | ✓ Creat | - |
| `docs/etapa6_optimizare_concluzii.md` | - | - | - | ✓ Creat |
| `docs/confusion_matrix_optimized.png` | - | - | - | ✓ Creat |
| `docs/screenshots/` | - | ✓ Creat | Actualizat | Actualizat |
| `results/training_history.csv` | - | - | ✓ Creat | - |
| `results/optimization_experiments.csv` | - | - | - | ✓ Creat |
| `results/final_metrics.json` | - | - | - | ✓ Creat |
| **README.md** (acest fișier) | Draft | Actualizat | Actualizat | **FINAL** |

*\* Actualizat dacă s-au adăugat date noi în Etapa 4*

### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset analizat și preprocesat" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură SIA funcțională" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - Accuracy=X.XX, F1=X.XX" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - Accuracy=X.XX, F1=X.XX (optimizat)" |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

```
Python >= 3.8 (recomandat 3.10+)
pip >= 21.0
[sau LabVIEW >= 2020 pentru proiecte LabVIEW]
```

### 9.2 Instalare

```bash
# 1. Clonare repository
git clone [URL_REPOSITORY]
cd proiect-rn-[nume-prenume]

# 2. Creare mediu virtual (recomandat)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# sau: venv\Scripts\activate    # Windows

# 3. Instalare dependențe
pip install -r requirements.txt
```

### 9.3 Rulare Pipeline Complet

```bash
# Pasul 1: Preprocesare date (dacă rulați de la zero)
python src/preprocessing/data_cleaner.py
python src/preprocessing/data_splitter.py --stratify --random_state 42

# Pasul 2: Antrenare model (pentru reproducere rezultate)
python src/neural_network/train.py --config config/optimized_config.yaml

# Pasul 3: Evaluare model pe test set
python src/neural_network/evaluate.py --model models/optimized_model.h5

# Pasul 4: Lansare aplicație UI
streamlit run src/app/main.py
# sau: python src/app/main.py (pentru Flask/FastAPI)
# sau: [instrucțiuni LabVIEW dacă aplicabil]
```

### 9.4 Verificare Rapidă 

```bash
# Verificare că modelul se încarcă corect
python -c "from src.neural_network.model import load_model; m = load_model('models/optimized_model.h5'); print('✓ Model încărcat cu succes')"

# Verificare inferență pe un exemplu
python src/neural_network/evaluate.py --model models/optimized_model.h5 --quick-test
```

### 9.5 Structură Comenzi LabVIEW (dacă aplicabil)

```
[Completați dacă proiectul folosește LabVIEW]
1. Deschideți [nume_proiect].lvproj
2. Rulați Main.vi
3. ...
```

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|--------------------------------|--------|----------|--------|
| [Obiectiv 1 din 2.2] | [target] | [realizat] | [✓/✗] |
| [Obiectiv 2 din 2.2] | [target] | [realizat] | [✓/✗] |
| Accuracy pe test set | ≥70% | [X.XX%] | [✓/✗] |
| F1-Score pe test set | ≥0.65 | [X.XX] | [✓/✗] |
| [Metric specific domeniului] | [target] | [realizat] | [✓/✗] |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

*[Fiți onești - evaluatorul apreciază identificarea clară a limitărilor]*

1. **Limitare 1:** [ex: Modelul eșuează pe imagini cu iluminare <50 lux - accuracy scade la 45%]
2. **Limitare 2:** [ex: Latența depășește 100ms pentru batch size >32 - neadecvat pentru real-time]
3. **Limitare 3:** [ex: Clasa "defect_minor" are recall doar 52% - date insuficiente]
4. **Funcționalități planificate dar neimplementate:** [ex: Export ONNX, integrare API extern]

### 10.3 Lecții Învățate (Top 5)

1. **[Lecție 1]:** [ex: Importanța EDA înainte de antrenare - am descoperit 8% valori lipsă care afectau convergența]
2. **[Lecție 2]:** [ex: Early stopping a prevenit overfitting sever - fără el, val_loss creștea după epoca 20]
3. **[Lecție 3]:** [ex: Augmentările specifice domeniului (zgomot gaussian calibrat) au adus +5% accuracy vs augmentări generice]
4. **[Lecție 4]:** [ex: Threshold-ul default 0.5 nu e optim pentru clase dezechilibrate - ajustarea la 0.35 a redus FN cu 40%]
5. **[Lecție 5]:** [ex: Documentarea incrementală (la fiecare etapă) a economisit timp major la integrare finală]

### 10.4 Retrospectivă

**Ce ați schimba dacă ați reîncepe proiectul?**

*[1-2 paragrafe: Decizii pe care le-ați lua diferit, cu justificare bazată pe experiența acumulată]*

[Completați aici]

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** (1-2 săptămâni) | [ex: Augmentare date pentru clasa subreprezentată] | [ex: +10% recall pe clasa "defect_minor"] |
| **Medium-term** (1-2 luni) | [ex: Implementare model ensemble] | [ex: +3-5% accuracy general] |
| **Long-term** | [ex: Deployment pe edge device (Raspberry Pi)] | [ex: Latență <20ms, cost hardware redus] |

---

## 11. Bibliografie

*[Minimum 3 surse cu DOI/link funcțional - format: Autor, Titlu, Anul, Link]*

1. [Autor], [Titlu articol/carte], [Anul]. DOI: [link] sau URL: [link]
2. [Autor], [Titlu articol/carte], [Anul]. DOI: [link] sau URL: [link]
3. [Autor], [Titlu articol/carte], [Anul]. DOI: [link] sau URL: [link]
4. [Surse suplimentare dacă este cazul]

**Exemple format:**
- Abaza, B., 2025. AI-Driven Dynamic Covariance for ROS 2 Mobile Robot Localization. Sensors, 25, 3026. https://doi.org/10.3390/s25103026
- Keras Documentation, 2024. Getting Started Guide. https://keras.io/getting_started/

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [ ] **Accuracy ≥70%** pe test set (verificat în `results/final_metrics.json`)
- [ ] **F1-Score ≥0.65** pe test set
- [ ] **Contribuție ≥40% date originale** (verificabil în `data/generated/`)
- [ ] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [ ] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [ ] **Confusion matrix** generată și interpretată (Secțiunea 6.2)
- [ ] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [ ] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [ ] **Demonstrație end-to-end** disponibilă în `docs/demo/`

### Repository și Documentație

- [ ] **README.md** complet (toate secțiunile completate cu date reale)
- [ ] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [ ] **Screenshots** prezente în `docs/screenshots/`
- [ ] **Structura repository** conformă cu Secțiunea 8
- [ ] **requirements.txt** actualizat și funcțional
- [ ] **Cod comentat** (minim 15% linii comentarii relevante)
- [ ] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare

- [ ] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [ ] **Tag `v0.6-optimized-final`** creat și pushed
- [ ] **Commit-uri incrementale** vizibile în `git log` (nu 1 commit gigantic)
- [ ] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

### Verificare Anti-Plagiat

- [ ] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [ ] **Minimum 40% date originale** (nu doar subset din dataset public)
- [ ] Cod propriu sau clar atribuit (surse citate în Bibliografie)

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** [DD.MM.YYYY]  
**Tag Git:** `v0.6-optimized-final`

---

*Acest README servește ca documentație principală pentru Livrabilul 1 (Aplicație RN). Pentru Livrabilul 2 (Prezentare PowerPoint), consultați structura din RN_Specificatii_proiect.pdf.*
