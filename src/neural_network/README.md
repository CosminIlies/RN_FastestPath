# Rețea Neuronală de Grafuri pentru Aglomerări Urbane

## Prezentare Generală

Această implementare oferă o **rețea neuronală de grafuri (GNN)** specializată pentru analiza aglomerărilor urbane. Modelul utilizează **Rețele Convoluționale de Grafuri (GCN)** pentru procesarea eficientă a datelor structurate sub formă de graf, reprezentând orașe și conexiunile complexe dintre acestea.
## Arhitectura Tehnică

### Structura Straturilor

Modelul implementează o arhitectură GCN cu **trei straturi convoluționale** și un strat de ieșire liniar:

#### 1. Strat de Intrare
- **Tip**: GCNConv
- **Dimensiuni**: `input_dim → hidden_dim`
- **Activare**: ReLU
- **Regularizare**: Dropout (p=0.3)

#### 2. Primul Strat Ascuns
- **Tip**: GCNConv  
- **Dimensiuni**: `hidden_dim → hidden_dim`
- **Activare**: ReLU
- **Regularizare**: Dropout (p=0.3)

#### 3. Al Doilea Strat Ascuns
- **Tip**: GCNConv
- **Dimensiuni**: `hidden_dim → hidden_dim // 2`
- **Activare**: ReLU

#### 4. Strat de Ieșire
- **Tip**: Linear
- **Dimensiuni**: `hidden_dim // 2 → output_dim`
- **Activare**: Niciuna (raw output)

### Caracteristici de Regularizare

- **Dropout**: Aplicat cu probabilitatea 0.3 după primele două straturi GCN pentru prevenirea overfitting-ului
- **Activarea ReLU**: Utilizată după fiecare strat GCN, exceptând stratul final

## Parametrii de Configurare

### Parametri de Intrare

| Parametru | Tipul | Valoare Implicită | Descriere |
|-----------|--------|-------------------|-----------|
| `input_dim` | `int` | **obligatoriu** | Numărul caracteristicilor de intrare per nod (oraș) |
| `hidden_dim` | `int` | `64` | Numărul unităților ascunse în straturile GCN |
| `output_dim` | `int` | `1` | Dimensiunea vectorului de ieșire |

