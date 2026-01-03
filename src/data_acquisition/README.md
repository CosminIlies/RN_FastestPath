# Generarea Datelor pentru Rețeaua de Orașe

Acest modul conține funcțiile pentru generarea automată de date sintetice pentru o rețea de orașe și conexiunile dintre acestea, utilizate pentru antrenarea rețelei neurale de tip Graph Neural Network (GNN).

## Metoda de Generare

### 1. Generarea Caracteristicilor Orașelor

Funcția `generate_data(city_name, seed=1)` generează caracteristici sintetice pentru un oraș folosind următoarele distribuții probabilistice:

#### Caracteristici Demografice și Economice:
- **Populație (`population`)**: Generată folosind o distribuție log-normală cu `mean=10` și `sigma=1.5`, rezultând în valori realiste pentru populația orașelor
- **Suprafața (`area`)**: Distribuție uniformă între 50 și 500 km²
- **PIB per capita (`gdp_per_capita`)**: Distribuție uniformă între 20,000 și 80,000 de unități monetare

#### Score-uri de Dezvoltare (0.1 - 1.0):
- **Score Educațional (`education_score`)**: Indică calitatea sistemului educațional
- **Score Infrastructură (`infrastructure_score`)**: Reflectă dezvoltarea infrastructurii
- **Score Localizare (`location_score`)**: Evaluează avantajele geografice

#### Coordonate Geografice:
- **Coordonata X (`x`)**: Uniformă între 0 și 1,000 
- **Coordonata Y (`y`)**: Uniformă între 0 și 1,000

### 2. Generarea Conexiunilor (Edges)

Funcția `generate_edges(cities, num_cities)` creează legături între orașe bazate pe criterii geografice și economice:

#### Algoritm de Conectare:

1. **Calcul Distanță Geografică**:
   ```
   geo_dist = √[(x₁ - x₂)² + (y₁ - y₂)²]
   ```

2. **Calcul Similaritate Economică**:
   ```
   econ_similarity = 1 - |PIB_capita₁ - PIB_capita₂| / 60,000
   ```

3. **Criterii de Conectare**:
   - Orașele sunt conectate dacă:
     - Distanța geografică < 300 unități, SAU
     - Distanța geografică < 500 unități ȘI similaritatea economică > 0.7

#### Caracteristici ale Rețelei Generate:

- **Graful rezultat este neorientat** (conexiunile sunt bidireccionale)
- **Greutatea muchiilor** este egală cu distanța geografică între orașe
- **Densitatea conexiunilor** depinde de distribuția spațială și economică a orașelor
- **Clusterele economice** tind să fie mai dense datorită criteriului de similaritate economică

## Avantajele Metodei

1. **Realism**: Distribuțiile folosite reflectă caracteristicile reale ale orașelor
2. **Reproductibilitate**: Folosirea unui seed permite regenerarea exactă a datelor
3. **Flexibilitate**: Parametrii pot fi ajustați pentru diferite scenarii
4. **Scalabilitate**: Metoda funcționează pentru orice număr de orașe
