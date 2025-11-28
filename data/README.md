  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Internet (Wikipedia, diverse site-uri si statistici)
* **Modul de achiziție:** Normal prin webscraping, dar pentru testare le voi genera random
* **Perioada / condițiile colectării:** an 2021

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 1
* **Număr de caracteristici (features):** 11
* **Tipuri de date:** Numerice
* **Format fișiere:** JSON

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| numele orasului | string | - | folosit pentru a identifica orasul | - |
| populatie | numeric | oameni | populatia totala a orasului | 0-10000000 |
| arie | numeric | m | aria totala a orasului | 50 - 500 |
| GDP Orasului pe cap de locuitor | numeric | EUR | Gross Domestic Product pe cap de locuitor. Folosit pentru a determina cat de atragator economic este orasul | 20000 - 80000 |
| scor infrastructura | numeric | % | cat de atragator este d.p.d.v. al infrastructurii | 0-1 |
| scor educatie | numeric | % | cat de atragator este d.p.d.v. al educatiei  | 0-1 |
| scor locatie | numeric | % | cat de atragator este d.p.d.v. al locatiei | 0-1 |
| x | numeric | km | pozitia orasului pe axa x | 0-1000 |
| y | numeric | km | positia orasului pe axa y | 0-1000 |
| aglomerare | numeric | % | aglomeratia  | 0-1 |
