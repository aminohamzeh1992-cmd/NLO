# NLO

Salesman/StructRL-Projekt fuer eine 9-Raeume-GridWorld.

Der aktuelle GitHub-Stand enthaelt den lauffaehigen Salesman-Kern mit exakter Oracle-Berechnung fuer den optimalen Weg:

- `Salesman/env.py` baut 4-, 6- und 9-Raeume-Umgebungen.
- `Salesman/oracle.py` berechnet den optimalen Weg exakt per kuerzesten Wegen + Held-Karp-DP.
- `Salesman/main.py` ist der einfache Einstieg fuer den Oracle-Lauf.
- `Salesman/results_oracle/` enthaelt die berechnete Referenz fuer 9 Raeume, 5 Items, Seed 42.

## Installation

```powershell
pip install -r requirements.txt
```

## Starten

```powershell
cd Salesman
python main.py --oracle_only
```

Oder explizit fuer den geprueften Fall:

```powershell
python main.py --rooms 9 --items 5 --item_seed 42 --oracle_only
```

## Geprueftes Ergebnis

Fuer `rooms=9`, `items=5`, `item_seed=42`:

- Optimal: `42` Schritte
- Optimale Item-Reihenfolge: `2-5-3-4-1`
- Naive Reihenfolge: `108` Schritte
- Ersparnis: `66` Schritte (`61.11%`)

Die CSVs liegen unter:

```text
Salesman/results_oracle/
```
