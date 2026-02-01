# OCR AI Methodenvergleich

Dieses Repository implementiert und bewertet zwei leichte Feed-Forward-Klassifikatoren auf dem MNIST-Datensatz unter Verwendung zweier unterschiedlicher Trainingsparadigmen: Genetischer Algorithmus (GA) und Gradientenabstieg (GD). 

Es legt besonderen Wert auf Reproduzierbarkeit, konfigurierbare Experimente und klar definierte Evaluationsmetriken, wodurch es sich für Forschung, Benchmarking und Bildungszwecke eignet.

## Inhaltsverzeichnis
- [Übersicht](#übersicht)
- [Warum dieses Projekt? 📝](#warum-dieses-projekt-)
- [Schnellstart & Installation 🔧](#schnellstart--installation-)
- [Modelle einzeln trainieren und testen 🧪](#modelle-einzeln-trainieren-und-testen-)
- [GA vs GD programmatisch vergleichen ⚖️](#ga-vs-gd-programmatisch-vergleichen-)
- [Visuelle Ergebnisse 📊](#visuelle-ergebnisse-)
- [Daten-Pipeline](#daten-pipeline)
- [Lizenz](#lizenz)

## Übersicht

Enthaltene Ansätze:

- **Genetischer Algorithmus (GA)**: Population-basierte Optimierung der Modellparameter über Generationen.
- **Gradientenabstieg (GD)**: Klassisches neuronales Netz, trainiert mit stochastischem Gradientenabstieg.

## Warum dieses Projekt? 📝

Ich habe dieses Projekt als Facharbeit der 9. Klasse gewählt, um eine grundlegende Fragestellung in der KI zu untersuchen: Ob unterschiedliche Trainingsparadigmen zu vergleichbaren Ergebnissen führen und, falls nicht, wie sich ihre Effizienz und Leistung unterscheiden. Dieses Projekt ermöglichte es mir, zwei verschiedene Methoden, Genetische Algorithmen und Gradientenabstieg, in einem reproduzierbaren, modularen Rahmen zu implementieren und ihre Ergebnisse auf dem MNIST-Datensatz vergleichend zu analysieren.

## Schnellstart & Installation 🔧

Klonen Sie das Repository und installieren Sie die Abhängigkeiten:

```bash
git clone https://github.com/leopoldsprenger/ai-training-comparison.git
cd ai-training-comparison
pip install -r requirements.txt
```

Hinweis: Diese Anleitung nennt keine spezifische Python-Version. Stellen Sie sicher, dass PyTorch und die in `requirements.txt` gelisteten Pakete installiert sind.

Führen Sie die Unit-Tests aus mit:

```bash
pytest -q
```

## Modelle einzeln trainieren und testen 🧪

Die Trainings- und Testskripte verwenden `argparse`. Beispiele:

- Trainiere das Genetische Algorithmus-Modell:
  ```bash
  python3 src/genetic_algorithm/train.py --name model_name
  ```

- Teste ein gespeichertes Genetisches Algorithmus-Modell:
  ```bash
  python3 src/genetic_algorithm/test.py --name model_name
  ```

- Trainiere mit Gradientenabstieg:
  ```bash
  python3 src/gradient_descent/train.py --name model_name
  ```

- Teste ein gespeichertes Gradientenabstiegs-Modell:
  ```bash
  python3 -m src/gradient_descent/test.py --name model_name
  ```

Wichtige Flags:
- `--name` / `-n`: Basisname für gespeicherte Modelle (ohne Erweiterung). Modelle werden im Ordner `models/` abgelegt.
- `--generations` / `-g`: Anzahl Generationen (für GA) bzw. Epochen (für GD), verwendet vom Vergleichsskript.

## Vergleichsskript ⚖️

Das Vergleichsskript trainiert GA und GD, speichert beide Modelle und plottet die Genauigkeiten:

```bash
python -m src.utils.compare_trainings --name vergleich1 --generations 50
```

Die Modelle werden nach `models/{name}_ga.pt` und `models/{name}_gd.pt` geschrieben.

## Visuelle Ergebnisse 📊

Beispielbilder befinden sich im `imgs/`-Ordner. Unten sind exemplarische Ergebnisse mit kurzen Bildunterschriften:

| GA — nach Hyperparameter-Anpassungen | GA — 50 Generationen |
|------------------------------------|--------------------|
| ![GA after hyperparameter](../imgs/genetic_algorithm/test_3_after_adjusting_hyper_parameters.png) | ![GA 50 generations](../imgs/genetic_algorithm/test_4_test_with_50_gens.png) |
| zeigt das Verhalten früher Generationen nach Anpassungen an Selektion und Mutation. | Verlauf von Best- und Durchschnittsgenauigkeit über 50 Generationen. |

| GD — Kreuzentropie pro Epoche | GD — Testdatensatz-Vorhersagen |
|-------------------------------|-------------------------------|
| ![Cross entropy per epoch](../imgs/gradient_descent/cross_entropy_per_epochs_graph.png) | ![GD predictions](../imgs/gradient_descent/gradient_descent_test_dataset_predictions.png) |
| Hilfreich zum Beurteilen von Konvergenz und Stabilität. | Beispielvorhersagen auf dem Testdatensatz (erste 40 Bilder) mit prognostizierten Labels. |

![Vergleich](../imgs/accuracy_comparison_gradient_descent_and_genetic_algorithm.png)  
*Vergleich — GA (best), GA (durchschnittlich) und GD (pro Epoche) zusammengeführt für direkten Vergleich.*

## Daten-Pipeline

Das Modul `src/data_manager.py` stellt eine kleine, robuste Dataset-Abstraktion bereit, die im Projekt verwendet wird:

- `MNISTDataset(filepath)` erwartet eine gespeicherte `(images, labels)`-Tupeldatei im torch `.pt`-Format und validiert Existenz und Typen.
- `images` werden auf `(N, TENSOR_SIZE)` umgeformt und auf den Bereich [0, 1] normalisiert.
- `labels` werden in Integer-Datentyp konvertiert und als One-Hot-Floats kodiert (`NUM_CLASSES = 10`).
- `TRAIN_DATASET` und `TEST_DATASET` werden aus `data/processed/training.pt` und `data/processed/test.pt` konstruiert; bei Ladefehlern fallen sie auf leere Listen zurück und warnen.

Exponierte Konstanten: `IMAGE_SIZE`, `TENSOR_SIZE`, `NUM_CLASSES` und Pfad-Helper wie `MODEL_WEIGHTS_DIR`.

In Trainings- und Evaluationsskripten werden die Datasets typischerweise via `DataLoader` verwendet, z. B.:

```python
from torch.utils.data import DataLoader
import data_manager as data
train_loader = DataLoader(data.TRAIN_DATASET, batch_size=32)
```

Anmerkung zu Konfigurationen: Die beiden Trainingspipelines verwenden unterschiedliche Modellklassen und algorithmische Einstellungen, die sich aus empirischer Exploration ergaben. Für reproduzierbare und systematische Experimente sind die Hyperparameter in `src/genetic_algorithm/config.py` und `src/gradient_descent/config.py` deklariert; Forschende sollten diese Werte gezielt variieren, um Sensitivitäts- und Konvergenzeigenschaften zu untersuchen ("spielen Sie mit der Konfiguration", um kontrollierte Studien durchzuführen).

## Lizenz

Dieses Projekt verwendet die MIT-Lizenz (siehe `LICENSE`).