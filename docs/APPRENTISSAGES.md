# Apprentissages du Projet

Ce document recense les apprentissages et connaissances acquises au cours du développement de ce projet de traitement d'images pour systèmes embarqués de véhicules autonomes.

---

## Gestion des Datasets Volumineux avec Keras DataGenerator

### Contexte

Lorsqu'on travaille avec de grands ensembles de données, il est souvent impossible de charger toutes les données en mémoire simultanément. C'est particulièrement vrai pour les projets de vision par ordinateur où les images peuvent être nombreuses et volumineuses. Keras propose des solutions élégantes pour gérer ce problème via l'utilisation de générateurs de données.

### Problématique

La méthode classique de chargement des données consiste à charger l'ensemble du dataset en mémoire avant l'entraînement :

```python
import numpy as np
from keras.models import Sequential

# Chargement de tout le dataset en mémoire
X, y = np.load('some_training_set_with_labels.npy')

# Entraînement
model.fit(x=X, y=y)
```

Cette approche peut causer des problèmes de mémoire lorsque le dataset est trop volumineux pour tenir en RAM.

### Solution : DataGenerator personnalisé avec `keras.utils.Sequence`

#### Architecture recommandée

Pour une meilleure organisation, il est recommandé de structurer le projet ainsi :

```
project/
├── data_generator.py    # Classe DataGenerator personnalisée
├── train_model.py       # Script d'entraînement Keras
└── data/                # Répertoire contenant les données
```

#### Organisation des données

Avant de créer le générateur, il est important d'organiser les données avec deux structures :

1. **`partition`** : Un dictionnaire contenant les listes d'IDs pour chaque split
   ```python
   partition = {
       'train': ['id-1', 'id-2', 'id-3'],
       'validation': ['id-4']
   }
   ```

2. **`labels`** : Un dictionnaire associant chaque ID à son label
   ```python
   labels = {
       'id-1': 0,
       'id-2': 1,
       'id-3': 2,
       'id-4': 1
   }
   ```

#### Implémentation du DataGenerator

La classe `DataGenerator` doit hériter de `keras.utils.Sequence` pour bénéficier de fonctionnalités avancées comme le multiprocessing :

**Initialisation** :
- Stocke les paramètres essentiels (dimensions, batch_size, nombre de classes, etc.)
- Conserve les listes d'IDs et les labels
- Appelle `on_epoch_end()` pour initialiser les index

**Méthodes principales** :

1. **`__len__()`** : Retourne le nombre de batches par époque
   ```python
   return int(np.floor(len(self.list_IDs) / self.batch_size))
   ```

2. **`__getitem__(index)`** : Génère un batch de données pour un index donné
   - Sélectionne les IDs correspondant au batch
   - Appelle `__data_generation()` pour charger les données

3. **`on_epoch_end()`** : Appelée à la fin de chaque époque
   - Réinitialise les index
   - Mélange les index si `shuffle=True` pour éviter que les batches soient identiques entre époques

4. **`__data_generation(list_IDs_temp)`** : Méthode privée qui charge réellement les données
   - Charge chaque échantillon depuis son fichier (ex: `np.load('data/' + ID + '.npy')`)
   - Convertit les labels en format catégoriel avec `keras.utils.to_categorical()`

#### Utilisation dans le script d'entraînement

Une fois le DataGenerator créé, il suffit de l'utiliser avec `fit_generator()` au lieu de `fit()` :

```python
from my_classes import DataGenerator

# Paramètres
params = {
    'dim': (32, 32, 32),
    'batch_size': 64,
    'n_classes': 6,
    'n_channels': 1,
    'shuffle': True
}

# Création des générateurs
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Entraînement
model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=6
)
```

#### Avantages du multiprocessing

L'utilisation de `use_multiprocessing=True` avec plusieurs `workers` permet de :
- Générer les batches en parallèle sur plusieurs cœurs CPU
- Assurer que le goulot d'étranglement soit les opérations GPU (forward/backward) et non la génération de données
- Optimiser l'utilisation des ressources système

### Points clés à retenir

1. **Chargement à la volée** : Les données sont chargées batch par batch, évitant la saturation de la mémoire
2. **Multiprocessing** : La génération de données peut être parallélisée pour améliorer les performances
3. **Shuffling** : Le mélange des données entre époques améliore la robustesse du modèle
4. **Flexibilité** : Cette approche permet d'effectuer des opérations complexes (préprocessing, augmentation) sans ralentir l'entraînement
5. **Modularité** : Séparer le générateur du script d'entraînement améliore la maintenabilité du code

### Implémentation concrète pour le projet Cityscapes

Dans le contexte de ce projet de segmentation sémantique sur le dataset Cityscapes, un `CityscapesDataGenerator` a été implémenté dans `src/data_generator.py`.

#### Caractéristiques spécifiques

1. **Organisation des IDs** : Au lieu d'utiliser des IDs simples, le générateur utilise des tuples `(city, sequence, frame)` qui correspondent à la structure du dataset Cityscapes.

2. **Chargement des données** :
   - Images RGB chargées depuis `data/raw/leftImg8bit/`
   - Masques de segmentation chargés depuis `data/raw/gtFine/`
   - Conversion automatique des 34 classes Cityscapes vers 8 catégories

3. **Préprocessing intégré** :
   - Redimensionnement des images et masques (par défaut 512x512)
   - Normalisation des images (optionnelle, par défaut activée)
   - Support de l'augmentation de données via Albumentations

4. **Fonction helper** : La fonction `create_data_generators()` simplifie la création des générateurs d'entraînement et de validation.

#### Exemple d'utilisation

```python
from src.data_generator import create_data_generators
from tensorflow import keras

# Création des générateurs
train_gen, val_gen = create_data_generators(
    train_split="train",
    val_split="val",
    batch_size=16,
    dim=(512, 512),
    normalize=True
)

# Avec augmentation de données (optionnel)
import albumentations as A
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

train_gen_aug, val_gen = create_data_generators(
    train_split="train",
    val_split="val",
    batch_size=16,
    dim=(512, 512),
    augmentation=train_aug,
    normalize=True
)

# Utilisation avec model.fit()
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    use_multiprocessing=True,
    workers=4
)
```

#### Avantages de cette implémentation

- **Adapté au dataset** : Gestion native de la structure Cityscapes
- **Efficace en mémoire** : Chargement batch par batch
- **Flexible** : Support du redimensionnement et de l'augmentation
- **Prêt pour le multiprocessing** : Compatible avec `use_multiprocessing=True`

### Références

- [A detailed example of how to use data generators with Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) - Stanford University
- [Keras Training on Large Datasets](https://medium.datadriveninvestor.com/keras-training-on-large-datasets-3e9d9dbc09d4) - Medium DataDrivenInvestor

---
