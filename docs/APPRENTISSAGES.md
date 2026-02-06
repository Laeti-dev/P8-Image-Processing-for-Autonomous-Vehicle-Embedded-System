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
## Architecture U-Net pour la Segmentation d'Images

### Contexte

L'architecture U-Net est un réseau de neurones convolutifs (CNN) spécifiquement conçu pour la segmentation d'images, c'est-à-dire la tâche de prédire pour chaque pixel d'une image à quelle classe il appartient. Développée initialement pour la segmentation d'images biomédicales, U-Net a rapidement gagné en popularité dans divers domaines en raison de sa performance robuste, surtout dans les cas où les données d'entraînement sont limitées.

### Origine du nom

Le réseau U-Net tire son nom de sa forme distinctive en "U". Cette architecture comporte deux chemins : un chemin contractant (ou de descente) qui capture le contexte et un chemin symétrique d'expansion (ou de montée) qui permet une localisation précise. La dénomination "U-Net" est donc directement inspirée par cette conception architecturale, où la forme en "U" est centrale pour la fonctionnalité du réseau.

### Structure de l'architecture U-Net

L'architecture U-Net se compose de trois parties principales :

#### 1. Chemin de contraction (encodeur)

Le chemin de contraction se compose de blocs successifs qui réduisent progressivement la résolution spatiale tout en augmentant la profondeur des caractéristiques :

- **Entrée** : L'image commence par être introduite dans le réseau avec sa dimension initiale (par exemple, 572x572 pixels pour 3 canaux RGB).
- **Couches convolutives** : Des couches de convolution (typiquement 3x3) suivies d'une fonction d'activation ReLU sont appliquées pour extraire des caractéristiques de l'image à différents niveaux.
Couches de convolution : C'est comme passer une petite loupe carrée (souvent de 3x3 pixels) sur toute l'image, petit à petit. Cette loupe ne se contente pas de regarder ; elle est programmée pour réagir à des motifs précis : un bord vertical, un coin, ou une couleur spécifique.
-> **Le principe** : On multiplie les pixels sous la loupe par des chiffres (les "filtres") pour faire ressortir une caractéristique.
-> **Le résultat** : On obtient une "carte de caractéristiques" (feature map) qui dit : "Ici, il y a un contour", "Là, il y a une texture", etc.

ReLU signifie Rectified Linear Unit. C'est une règle mathématique ultra-simple : si le résultat de la convolution est négatif, on le transforme en 0. S'il est positif, on le garde tel quel.

-> **En clair** : C'est un interrupteur. On dit au réseau : "Si tu n'as rien trouvé d'intéressant (valeur négative), oublie ce pixel. Si tu as trouvé quelque chose (valeur positive), garde l'info." Cela permet de ne garder que les informations pertinentes et d'ajouter de la "complexité" au modèle.

- **Pooling** : Après chaque série de couches convolutives, une opération de max pooling (2x2) est appliquée, réduisant les dimensions spatiales par un facteur de 2 et augmentant la profondeur des cartes de caractéristiques.
Le Max Pooling sert à réduire la taille de l'image. On prend un petit carré (par exemple 2x2 pixels) et on ne garde que la valeur la plus forte (le maximum) parmi les quatre.

Ce processus crée une structure pyramidale qui réduit progressivement la dimension spatiale des données tout en augmentant la profondeur, permettant de capturer des informations de plus en plus abstraites et globales de l'image.

#### 2. Partie centrale (le fond du "U")

La partie la plus basse de l'U correspond au niveau de contraction le plus profond, avec le plus grand nombre de caractéristiques (par exemple, 1024 caractéristiques) et la résolution spatiale la plus faible. Il y a un bloc de convolution supplémentaire sans max pooling suivant, représentant le point de transition entre la contraction et l'expansion.

#### 3. Chemin d'expansion (décodeur)

Le chemin d'expansion inverse le processus de contraction pour reconstruire une carte de segmentation détaillée :

- **Up-convolution** : L'opération d'up-convolution (ou deconvolution) augmente les dimensions spatiales des caractéristiques, permettant de restaurer progressivement la résolution originale de l'image. C'est l'inverse du Max Pooling : on veut reprendre une petite image pleine d'informations abstraites et l'agrandir pour retrouver la taille originale. Au lieu de perdre des détails, on essaie de "deviner" ou d'apprendre comment reconstruire les pixels manquants pour rendre l'image plus nette et plus grande.
- **Connexions de saut (skip connections)** : Après chaque up-convolution, une opération de "copie et rognage" est effectuée, où les cartes de caractéristiques correspondantes du chemin de contraction sont copiées et fusionnées avec la sortie actuelle. Cette fusion se fait via une opération de concaténation.
C'est le secret de l'U-Net. Lors de la descente (contraction), on perd beaucoup de précision sur l'emplacement exact des objets à cause du Max Pooling. On prend l'image précise du début (l'encodeur) et on vient la "coller" directement sur l'image floue en train d'être reconstruite (le décodeur). Le réseau a le meilleur des deux mondes : il sait ce qu'il y a dans l'image (grâce à la contraction) et il sait exactement où ça se trouve (grâce aux copies directes).
- **Raffinement** : Le chemin d'expansion se poursuit avec des couches de convolution pour raffiner les caractéristiques et reconstruire l'image à sa résolution originale.

#### 4. Sortie

La dernière couche du réseau est une couche convolutive avec un nombre de filtres égal au nombre de classes de segmentation désirées. Elle produit une carte de segmentation où chaque pixel est classifié dans une catégorie spécifique.

### Fonctionnement des connexions de saut

Les connexions de saut (skip connections) sont un élément crucial de l'architecture U-Net. Elles permettent :

- **Fusion du contexte et de la localisation** : Les caractéristiques de contexte à partir du chemin contractant sont fusionnées avec des informations de localisation plus détaillées à partir du chemin d'expansion.
- **Préservation des détails** : Les informations spatiales perdues lors de la contraction sont restaurées grâce à ces connexions.
- **Segmentation précise** : Cette combinaison de contexte et de localisation est cruciale pour effectuer une segmentation précise à l'échelle des pixels.


### La structure en "U" : résumé

- **Chemin de contraction (Gauche du U)** : On réduit la taille pour comprendre le "QUOI" (Est-ce un chat ? Une tumeur ?).
- **Bas du U** : Le résumé maximum de l'image.
- **Chemin d'expansion (Droite du U)** : On agrandit pour retrouver le "OÙ" (Où sont les contours exacts ?).


### Comment le réseau apprend : la rétropropagation

Au début, les filtres ne savent strictement rien faire. Ils sont remplis de nombres au hasard. Si on visualisait ce qu'ils voient, ce serait juste du bruit, comme de la neige sur une vieille télévision.

Pour apprendre, le réseau utilise une méthode de "Correction d'erreurs" (qu'on appelle techniquement la Rétropropagation).

**Voici comment ça se passe en 3 étapes :**

1. **La phase d'essai (Le "Devine !")** : Le réseau reçoit une image (par exemple un chat). Ses filtres, encore aléatoires, extraient n'importe quoi. À la fin, le réseau dit : "Je pense que c'est un grille-pain à 80%."

2. **La sanction (Le "Perdu !")** : On compare sa réponse avec la réalité (la "vérité terrain"). L'erreur est énorme : "Non, c'était un chat !". Le réseau calcule alors mathématiquement l'écart entre sa bêtise et la vérité.

3. **Le réglage (L'ajustement)** : C'est l'étape clé. Le réseau fait "marche arrière" (du bas du U vers le haut). Il regarde chaque filtre et se demande : "Comment devrais-je changer les petits nombres dans ma loupe pour que, la prochaine fois, je détecte mieux les oreilles pointues plutôt que la forme carrée du grille-pain ?"
   - Si un nombre dans le filtre a aidé à voir le chat, on l'augmente.
   - S'il a induit le réseau en erreur, on le diminue.

#### La hiérarchie de l'apprentissage

Ce qui est fascinant, c'est que les filtres s'organisent naturellement par niveaux de complexité au fur et à mesure qu'on descend dans le "U" :

- **Premières couches (Haut du U)** : Les filtres apprennent des choses très simples. Ils deviennent des experts en lignes, en points ou en couleurs.
- **Couches intermédiaires** : En combinant les lignes des filtres précédents, ces nouveaux filtres apprennent des formes plus complexes : des cercles, des textures (poils, métal), des angles.
- **Couches profondes (Bas du U)** : Ici, les filtres reconnaissent des concepts entiers. Un filtre peut s'activer uniquement s'il voit un "œil", un autre pour une "roue".


### Réduction de résolution spatiale vs augmentation des canaux

Concrètement, la réduction de la résolution spatiale signifie que l'image devient plus petite en termes de pixels, mais qu'elle devient aussi plus "intelligente".

Imagine que tu regardes une photo de très près, le nez collé sur l'écran. Tu vois parfaitement chaque pixel, mais tu es incapable de dire si tu regardes un chat ou un paysage. Pour comprendre, tu dois prendre du recul. C'est exactement ce que fait le Max Pooling.

**1. La perte de précision géographique**

Si ton image de départ fait 512x512 pixels, après un premier Max Pooling (2x2), elle ne fait plus que 256x256 pixels.

- **Physiquement** : L'image occupe 4 fois moins de place en mémoire.
- **Spatialement** : Tu as "fusionné" des détails. Si une petite tache de 1 pixel représentait un défaut précis, après le pooling, cette tache est mélangée avec ses voisins. On sait toujours qu'il y a quelque chose dans cette zone, mais on ne sait plus exactement sur quel pixel précis c'était.

**2. L'augmentation du "Champ Récepteur"**

C'est le point le plus important. À chaque fois que la résolution diminue, chaque pixel restant "représente" une plus grande surface de l'image originale.

- **Au début (Haute résolution)** : Un filtre 3x3 voit une minuscule zone (quelques millimètres de l'objet).
- **Après 3 Max Poolings (Basse résolution)** : Un filtre 3x3, parce qu'il travaille sur une image très contractée, "voit" en réalité l'équivalent d'une énorme zone de l'image de départ.

C'est ainsi que le réseau passe d'une vision "locale" (je vois des textures) à une vision "globale" (je vois une structure entière, comme un bras ou une jambe).

**3. Le paradoxe de la profondeur (Les canaux)**

Tu as sans doute lu que la "profondeur des cartes de caractéristiques augmente". C'est là que le terme "contraction" est un peu trompeur : l'image devient plus petite en surface, mais elle devient plus épaisse.

- **Image d'entrée** : 512 x 512 x 3 (Rouge, Vert, Bleu).
- **Plus bas dans le U** : On pourrait avoir du 64 x 64 x 256.

Cela signifie qu'on a sacrifié la résolution spatiale (la précision du "où") pour gagner en richesse d'information (le "quoi"). Au lieu d'avoir 3 couleurs, on a maintenant 256 détecteurs différents qui disent : "Ici il y a un contour de rein", "Ici il y a une zone sombre", "Ici il y a une texture suspecte", etc.

**Pour comprendre comment on passe de 1 canal (gris) à 512 canaux :**

Oublie l'image physique et pense à une équipe d'experts.

- **Un canal = Un point de vue** : Au départ, ton image en niveaux de gris n'a qu'un canal : l'intensité lumineuse. C'est une seule grille de nombres.

Passer à 512 canaux, c'est comme donner la même image à 512 détecteurs différents, où chaque détecteur a une spécialité unique :
  - Le détecteur n°1 cherche les lignes verticales.
  - Le détecteur n°2 cherche les arrondis.
  - Le détecteur n°3 cherche les textures granuleuses.
  - ...
  - Le détecteur n°512 cherche les contrastes de luminosité spécifiques.

- **Comment ça se passe techniquement ?** : C'est la couche de convolution qui crée ces canaux. Quand on définit une couche de convolution, on choisit le nombre de "filtres" (ou "noyaux"). Si tu dis à ton code : `Conv2D(out_channels=512)`, tu es en train de dire : "Prépare 512 filtres différents."

Chaque filtre va balayer l'image d'entrée. Chaque filtre produit sa propre carte de résultats (sa propre grille de nombres). On empile ces 512 grilles les unes sur les autres.

**Résultat** : Ton "image" a maintenant une épaisseur de 512. Ce n'est plus une image que l'on peut "regarder" avec nos yeux humains, c'est un volume de données où chaque étage répond à la question : "Est-ce que ma spécialité (ligne, angle, forme) est présente à cet endroit ?"

- **L'analogie du prisme** : Imagine que tu regardes un objet à travers un prisme qui décompose la lumière. Au début, tu as une lumière blanche (1 canal). Après le prisme, tu as un arc-en-ciel (plusieurs canaux de couleurs). Dans l'U-Net, c'est pareil, sauf que l'IA ne décompose pas seulement les couleurs, elle décompose les concepts visuels. Plus tu descends dans le "U", plus tu augmentes le nombre de canaux (64, 128, 256, 512) parce que tu demandes au réseau de chercher des combinaisons de formes de plus en plus complexes et variées.

**Pourquoi augmenter les canaux alors qu'on réduit la taille ?**

C'est un échange :
- On accepte de perdre en résolution spatiale (l'image devient toute petite, genre 64x64 pixels).
- Mais en échange, on gagne en profondeur sémantique (on a 512 canaux qui décrivent très précisément ce qui se trouve dans ces quelques pixels).

**Concrètement** : À la pointe du "U", le réseau ne voit plus une image, il voit un "vecteur" (une liste de caractéristiques) très riche qui lui dit par exemple : "Il y a 90% de chances qu'il y ait un contour de cellule ici, avec une texture fibreuse et un noyau sombre à proximité."

#### La remontée et la mise en mémoire

Pour faire de la segmentation (dessiner les contours précis d'un organe ou d'un objet), il faut que l'image finale ait la même taille que l'image de départ. C'est là qu'interviennent les Skip Connections (le "copie et rognage"), qui sont la véritable signature de l'U-Net.

**Le problème de la remontée**

Quand on fait l'Up-convolution (on agrandit l'image), c'est un peu comme si on zoomait sur une photo de mauvaise qualité : ça devient pixelisé et les bords sont flous. Le réseau a "oublié" les détails précis du début.

**La solution : Le "Copie et Rognage" (Skip Connections)**

C'est l'idée de génie de l'U-Net : on crée des passerelles horizontales.

- On prend la carte très précise qui sort de l'encodeur (à gauche).
- On la "copie" et on vient la coller directement à côté de la carte floue qui est en train de remonter (à droite).

**Pourquoi "Rognage" ?**

Dans l'article original de l'U-Net, les convolutions faisaient perdre quelques pixels sur les bords (à cause de la taille du filtre 3x3). Du coup, l'image de gauche était un tout petit peu plus grande que celle de droite. Il fallait donc "rogner" (couper les bords) de l'image de gauche pour qu'elle s'ajuste parfaitement à celle de droite avant de les fusionner. Aujourd'hui, on utilise souvent des techniques pour garder la même taille, donc le "rognage" est devenu moins fréquent, mais le nom est resté.

**Le résultat final**

Grâce à ce collage, la partie droite du réseau possède deux types d'informations :
- **L'info "Intelligente"** (qui vient du bas) : "Je sais que c'est une cellule."
- **L'info "Visuelle"** (qui vient du pont de gauche) : "Je sais exactement où se trouvaient les bords au pixel près."

En combinant les deux, le réseau peut reconstruire une carte de segmentation parfaite.

**La toute dernière couche**

Une fois arrivé en haut à droite, le réseau utilise une dernière convolution pour transformer ses nombreux canaux (par exemple 64) en autant de canaux que de classes.

- Si tu veux séparer "Cellule" vs "Fond", tu finis avec 2 canaux.
- Le premier canal s'allume pour les pixels "Cellule".
- Le deuxième canal s'allume pour les pixels "Fond".

C'est ainsi que l'U-Net transforme une photo complexe en un plan de découpe ultra-précis.

#### Le coût mémoire : la stratégie de sauvegarde

Dans un réseau de neurones classique (en ligne droite), on peut se permettre de "jeter" les informations des couches précédentes au fur et à mesure qu'on avance pour libérer de la mémoire. Mais dans l'U-Net, le réseau doit effectivement garder en mémoire (stocker) les résultats de chaque bloc de la partie gauche.

**1. Le garde-meuble de l'encodeur**

À chaque étape du chemin de contraction (la descente), le réseau fait deux choses en parallèle :
- **Il continue son chemin** : Il passe l'image au Max Pooling pour la compresser et descendre plus bas dans le U.
- **Il met de côté** : Il prend une "copie" de la carte de caractéristiques (avant qu'elle ne soit réduite par le pooling) et la stocke dans la mémoire vive de l'ordinateur (la RAM ou la mémoire de la carte graphique).

**2. La livraison au décodeur**

Quand le réseau entame sa remontée (le chemin d'expansion), il arrive en face de "ponts" vides. C'est là qu'il va chercher dans sa mémoire la copie qu'il avait rangée au début :
- L'image qui remonte du bas apporte le "Savoir" (le concept abstrait).
- La copie stockée apporte la "Mémoire visuelle" (les détails nets).

**Pourquoi est-ce coûteux ?**

Cette "mise en mémoire" est la raison pour laquelle l'U-Net est un modèle gourmand. Plus ton image de départ est grande (par exemple une image 4K), plus les copies que tu dois garder en mémoire sont lourdes.

**Analogie** : C'est un peu comme si tu construisais un pont :
- Tu laisses des briques de précision sur la rive gauche.
- Tu traverses la rivière en bateau (le bas du U).
- Une fois sur la rive droite, tu te rends compte que pour construire un château bien droit, tu as besoin des briques précises que tu as laissées derrière toi. Heureusement, tu as tendu des câbles (les Skip Connections) pour les ramener directement.

**Le rôle de la fusion**

Une fois que la copie est "livrée" à droite, on ne se contente pas de la regarder. On fait une concaténation. Si la remontée nous donne 128 canaux flous et que la copie nous donne 128 canaux nets, on les colle ensemble pour obtenir un bloc de 256 canaux. Les couches de convolution suivantes vont ensuite "mélanger" ces deux sources pour créer une image finale qui est à la fois intelligente et ultra-précise.

**En résumé** : Oui, c'est une véritable stratégie de sauvegarde. Sans cette mise en mémoire, l'U-Net serait comme une personne qui a très bien compris un concept global mais qui a oublié tous les détails pratiques pour le dessiner.

### Avantages de l'architecture U-Net

1. **Performance avec peu de données** : U-Net excelle particulièrement dans les cas où les données d'entraînement sont limitées, grâce à sa capacité à apprendre efficacement à partir d'un nombre réduit d'exemples.
2. **Segmentation précise** : La combinaison du chemin de contraction (contexte global) et du chemin d'expansion (détails locaux) permet une segmentation très précise à l'échelle des pixels.
3. **Architecture symétrique** : La symétrie de l'architecture facilite la compréhension et l'implémentation.
4. **Flexibilité** : L'architecture peut être adaptée à différents types de données d'entrée et différents nombres de classes de sortie.

### Implémentation dans le projet

Dans le contexte de ce projet de segmentation sémantique sur le dataset Cityscapes, une architecture U-Net a été implémentée dans `src/model_architecture.py`.

#### Caractéristiques de l'implémentation

1. **Architecture modulaire** : L'implémentation utilise des blocs réutilisables (`conv_block`) pour l'encodeur et le décodeur, facilitant la modification et l'expérimentation. Chaque bloc contient deux couches de convolution suivies de BatchNormalization et d'activation ReLU.

2. **BatchNormalization** : Chaque bloc convolutif utilise la normalisation par batch, ce qui permet :
   - Une stabilisation de l'entraînement
   - Une accélération de la convergence
   - Une réduction de la sensibilité à l'initialisation des poids
   - Une régularisation légère

3. **Connexions de saut** : Les connexions de saut sont implémentées via des opérations de concaténation (`layers.concatenate`) entre les couches correspondantes de l'encodeur et du décodeur. Cette approche permet de fusionner les caractéristiques de différentes résolutions.

4. **Up-sampling avec Conv2DTranspose** : L'up-sampling est réalisé via des couches de transposition de convolution (`Conv2DTranspose`), qui permettent :
   - Un apprentissage des paramètres d'upsampling (contrairement à `UpSampling2D` qui est déterministe)
   - Une meilleure reconstruction des détails spatiaux
   - Un contrôle plus fin sur la résolution de sortie

5. **Gestion des dimensions** : L'utilisation de `padding="same"` dans toutes les couches convolutives garantit que les dimensions spatiales sont préservées (ou restaurées) à chaque niveau, simplifiant la gestion des connexions de saut.

6. **Dropout dans le bottleneck** : Un taux de dropout est appliqué dans la partie la plus profonde du réseau (bottleneck) pour réduire le surapprentissage, particulièrement important dans cette zone où le nombre de paramètres est le plus élevé.

7. **Progression des filtres** : Le nombre de filtres double à chaque niveau de descente (64 → 128 → 256 → 512 → 1024), permettant de capturer des caractéristiques de plus en plus complexes et abstraites.

8. **Sortie multi-classes** : La couche de sortie utilise une convolution 1x1 avec une activation softmax pour produire une carte de segmentation où chaque pixel est classifié dans une catégorie spécifique.

#### Détails techniques de l'implémentation

**Structure des blocs convolutifs** :
- Chaque bloc (`conv_block`) contient deux couches de convolution 3x3
- BatchNormalization après chaque convolution pour stabiliser l'entraînement
- Activation ReLU pour introduire la non-linéarité
- Initialisation des poids avec `he_normal` (adaptée à ReLU)

**Gestion des dimensions spatiales** :
- **Encodeur** : MaxPooling2D (2x2) réduit les dimensions par un facteur de 2 à chaque niveau
- **Décodeur** : Conv2DTranspose (2x2, stride=2) double les dimensions à chaque niveau
- Le padding "same" garantit que les dimensions correspondent exactement pour les connexions de saut

**Architecture à 4 niveaux** :
- 4 niveaux de descente dans l'encodeur (plus le bottleneck)
- 4 niveaux de montée dans le décodeur
- Chaque niveau correspond à une résolution différente (512 → 256 → 128 → 64 → 32 → 64 → 128 → 256 → 512)

**Variante plus légère** :
- Une version `build_unet_small` est également disponible avec seulement 3 niveaux, utile pour les tests rapides ou les ressources limitées

#### Exemple d'utilisation

```python
from src.model_architecture import build_unet

# Construction du modèle U-Net
model = build_unet(
    input_shape=(512, 512, 3),  # Dimensions de l'image d'entrée
    n_classes=8,                # Nombre de classes de segmentation
    filters=64,                 # Nombre de filtres de base (sera doublé à chaque niveau)
    dropout=0.5,                # Taux de dropout dans le bottleneck
    activation='softmax'         # Activation finale pour classification multi-classes
)

# Compilation du modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Pour une version plus légère (tests rapides)
model_small = build_unet_small(
    input_shape=(512, 512, 3),
    n_classes=8,
    filters=32  # Moins de filtres pour réduire la taille du modèle
)
```

#### Considérations pratiques

**Mémoire et performance** :
- Le nombre de paramètres augmente rapidement avec la profondeur et le nombre de filtres
- Pour des images plus grandes ou des ressources limitées, considérer `build_unet_small` ou réduire `filters`
- L'utilisation de BatchNormalization peut légèrement augmenter la consommation mémoire mais améliore la stabilité

**Choix des hyperparamètres** :
- **filters** : Commencer avec 64 pour un bon équilibre performance/mémoire. Augmenter pour plus de capacité, diminuer pour des contraintes mémoire.
- **dropout** : Typiquement entre 0.3 et 0.5 dans le bottleneck. Plus élevé si surapprentissage, plus faible si sous-apprentissage.
- **input_shape** : Les dimensions doivent être des multiples de 16 (ou 2^n selon le nombre de niveaux) pour éviter les problèmes de dimensions lors de l'upsampling.

**Alternatives d'up-sampling** :
- **Conv2DTranspose** (utilisé ici) : Apprend les paramètres, plus flexible mais plus de paramètres
- **UpSampling2D + Conv2D** : Déterministe, moins de paramètres mais peut nécessiter plus de couches pour de bons résultats

### Points clés à retenir

1. **Forme en U** : L'architecture tire son nom de sa forme distinctive, avec un chemin de contraction (gauche) et un chemin d'expansion (droite).
2. **Encodeur-Décodeur** : L'encodeur réduit les dimensions spatiales et extrait des informations de haut niveau, tandis que le décodeur reconstruit une carte de segmentation détaillée.
3. **Connexions de saut** : Essentielles pour combiner le contexte global (encodeur) avec les détails locaux (décodeur).
4. **Efficacité avec peu de données** : Particulièrement adapté aux situations où les données d'entraînement sont limitées.
5. **Segmentation pixel-par-pixel** : Chaque pixel de l'image d'entrée est classifié dans une catégorie spécifique.

### Références

- [Architecture U-Net: Une explication détaillée](https://datasciencetoday.net/index.php/en-us/deep-learning/228-unet) - DataScienceToday
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) - Ronneberger et al., 2015

---

## Utilisation d'un Backbone et Fine-Tuning

### Contexte

Dans l'architecture U-Net classique, l'encodeur (la partie gauche du U) est construit à partir de zéro : on empile des blocs convolutifs (`conv_block`) suivis de Max Pooling, et tous les poids sont initialisés aléatoirement. Le réseau doit donc **tout apprendre** à partir de nos seules données de segmentation.

L'idée du **backbone** (ou "colonne vertébrale") est de remplacer cet encodeur artisanal par un réseau de classification déjà entraîné sur des millions d'images (typiquement ImageNet, qui contient 1,2 million d'images réparties en 1000 catégories). On récupère ainsi des filtres qui savent déjà reconnaître des formes, des textures, des contours, des objets... et on les réutilise comme point de départ. C'est ce qu'on appelle le **transfert learning** (apprentissage par transfert).

### Qu'est-ce qu'un backbone ?

Un backbone est un réseau de neurones pré-entraîné dont on retire la couche de classification finale (la "tête") pour ne garder que les couches d'extraction de features. Parmi les backbones les plus utilisés :

- **MobileNetV2** : Léger et rapide, conçu pour les appareils mobiles. Utilise des blocs "inverted residual bottleneck" qui sont plus efficaces que les convolutions classiques.
- **ResNet50** : Plus profond et plus puissant, avec des connexions résiduelles qui permettent d'entraîner des réseaux très profonds sans dégradation.
- **VGG16** : Un des premiers backbones populaires, simple mais gourmand en mémoire.

### Comment le backbone remplace l'encodeur

#### U-Net classique (encodeur fait maison)

Dans le U-Net simple, l'encodeur est construit manuellement avec des `conv_block` + `MaxPooling` :

```
Entrée (512x512x3)
  → conv_block (64 filtres)  → MaxPooling → 256x256
  → conv_block (128 filtres) → MaxPooling → 128x128
  → conv_block (256 filtres) → MaxPooling → 64x64
  → conv_block (512 filtres) → MaxPooling → 32x32
  → conv_block (1024 filtres) → Bottleneck
```

Chaque `conv_block` contient 2 convolutions + BatchNorm + ReLU. Les skip connections sont prises directement à la sortie de chaque `conv_block`, avant le Max Pooling. L'architecture est **parfaitement symétrique** : 4 niveaux de descente, 4 niveaux de remontée, et la sortie a exactement la même résolution que l'entrée.

#### U-Net avec backbone (encodeur pré-entraîné)

Quand on utilise MobileNetV2 comme backbone, on remplace **tout l'encodeur** par ce réseau :

```
Entrée (512x512x3)
  → [MobileNetV2 pré-entraîné sur ImageNet]
      ├─ block_1_expand_relu  → 256x256 (skip connection 1)
      ├─ block_3_expand_relu  → 128x128 (skip connection 2)
      ├─ block_6_expand_relu  →  64x64  (skip connection 3)
      ├─ block_13_expand_relu →  32x32  (skip connection 4)
      └─ sortie finale        →  16x16  (bottleneck)
```

Les skip connections ne viennent plus de nos propres `conv_block`, mais des couches intermédiaires de MobileNetV2. On va "piocher" dans le réseau pré-entraîné les sorties à différentes résolutions.

**Différence clé** : MobileNetV2 est beaucoup plus profond (~155 couches) que notre encodeur artisanal (~8 couches convolutives). Ses features sont aussi de nature différente : elles ont été apprises sur la classification d'objets du quotidien, pas sur la segmentation de scènes urbaines. C'est pour cela que le fine-tuning est important (voir plus bas).

### Le décodeur reste custom

Le décodeur (partie droite du U) est toujours construit manuellement, exactement comme dans le U-Net classique : des `Conv2DTranspose` pour agrandir la résolution, des concaténations avec les skip connections, et des `conv_block` pour raffiner. C'est le décodeur qui apprend à transformer les features du backbone en masques de segmentation.

### Pièges courants et erreurs à éviter

Lors de l'implémentation du U-Net avec backbone dans ce projet, plusieurs erreurs ont été identifiées et corrigées. Ces pièges sont fréquents et méritent d'être documentés.

#### Piège 1 : Le prétraitement des données (normalisation)

C'est probablement l'erreur la plus insidieuse car elle ne provoque pas de plantage, mais dégrade silencieusement les performances.

**Le problème** : Chaque backbone pré-entraîné attend ses données d'entrée dans une plage de valeurs spécifique. MobileNetV2, entraîné sur ImageNet, attend des pixels dans la plage **[-1, 1]**. Or, notre `CityscapesDataGenerator` normalise les images en **[0, 1]** (via `image / 255.0`).

Si on envoie des valeurs en [0, 1] à un réseau qui attend du [-1, 1], les features extraites seront dégradées. Le backbone ne "reconnaît" pas ce qu'il voit car les valeurs numériques sont dans la mauvaise plage. C'est comme si on parlait en chuchotant à quelqu'un qui attend qu'on crie : l'information passe, mais très mal.

**La solution** : Ajouter une couche de prétraitement directement dans le modèle qui convertit [0, 1] en [-1, 1] :

```python
# Conversion de [0, 1] vers [-1, 1] : x * 2.0 - 1.0
preprocessed = layers.Lambda(
    lambda x: x * 2.0 - 1.0,
    name='mobilenet_preprocess'
)(inputs)
```

L'avantage d'intégrer cette conversion dans le modèle lui-même (plutôt que dans le pipeline de données) est qu'on n'a rien à changer dans le `DataGenerator`. Le modèle gère son propre prétraitement de manière transparente.

#### Piège 2 : La résolution de sortie

**Le problème** : MobileNetV2 effectue **5 niveaux de sous-échantillonnage** (la résolution est divisée par 2 à chaque fois), arrivant à 1/32 de la résolution d'entrée. Avec une entrée de 512x512, le bottleneck fait 16x16.

Si le décodeur ne fait que **4 étapes d'upsampling** (chacune ×2), la sortie ne remonte qu'à 1/2 de la résolution d'entrée (256x256 au lieu de 512x512). Les prédictions sont alors à la **moitié de la taille** des masques de vérité terrain, ce qui fausse complètement la loss et les métriques.

```
Problème :
  16x16 (bottleneck)
  → ×2 → 32x32
  → ×2 → 64x64
  → ×2 → 128x128
  → ×2 → 256x256  ← Sortie à 1/2 résolution ! ✗
```

**Comparaison avec le U-Net simple** : Dans le U-Net classique, l'encodeur ne fait que **4 niveaux** de descente (1/16), et le décodeur fait 4 étapes de remontée. L'architecture est parfaitement symétrique et la sortie est à la bonne résolution.

**La solution** : Ajouter une **5ème étape d'upsampling** dans le décodeur pour retrouver la résolution complète :

```
Correction :
  16x16 (bottleneck)
  → ×2 → 32x32   + skip connection (block_13)
  → ×2 → 64x64   + skip connection (block_6)
  → ×2 → 128x128  + skip connection (block_3)
  → ×2 → 256x256  + skip connection (block_1)
  → ×2 → 512x512  (pas de skip connection à ce niveau) ✓
```

Cette dernière étape n'a pas de skip connection car il n'y a pas de couche intermédiaire de MobileNetV2 à la résolution complète. Le décodeur apprend seul à reconstruire les derniers détails.

#### Piège 3 : Ne pas geler le backbone (absence de stratégie de transfert learning)

**Le problème** : Si on entraîne directement le modèle complet (backbone + décodeur) avec un learning rate standard (par ex. 1e-4), les poids pré-entraînés du backbone sont rapidement écrasés. Les gradients du décodeur (qui apprend de zéro) sont très grands et se propagent dans le backbone, détruisant les features déjà apprises sur ImageNet.

Résultat : on perd tout le bénéfice du transfert learning, et on se retrouve dans la même situation qu'un entraînement de zéro, mais avec une architecture moins bien adaptée que le U-Net simple.

**La solution** : Utiliser une stratégie en deux phases (voir section suivante).

### Le transfert learning en deux phases

#### Phase 1 : Backbone gelé (entraîner le décodeur seul)

Dans cette première phase, on **verrouille** les poids du backbone MobileNetV2 :

```python
encoder.trainable = False
```

Concrètement, cela signifie que lors de la rétropropagation, les gradients ne modifient **que** les poids du décodeur. Le backbone reste exactement dans l'état où ImageNet l'a laissé.

**Ce qui se passe** : Le décodeur apprend à transformer les features génériques d'ImageNet (lignes, textures, formes d'objets) en masques de segmentation pour nos 8 classes Cityscapes. C'est rapide car il y a beaucoup moins de paramètres à mettre à jour.


**À la fin de la Phase 1**, le modèle fonctionne correctement mais de manière sous-optimale : le backbone extrait des features pensées pour la classification ImageNet, pas pour la segmentation de scènes de conduite.

#### Phase 2 : Fine-tuning (tout dégeler avec un learning rate bas)

Dans cette deuxième phase, on **déverrouille** tout le réseau :

```python
for layer in model.layers:
    layer.trainable = True
```

La différence cruciale : on **recompile** le modèle avec un learning rate **beaucoup plus bas** (typiquement 10× plus petit) :

```python
# Phase 1 : LR = 1e-4 (standard)
# Phase 2 : LR = 1e-5 (10× plus bas)
model.compile(optimizer=Adam(learning_rate=1e-5), ...)
```

**Pourquoi un learning rate si bas ?** Parce que les poids du backbone sont déjà de **bons poids**. On veut les **ajuster finement**, pas les écraser. Un learning rate trop élevé détruirait les features pré-apprises, ce qui reviendrait à entraîner de zéro.

**Ce qui se passe** : Le backbone s'adapte progressivement. Ses features passent de "reconnaître des objets ImageNet" à "reconnaître des éléments de scènes urbaines" (routes, piétons, véhicules...). Les ajustements sont subtils : les filtres qui détectaient des "oreilles de chat" apprennent maintenant à mieux détecter des "rétroviseurs" ou des "bords de trottoir".

**Analogie** : On dit au peintre expert : "Maintenant tu peux adapter ta technique." Il ajuste doucement son style pour mieux coller aux scènes urbaines, sans tout oublier de ce qu'il savait faire.

#### Le modèle final

Le modèle final est celui produit par la **Phase 2**. La variable `model` dans le notebook est le **même objet Python** tout au long du processus. Il n'y a pas de "nouveau" modèle créé en Phase 2 : on continue simplement l'entraînement du même modèle, mais cette fois avec tous les poids débloqués et un learning rate plus doux.

Le `ModelCheckpoint` dans les callbacks sauvegarde automatiquement le meilleur modèle (celui avec le meilleur `val_iou_coefficient`), quelle que soit la phase.

### Récapitulatif du workflow complet

```
1. Construire le modèle
   └─ build_unet_mobilenet(weights="imagenet", freeze_backbone=True)
       ├─ Couche de preprocessing : [0,1] → [-1,1]
       ├─ Encodeur : MobileNetV2 (poids ImageNet, gelé)
       └─ Décodeur : Custom (poids aléatoires, entraînable)

2. Phase 1 : Entraîner le décodeur seul
   └─ model.fit(...) avec LR = 1e-4
       → Le décodeur apprend à segmenter avec les features ImageNet

3. Phase 2 : Fine-tuning complet
   ├─ Dégeler toutes les couches
   ├─ Recompiler avec LR = 1e-5
   └─ model.fit(...) à nouveau
       → Le backbone s'adapte aux scènes urbaines

4. Résultat final
   └─ Un modèle dont le backbone a été affiné pour la segmentation
      de scènes de conduite, avec de meilleures performances qu'un
      U-Net entraîné de zéro
```

### Implémentation dans le projet

L'implémentation du U-Net avec backbone MobileNetV2 se trouve dans `src/model_architecture.py` (fonction `build_unet_mobilenet`). Le notebook `notebooks/02-training_pipeline.ipynb` contient le pipeline d'entraînement complet avec les deux phases.

#### Paramètres de configuration clés

Dans le notebook, les hyperparamètres suivants contrôlent le transfert learning :

```python
USE_MOBILENET_BACKBONE = True       # Activer le backbone MobileNetV2
MOBILENET_WEIGHTS = "imagenet"      # Utiliser les poids pré-entraînés ImageNet
FREEZE_BACKBONE = True              # Geler l'encodeur en Phase 1
LEARNING_RATE = 1e-4                # LR standard pour la Phase 1
FINE_TUNING_LR = 1e-5               # LR réduit pour la Phase 2
```

### Points clés à retenir

1. **Un backbone n'est pas magique** : Il faut le prétraitement correct (normalisation), la bonne résolution de sortie, et une stratégie de gel/dégel pour en tirer profit.
2. **Le preprocessing est critique** : Chaque backbone attend ses données dans une plage spécifique. Une mauvaise normalisation dégrade silencieusement les performances sans provoquer d'erreur.
3. **La résolution de sortie doit correspondre** : Le nombre d'étapes d'upsampling dans le décodeur doit compenser exactement le nombre de niveaux de sous-échantillonnage du backbone.
4. **Le transfert learning se fait en deux phases** : D'abord entraîner le décodeur seul (backbone gelé), puis fine-tuner l'ensemble avec un learning rate réduit.
5. **Le learning rate de fine-tuning est crucial** : Trop élevé, il détruit les features pré-apprises. Trop bas, le modèle n'apprend rien de nouveau. Un facteur 10× plus petit que le LR initial est un bon point de départ.
6. **Le modèle final est celui de la Phase 2** : Les deux phases opèrent sur le même objet modèle ; la Phase 2 affine ce que la Phase 1 a commencé.

### Références

- [Transfer Learning and Fine-Tuning](https://www.tensorflow.org/tutorials/images/transfer_learning) - TensorFlow Official Tutorial
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) - Sandler et al., 2018
- [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526) - Pan & Yang, 2010

---

## Architecture DeepLabV3 pour la Segmentation Sémantique

### Contexte

DeepLabV3 est une architecture de segmentation sémantique développée par Google en 2017. Contrairement à U-Net qui utilise une structure encodeur-décodeur symétrique avec des skip connections, DeepLabV3 adopte une approche fondamentalement différente : il se concentre sur la capture d'informations **multi-échelles** grâce à un module appelé **ASPP** (Atrous Spatial Pyramid Pooling).

DeepLabV3 fait partie de la famille "DeepLab" (V1, V2, V3, V3+), chaque version apportant des améliorations. La version 3 est celle implémentée dans ce projet.

### Le problème que DeepLabV3 résout

En segmentation sémantique, un défi majeur est de capturer simultanément :
- Le **contexte global** : "Je suis dans une scène de rue" (nécessite de voir une large zone de l'image)
- Les **détails locaux** : "Ce pixel précis est un piéton" (nécessite de voir une zone très petite)

U-Net résout ce problème avec ses skip connections (le décodeur récupère les détails de l'encodeur). DeepLabV3 le résout autrement : avec les **convolutions atrous** (aussi appelées "convolutions dilatées").

### La convolution atrous : l'idée clé

#### Convolution classique vs convolution atrous

Dans une convolution classique 3x3, le filtre regarde 9 pixels **adjacents** :

```
Convolution classique (3x3) :

  [x] [x] [x]
  [x] [x] [x]
  [x] [x] [x]

  → Le filtre voit une zone de 3x3 pixels
```

Dans une convolution atrous, on **espace** les pixels que le filtre regarde. Le paramètre `dilation_rate` (ou taux de dilatation) contrôle cet espacement :

```
Convolution atrous (3x3, rate=2) :

  [x]  .  [x]  .  [x]
   .   .   .   .   .
  [x]  .  [x]  .  [x]
   .   .   .   .   .
  [x]  .  [x]  .  [x]

  → Le filtre voit toujours 9 pixels, mais couvre une zone de 5x5
```

```
Convolution atrous (3x3, rate=4) :

  [x]  .   .   .  [x]  .   .   .  [x]
   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .
  [x]  .   .   .  [x]  .   .   .  [x]
   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .
  [x]  .   .   .  [x]  .   .   .  [x]

  → Le filtre voit toujours 9 pixels, mais couvre une zone de 9x9
```

**L'avantage** : On agrandit le **champ récepteur** (la zone vue par le filtre) **sans augmenter le nombre de paramètres**. Un filtre 3x3 avec rate=6 voit une zone aussi large qu'un filtre 13x13, mais avec seulement 9 poids au lieu de 169.

**Analogie** : Imagine que tu regardes un paysage. Une convolution classique, c'est regarder à travers un petit trou dans un mur : tu vois très précisément mais très peu. Une convolution atrous, c'est regarder à travers une grille : tu vois toujours le même nombre de "points", mais ils sont espacés, ce qui te donne une vue d'ensemble bien plus large.

### Le module ASPP (Atrous Spatial Pyramid Pooling)

Le module ASPP est le coeur de DeepLabV3. L'idée est simple mais puissante : au lieu d'utiliser une seule convolution atrous, on en utilise **plusieurs en parallèle** avec des taux de dilatation différents, puis on combine les résultats.

#### Structure du module ASPP

Le module ASPP prend les features du backbone et les traite avec **5 branches parallèles** :

```
Features du backbone
        │
        ├──→ [Global Average Pooling]    → Vue "ultra-globale" (1 valeur par canal)
        │         → Conv 1x1 → BN → ReLU → Resize
        │
        ├──→ [Conv 1x1]                  → Vue "pixel par pixel"
        │         → BN → ReLU
        │
        ├──→ [Conv 3x3, rate=6]          → Vue "moyenne" (champ récepteur ~13x13)
        │         → BN → ReLU
        │
        ├──→ [Conv 3x3, rate=12]         → Vue "large" (champ récepteur ~25x25)
        │         → BN → ReLU
        │
        └──→ [Conv 3x3, rate=18]         → Vue "très large" (champ récepteur ~37x37)
                  → BN → ReLU
        │
        ▼
  [Concaténation des 5 branches]
        │
        ▼
  [Conv 1x1 + BN + ReLU + Dropout]      → Fusion et projection finale
```

Chaque branche "regarde" la même feature map mais à une **échelle différente** :
- La branche **1x1** voit chaque pixel individuellement (détails très fins)
- La branche **rate=6** voit un contexte moyen (un objet entier)
- La branche **rate=12** voit un contexte large (un objet et ses voisins)
- La branche **rate=18** voit un contexte très large (une scène entière)
- Le **Global Average Pooling** résume toute l'image en une seule valeur par canal (contexte ultra-global)

**Analogie** : Imagine 5 photographes qui regardent la même scène de rue :
- Le premier utilise un objectif macro (il voit les détails d'un panneau)
- Le deuxième utilise un objectif standard (il voit un piéton entier)
- Le troisième utilise un grand angle (il voit le piéton et la voiture à côté)
- Le quatrième utilise un ultra grand angle (il voit toute la rue)
- Le cinquième fait une photo satellite (il voit le quartier entier)

En combinant les 5 photos, on a une compréhension complète de la scène à toutes les échelles.

#### Pourquoi c'est efficace

Le ASPP résout élégamment le problème des objets de **tailles variées** dans une même image. Dans une scène de conduite Cityscapes :
- Un **feu de signalisation** occupe quelques pixels → la branche rate=6 suffit
- Une **voiture** occupe une zone moyenne → la branche rate=12 est idéale
- La **route** occupe une grande partie de l'image → les branches rate=18 et le pooling global la capturent bien

Sans ASPP, un réseau avec un seul champ récepteur aurait du mal à segmenter correctement des objets de tailles très différentes dans la même image.

### Architecture complète de DeepLabV3

```
Image d'entrée (512x512x3)
        │
        ▼
┌─────────────────────┐
│   Backbone           │
│   (MobileNetV2 ou   │  ← Pré-entraîné sur ImageNet
│    ResNet50)         │
│                     │
│   Sortie : 16x16    │  ← 1/32 de la résolution d'entrée
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Conv atrous         │
│  (rate=2)            │  ← Transition avant ASPP
│  + BN + ReLU         │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Module ASPP         │
│  (5 branches         │  ← Capture multi-échelle
│   parallèles)        │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Resize bilinéaire   │  ← Remontée directe à 512x512
│  (16x16 → 512x512)  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Conv 1x1 + softmax │  ← Prédiction par pixel (8 classes)
└─────────────────────┘
        │
        ▼
  Masque de segmentation (512x512x8)
```

### Différences fondamentales avec U-Net

| Aspect | U-Net | DeepLabV3 |
|--------|-------|-----------|
| **Encodeur** | Custom ou backbone | Toujours un backbone pré-entraîné |
| **Décodeur** | Symétrique (Conv2DTranspose + skip connections) | Très simple (juste un resize bilinéaire) |
| **Skip connections** | Oui, à chaque niveau | Non (pas dans la version de base) |
| **Multi-échelle** | Implicite (via les niveaux de l'encodeur) | Explicite (via le module ASPP) |
| **Remontée en résolution** | Progressive (×2 à chaque étape) | Directe (un seul resize de 1/32 à 1/1) |
| **Nombre de paramètres** | Élevé (encodeur + décodeur complet) | Plus léger (pas de décodeur lourd) |

#### Décodeur simple vs décodeur complexe

La différence la plus frappante est le **décodeur**. Dans U-Net, le décodeur est aussi complexe que l'encodeur : il remonte progressivement la résolution avec des `Conv2DTranspose`, en fusionnant à chaque étape les skip connections. C'est un processus appris en plusieurs étapes.

Dans DeepLabV3, le "décodeur" est brutalement simple : un unique **resize bilinéaire** qui passe directement de 16x16 à 512x512. Il n'y a pas de couches apprenables dans cette remontée. L'hypothèse de DeepLabV3 est que le module ASPP a déjà capturé suffisamment d'informations multi-échelles pour qu'un simple agrandissement suffise.

**Analogie** : U-Net reconstruit un puzzle pièce par pièce en vérifiant à chaque étape (les skip connections). DeepLabV3 préfère d'abord très bien comprendre le puzzle à petite échelle (via ASPP), puis agrandir directement l'image avec un photocopieur (resize bilinéaire).

C'est pour cette raison que la version DeepLabV3+ (avec un "+") a été créée par la suite : elle ajoute un décodeur plus sophistiqué avec des skip connections, combinant le meilleur des deux mondes.

### L'output stride : un concept important

L'**output stride** est le rapport entre la résolution de l'image d'entrée et la résolution de la feature map du backbone. Dans notre implémentation :

- MobileNetV2 produit des features à 1/32 de la résolution → output stride = 32
- Avec une entrée 512x512, le backbone sort des features de 16x16

Un output stride plus petit (par exemple 16 ou 8) donne des features de meilleure résolution mais coûte plus cher en calcul. Dans le code, une convolution atrous avec `dilation_rate=2` est appliquée après le backbone pour simuler un output stride plus fin sans changer la résolution réelle :

```python
# Simule un output_stride=16 en appliquant une convolution atrous
x = layers.Conv2D(
    aspp_filters, 3,
    padding='same',
    dilation_rate=2,
    kernel_initializer='he_normal'
)(encoder_output)
```

### Implémentation dans le projet

L'implémentation de DeepLabV3 se trouve dans `src/model_architecture.py` et comprend :

1. **`aspp_block()`** : Le module ASPP avec ses 5 branches parallèles (global pooling, conv 1x1, et 3 convolutions atrous avec rates 6, 12, 18).

2. **`build_deeplabv3()`** : La fonction principale qui assemble backbone + convolution atrous + ASPP + resize + couche de sortie. Supporte MobileNetV2 ou ResNet50 comme backbone.

3. **`build_deeplabv3_mobilenet()`** : Wrapper de commodité qui appelle `build_deeplabv3` avec MobileNetV2.

#### Exemple d'utilisation

```python
from src.model_architecture import build_deeplabv3_mobilenet

# Construction du modèle DeepLabV3 avec MobileNetV2
model = build_deeplabv3_mobilenet(
    input_shape=(256, 512, 3),
    n_classes=8,
    alpha=1.0,
    weights="imagenet",
    aspp_filters=256,
    activation="softmax"
)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=['accuracy', DiceCoefficient(), IoUCoefficient()]
)
```

### Quand choisir DeepLabV3 plutôt que U-Net ?

- **DeepLabV3** est particulièrement adapté quand la scène contient des objets de **tailles très variées** (comme les scènes de conduite : petits panneaux, piétons moyens, grandes routes). Le module ASPP est conçu pour ce cas.

- **U-Net** est souvent meilleur quand on a besoin de **frontières très précises** entre les objets (comme en imagerie médicale). Les skip connections préservent les détails spatiaux que le resize bilinéaire de DeepLabV3 peut lisser.

- **DeepLabV3+** (non implémenté ici) combine les avantages des deux : le ASPP pour le multi-échelle et un décodeur avec skip connections pour la précision des contours.

### Points clés à retenir

1. **La convolution atrous** agrandit le champ récepteur sans augmenter le nombre de paramètres, en espaçant les pixels regardés par le filtre.
2. **Le module ASPP** applique plusieurs convolutions atrous en parallèle avec des taux de dilatation différents, permettant de capturer des informations à différentes échelles simultanément.
3. **DeepLabV3 a un décodeur très simple** (un unique resize bilinéaire), contrairement à U-Net qui reconstruit progressivement la résolution.
4. **L'architecture mise tout sur le ASPP** : l'hypothèse est qu'une excellente compréhension multi-échelle des features compense l'absence d'un décodeur sophistiqué.
5. **Le choix entre U-Net et DeepLabV3** dépend du problème : U-Net pour la précision des contours, DeepLabV3 pour la diversité des tailles d'objets.

### Références

- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) - Chen et al., 2017 (article original DeepLabV3)
- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) - Chen et al., 2018 (DeepLabV3+)
- [TensorFlow Model Garden - Semantic Segmentation](https://www.tensorflow.org/tfmodels/vision/semantic_segmentation) - Tutorial officiel TensorFlow

---
