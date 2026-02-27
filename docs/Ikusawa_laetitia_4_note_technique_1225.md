## Ikusawa_laetitia_4_note_technique_1225

### 1. Introduction et contexte

Les véhicules autonomes reposent fortement sur la perception de l’environnement pour prendre des décisions sûres en temps réel. Parmi les différents modules de perception, la **segmentation sémantique d’images** joue un rôle central : elle consiste à attribuer une étiquette de classe à **chaque pixel** d’une image, afin de distinguer précisément la route, les bâtiments, les piétons, les véhicules, le ciel, etc. Cette compréhension fine de la scène est indispensable pour des tâches telles que le suivi de voie, la détection d’obstacles ou la planification de trajectoire.

Dans un contexte embarqué, les contraintes sont particulièrement fortes :

- **Temps réel** : latence d’inférence limitée pour chaque image.
- **Ressources matérielles restreintes** : GPU/CPU embarqué avec mémoire limitée.
- **Robustesse** : adaptation à des conditions variées (villes, météo, lumière, points de vue).
- **Sécurité fonctionnelle** : erreurs de segmentation pouvant impacter la prise de décision.

Le projet présenté ici vise à développer un **système complet de segmentation sémantique** pour véhicules autonomes, à partir du jeu de données **Cityscapes**, en agrégeant les 34 classes originales en **8 catégories** principales (Void, Flat, Construction, Object, Nature, Sky, Human, Vehicle). L’architecture globale combine :

- un **pipeline de données** permettant le chargement des images Cityscapes, la conversion des masques 34→8 classes et la préparation des lots pour l’entraînement ;
- plusieurs **architectures de deep learning** pour la segmentation (U-Net, U-Net avec backbone MobileNetV2, DeepLabV3 avec MobileNetV2) ;
- un **suivi d’expérimentations** via MLflow sur DagsHub pour comparer les modèles et leurs hyperparamètres ;
- une **API FastAPI** de prédiction (`/predict`) capable de charger le modèle depuis Azure Blob Storage et de servir des prédictions en temps quasi réel ;
- une **application Streamlit** de démonstration pour visualiser les masques prédits et les superposer aux images d’entrée.

Le cœur de cette étude est double :

- **Évaluer différentes architectures de segmentation adaptées au contexte embarqué**, en comparant un U-Net “classique”, un U-Net avec backbone MobileNetV2 et un DeepLabV3+MobileNetV2.
- **Quantifier l’impact de la data augmentation** sur les performances de segmentation, en particulier en termes d’**IoU (Intersection-over-Union)** et de **Dice coefficient**, et analyser dans quelle mesure ces techniques améliorent la robustesse du modèle face à la variabilité des scènes.

Les expérimentations sont organisées en plusieurs ensembles de runs MLflow :

- une première série pour **comparer les architectures de base** (expérience 4) ;
- une seconde pour **évaluer l’apport de la data augmentation** (expérience 6) ;
- enfin, un ensemble de runs dédiés au **modèle sélectionné et à son hypertuning** (expérience 5), focalisés sur U-Net avec backbone MobileNetV2.

La suite de cette note présente d’abord un **état de l’art** des approches de segmentation sémantique et des techniques d’augmentation de données pertinentes pour la conduite autonome. Elle détaille ensuite **l’architecture du modèle retenu** et du pipeline d’entraînement, puis discute **les résultats obtenus**, notamment l’apport mesurable de la data augmentation, avant de conclure sur les **limites actuelles** et les **pistes d’amélioration** envisageables.

### 2. État de l’art et approches existantes

#### 2.1. Segmentation sémantique pour la conduite autonome

La segmentation sémantique est une tâche de vision par ordinateur où l’on cherche à attribuer à chaque pixel une classe parmi un ensemble prédéfini. Dans le contexte des véhicules autonomes, cette tâche est généralement formulée sur des jeux de données urbains comme **Cityscapes**, qui fournissent des annotations fines pour des scènes de conduite en milieu urbain. Les systèmes de perception doivent distinguer des entités comme :

- la **chaussée** et les surfaces “roulables” (Flat) ;
- les **bâtiments, murs, barrières** (Construction) ;
- les **objets** de mobilier urbain (poteaux, panneaux, feux) ;
- la **végétation** et le **terrain** (Nature) ;
- le **ciel** ;
- les **piétons** et autres usagers vulnérables (Human) ;
- les **véhicules** motorisés et non motorisés (Vehicle).

Traditionnellement, des approches basées sur des descripteurs manuels (HOG, SIFT) associés à des classifieurs (SVM, Random Forests) ont été explorées, mais ces méthodes se sont rapidement révélées insuffisantes pour capturer la richesse des scènes urbaines et la complexité des objets. L’avènement des **réseaux de neurones convolutifs (CNN)** a profondément transformé le domaine, donnant naissance à une famille d’architectures dédiées à la segmentation dense.

#### 2.2. Architectures de segmentation par deep learning

Les architectures modernes de segmentation se basent typiquement sur une structure **encodeur–décodeur** :

- l’**encodeur** réduit progressivement la résolution spatiale tout en augmentant le nombre de canaux, afin de capturer des représentations de haut niveau ;
- le **décodeur** reconstruit une carte de segmentation à la résolution de l’image d’origine.

##### U-Net

L’architecture **U-Net** est devenue un standard pour la segmentation d’images. Elle est composée :

- d’un **chemin contractant** (encodeur) qui applique des blocs Conv2D + activation + pooling pour extraire des caractéristiques à des échelles de plus en plus larges ;
- d’un **chemin expansif** (décodeur) qui remonte en résolution par des opérations de upsampling suivies de convolutions ;
- de **skip connections** qui relient symétriquement les couches de même niveau entre encodeur et décodeur.

Ces connexions de saut permettent de **réinjecter l’information de bas niveau** (détails spatiaux, contours) perdue lors des downsamplings, ce qui rend U-Net particulièrement efficace pour des tâches où la précision des frontières est cruciale. Dans ce projet, une implémentation complète de U-Net est disponible dans `src/model_architecture.py` et a servi de **premier modèle de référence**.

##### U-Net avec backbone MobileNetV2

Une limite importante de l’U-Net “plein” est sa **consommation mémoire** et son **coût de calcul**, peu adaptés à un déploiement embarqué. Une approche courante consiste à **remplacer l’encodeur** par un backbone convolutionnel pré-entraîné et optimisé, tel que **MobileNetV2**, qui a été conçu pour des scénarios “mobile/embedded”.

L’architecture **U-Net MobileNetV2** garde la structure en U et les skip connections, mais :

- utilise MobileNetV2 comme **encodeur** (partie descendante), bénéficiant de ses blocs “inverted residual” et convolutions depthwise-separable pour réduire le nombre de paramètres ;
- construit un **décodeur** symétrique qui remonte progressivement en résolution à partir des features de MobileNetV2, en concaténant les activations intermédiaires (skip connections) aux cartes de caractéristiques upsamplées.

Cette variante offre un **compromis très intéressant** pour le projet :

- elle reste capable de capter des détails fins grâce aux skip connections ;
- elle est plus **légère et rapide** qu’un U-Net entièrement “plein”, ce qui la rend plus compatible avec un déploiement embarqué ;
- elle tire parti d’un **pré-entraînement** sur ImageNet, ce qui accélère et stabilise la convergence sur Cityscapes.

C’est cette architecture qui est au centre des expérimentations avancées du projet (runs `Augmented_U-Net_MobileNetV2_*` loggés dans MLflow) et du pipeline de fine-tuning mis en place dans `notebooks/02-training_pipeline.ipynb`.

##### DeepLabV3 avec MobileNetV2

**DeepLabV3** propose une approche différente de U-Net pour la segmentation sémantique. Plutôt que de s’appuyer sur un décodeur complexe, DeepLabV3 introduit le module **ASPP (Atrous Spatial Pyramid Pooling)** pour capturer des informations à **plusieurs échelles** sans réduire excessivement la résolution.

Les points clés de DeepLabV3 sont :

- l’utilisation de **convolutions atrous (dilatées)** pour obtenir un grand champ réceptif sans perte de résolution excessive ;
- un module **ASPP** qui applique plusieurs convolutions atrous en parallèle avec des taux de dilatation différents, puis agrège les résultats pour capturer à la fois des détails locaux et un contexte global ;
- un **décodeur simplifié**, souvent réduit à un simple upsampling bilinéaire, contrairement au décodeur en escalier de U-Net.

Dans ce projet, DeepLabV3 est implémenté dans `src/model_architecture.py` avec un **backbone MobileNetV2**. Cette combinaison permet :

- de bénéficier des **capacités multi-échelles** de DeepLabV3 pour des scènes complexes (petits panneaux, piétons, grands bâtiments, routes) ;
- de **limiter le coût** grâce à MobileNetV2 comme encodeur léger.

DeepLabV3+MobileNetV2 a été évalué dans le cadre de la **comparaison d’architectures** (expériences MLflow correspondantes) afin de juger son intérêt par rapport à U-Net et U-Net MobileNetV2 sur Cityscapes 8-classes.

#### 2.3. Contraintes embarquées et choix d’architecture

Dans le cadre d’un système embarqué, le choix d’architecture ne se fait pas uniquement sur la base des métriques de segmentation (IoU, Dice) mais aussi sur :

- le **temps d’inférence par image** ;
- la **taille mémoire** du modèle et des activations ;
- la **facilité d’intégration** dans une API et un pipeline de déploiement.

Les expérimentations menées dans ce projet montrent que **U-Net MobileNetV2** représente un compromis intéressant :

- il bénéficie des **skip connections** et de la structure en U pour des frontières précises ;
- grâce à MobileNetV2, il reste **suffisamment léger** pour une exécution réaliste dans un contexte embarqué ;
- il supporte bien les phases de **fine-tuning** du backbone (avec une IoU de validation d’environ 0,55–0,60), ce qui permet de spécialiser le modèle à Cityscapes.

Les runs loggés dans MLflow (expériences 4 et 6) fournissent une **base quantitative** pour ce choix, tandis que l’expérience 5 consolide le **modèle final** retenu pour le déploiement.

#### 2.4. Approches d’augmentation de données

La **data augmentation** est un levier essentiel pour améliorer la généralisation des modèles de segmentation, en particulier lorsque :

- les scènes sont **variées** (villes, saisons, météo, trafic) ;
- certains objets d’intérêt (piétons, cyclistes) sont **sous-représentés**, comme le montre l’analyse du déséquilibre des classes (classe Human ≈ 0,75 % des pixels).

Pour la conduite autonome, les transformations les plus pertinentes sont :

- **Transformations géométriques** :
  - rotations modérées, translations horizontales, flips gauches/droites (pour simuler des changements de perspective et de trajectoire) ;
  - crops aléatoires et léger zoom pour forcer le modèle à être robuste à la position et à l’échelle des objets.
- **Transformations photométriques** :
  - variations de **luminosité, contraste, saturation** pour simuler différentes conditions météo et d’éclairage (jour/nuit, ombres, contre-jour) ;
  - ajout de bruit ou de flou léger pour simuler des capteurs moins parfaits ou des mouvements.

Dans le pipeline d’entraînement du projet, ces augmentations sont intégrées dans le **générateur de données** et activées via des indicateurs comme `USE_AUGMENTATION` ou `USE_LIGHT_AUGMENTATION`. Les expériences MLflow dédiées (expérience 6) comparent explicitement les modèles avec et sans augmentation, en suivant notamment la **val_iou_coefficient** et la **val_dice_coefficient**.

Les résultats (détaillés dans la section 5) montrent que l’augmentation de données permet :

- d’**améliorer l’IoU de validation** par rapport aux entraînements sans augmentation ;
- de **réduire le sur-apprentissage** sur les classes majoritaires ;
- d’augmenter la **robustesse visuelle** du modèle sur des cas plus difficiles (variations d’éclairage, petites instances d’objets).

### 3. Données et préparation

Le projet s’appuie sur le jeu de données **Cityscapes**, largement utilisé pour l’évaluation des systèmes de perception en environnement urbain. Le dataset fournit des images haute résolution (1024×2048) annotées pixel par pixel. Dans le cadre de ce travail, seules les **images et masques de segmentation** ont été utilisées, avec la structure suivante :

- **Entraînement** : 2 964 images issues de 18 villes ;
- **Validation/Test** : 1 525 images issues de 6 villes ;
- **Images** : format 1024×2048, ratio 2:1 typique de la vision embarquée frontale.

Afin de simplifier la tâche et de se concentrer sur les entités pertinentes pour la conduite autonome, les **34 classes d’origine** fournies par Cityscapes sont regroupées en **8 grandes catégories** : Void, Flat, Construction, Object, Nature, Sky, Human, Vehicle. Cette consolidation est implémentée dans les fonctions utilitaires et détaillée dans le notebook d’exploration et la documentation. Elle permet :

- de réduire la complexité de la tâche de segmentation ;
- de mieux refléter les besoins opérationnels (par exemple, distinguer globalement “véhicule” sans séparer tous les sous-types) ;
- de stabiliser les métriques sur des classes moins peuplées.

L’analyse exploratoire a mis en évidence un **fort déséquilibre de classes** : certaines catégories (Flat, Construction) occupent une grande proportion des pixels, alors que d’autres (notamment Human) sont très sous-représentées (moins de 1 % des pixels). Cette observation a motivé :

- l’utilisation de **fonctions de perte adaptées** (comme une `combined_loss` qui combine Dice et IoU) pour mieux prendre en compte les classes rares ;
- l’introduction de **data augmentation** ciblée afin d’augmenter la diversité et de limiter le sur-apprentissage.

Avant l’entraînement, les images sont :

- **redimensionnées** à une taille plus modeste (par exemple 256×512) afin de réduire le coût mémoire tout en préservant l’aspect 2:1 ;
- **normalisées** dans \([0,1]\) (division par 255.0) ;
- associées à leurs masques correspondants, eux-mêmes encodés en entiers \([0,7]\) pour les 8 catégories.

Ces opérations sont encapsulées dans un **générateur de données** qui gère également l’application (ou non) des transformations d’augmentation au vol pendant l’entraînement.

### 4. Modèle et architecture retenue

Plusieurs architectures ont été implémentées et évaluées dans le module de modèles, puis orchestrées via le notebook `notebooks/02-training_pipeline.ipynb` :

- un **U-Net “classique”** convolutif ;
- un **U-Net avec backbone MobileNetV2** (U-Net MobileNetV2) ;
- un **DeepLabV3 avec MobileNetV2**.

Ces modèles ont été comparés à l’aide d’expériences MLflow (expérience 4), en suivant principalement les métriques **IoU** et **Dice** sur le set de validation. Les résultats montrent que **U-Net MobileNetV2** offre un compromis particulièrement intéressant :

- des **performances de segmentation** comparables voire supérieures aux autres variantes sur Cityscapes 8-classes ;
- une **taille de modèle** et un **temps d’inférence** compatibles avec un déploiement embarqué ;
- la possibilité d’exploiter un **pré-entraînement** sur ImageNet via MobileNetV2, puis de procéder à un fine-tuning contrôlé.

L’architecture retenue pour le modèle final se compose donc :

- d’un **encodeur MobileNetV2** tronqué, initialisé par des poids pré-entraînés ;
- d’un **décodeur de type U-Net** qui remonte progressivement en résolution, avec des **skip connections** reliant les couches du backbone à leurs homologues dans le décodeur ;
- d’une **tête de segmentation** (convolution 1×1 + softmax) produisant une carte de labels sur 8 canaux.

L’entraînement suit une stratégie en **deux phases** :

1. **Phase 1 – Entraînement du décodeur (backbone gelé)** :
   - MobileNetV2 est figé (non entraînable), seul le décodeur et la tête de segmentation sont entraînés.
   - Un taux d’apprentissage relativement “standard” est utilisé (par exemple 1e-3).
   - Cette phase permet d’apprendre rapidement une carte de segmentation raisonnable sans perturber les poids génériques du backbone.

2. **Phase 2 – Fine-tuning de l’ensemble du réseau** :
   - Le backbone est progressivement **dégelé** pour adapter ses filtres au domaine Cityscapes.
   - Un **learning rate réduit** (par exemple 1e-5) est utilisé pour éviter de détruire l’information pré-entraînée.
   - Des callbacks supplémentaires sont activés pour surveiller les métriques et sauvegarder les meilleurs modèles (via des checkpoints locaux ou Azure).

Les métriques d’entraînement et de validation sont suivies avec des callbacks personnalisés ainsi que via **MLflow** (paramètres de configuration, métriques par époque, artefacts). Un run typique de fine-tuning (`Augmented_UNet_MobileNetV2_finetune_...`) montre une **IoU de validation** stabilisée autour de **0,55–0,60**, ce qui constitue une base solide pour la suite du projet.

D’un point de vue système, le modèle final est **sérialisé** au format `.keras` et :

- sauvegardé localement dans le dossier des modèles ou checkpoints ;
- **téléversé sur Azure Blob Storage** via un gestionnaire de stockage ;
- **chargé à l’inférence** par une classe de type `SegmentationPredictor`, qui gère le téléchargement depuis Azure, le cache local, la normalisation des entrées et la conversion des prédictions en masques de classes.

Ce pipeline permet d’intégrer le modèle dans une **API FastAPI** exposant les endpoints `/health` et `/predict`, et de proposer une **interface Streamlit** pour la démonstration interactive.

### 5. Résultats expérimentaux

#### 5.1. Métriques et protocole

Les performances des modèles sont évaluées principalement à l’aide de :

- **IoU (Intersection-over-Union)** par pixel et moyenne sur les classes ;
- **Dice coefficient**, fortement corrélé à l’IoU mais plus sensible aux erreurs sur des petites régions ;
- **Accuracy globale** (moins informative en cas de fort déséquilibre, mais suivie à titre indicatif).

Ces métriques sont calculées à la fois sur le **jeu d’entraînement** et sur le **jeu de validation**, à chaque époque, via des métriques personnalisées et remontées dans MLflow.

#### 5.2. Comparaison des architectures (sans et avec augmentation)

Les expériences loggées dans l’expérience 4 de MLflow permettent de comparer les architectures de base (U-Net, U-Net MobileNetV2, DeepLabV3+MobileNetV2) sur un protocole d’entraînement homogène. Globalement :

- U-Net fournit une **ligne de base solide**, mais avec un coût mémoire plus élevé ;
- U-Net MobileNetV2 atteint une **IoU de validation supérieure** tout en étant plus léger, ce qui en fait un **candidat privilégié pour l’embarqué** ;
- DeepLabV3+MobileNetV2 est compétitif, en particulier pour des objets de tailles variées, mais la simplicité et la lisibilité de U-Net MobileNetV2, combinées à ses performances, ont motivé son choix comme **modèle principal**.

L’impact de la **data augmentation** est évalué de manière plus systématique dans l’expérience 6, en comparant des entraînements :

- **sans augmentation** (`augmentation = "None"`) ;
- avec **augmentation “light”** (transformations modérées) ;
- avec **augmentation “full”** (combinaison plus agressive de transformations géométriques et photométriques).

Les courbes de `val_iou_coefficient` et `val_dice_coefficient` montrent que :

- l’activation de la data augmentation permet de **gagner plusieurs points d’IoU** sur le set de validation par rapport au modèle entraîné sans augmentation ;
- les modèles sans augmentation ont tendance à **sur-apprendre** les configurations visuelles dominantes (routes dégagées, conditions lumineuses standard) et à se dégrader plus rapidement sur la validation ;
- une augmentation trop agressive peut parfois ralentir la convergence ; le mode “light” ou un compromis calibré s’avère souvent le plus efficace.

Ces tendances s’observent également en regardant les **exemples visuels** générés pendant l’entraînement (batchs d’images, masques prédits vs ground truth, feature maps), qui montrent une meilleure **robustesse** aux variations d’éclairage et à la présence de petits objets lorsque l’augmentation est activée.

#### 5.3. Fine-tuning et modèle final

Les expérimentations dédiées au **modèle retenu** sont regroupées dans l’expérience 5, où le **U-Net MobileNetV2 avec augmentation** est affiné via :

- un ajustement du **taux d’apprentissage** et du **nombre d’époques** ;
- une phase explicite de **fine-tuning** du backbone ;
- une sélection stricte du **meilleur modèle** selon la meilleure `val_iou_coefficient`.

Le run de fine-tuning illustré dans le notebook montre par exemple une montée de l’IoU de validation jusqu’à environ **0,56**, avec un compromis satisfaisant entre précision, temps d’entraînement et stabilité des métriques. Ce modèle est finalement retenu et déployé via Azure pour l’API de prédiction.

### 6. Conclusion et pistes d’amélioration

Ce projet a permis de concevoir et d’évaluer un **système de segmentation sémantique complet** pour la conduite autonome, depuis l’analyse de données Cityscapes jusqu’au déploiement d’un modèle dans une API de prédiction et une interface web. Après comparaison de plusieurs architectures (U-Net, U-Net MobileNetV2, DeepLabV3+MobileNetV2) et exploration de différentes stratégies de data augmentation, le **U-Net MobileNetV2 avec augmentation et fine-tuning** s’est imposé comme **modèle de référence**. Il offre un bon compromis entre **qualité de segmentation** (IoU et Dice sur Cityscapes 8-classes) et **contraintes embarquées** (taille de modèle, temps d’inférence).

L’**impact de la data augmentation** est particulièrement notable : les expériences MLflow montrent une amélioration systématique des métriques de validation et une meilleure robustesse visuelle, confirmant que la diversité des perturbations appliquées aux images (géométriques et photométriques) permet au modèle de mieux généraliser à des conditions de conduite variées.

Malgré ces résultats encourageants, plusieurs **limites** subsistent :

- le dataset Cityscapes, bien que riche, ne couvre pas l’ensemble des scénarios réels (conditions météo extrêmes, nuit, campagne, etc.) ;
- certaines classes minoritaires (notamment Human) restent difficiles à segmenter avec une grande précision ;
- le modèle, bien que plus léger qu’un U-Net “plein”, pourrait encore être optimisé (quantization, pruning) pour des cibles matérielles très contraintes.

Plusieurs **pistes d’amélioration** sont envisageables :

- **Côté données** :
  - enrichir le corpus avec d’autres datasets (BDD100K, Mapillary) ;
  - générer des **données synthétiques** via simulation (par exemple moteurs de rendu 3D) ;
  - expérimenter des techniques d’augmentation plus avancées (Mixup, CutMix, Random Erasing).
- **Côté modèle** :
  - explorer des backbones encore plus compacts (MobileNetV3, EfficientNet-Lite) ;
  - recourir à la **distillation de connaissances** pour transférer les performances d’un modèle lourd vers un modèle embarqué plus petit ;
  - tester des variantes comme **DeepLabV3+** ou d’autres architectures encoder–decoder modernes.
- **Côté système** :
  - optimiser davantage la chaîne d’inférence (conversion vers ONNX, TensorRT, etc.) ;
  - mettre en place un **monitoring en production** (latence, dérive de données, qualité des prédictions) ;
  - intégrer la segmentation dans un pipeline plus large de perception et de décision (fusion avec détection d’objets, suivi de trajectoires, etc.).

En synthèse, le travail réalisé fournit une **base solide** pour un système de segmentation embarqué, avec un pipeline reproductible (notebooks, code de la librairie, suivi MLflow) et un modèle opérationnel déployé via API. Les améliorations futures porteront principalement sur l’extension du domaine de validité (plus de données, plus de diversité) et l’optimisation fine pour des plateformes embarquées spécifiques.

