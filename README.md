# CY_NLP_MiniProjet

# Classification de cuisines à partir d'ingrédients — *What's Cooking?*

Projet de NLP : prédire le **type de cuisine** (20 classes : italian, mexican, indian, thai, …) d'une recette à partir de sa **liste d'ingrédients**.

Le notebook `projetNLP.ipynb` compare trois approches, de la plus simple à la plus puissante :

| # | Approche | Apprentissage | Support |
|---|----------|---------------|---------|
| 1 | Vecteurs de poids par cuisine (IDF) + produit scalaire + softmax | Non | CPU |
| 2 | Réseau de neurones dense (MLP, PyTorch) | Oui | CPU |
| 3 | Transformer pré-entraîné `roberta-base` (fine-tuning) | Oui | GPU (Google Colab) |

---

## Données

- **Dataset** : [What's Cooking?](https://www.kaggle.com/c/whats-cooking) (Kaggle / Yummly)
- **Fichiers** : `train.json` (recettes étiquetées) et `test.json` (recettes sans étiquette, utilisé uniquement pour la partie RoBERTa)
- **Format** : chaque recette contient un `id`, une liste `ingredients` et (pour le train) une `cuisine`
- **Particularité** : le dataset est **déséquilibré**. L'italien et le mexicain représentent à eux seuls plus d'un tiers des recettes, alors que les cuisines brésilienne ou russe en représentent environ 1 %.

> Les fichiers de données ne sont pas inclus dans ce dépôt : à télécharger sur Kaggle et à placer à côté du notebook (ou dans `/content` sous Colab).

---

## Structure du notebook

### Partie 1 — Classification sans apprentissage

1. **Exploration** : nombre de recettes, de classes, répartition par cuisine.
2. **Découpage** stratifié par cuisine : 70 % train / 15 % validation / 15 % test (seed 42).
3. **Construction du modèle** :
   - comptage, pour chaque cuisine, du nombre de recettes contenant chaque ingrédient ;
   - sélection des **30 ingrédients les plus fréquents** de chaque cuisine ;
   - vocabulaire = union de ces top-30 ;
   - **IDF** = `log(nb de cuisines / nb de cuisines dont le top-30 contient l'ingrédient)` : un ingrédient présent partout a un poids nul, un ingrédient rare est très discriminant ;
   - vecteur de poids de chaque cuisine (IDF aux positions de ses ingrédients du top-30).
4. **Prédiction** : la recette est encodée en vecteur binaire, le score de chaque cuisine est le produit scalaire avec son vecteur de poids, puis un **softmax** donne une probabilité par cuisine.

### Partie 2 — Avec apprentissage (MLP PyTorch)

- **Entrée** : vecteur binaire de la taille du vocabulaire de la partie 1.
- **Architecture** : `Linear(n_vocab → 128) → ReLU → Dropout(0.3) → Linear(128 → 64) → ReLU → Dropout(0.3) → Linear(64 → 20)`, softmax en sortie.
- **Entraînement** : `CrossEntropyLoss`, Adam (lr = 1e-3), batch de 64, 30 epochs, conservation des poids de la meilleure `val_loss` (early stopping sur les poids).
- **Analyse d'erreurs** : affichage des erreurs les plus « confiantes » et commentaire de 5 cas (déséquilibre des classes, cuisines proches comme thaï/vietnamien ou chinois/coréen, ingrédients trop génériques).

### Partie 3 — Transformer `roberta-base` (Google Colab)

| Élément | Choix |
|---|---|
| Modèle | `roberta-base` (125 M de paramètres) |
| Tête | `RobertaClassificationHead` (768→768, tanh, dropout, 768→20) sur le token `<s>` |
| Texte d'entrée | ingrédients en minuscules, séparés par `", "` |
| Tokenisation | BPE byte-level de RoBERTa, padding dynamique par batch |
| Longueur max | percentile 99,9 des longueurs en tokens, plafonné à 128 |
| Validation | 10 % du train, split stratifié (seed 42), doublons exacts supprimés |
| Sélection | meilleure epoch selon le **macro-F1** |
| Phase 1 | encodeur gelé, tête seule (4 epochs, lr 1e-3, batch 64) |
| Phase 2 | fine-tuning complet (4 epochs, lr 2e-5, batch 32, warmup 6 %, weight decay 0,01) |

Sorties générées dans `/content/outputs` : `submission.csv` (prédictions sur `test.json`), `confusion_matrix.png`, `model/` (modèle + tokenizer) et `model_card.json` (hyperparamètres et scores).

---

## Métriques d'évaluation

Toutes calculées à partir de la **matrice de confusion** : accuracy, précision, rappel et F1 par classe, moyennes **macro** (toutes les classes pèsent pareil) et **pondérée** (par le support). Matrices de confusion normalisées par ligne (la diagonale correspond au rappel).

Le notebook produit un tableau comparatif final (sans apprentissage vs MLP) sur le jeu de test, ainsi que `confusion_sans_apprentissage.png` et `confusion_avec_apprentissage.png`.

### Résultats

| Modèle (test) | Accuracy | Précision (macro) | Rappel (macro) | F1 (macro) |
|---|---|---|---|---|
| Sans apprentissage (IDF) | _à compléter_ | _à compléter_ | _à compléter_ | _à compléter_ |
| MLP (PyTorch) | _à compléter_ | _à compléter_ | _à compléter_ | _à compléter_ |
| RoBERTa fine-tuné (validation) | _à compléter_ | — | — | _à compléter_ |

> Remarque : RoBERTa est évalué sur sa propre validation (split 90/10 du train), pas sur le même jeu de test que les deux premières approches. La comparaison directe est donc à interpréter avec prudence.

---

## Installation et exécution

**Parties 1 et 2 (local)**

```bash
pip install numpy matplotlib torch
jupyter notebook projetNLP.ipynb
```

**Partie 3 (Google Colab)**

1. Ouvrir le notebook dans Colab et activer un GPU (*Exécution → Modifier le type d'exécution → GPU*).
2. Lancer la cellule d'installation (`transformers`, `datasets`, `accelerate`, `evaluate`, `scikit-learn`).
3. Importer `train.json` et `test.json` quand la fenêtre d'upload s'ouvre.
4. Exécuter les cellules dans l'ordre ; les résultats sont écrits dans `/content/outputs`.

---

## Limites et pistes d'amélioration

- Le vocabulaire (top-30 par cuisine) est restreint : des ingrédients discriminants comme `Thai fish sauce` en sont absents. Augmenter le top-N ou utiliser tout le vocabulaire devrait aider.
- Le déséquilibre des classes biaise les modèles vers l'italien et le mexicain : pondération de la loss ou rééchantillonnage possibles.
- Dans la partie 1, la fréquence d'un ingrédient dans une cuisine est calculée mais non utilisée : seul l'IDF pèse dans le vecteur. Un vrai TF-IDF est une piste directe.
- Les cuisines proches géographiquement (thaï/vietnamien, chinois/coréen) restent les principales sources de confusion.
- Aligner l'évaluation de RoBERTa sur le même jeu de test que les autres modèles.

---

## Remarques

Certaines parties du code (découpage, métriques, boucle d'entraînement, choix de l'architecture du MLP) ont été réalisées avec l'aide de ChatGPT, comme indiqué dans les commentaires du notebook.
