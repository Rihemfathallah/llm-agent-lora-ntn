# AGENTS.md — règles pour tout assistant de code travaillant sur ce dépôt

Projet : agent LLM événementiel pour le contrôle adaptatif des paramètres LoRa
(SF, puissance, canal, lien) dans un réseau IoT hybride terrestre + LEO.
Cahier des charges : `docs/PROJECT_SPEC.md`. Lis-le avant toute tâche.

## Règles non négociables

1. **Aucun chiffre écrit à la main.** Tout nombre dans le README, le LOG ou le rapport
   vient d'un CSV de `results/`, produit par un script de `experiments/`.
2. **Aucun résultat scripté, mocké ou simulé « à la main » n'est présenté comme une mesure.**
   Si une expérience utilise un mock, son nom de fichier, sa colonne CSV et sa légende
   de figure le disent explicitement (`mock_`, `scripted_`).
3. **Les résultats négatifs sont valides.** Ne règle jamais un paramètre dans le but
   de « faire gagner » l'agent. Si l'agent ne bat pas ADR, on le rapporte tel quel.
4. **Ne modifie que ce que la tâche demande.** Pas de refactor opportuniste.
   Si tu vois un autre problème, note-le dans `LOG.md` (section « À traiter ») sans le corriger.
5. **Tests d'abord.** `python -m pytest -q` doit passer avant et après chaque modification.
   Toute nouvelle fonction de `phy/`, `channel/`, `netsim/`, `policies/` reçoit un test.
6. **Reproductibilité.** Paramètres dans `configs/*.yaml`, seeds fixes, résultats bruts en CSV,
   un script par figure qui la régénère depuis le CSV. 5 seeds, moyenne ± écart-type.
7. **Comparaison équitable.** Mêmes scénarios, mêmes seeds, mêmes réalisations de canal
   pour toutes les politiques. Si l'agent voit le rapport d'anomalie, le classifieur ML
   reçoit le flag d'anomalie aussi.
8. **Hypothèses explicites.** Toute simplification (pas de rotation terrestre, tables
   TR 38.811 en bande S pour 868 MHz, SF orthogonaux, PER depuis un SER non codé…) est
   listée dans `docs/ASSUMPTIONS.md`. Ajoute-y toute nouvelle hypothèse.
9. **Le simulateur réseau ne simule jamais d'échantillons de chirp.** Il lit les tables PER.
10. **Une tâche = une branche git = des commits petits.** Message de commit : `type: description`.

## Fin de chaque session (obligatoire)

Ajouter à `LOG.md` une entrée : date, ce qui a été lancé (commande exacte), ce qui a été
observé (chiffres tirés des CSV), ce qui a échoué ou surprend, prochaine étape.
Ne jamais écrire dans le LOG une conclusion que les CSV ne montrent pas.

## Définitions de métriques (à ne pas réinterpréter)

- PDR = paquets livrés / paquets envoyés.
- Énergie par paquet livré (mJ) = énergie TX totale / paquets livrés.
- Temps de récupération = temps entre le début de l'événement et le retour du PDR à ±5 % du niveau pré-événement.
- Taux d'actions invalides = appels `set_tx_params` rejetés / appels `set_tx_params` totaux.

## Commandes

```bash
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```
