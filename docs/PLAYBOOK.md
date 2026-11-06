> **Statut (6 nov. 2026) :** les étapes de la feuille de route (cooldown/TTL de l'agent, E4 avec
> l'agent réel, E5/E6 réels, SNR par device, tables PER, Doppler au niveau paquet, satellite dans le
> simulateur, tests + RNG à clés) sont faites. Voir `LOG.md` (entrée du 6 nov.) et `docs/GUIDE_FR.md`
> pour la suite (runs qwen2.5:3b/7b). Ce fichier est conservé comme trace de la méthode de travail.

# PLAYBOOK — corriger le projet LoRa NTN étape par étape

## Comment utiliser tes deux outils

| Outil | Rôle | Pourquoi |
|---|---|---|
| **Antigravity** | Exécuter : écrire le code, lancer les scripts, régénérer les figures | Il peut tourner longtemps et lancer des commandes |
| **Claude Sonnet (local)** | Relire : critiquer le diff et les résultats, rédiger README / rapport | Un deuxième regard indépendant attrape les erreurs du premier |

**Boucle pour chaque étape (toujours la même) :**
1. `git checkout -b <branche>`
2. Colle le prompt de l'étape dans **Antigravity**.
3. Lance toi-même `python -m pytest -q` et regarde les figures/CSV produits.
4. Colle le diff (`git diff main`) + les CSV dans **Sonnet** avec le *prompt de relecture* (en bas).
5. Corrige ce que Sonnet a trouvé (retour à Antigravity si besoin).
6. `git checkout main && git merge <branche>`.

Un prompt = une branche. Ne fusionne jamais deux étapes d'un coup.

---

## ÉTAPE 0 — Mise en place (toi, 20 min, sans IA)

Fichiers à copier dans ton dépôt (ils sont dans le dossier `kit/`) :

| Fichier du kit | Destination dans le dépôt |
|---|---|
| `AGENTS.md` | racine |
| `CLAUDE.md` | racine (Sonnet/Claude Code le lit automatiquement) |
| `requirements.txt` | racine |
| `.gitignore` | racine (remplace l'ancien, qui était en UTF-16 donc ignoré par Git) |
| `tests/test_basics.py` | `tests/` |
| `docs/ASSUMPTIONS.md` | `docs/` |

Dans Antigravity, si `AGENTS.md` n'est pas pris en compte automatiquement, colle son contenu dans les règles (Rules) du workspace.

```bash
cd llm-agent-lora-ntn
git init            # si pas déjà fait
git rm -r --cached -q . 2>/dev/null; git add -A     # reprend le .gitignore corrigé
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q          # attendu : 11 passed
git add -A && git commit -m "chore: AGENTS.md, tests de base, requirements, gitignore UTF-8"
```
Si les 11 tests passent, tu as un point de départ propre.

---

## ÉTAPE 1 — Remettre l'honnêteté du dépôt (branche `fix/honesty`)

But : que README, LOG, noms de fichiers et figures disent la même chose.

**Prompt pour Antigravity :**
```
Lis AGENTS.md, README.md, LOG.md et docs/PROJECT_SPEC.md.

Tâche (branche fix/honesty), sans changer aucune logique de simulation :
1. E4 : la politique nommée "LLM-Agent" est en réalité un oracle scripté. Renomme-la
   "Scripted-Oracle" dans experiments/e4_anomalies.py, dans les colonnes CSV, les légendes
   de figures et les prints. Régénère results/e4_anomalies.csv/.png.
2. E5 et E6 utilisent un mock / des variantes scriptées : préfixe leurs fichiers de sortie
   par "mock_" (mock_e5_agent_cost.*, scripted_e6_ablations.*), ajoute "MOCK — pas une mesure"
   dans le titre des figures.
3. LOG.md : réécris les entrées E4–E6 pour dire explicitement qu'elles sont scriptées/mockées.
   Ajoute une section "À traiter" reprenant la roadmap du README.
4. README.md : supprime les références à des fichiers qui n'existent pas
   (docs/ARCHITECTURE.md, docs/PLAYBOOK.md) ou crée-les s'ils sont vraiment utiles.
   Corrige le tableau Status si besoin.
5. Mets à jour la liste des fichiers dans la section "Repository layout".
Ne touche à rien d'autre. Lance pytest. Commit par sous-tâche. Mets à jour LOG.md.
```
**Tu vérifies :** `grep -rn "LLM-Agent" experiments results README.md` ne renvoie plus rien de trompeur.

---

## ÉTAPE 2 — Aléatoire reproductible et appariable (branche `fix/keyed-rng`)

But : toutes les politiques voient exactement le même canal (condition pour comparer ADR vs agent).

**Prompt pour Antigravity :**
```
Lis AGENTS.md et netsim/simulator.py.

Problème : le simulateur utilise un seul self.rng séquentiel. Si deux politiques prennent
des décisions différentes, elles consomment le flux aléatoire différemment et ne voient plus
les mêmes réalisations de canal (shadowing, tirage PER, jitter de trafic).

Tâche (branche fix/keyed-rng) :
1. Introduis un générateur "keyed" : les tirages de shadowing, de PER et de trafic sont dérivés
   de (seed, device_id, numéro_de_transmission, type_de_tirage) via np.random.SeedSequence ou
   un hash déterministe, indépendamment des décisions de la politique.
2. Ajoute tests/test_rng.py :
   - deux politiques différentes (fixed SF7 vs ADR) avec la même seed obtiennent exactement le
     même shadowing pour le même (device, n° de transmission) ;
   - même seed => résultats strictement identiques ; seeds différentes => résultats différents.
3. Les expériences E3 existantes doivent tourner sans modification de leurs résultats
   statistiques majeurs : régénère E3 et dis-moi quelles valeurs ont changé et de combien.
4. Ajoute des fonctions d'analyse appariée dans un nouveau module analysis/paired.py
   (différence par seed, moyenne ± écart-type, intervalle de confiance par bootstrap).
Ne modifie aucune autre logique. pytest doit passer. Mets à jour LOG.md et docs/ASSUMPTIONS.md.
```

---

## ÉTAPE 3 — Doppler au niveau du paquet (branche `feat/packet-level-doppler`)

C'est l'étape **scientifiquement la plus importante** : elle teste ton « insight clé ». Aujourd'hui le Doppler n'est appliqué qu'à l'intérieur de chaque symbole, donc ton E1b ne peut pas répondre à la question.

**Prompt pour Antigravity :**
```
Lis AGENTS.md, docs/PROJECT_SPEC.md (section 3.3 "Doppler" et "The key physical insight"),
phy/css.py, phy/ser_montecarlo.py et experiments/e1_phy_validation.py.

Problème : apply_doppler() est utilisé symbole par symbole, donc la dérive du décalage
fréquentiel pendant tout le paquet (préambule + payload) n'est pas modélisée. Un récepteur réel
estime le décalage statique sur le préambule ; ce qui reste est l'erreur résiduelle plus la
dérive f_D' pendant le payload.

Tâche (branche feat/packet-level-doppler) :
1. Génère un paquet complet : préambule (8 upchirps + 2 downchirps de sync, simplifié si besoin)
   puis les symboles du payload (nombre exact via payload_symbols()).
2. Applique apply_doppler() UNE FOIS sur tout le paquet (f_d initial + f_d_rate) puis du bruit AWGN.
3. Récepteur A ("preamble-only") : estime le décalage fréquentiel sur le préambule
   (dechirp + FFT + interpolation du pic), le retire, puis démodule le payload sans suivi.
   Récepteur B ("with tracking") : même chose, avec une mise à jour du décalage pendant le payload
   (par exemple estimation sur un chirp de référence, ou boucle de suivi simple décrite dans le code).
4. Simule SER vs Doppler rate (0 à 1200 Hz/s) pour chaque SF 7–12, pour A et B, à SNR = seuil + 5 dB,
   500+ essais par point, avec seeds fixes. Écris les CSV dans results/.
5. Régénère la figure E1b avec les deux récepteurs. Ajoute en lecture seule les lignes verticales
   du Doppler rate max au zénith (calculé par channel/satellite.py) pour 550, 600 et 800 km.
6. Documente le modèle de récepteur retenu comme hypothèse A5 dans docs/ASSUMPTIONS.md.
7. Tests : sans Doppler, 0 erreur à SNR élevé ; avec Doppler nul et SNR seuil + 5 dB, SER faible ;
   la dérive sur le payload SF12 en bins est cohérente avec f_D' * ToA / bin_width.
N'interprète pas les résultats dans le README : décris seulement ce que montrent les CSV.
Compare qualitativement avec Doroshkin et al. (IEEE Access 2019) cité dans le README,
et signale clairement tout écart. Mets à jour LOG.md.
```
**Tu vérifies :** quel que soit le résultat (SF12 fragile ou non), tu peux l'expliquer. Les deux sont publiables. Si SF12 tient jusqu'à 1000 Hz/s même avec le récepteur A, ton modèle de récepteur est peut-être trop clément : demande à Sonnet de le challenger.

---

## ÉTAPE 4 — Tables PER branchées au simulateur (branche `feat/per-tables`)

**Prompt pour Antigravity :**
```
Lis AGENTS.md, phy/ser_montecarlo.py (génération de lookup tables) et netsim/simulator.py
(argument per_lookup(sf, snr_db, doppler_rate) -> float).

Tâche (branche feat/per-tables) :
1. Génère PER(SNR, SF, Doppler rate) à partir du Monte Carlo packet-level de l'étape 3
   (récepteur choisi dans docs/ASSUMPTIONS.md), avec PER = 1 - (1 - SER)^n_symboles (hypothèse A4).
   Grille : SNR de seuil-8 dB à seuil+8 dB, Doppler rate 0–1200 Hz/s. Sauvegarde dans phy/tables/
   (CSV + .npz). Un script phy/make_tables.py reproductible, seed fixe.
2. Module phy/per_table.py : chargement + interpolation (RegularGridInterpolator), clamp aux bornes,
   fonction per_lookup(sf, snr_db, doppler_rate) compatible avec le simulateur.
3. Branche-la comme valeur par défaut du Simulator. Garde le modèle à seuil 0/1 sous un nom
   explicite pour les comparaisons.
4. Tests : PER dans [0,1], monotone décroissant en SNR, valeur ~0.5 près du seuil SF correspondant
   (tolérance raisonnable, à justifier), accord avec le SER de l'étape 3.
5. Régénère E3 avec les tables. Dis-moi quelles conclusions changent par rapport au modèle à seuil.
Mets à jour LOG.md et docs/ASSUMPTIONS.md.
```

---

## ÉTAPE 5 — SNR par device et configs YAML (branche `feat/per-device-snr`)

But : qu'ADR soit enfin différent de SF7 fixe.

**Prompt pour Antigravity :**
```
Lis AGENTS.md, netsim/devices.py, channel/terrestrial.py, channel/link_budget.py et
experiments/e3_scalability.py.

Problème : tous les devices ont le même SNR fixe (5 dB), donc ADR n'a rien à adapter.

Tâche (branche feat/per-device-snr) :
1. Chaque device calcule son SNR à partir de sa distance à la gateway (Okumura-Hata rural +
   shadowing log-normal, écart-type en paramètre, pas de valeur codée en dur) via link_budget.py.
   Positions tirées uniformément dans un disque (rayon en config, distance minimale 100 m).
2. Crée configs/ avec scenarios S1 à S5 en YAML : S1 baseline terrestre sans anomalie,
   S2 jamming, S3 deep fade, S4 gateway outage, S5 traffic surge. Chaque YAML contient tous les
   paramètres (devices, durée, intervalle de trafic avec jitter, canaux, shadowing, seeds, anomalies).
   Écris un chargeur configs/loader.py avec validation.
3. Traffic : remplace les arrivées exponentielles par un intervalle ~10 min + jitter aléatoire
   borné, comme dans le cahier des charges (garde Poisson en option documentée).
4. Refais E3 depuis les YAML. Vérifie qu'ADR se distingue maintenant de SF7 fixe et décris
   les écarts mesurés (PDR, énergie par paquet livré) sans interpréter au-delà des CSV.
5. Tests : le SNR décroît avec la distance en moyenne ; le chargeur YAML rejette une config invalide.
Mets à jour LOG.md.
```

---

## ÉTAPE 6 — Lien satellite dans le simulateur (branche `feat/satellite-netsim`)

**Prompt pour Antigravity :**
```
Lis AGENTS.md, channel/satellite.py (simulate_pass, satellite_link_budget), netsim/simulator.py,
netsim/anomalies.py et docs/PROJECT_SPEC.md sections 3.2 et 3.4.

Problème : link="satellite" n'est relié à rien dans le simulateur.

Tâche (branche feat/satellite-netsim) :
1. Le simulateur gère un ou plusieurs passages de satellite (horaires en config). Pendant un passage,
   chaque transmission en lien satellite calcule : élévation, distance oblique, délai, Doppler shift
   et Doppler rate, SNR via satellite_link_budget (gain d'antenne de réception satellite en
   paramètre de config), puis le PER via les tables (Doppler rate inclus).
2. Hors passage ou élévation < minimum : la transmission satellite échoue explicitement (cause
   enregistrée), jamais silencieusement.
3. Le lien "terrestrial" continue de fonctionner comme avant ; "wait" met le paquet en attente.
4. Anomalie "gateway outage" : la gateway terrestre est indisponible 30 min ; les devices en bordure
   peuvent passer en satellite pendant un passage.
5. Les outils de l'agent get_satellite_status et get_link_stats lisent ces vraies valeurs.
6. Nouveau scénario YAML S6 : terrestre + 2 passages + gateway outage.
7. Tests : un paquet satellite hors passage échoue avec la bonne cause ; le SNR satellite est cohérent
   avec channel/link_budget.py à une élévation donnée.
Fais ensuite tourner S6 avec ADR et les politiques fixes, sans agent, et décris les résultats.
Mets à jour LOG.md et docs/ASSUMPTIONS.md.
```

---

## ÉTAPE 7 — Agent : cooldown, expiration des overrides, traces (branche `fix/agent-cooldown-ttl`)

**Prompt pour Antigravity :**
```
Lis AGENTS.md, policies/llm_agent/agent.py, tools.py et prompt.py.

Problèmes connus : pas de cooldown entre déclenchements, les overrides posés par l'agent n'expirent
jamais, et les décisions ne sont pas sauvegardées dans un format réutilisable.

Tâche (branche fix/agent-cooldown-ttl) :
1. Cooldown par scope (device/groupe/canal) : pas plus d'une invocation par fenêtre configurable.
2. TTL sur les overrides : après N minutes simulées, le device revient à ADR. Le TTL est dans la config.
3. Traces : pour chaque décision, sauvegarde en JSONL dans results/traces/ : temps simulé, trigger,
   messages, appels d'outils, arguments, résultats, validité, latence réelle, tokens si disponibles,
   fallback ou non. Un fichier par (scénario, politique, seed).
4. Valide les appels d'outils avec un schéma JSON, borne à 6 appels par décision, enregistre chaque rejet.
5. Fournis un client LLM factice déterministe (pour les tests uniquement, jamais pour les résultats)
   afin de tester : cooldown, TTL, validation, fallback sur timeout.
6. Tests pour chacun de ces points, sans Ollama.
Mets à jour LOG.md.
```

---

## ÉTAPE 8 — Vrai E4 avec Ollama (branche `feat/e4-real-agent`)

Prérequis : `ollama pull qwen2.5:3b` et Ollama qui tourne (`ollama serve`).

**Prompt pour Antigravity :**
```
Lis AGENTS.md, experiments/e4_anomalies.py, policies/llm_agent/agent.py et configs/.

Tâche (branche feat/e4-real-agent) :
1. Réécris E4 pour qu'il utilise le VRAI agent Ollama (qwen2.5:3b, temperature 0, seed fixe),
   plus ADR, plus l'oracle scripté renommé Scripted-Oracle comme référence supérieure explicite.
2. Commence par le scénario jamming uniquement, 1 seed, pour valider la chaîne de bout en bout.
   Montre-moi les traces JSONL avant de lancer plus.
3. Ensuite : S2 à S6, 5 seeds, mêmes réalisations de canal pour toutes les politiques.
4. Sortie : delivery ratio dans le temps autour de chaque événement, temps de récupération par
   politique (mean ± std), CSV bruts + figures.
5. Le script doit reprendre là où il s'est arrêté si Ollama plante (résultats partiels sauvegardés).
6. N'ajuste ni le prompt ni les seuils pour améliorer les résultats après avoir vu les courbes ;
   si tu veux changer le prompt, crée une version nommée (prompt_v2) et garde les deux, en notant
   le changement dans LOG.md.
Décris les résultats tels quels, y compris si l'agent n'améliore rien.
```
**Remarque :** sur CPU, 5 seeds × 5–6 scénarios peut prendre des heures. Lance-le la nuit, c'est normal.

---

## ÉTAPE 9 — Vrais E5 et E6 (branche `feat/e5-e6-real`)

**Prompt pour Antigravity :**
```
Lis AGENTS.md, experiments/e5_agent_cost.py, e6_ablations.py et les traces de l'étape 8.

Tâche (branche feat/e5-e6-real) :
1. E5 : calcule depuis les traces réelles : latence médiane et P95, nombre d'appels LLM par heure
   simulée, taux d'actions invalides, taux de fallback. Compare qwen2.5:3b et qwen2.5:7b
   (ou llama3.2:3b si la RAM est insuffisante pour 7b). Supprime tout code de mock du chemin
   des résultats ; le mock reste uniquement dans tests/.
2. E6 : ablations réelles de l'agent : (a) sans l'outil get_anomaly_report, (b) sans les faits LoRa
   dans le prompt. Même scénarios, mêmes seeds que E4.
3. Discussion de latence : écris dans results/e5_latency_note.md si la latence réelle est
   compatible avec un réseau où les devices émettent toutes les ~10 minutes, avec les chiffres.
4. CSV bruts + figures, mean ± std sur 5 seeds.
Mets à jour LOG.md.
```

---

## ÉTAPE 10 — Baselines manquantes et détecteur (branche `feat/full-baselines`)

**Prompt pour Antigravity :**
```
Lis AGENTS.md, policies/oracle.py, policies/ml_classifier.py, netsim/anomalies.py, docs/PROJECT_SPEC.md
(sections 3.5 Phase 4 et 5, et pitfall 3).

Tâche (branche feat/full-baselines) :
1. Intègre l'oracle (upper bound) et le classifieur ML (HistGradientBoostingClassifier entraîné sur
   labels oracle) dans E3 et E4 avec les mêmes scénarios et seeds. Entraînement et test sur des seeds
   disjointes, pas de fuite de données.
2. Équité : si l'agent reçoit les rapports d'anomalie, le classifieur reçoit aussi le flag d'anomalie.
   Fais aussi tourner une variante du classifieur sans le flag et rapporte les deux.
3. Détecteur : calcule précision et rappel du RollingZScoreDetector contre les événements injectés
   (par type d'événement, par canal), plus le délai de détection. Ajoute Isolation Forest en option
   et compare.
4. Tableau final consolidé (CSV + figure) : PDR, énergie par paquet livré, temps de récupération
   pour toutes les politiques et tous les scénarios, mean ± std sur 5 seeds, avec comparaison
   appariée (analysis/paired.py).
Mets à jour LOG.md.
```

---

## ÉTAPE 11 — README final et rapport IEEE (branche `docs/final`) — avec Sonnet

C'est ici que Sonnet est le plus utile. Colle-lui `results/` (CSV), `LOG.md`, `docs/ASSUMPTIONS.md` et le cahier des charges.

**Prompt pour Sonnet :**
```
Voici le cahier des charges, mon LOG, mes hypothèses et mes CSV finaux.
Rédige la structure et un premier jet du rapport IEEE de 4 à 6 pages :
Introduction, System Model, Proposed Method, Results, Conclusion, plus une section
"Threats to validity" tirée de docs/ASSUMPTIONS.md.
Règles : chaque nombre cité doit venir d'un des CSV que je t'ai donnés, avec le nom du fichier
entre crochets. N'écris aucune affirmation que les données ne soutiennent pas.
Si un résultat est négatif ou inattendu, formule-le honnêtement.
Liste à la fin les affirmations pour lesquelles il te manque une donnée.
```
Puis, pour le README : figure principale en haut, tableau des résultats, commandes pour tout reproduire. Mets à jour ta ligne de CV **seulement** avec des chiffres issus de `results/`.

---

## PROMPT DE RELECTURE (pour Sonnet, après chaque étape)

```
Tu es relecteur critique d'un projet de recherche. Voici le diff de la branche et les CSV/figures
générés. Cherche : (1) bugs et erreurs de formule ; (2) résultats qui semblent trop beaux ou
incohérents avec la théorie ; (3) tout ce qui viole AGENTS.md ; (4) hypothèses non documentées ;
(5) fuites de données ou comparaison inéquitable entre politiques.
Réponds par une liste classée par gravité, avec le fichier et la ligne. N'écris pas de code,
sauf pour illustrer un bug précis. Dis clairement si tu n'as pas assez d'information pour juger.
```

## PROMPT DE FIN DE SESSION (Antigravity)

```
Termine la session : ajoute à LOG.md une entrée datée (commandes lancées, chiffres observés tirés
des CSV, problèmes, prochaine étape), lance pytest, et commit. N'écris aucune conclusion non
supportée par les CSV.
```

---

## Ordre et durée indicative

| Étapes | Contenu | Ordre de grandeur |
|---|---|---|
| 0 – 2 | Hygiène, honnêteté, RNG | 1 à 2 jours |
| 3 – 4 | Doppler paquet, tables PER | ~1 semaine (la plus délicate) |
| 5 – 6 | SNR par device, satellite | ~1 semaine |
| 7 – 9 | Agent, vrais E4–E6 | ~1 semaine + temps de calcul Ollama |
| 10 – 11 | Baselines, détecteur, rapport | ~1 semaine |

Si tu manques de temps : **coupe les extensions (Isolation Forest, 7B, S6), jamais les baselines ni le vrai E4.**
