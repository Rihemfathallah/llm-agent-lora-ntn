# Guide (FR) — démarche, résultats, et ce qu'il te reste à faire

Ce guide est pour toi (et pour préparer l'entretien Mitacs). Les chiffres viennent de
`results/summary.md`, généré par `experiments/summarize.py` à partir des CSV : ne les recopie pas
ailleurs sans relancer le script.

## 1. La démarche en 6 étapes (le fil à raconter au professeur)

1. **Question.** Un petit LLM local, appelé seulement sur événements et qui agit via des outils
   validés, choisit-il mieux les paramètres LoRa (SF, puissance, canaux, lien) qu'ADR et qu'un
   classifieur ML, surtout pendant des événements imprévus ? À quel coût de latence ?
2. **D'abord la physique (E1–E2).** Le Doppler *rate* d'un passage LEO à 600 km fait dériver la
   fréquence pendant le paquet. Modèle au niveau paquet avec deux récepteurs (sans suivi / avec
   suivi de fréquence), validé contre la formule exacte du SER et contre une mesure CubeSat publiée
   (Doroshkin et al., 2019).
3. **Un simulateur réseau honnête (E3).** ALOHA vérifié contre la théorie, contrôle LoRaWAN classe A
   (une commande n'arrive qu'après un uplink reçu), mêmes réalisations du canal pour toutes les
   politiques (comparaisons appariées).
4. **Des baselines qui piègent les conclusions faciles.** Fixed SF7/SF12, ADR, ML entraîné sur un
   oracle, oracle, et surtout **Triggered-Rules** : mêmes déclencheurs et mêmes infos que l'agent,
   mais des règles écrites à la main. C'est ce qui permet de séparer « l'architecture
   événementielle aide » de « le LLM raisonne bien ».
5. **L'agent réel (E4–E6).** Vrai LLM, 5 outils, garde-fous, traces complètes de chaque décision.
6. **Résultats honnêtes et limites.** Les résultats négatifs sont présentés comme des résultats.

## 2. Ce que montrent les résultats (à connaître par cœur)

- **Physique.** Sans suivi de fréquence, SF12 casse dès 50 Hz/s ; avec suivi, au-delà de
  250–300 Hz/s. Au zénith à 600 km le taux est de 252 Hz/s. Le modèle avec suivi prédit l'échec
  de SF12 au zénith sous ~650 km ; Doroshkin l'a mesuré sous ~550 km. Le modèle est donc cohérent
  avec le matériel réel, et un peu pessimiste.
- **Le SF « le plus robuste au sol » (SF12) est le plus fragile en orbite** sans suivi :
  utilisable sur ~39 % du passage seulement.
- **ADR ne réagit pas** dans la durée d'un événement : il n'augmente jamais le SF et décide sur
  20 uplinks (plus de 3 h à un uplink toutes les 10 min).
- **L'architecture événementielle vaut ~10 points.** Triggered-Rules bat ADR de +10,5 (brouillage),
  +15,0 (fin de passage) et +12,9 points (panne de passerelle) de PDR, sur les 5 seeds.
- **Le fading profond est impossible à corriger côté réseau** (classe A : les devices « éteints » ne
  reçoivent jamais la commande). Seul l'oracle s'en sort.
- **L'oracle n'est pas une borne supérieure sous forte charge.** Il perd 7,8 points en surcharge, car
  il met tout le monde en SF7. Le classifieur qui l'imite sous-provisionne (−16 à −20 points sur S1, S2, S3 et S5).
- **Agent Qwen3-0.6B.** Il lit correctement le réseau mais n'agit presque jamais (6 actions sur 287
  décisions) : identique à ADR sur S1–S3. En surcharge (S5), il a mis tous les devices en SF12
  sur 2 seeds : −17,5 ± 23,9 points. L'action était *valide* (dans les bornes) : la validation par
  schéma ne suffit pas.
- **Sans anomalie (E3), l'agent fait aussi du tort sous charge.** Identique à ADR jusqu'à 200 devices,
  il est en dessous sur 7 de ses 30 runs à plus forte charge (pire seed : −15,5 points à N = 800).
  Chaque fois, une fausse alerte du détecteur tombe en même temps qu'une revue périodique, et il met
  tout le monde en SF12. Sur E3 + E4, 15 de ses 16 actions appliquées sont **la même action**.
  C'est un réflexe, pas une réponse à l'événement observé.
- **Ablations (E6).** Sans l'outil d'anomalie, ou sans les faits LoRa dans le prompt, le 0.6B
  n'appelle plus jamais `set_tx_params` (0 appel sur 271 décisions par variante). Les deux
  variantes sont donc identiques à ADR, et elles évitent le dégât de S5. À cette taille, l'ablation
  mesure la propension du modèle à agir, pas la valeur de l'information : à refaire avec le 3B.
- **Coût.** 5,8 s de latence médiane par décision sur 2 threads CPU, 7,7 appels LLM par heure
  simulée. C'est négligeable devant les 10 min entre deux uplinks. Le vrai délai est celui de la
  classe A, pas du LLM.

## 3. Ce qu'il te reste à faire sur ton laptop (la mesure décisive)

Ollama et Hugging Face étaient bloqués dans l'environnement où le projet a été finalisé. Les
résultats d'agent viennent donc de **Qwen3-0.6B via llama.cpp**. Le protocole prévoit **qwen2.5:3b**
(et 7b si tu as 16 Go de RAM). Les scripts tournent tels quels :

```bash
git pull                        # ou copie le dossier
python -m venv .venv && source .venv/bin/activate    # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q             # doit afficher 60 passed

ollama pull qwen2.5:3b
python experiments/e4_anomalies.py --policies LLM-Agent --backend ollama --model qwen2.5:3b
python experiments/e6_ablations.py --backend ollama --model qwen2.5:3b
python experiments/e5_agent_cost.py
python experiments/summarize.py
python report/make_numbers.py   # puis recompile le rapport (Overleaf ou pdflatex)
```

Durée estimée : 25 runs × ~12 décisions × 5–15 s ≈ 30–75 min pour E4 avec un 3B sur un laptop
récent. C'est reprenable : si tu interromps, relance la même commande. Les runs déjà faits sont
sautés et les réponses déjà obtenues sont rejouées depuis le cache.

Les résultats du 3B s'ajoutent à côté de ceux du 0.6B (colonne `model_key`) : `summary.md` affiche
une ligne par modèle. **Ne choisis pas le modèle ou le prompt en fonction du résultat** (règle 3 de
`AGENTS.md`). Si le 3B ne bat pas Triggered-Rules, c'est le résultat.

## 4. Questions d'entretien — réponses basées sur ce que tu as réellement fait

- **Pourquoi SF12 peut-il échouer en LEO alors que le SNR suffit ?** Un paquet SF12 dure 1,3 s.
  Au zénith la fréquence dérive de 252 Hz/s, alors qu'un bin FFT ne fait que 30,5 Hz. Sans suivi
  de fréquence, la fin du paquet est décalée de ~9 bins. Avec suivi, la boucle tient tant que la
  dérive reste sous un demi-bin par symbole (466 Hz/s en théorie, ~250–300 Hz/s mesuré en
  simulation avec bruit).
- **Pourquoi ne pas appeler le LLM à chaque paquet ?** 300 devices × 6 uplinks/h = 1800 décisions/h,
  contre 7,7 appels/h en mode événementiel. Et une commande n'atteint de toute façon le device
  qu'après son prochain uplink.
- **Comment empêchez-vous les actions invalides ou dangereuses ?** Validation par schéma, bornes,
  « read before write », plafond de 6 appels, repli sur ADR, baux qui expirent. Et la partie honnête :
  **ça ne suffit pas**. L'agent a fait une action valide mais nuisible (SF12 partout en surcharge).
  Il faut des garde-fous sur les conséquences : rollback si la perte augmente, déploiement d'abord
  sur un sous-ensemble de devices, ou simulation de l'action dans un jumeau numérique avant de
  l'appliquer. J'ai aussi testé un contrôle de charge prédite, et il ne l'aurait pas bloquée.
- **Comment savez-vous que l'agent apporte quelque chose par rapport à ADR ?** Mêmes scénarios,
  mêmes seeds, mêmes réalisations du canal (différences appariées), ablations, et surtout la
  baseline Triggered-Rules sur les mêmes déclencheurs.
- **Limites du modèle satellite ?** Pas de rotation terrestre, tables TR 38.811 en bande S
  utilisées à 868 MHz, SF orthogonaux, pas de trafic de fond au satellite, voie descendante
  satellite en broadcast supposée.

## 5. Ligne de CV (honnête, avec les chiffres actuels)

> *Built a Doppler-aware hybrid terrestrial–LEO LoRa simulator (packet-level PHY validated against
> CubeSat measurements) and an event-triggered LLM agent with guarded tools; showed that
> event-triggered control gains 10–15 points of delivery ratio over ADR during jamming, pass ends and
> gateway outages, and that a 0.6B agent's valid-but-harmful actions call for consequence-aware
> guardrails (median decision latency 5.8 s on CPU).*

Mets à jour avec les chiffres du 3B quand tu les auras. Tous les nombres doivent venir de
`results/summary.md`.
