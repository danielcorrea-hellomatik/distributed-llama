<!-- paper-url: https://zenodo.org/records/20357376 -->

Un modèle Mixture-of-Experts de 30 milliards de paramètres (Qwen3-30B-A3B) tourne sur **quatre cartes Raspberry Pi 5** (matériel CPU uniquement, sans GPU ni NPU) à **15,143 tok/s en décodage** : bit-exact et **+16,1 % au-dessus du meilleur résultat publiquement documenté** pour ce modèle et cette classe de matériel (13,04 tok/s, b4rtaz #255). Sur un silicium 16 Go identique, le même résultat représente **+15,2 %** par rapport à la version vanilla de `distributed-llama` (13,15 → 15,143 tok/s) — un chiffre sans biais attribuable à notre seul travail logiciel et de configuration ; la comparaison inter-SKU ne comporte elle non plus aucun biais matériel mesurable, puisque la version vanilla sur notre cluster 16 Go atteint 13,15 tok/s, dans le bruit de mesure du record 8 Go. Il s'agit d'une synthèse technique condensée de notre rapport ; l'article complet (chaque tableau, figure et log brut) est à un clic.

## Résultats en un coup d'œil

| Métrique | Valeur |
|---|---|
| Débit en décodage (vs le plafond public) | **15,143 tok/s** |
| Amélioration vs b4rtaz #255 (13,04 tok/s, décodage) | **+16,1 %** |
| Amélioration vs vanilla sur silicium 16 Go identique (sans biais) | **+15,2 %** |
| Débit de service soutenu (prefill inclus) | 14,449 tok/s |
| Time-to-first-token (TTFT) | 557 ms |
| Bande passante DRAM par nœud (soutenue / plafond constructeur) | 11,4 / 17 GB/s |
| Matériel | 4× Pi 5 16GB |
| Contraintes respectées | bit-exact, pas d'overclock, pas de changement de modèle |

```chart
{"type":"bar","xKey":"system","yDomain":[0,16.6],"highlightLast":"#1f9d57","unit":"tok/s",
"series":[{"key":"decode","label":"Décodage tok/s","color":"#9aa3b2"}],
"data":[{"system":"b4rtaz #255 (public record)","decode":13.04},{"system":"This work (clean-room)","decode":15.143}],
"caption":"Même modèle et même classe de matériel (Qwen3-30B-A3B Q40, 4× Raspberry Pi 5), métrique de décodage : une amélioration bit-exact de +16,1 % (n=20)."}
```

## Pourquoi c'est difficile

L'inférence en périphérie (edge) des LLM sur des ordinateurs monocartes grand public est de plus en plus étudiée comme alternative au cloud, respectueuse de la vie privée et à faible coût. Le Raspberry Pi 5 est la carte SBC ARM la plus déployée disposant d'assez de RAM (16 GB) pour héberger des modèles quantisés, et son Gigabit Ethernet permet de former de petits clusters. Mais le Pi 5 n'a **aucun accélérateur utilisable** (le GPU VideoCore VII ne dispose pas de compute shaders généralistes), si bien que l'inférence tourne sur le CPU, où le décodage est **limité par la bande passante mémoire**.

Nous adoptons **distributed-llama** v0.16.5 avec **douze modifications au niveau du code source** développées dans ce travail (huit correctifs du framework plus quatre optimisations bit-exact de kernels et de fusion d'opérations), ainsi qu'un réglage persistant du kernel à l'exécution.

### Cinq contraintes strictes

Chaque optimisation obéit à cinq règles, adoptées comme règles de conception. Elles écartent la plupart des accélérations à perte présentes dans la littérature, ce qui est précisément ce qui rend déployable en production ce qui subsiste, avec **zéro risque de régression de qualité** :

- **Sortie bit-exact** : le SHA-256 des 100 premiers token-ids générés correspond à une référence fixe (`seed=42`, `temperature=0`).
- **Pas de recompilation du kernel** : kernel Pi OS d'origine 6.12.75.
- **Pas de changement de modèle** : Qwen3-30B-A3B Q40 est fixé.
- **Pas d'overclock du CPU** : le silicium reste à la fréquence nominale de 2,4 GHz.
- **Pas de réduction de qualité** : top-k inchangé à 8, pas de sous-quantisation, pas d'élagage d'experts.

## Conception du système

**Par nœud :** Broadcom BCM2712 (4× Cortex-A76 @ 2.4 GHz, ARMv8.2-A) ; 16 GB LPDDR4X @ 4267 MT/s (≈17 GB/s théoriques) ; NVMe via PCIe Gen 2 ; Gigabit Ethernet intégré (latence intra-cluster mesurée 0,226 ms) ; Debian 13, kernel 6.12.75 aarch64 ; aucun accélérateur utilisable. Le cluster est constitué d'un coordinateur **root** (servant aussi l'API HTTP) plus trois **workers**, en Gigabit Ethernet full-duplex, avec synchronisation tensor-parallel par couche de transformer.

**Logiciel :** distributed-llama v0.16.5 + nos correctifs ; modèle Qwen3-30B-A3B Q40 (30B au total, 128 experts, top-k = 8, ~3B actifs/token) ; jemalloc 2 via `LD_PRELOAD` ; un petit proxy Python qui assainit les requêtes des clients en mode OpenAI strict.

## Méthodologie

Nous suivons les conventions de MLPerf Inference v5.1 : **deux runs de warm-up (écartés)**, **n = 20 runs de mesure**, prompt fixe, `temperature = 0`, client unique. Statistiques sous forme de moyenne, médiane, écart-type, IC à 95 % et p50/p90/p99. **Chaque revendication d'amélioration est validée bit-exact** en hachant en SHA-256 la séquence de token-ids générée par rapport à une référence fixe, sauf indication contraire.

## La trajectoire d'optimisation

```chart
{"type":"bar","xKey":"stage","yDomain":[0,16.4],"highlightLast":"#1f9d57","unit":"tok/s","height":440,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":13.04,"label":"plafond 13,04","color":"#d98a00","position":"insideTopLeft"}],
"data":[{"stage":"Llama 3.1 8B (dense)","toks":5.70},{"stage":"Qwen3-30B MoE switch","toks":11.40},{"stage":"max-seq + swap clean","toks":12.71},{"stage":"Stage 8 patches+flags","toks":13.72},{"stage":"Stage 9 TIER-0 sysctls","toks":14.011},{"stage":"Stage 10 SILU·MUL fuse","toks":14.081},{"stage":"Stage 13 true 1-pass","toks":14.27},{"stage":"Stage 14–15 chunk+NIC","toks":14.449},{"stage":"Stage 17 clean-room decode","toks":15.143}],
"caption":"Trajectoire d'optimisation et résultat final en clean-room. Les barres 1–8 indiquent le débit soutenu ; la dernière barre verte est la réplication clean-room #255 de l'étape 17 sur la métrique de décodage. Ligne en pointillés : le plafond public."}
```

### Le changement au plus grand effet : dense → MoE

Le changement au plus grand effet individuel a été la migration de **Llama 3.1 8B** (dense, ~5 GB de poids Q40, tous les paramètres actifs par token) vers **Qwen3-30B-A3B** (MoE, 128 experts, top-k = 8, ~3 GB actifs par token). Le débit limité par la bande passante s'est amélioré de **59 %** (7,18 → 11,4 tok/s) sans aucun autre changement. Le MoE *est* une activation creuse : pour un top-8-sur-128, le ratio d'activation est de 8/128 = 6,25 % des poids d'experts plus les paramètres partagés, et le multiplicateur effectif de bande passante correspond presque exactement à l'accélération observée.

### Huit correctifs du framework

Les correctifs réparent des bugs critiques et débloquent des flags d'optimisation. Les points saillants :

| # | Correctif | Pourquoi c'était important |
|---|---|---|
| 1 | `NnByte` → `NnUint` pour `nBatches` | Débordement de `uint8_t` : `nbatches=256` devenait silencieusement 0 (256 mod 256), déclenchant une assertion de la couche d'embedding. |
| 2 | Forcer `finish_reason` = `stop`/`length` | Un `finish_reason` vide envoyait les clients OpenAI stricts dans des boucles de retry infinies. |
| 3 | `try/catch` autour de `json::parse` | Des corps malformés levaient des exceptions non capturées, faisant planter le daemon par abort-trap (SIGABRT). |
| 6 | `posix_memalign(64, n)` pour les pipes | Le `new[]` par défaut sous ARM64 est aligné sur 16 B ; l'alignement sur ligne de cache (64 B) est requis pour le NEON vectorisé. |
| 7 | TCP `SO_RCVBUF`/`SO_SNDBUF` = 8 MB | Les buffers par défaut de 208 KiB provoquaient un blocage en écriture lors des pics de synchronisation en fin de couche. |

### Deux changements bit-exact contre-intuitifs

Tous deux vont à l'encontre des idées reçues sur la même base de code :

1. **Supprimer le prefetch logiciel (étape 12, +1,05 %).** La boucle interne du matmul NEON+dotprod portait deux appels `__builtin_prefetch`. Parmi cinq variantes balayées, *supprimer* les deux était la seule à aider : le prefetcher matériel du Cortex-A76 (détection de stride sur l'accès séquentiel aux poids) surpasse les indices manuels, qui se disputent les slots d'émission dans le décodeur 4 voies.

2. **Une véritable fusion SILU·MUL en une seule passe (étape 13, +1,5 %).** La fusion précédente était un wrapper à deux appels qui gardait l'intermédiaire résident en mémoire (3 lectures + 2 écritures par élément). Nous l'avons réécrite en une seule boucle NEON conservant les valeurs en registres : 2 lectures + 1 écriture par élément, bit-exactitude préservée (arithmétique identique ; seule la réécriture de l'intermédiaire est supprimée).

![Avant (deux passes) vs après (une seule passe) : la fusion SILU·MUL supprime l'écriture/lecture de l'intermédiaire entre les deux boucles : 2 lectures + 1 écriture par itération au lieu de 3 + 2, bit-exact.](/api/contents/research/0001_19d46624-4ce1-4a02-b01d-db23007f1cc5/images/silu-mul-fusion-fr.svg)

### Taille de chunk réseau et réglage NIC à l'exécution (étapes 14–15)

La boucle all-reduce `writeMany`/`readMany` plafonnait chaque syscall `send()`/`recv()` à 4 KB. Avec les buffers TCP plus grands de l'étape 9, cela représente 4× plus de syscalls que nécessaire ; en élargissant à 16 KB, on a divisé le nombre par quatre (+1,25 %). Un bundle NIC à l'exécution (ring RX agrandi, Receive Flow Steering hors du CPU 0, NAPI différé), persistant comme unité `systemd`, a ajouté +1,99 % au total. Les deux sont bit-exact : seule l'ordonnancement de la NIC change, pas l'arithmétique du modèle.

## Régler le Linux hôte a compté autant que le code

Il est tentant d'attribuer un tel résultat au moteur d'inférence. Sur un silicium 16 GB identique, **la majeure partie de l'accélération déployable est venue d'un réglage du système d'exploitation hôte aussi agressif que celui de l'application**, et chaque levier ci-dessous respecte la contrainte bit-exact, donc aucun ne coûte de qualité de sortie :

- **`SDRAM_BANKLOW=1`** dans l'EEPROM du bootloader : une modification d'entrelacement mémoire au niveau du firmware qui a porté la bande passante de lecture effective de 8,3 à 12,5 GB/s par nœud. **+5,9 %**, bit-exact, sans overclock.
- **`--nthreads 3` au lieu de 4**, libérant un cœur, avec l'IRQ Ethernet et le Receive Packet Steering épinglés sur ce cœur libéré (CPU 3), plus **jemalloc** (arena/tcache réglés). Ensemble **+10,3 %**.
- **Gouverneur CPU `performance`** à 2,4 GHz, et **poids `mlock`-és** (~4,45 GB verrouillés/worker) pour que le modèle ne soit jamais paginé sur disque.
- **Pile réseau / VM (étape 9, TIER 0)** : TCP BBR, `SO_RCVBUF`/`SO_SNDBUF` agrandis, **GRO désactivé** (il ajoute 50–200 µs de pur surcoût aux pics all-reduce de 510 kB en 1 GbE), `busy_poll`, `vm.swappiness=1`. **+2,12 %**.

L'ensemble de l'état est figé dans des unités `systemd`, de sorte qu'un cluster froid démarre directement dans la configuration optimisée : aucune étape manuelle, entièrement reproductible. Le point est général : sur une charge limitée par la bande passante mémoire, **le système d'exploitation n'est pas un substrat neutre.** L'entrelacement mémoire du firmware, le placement cœur/IRQ, l'allocateur, la résidence des pages et la pile réseau font chacun bouger l'aiguille de quelques pourcents qui se composent, et sur le même matériel ils additionnent *plus que ne le font les changements de code au niveau source*.

## Résultats : débit et stabilité

La variance d'un run à l'autre est faible : un coefficient de variation de **0,52 %**. La distribution ci-dessous correspond à l'échantillon n = 20 d'un instantané intermédiaire de l'étape 6 (moyenne **12,708 tok/s**) ; la configuration finale atteint **14,449 tok/s** (meilleur run 14,557, IC à 95 % ±0,038), les étapes ultérieures augmentant le débit sans changer ce comportement de latence.

```chart
{"type":"line","xKey":"run","yDomain":[12.4,13.0],"unit":"tok/s","height":340,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":12.708,"label":"moyenne 12,708","color":"#1f9d57"}],
"data":[{"run":1,"toks":12.85},{"run":2,"toks":12.81},{"run":3,"toks":12.74},{"run":4,"toks":12.74},{"run":5,"toks":12.69},{"run":6,"toks":12.67},{"run":7,"toks":12.72},{"run":8,"toks":12.78},{"run":9,"toks":12.62},{"run":10,"toks":12.68},{"run":11,"toks":12.64},{"run":12,"toks":12.77},{"run":13,"toks":12.58},{"run":14,"toks":12.69},{"run":15,"toks":12.64},{"run":16,"toks":12.71},{"run":17,"toks":12.76},{"run":18,"toks":12.65},{"run":19,"toks":12.69},{"run":20,"toks":12.71}],
"caption":"Débit par run sur 20 runs de mesure (instantané étape 6, warm-ups exclus). Coefficient de variation 0,52 %."}
```

Le time-to-first-token est de **557 ms en moyenne** (p50 545 ms, max 638 ms). Le débit augmente avec la longueur de la réponse jusqu'à une asymptote : les réponses plus courtes sont dominées par le TTFT :

```chart
{"type":"line","xKey":"max_tokens","logX":true,"yDomain":[11.5,13],"unit":"tok/s","height":320,
"series":[{"key":"toks","label":"tok/s soutenus","color":"#F76B1C"}],
"data":[{"max_tokens":50,"toks":11.83},{"max_tokens":100,"toks":12.47},{"max_tokens":200,"toks":12.67},{"max_tokens":400,"toks":12.86},{"max_tokens":800,"toks":12.62}],
"caption":"Débit soutenu vs longueur de réponse (x logarithmique). Converge vers 12,86 tok/s pour les réponses ≥ 400 tokens."}
```

**Le prefill est la limite pratique.** Le taux de prefill reste au-dessus de 15 tok/s jusqu'à des prompts de 2K tokens, mais pour des prompts de 20K tokens (gros prompts système d'agent), le temps de prefill projeté dépasse 20 minutes : la borne supérieure de l'utilisabilité pour des charges agent complètes ici.

Sous charge soutenue, tous les nœuds restent à 54–56 °C avec **zéro throttling** (seuil 85 °C). Le root porte des buffers supplémentaires d'orchestration/service ; les workers conservent >9 GB de marge pour la croissance du cache KV.

```chart
{"type":"bar","xKey":"node","stacked":true,"unit":"GB","height":320,"yDomain":[0,16],
"series":[{"key":"used","label":"Utilisé (modèle + buffers)","color":"#2f6fed"},{"key":"available","label":"Disponible","color":"#A8C68A"}],
"data":[{"node":"rpi-1005 (root)","used":12,"available":3.0},{"node":"rpi-1006","used":6.5,"available":9.4},{"node":"rpi-1007","used":6.3,"available":9.5},{"node":"rpi-1008","used":6.3,"available":9.5}],
"caption":"Utilisation mémoire par nœud pendant une inférence soutenue."}
```

## Où passe le temps : le mur mémoire

Le profilage du PMU ARM (`perf stat`, 60 s d'inférence soutenue) cerne le goulot d'étranglement : **49 % de cycles bloqués sur le backend**, **11,4 sur ~17 GB/s** de DRAM par nœud, IPC 1,74. Les taux de TLB et de mauvaise prédiction de branche inférieurs à 0,1 % confirment que la configuration en pages de 16 KB est déjà optimale ; `objdump` montre 322 produits scalaires NEON `udot`/`sdot` dans la boucle interne, déjà à la limite de l'ARMv8.2-A. Les cœurs restent inactifs en attente de la DRAM, c'est pourquoi chaque expérience côté calcul ne rapporte aucun gain.

```chart
{"type":"bar","xKey":"phase","layout":"horizontal","stacked":true,"unit":"%","height":210,"xDomain":[0,100],
"series":[{"key":"matmul","label":"Matmul Q40 (MoE)","color":"#2f6fed"},{"key":"sync","label":"Barrière de sync","color":"#d98a00"},{"key":"syscalls","label":"Syscalls (send/recv)","color":"#9aa3b2"},{"key":"other","label":"Orchestration & autres","color":"#c7cfdb"}],
"data":[{"phase":"per-token","matmul":56,"sync":17,"syscalls":4.5,"other":22.5}],
"caption":"Répartition du temps mural par token. Le matmul Q40 domine ; la barrière de synchronisation est la seule marge adressable par logiciel."}
```

Le plafond est physique. Sur une échelle logarithmique, le LPDDR4X du Pi 5 est ~16× sous un Apple M4 Pro et ~200× sous un H100 :

```chart
{"type":"bar","xKey":"platform","layout":"horizontal","logX":true,"unit":"GB/s","height":300,
"series":[{"key":"bw","label":"Bande passante mémoire (GB/s)","color":"#1b2a4a"}],
"data":[{"platform":"Pi 5 LPDDR4X","bw":17},{"platform":"Mac M4 Pro","bw":273},{"platform":"RTX 3060","bw":360},{"platform":"Mac M3 Ultra","bw":800},{"platform":"H100 SXM5","bw":3350}],
"caption":"Bande passante mémoire multi-plateformes (échelle logarithmique). C'est le plafond physique que le cluster atteint."}
```

## Le constat de télémétrie

Lors du balayage final en clean-room, nous avons trouvé deux agents de monitoring en arrière-plan tournant sur les quatre nœuds (la « charge de fond inactive » de notre méthodologie antérieure). Sur une charge limitée par la bande passante mémoire, ils ne sont pas gratuits : ils parcourent périodiquement les statistiques mémoire système, consommant la bande passante DRAM dont la phase de décodage a besoin.

| Arrière-plan | n | Décodage (tok/s) | Prefill (tok/s) |
|---|---:|---:|---:|
| Monitoring activé | 10 | 14,397 ± 0,153 | 18,74 |
| **Monitoring désactivé (clean-room)** | 20 | **15,143 ± 0,097** | 18,81 |

Arrêter les agents a augmenté le **décodage de +5,18 %** (les IC ne se chevauchent pas) tout en laissant le **prefill inchangé** : une double dissociation, le décodage est limité par la bande passante, le prefill limité par le calcul est le contrôle négatif. C'est aussi une leçon pratique : l'observabilité co-localisée taxe silencieusement l'inférence limitée par la mémoire, et un nœud de production sous monitoring sous-performe un benchmark clean-room de plusieurs pourcents.

## Étape 16 : une barrière WFE/SEV

La barrière inter-étapes faisait du busy-spin sur un atomique avec un indice ARM `yield`, de sorte que trois threads en attente relisaient continuellement une ligne de cache que le thread en progression écrit : du trafic de cohérence qui entre en concurrence avec le thread pilotant les E/S réseau. Nous avons remplacé le spin par le mécanisme d'événements ARMv8 : les threads en attente émettent `wfe` (attente d'événement à basse consommation) et le thread en progression diffuse `sev`. Purement de la signalisation, donc bit-exact par construction ; cela réduit aussi la consommation et la chaleur sur les cœurs en attente.

```chart
{"type":"bar","xKey":"build","unit":"tok/s","yDomain":[14.0,14.6],"highlightLast":"#1f9d57","errorKey":"ci","height":320,
"series":[{"key":"mean","label":"tok/s moyens","color":"#9aa3b2"}],
"data":[{"build":"yield spin (Stage 15)","mean":14.261,"ci":0.034},{"build":"WFE/SEV (Stage 16)","mean":14.329,"ci":0.042}],
"caption":"Barrière WFE/SEV de l'étape 16 vs yield-spin, A/B apparié à froid sur la même session (n=40/bras). +0,48 %, bit-exact, Welch t=2,45, p=0,014. Barres d'erreur : IC à 95 %."}
```

## Ce qui n'a pas marché

Conformément aux normes de reproductibilité, nous documentons chaque tentative en intention de traiter. Une sélection des 26 impasses cataloguées :

| Configuration | Cause racine de l'échec |
|---|---|
| Llama 3.3 70B Q40 sur 4× Pi 5 | 38 GB de poids forcent un swap agressif ; 0,15 tok/s sous thrashing. |
| Framework EXO | Dépend d'Apple MLX (Metal + Neural Engine + UMA) ; ne se compile pas sous ARM Linux. |
| prima.cpp | La découverte de topologie ZMQ se bloque >10 min sur le Pi 5 ; n'amorce jamais. |
| llama.cpp + RPC | Régression de 25× vs nœud unique (parallélisme de pipeline dominé par le surcoût réseau). |
| Repack ARM I8MM / SMMLA | Le Cortex-A76 **n'implémente pas** I8MM (`grep -c i8mm /proc/cpuinfo` = 0) ; c'est une fonctionnalité A78+. |
| Transparent hugepages, jumbo frames, NUMA interleave, PGO | Testés et révertis : neutres ou nuisibles sur cette charge limitée par la DRAM. |
| Prefetch logiciel (tiered PLDL2KEEP+L1) | −0,47 % ; le prefetcher matériel gagne déjà (voir étape 12). |

La raison récurrente d'inapplicabilité est structurelle : les techniques modernes au plus fort levier nécessitent un kernel plus récent, une fonctionnalité matérielle qui manque au A76, ou un redémarrage/recompilation que nous avons exclus, ce qui est précisément ce qui rend les gains bit-exact survivants déployables sur du matériel d'origine.

## Réserves et limitations

- **Le chiffre phare de +16,1 % mêle plusieurs effets** : notre cluster 16 GB optimisé vs le cluster 8 GB vanilla de b4rtaz (et `--nthreads 3` vs 4). Sur **un silicium 16 GB identique**, vanilla → notre build → entièrement réglé donne 13,15 → 13,88 → **15,143** (+15,2 %), dont seulement **+5,6 % est attribuable au code source** ; le reste tient au choix du modèle, au flag firmware, à la configuration à l'exécution et à un environnement propre. Le tableau 22 de l'article décompose cela levier par levier.
- **« Bit-exact » signifie SHA-256 vs notre propre build canonique** (les flags incluent `-ffast-math`), pas une stricte conformité IEEE-754. L'énergie (J/token) est la seule métrique edge standard pas encore mesurée. Tous les chiffres concernent un modèle, un site, un lot de silicium.

## Conclusion et travaux futurs

Le chiffre phare importe moins que la méthode derrière lui. **L'essentiel de l'accélération n'est venu ni du modèle ni de kernels inédits, mais du traitement du système d'exploitation comme une partie réglable de la pile d'inférence** : entrelacement mémoire du firmware, placement des cœurs et des IRQ, allocateur, résidence des pages, et une pile réseau réglée pour le trafic all-reduce à faible latence et en rafales entre les nœuds. Rien de tout cela n'a touché au matériel, à l'horloge ou à la qualité de sortie du modèle. Voilà la leçon transférable : **cette catégorie d'optimisation au niveau de l'OS et du réseau s'applique à tout système d'inférence CPU distribué, pas seulement à distributed-llama.**

Nous avons atteint le mur mémoire que présente le Pi 5 : les octets déplacés par token sont fixés par le modèle et sa quantisation, et aucune réduction bit-exact supplémentaire de ce trafic n'est disponible. Le levier logiciel restant, le parallélisme tensoriel asynchrone (Async Tensor Parallelism), recouvre la communication par le calcul plutôt que de déplacer moins d'octets.

**La suite.** Nous prévoyons de publier une **image Linux réglée pour l'inférence distribuée** : un OS pré-configuré pour une faible latence inter-nœuds sur des clusters de type dllama, afin que ces gains soient disponibles d'emblée, **sans changer de matériel, sans overclocker, et sans abaisser la qualité du modèle.** Nous prévoyons aussi d'effectuer la même passe au niveau de l'OS et du réseau sur **d'autres modèles open-source**, pour tester jusqu'où l'approche se généralise au-delà de Qwen3-MoE.

## Ressources

- **Article, code et données brutes :** [github.com/hellomatik-org/distributed-llama](https://github.com/hellomatik-org/distributed-llama/tree/pi5-cluster) (branche `pi5-cluster`, `paper/`).
- **La référence publique à laquelle nous nous comparons :** [b4rtaz/distributed-llama discussion #255](https://github.com/b4rtaz/distributed-llama/discussions/255).
