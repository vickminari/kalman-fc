# 015 — Falsos positivos de pele: medidos, e a cor da bola (`cool`) no score combinado

**Data:** 22/11/2026

**Código:**
- `experiments/skin_check.py` (novo);
- `src/detector.py` (`cool_fraction`, `Candidate.cool`, `DetectorParams.cool_h/cool_v_min`);
- `src/scoring.py`;
- `experiments/candidate_features.py` (atributo `cool` e ablação);
- `experiments/replay_tracker.py` (cache invalidado quando o modelo do score muda).

**Testes:** `tests/test_skin_check.py` (3), `tests/test_detector.py` (+3). 56 no total.

## Contexto

- **A hipótese do [010](010-diagnostico-do-kalman-no-video.md):** pés e mãos aparecem como falsos positivos porque, sob os refletores, a pele clara tem saturação baixa e brilho alto, parecida com o branco da bola (S ≈ 18, V ≈ 216).
- **O plano (Tarefa 4):** medir a cor dos candidatos falsos contra as bolas e, se separar, acrescentar um atributo ao score combinado ([007](007-candidatos-falsos-e-detectores.md)).

## Medição (`experiments/skin_check.py`)

**Dados:** os candidatos do detector HSV em todos os quadros marcados (`outputs/experiments/<vídeo>/candidates.csv`, rótulo bola = a ≤ 10 px da marcação).

**O que mede em cada candidato:**
- a mediana de H, S e V no disco;
- **`skin`:** a fração de pixels "cor de pele" no disco (H ∈ [0, 25] ∪ [165, 180], S ≥ 25, V ≥ 60);
- **`skin_ring`:** a mesma fração num anel em volta (1,3 r a 2,5 r);
- **`cool`:** a fração de pixels de matiz fria no disco (H ∈ [85, 150], V ≥ 60, qualquer S).

**Recorte de interesse:** os candidatos que **passam no limiar do score combinado** atual são os que competem com a bola na trilha.

AUC (bola × falso; 1 = o atributo alto é sempre a bola; 0 = o atributo baixo é sempre a bola):

| | Lateral, todos | Lateral, que passam no score | Centro, todos | Centro, que passam no score |
|---|---|---|---|---|
| Candidatos (falsos / bolas) | 4851 / 353 | **83 / 329** | 2461 / 435 | **85 / 408** |
| H mediano | 0,966 | 0,964 | 0,940 | 0,963 |
| S mediano | 0,151 | 0,080 | 0,217 | 0,271 |
| `skin` | 0,698 | **0,167** | 0,731 | 0,530 |
| `skin_ring` | 0,667 | 0,198 | 0,652 | 0,407 |
| **`cool`** | **0,976** | **0,932** | **0,987** | **0,982** |

Medianas entre os que passam no score:

| | Bola (lat / cen) | Falso (lat / cen) |
|---|---|---|
| H | 122 / 105 | 18 / 24 |
| S | 15 / 14 | 35 / 22 |
| `skin` | 0,13 / 0,08 | **0,78** / 0,07 |
| `cool` | **0,42 / 0,39** | **0,00 / 0,00** |

**Mosaicos** (`outputs/experiments/<vídeo>/skin_mosaic.png`, 40 falsos de maior score + 10 bolas):

- **No lateral, a hipótese se confirma:** quase todos os falsos de score alto são **pés, mãos e pernas** (70–98% de pele). Alguns "falsos" com 0–10% de pele são a própria bola grande perto da câmera, com o centro a mais de 10 px da marcação.
- **No centro, os falsos são outros:** sobretudo **calções e a parede ou grade branca**, e alguns pés. Por isso `skin` quase não separa ali.
- **As bolas têm 2–42% de "pele"** por causa da estampa vermelha e rosa. Isso explica a sobreposição.
- **O que separa nos dois vídeos é a matiz da bola.** O branco dela, sob os refletores, puxa para o azul, e a estampa é azul (H ≈ 100–120). Pele, calção, parede e grama ficam em matizes quentes (H < 60).

Comando: `python -m experiments.skin_check videos/<vídeo>.MOV`.

## Reajuste do score com os atributos de cor

Regressão logística igual à do [007](007-candidatos-falsos-e-detectores.md), com a mesma divisão temporal (treino = 1ª metade dos quadros marcados), sobre o `skin_stats.csv`. Métricas no **teste**:

| | Lateral: AUC / melhor = bola / recall / falsos por quadro | Centro: AUC / melhor = bola / recall / falsos por quadro |
|---|---|---|
| Score atual (7 atributos) | 0,9907 / 98,1% / 91,1% / 0,138 | 0,9914 / 97,9% / 92,9% / 0,105 |
| + `skin` | 0,9950 / 98,7% / 93,7% / 0,098 | 0,9921 / 98,3% / 92,9% / 0,094 |
| + `skin` + `skin_ring` | 0,9946 / 98,7% / 93,0% / 0,098 | 0,9919 / 97,9% / 92,9% / 0,096 |
| **+ `cool`** | **0,9980 / 99,4% / 96,8% / 0,040** | **0,9983 / 99,6%** / 87,9% / **0,003** |
| + `skin` + `cool` | 0,9981 / 99,4% / 96,8% / 0,040 | 0,9983 / 99,6% / 87,9% / 0,003 |

(recall e falsos por quadro no limiar que mantém 95% das bolas do treino)

**Entre vídeos** (treina em um vídeo inteiro, testa na metade de teste do outro; é o caso de um vídeo sem marcações):

| | Centro → lateral | Lateral → centro |
|---|---|---|
| Score atual | AUC 0,986; melhor = bola 94,9%; recall 96,8%; falsos 0,40/quadro | 0,969; 90,0%; 84,1%; 0,149 |
| + `skin` | 0,989; 96,2%; 94,3%; 0,32 | 0,961; 89,1%; 84,1%; 0,132 |
| **+ `cool`** | **0,997; 100%; 94,3%; 0,058** | **0,988; 97,5%; 87,9%; 0,014** |

## Decisão

- **Novo atributo `cool`** (fração de matiz fria no disco do candidato), calculado no detector (`cool_fraction`) e usado pelo score combinado. O detector não filtra por ele.
- **`skin` não entra no modelo:** os falsos do lateral são mesmo pele, mas a cor da bola separa melhor nos dois vídeos e também pega calções e parede. `skin` ainda piora entre vídeos (AUC 0,969 → 0,961). A medida fica registrada aqui.
- **Os modelos `data/models/ball_scorer_campo_{lateral,centro}.json` são retreinados** com `cool` (`python -m experiments.candidate_features videos/<vídeo>.MOV`). Os anteriores estão guardados em `outputs/models_antes_cool/`.

## Alternativas

- **`skin` no score:** ajuda no lateral (falsos por quadro 0,138 → 0,098), quase nada no centro e piora entre vídeos. Descartado em favor de `cool`.
- **`skin_ring` (pele em volta, "pé perto"):** não acrescenta nada a `skin`. Além disso, a bola no pé é justamente o caso que interessa.
- **Atributo de "vizinhança de jogador"** (candidato colado numa região grande que não é grama): não testado. Com `cool`, os falsos que sobram são poucos (0,04 e 0,003 por quadro), e o atributo puniria a bola no pé.
- **Filtrar no detector por cor** (cortar candidatos sem matiz fria): descartado. Uma bola borrada ou muito pequena pode ter pouca cor, e a regressão já pesa o atributo junto com os outros.

## Limitação

`cool` é uma propriedade **desta bola sob esta luz**: branco frio com a estampa azul. Com outra bola (por exemplo, uma laranja numa gravação limpa) ou outra iluminação, o atributo precisa ser redefinido e o score retreinado. É a mesma situação do prior de tamanho, que depende da câmera. Os dois vídeos de teste foram gravados na mesma noite, então o teste entre vídeos não prova generalização para outra luz.

## Evidência: modelos retreinados e efeito na trilha

**Retreino** (`python -m experiments.candidate_features videos/<vídeo>.MOV`, metade de teste):

| | Lateral antes → com `cool` | Centro antes → com `cool` |
|---|---|---|
| AUC do score combinado | 0,991 → **0,998** | 0,991 → **0,998** |
| Melhor candidato = bola (quadros com a bola visível) | 65,4% → 66,2% | 66,3% → 67,4% |
| No limiar: recall / falsos por quadro | 60,8% / 0,14 → **64,6% / 0,04** | 62,9% / 0,10 → 59,5% / **0,00** |
| Peso de `cool` (atributos padronizados) | **2,45, o maior** (`size_fit` 1,90; `lss_sim` 1,52) | **2,64, o 2º maior** (depois de `size_fit`, 3,05; `motion` 1,50) |

Ablação do lateral ("melhor = bola" / falsos por quadro): atributos do detector 32,9% / 4,46 → **com `cool`, 63,3% / 0,77**. Só a cor já leva os atributos do detector quase ao nível de "movimento + LSS + tamanho" (65,4% / 0,14). Com tudo junto: 66,2% / 0,04.

**Trilha** (padrões do [014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md); `python -m experiments.replay_tracker videos/<v>.MOV [--metric]`, confirmado no pipeline completo, ver o [006](006-estado-atual-e-proximos-passos.md)):

| | Lateral F1 | Lateral F2 | Centro F1 | Centro F2 |
|---|---|---|---|---|
| Na bola (teste): sem → com `cool` | 65,0 → **71,7%** | 68,8 → **73,8%** | 67,7 → 68,3% | 71,1 → 71,1% |
| Na bola (vídeo todo) | 74,3 → **77,8%** | 75,1 → **78,6%** | 69,4 → **73,6%** | 74,0 → **76,8%** |
| Aceitas com a bola invisível | 11 → **5** | 13 → **5** | 9 → **0** | 10 → **0** |
| Aceitas na bola / com a bola visível | 305/320 → 329/336 | 306/323 → 328/335 | 355/369 → 375/378 | 384/395 → 389/391 |
| Saltos > 20 px | 110 → **46** | 144 → **60** | 28 → 17 | 85 → 66 |
| Trilhas iniciadas | 58 → **40** | 55 → **37** | 53 → **31** | 43 → **23** |
| Erro da predição no escuro (teste) | 23 → 17 px | 19 → 15 px | 26 → 10 px | 22 → 8 px |
| Velocidade p95 | 32 → 26 km/h | 36 → 23 km/h | 37 → 28 km/h | 33 → 27 km/h |
| Escuro recuperado de 10–30 quadros | 13 → 11 | 15 → 13 | 12 → 6 | 12 → 8 |

**Leitura:**

- **A cor tira os falsos da associação:** no centro, nenhuma medição é aceita com a bola invisível; as trilhas caem pela metade.
- **No centro, o "na bola" do teste quase não muda** (68,3 e 71,1%), mas o vídeo todo sobe 3–4 pontos. No teste do centro, o gargalo é o recall do detector (a bola não está entre os candidatos), não os falsos.
- **Menos episódios no escuro de 10–30 quadros recuperados:** com menos falsos aceitos, há menos interrupções e menos "recuperações". Para o E3 continuam 6–13 oclusões naturais de 10–30 quadros atravessadas por vídeo e filtro.
