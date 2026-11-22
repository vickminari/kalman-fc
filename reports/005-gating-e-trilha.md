# 005 — Gating, associação e lógica de trilha

**Data:** 21/11/2026 · **Código:** `src/gating.py`, `src/tracker.py`, `src/main.py` · **Testes:** `tests/test_tracker.py`

## Contexto

O detector entrega ~6 candidatos por quadro, a bola entre eles em ~69% dos quadros com ela visível ([002](002-detector.md)). O enunciado exige:

- uma região de busca derivada da covariância prevista, para descartar medições espúrias;
- seguir só com a predição por um número máximo de quadros e retomar a correção quando a medição volta.

## Decisão

**Gating** (`src/gating.py`):

- d² = (z − ẑ)ᵀ S⁻¹ (z − ẑ), com ẑ = M x⁻ e S = M P⁻ Mᵀ + Σm.
- Aceita se d² ≤ χ²₂(p). Padrão p = 0,99, limiar 9,21; também há 0,95 → 5,99.
- **Por que χ² com 2 graus de liberdade:** se a medição vem da bola, d² segue χ²(2), então p é a probabilidade de aceitar a bola verdadeira.
- A região de busca é a elipse {d² = limiar}, que cresce sozinha no escuro (P⁻ aumenta).
- **Teto de raio** `max_gate_px = 60`, no eixo maior: no escuro prolongado, a elipse não pode "engolir" metade do campo.

**Associação:** entre os candidatos dentro da região, fica o de menor custo **d² − w·ln(score)**, com w = 2. A distância de Mahalanobis manda, e a aparência desempata entre a bola e um pé próximo.

**Estados da trilha** (`src/tracker.py`):

- **BUSCANDO → TENTATIVA:** o candidato de maior score, fora da desordem estática, inicializa o filtro.
- **TENTATIVA → RASTREANDO:** depois de 3 medições aceitas. Uma trilha tentativa morre com mais de 2 quadros sem medição.
- **RASTREANDO ↔ NO_ESCURO:** sem medição aceita, o filtro só prediz. Com mais de **20 quadros no escuro** (~0,67 s), a trilha é perdida e volta a BUSCANDO.

**Desordem estática** (aproveita a estabilização):

- **Mapa:** média móvel (τ = 3 s) da ocupação de células de 6 px por candidatos, nas coordenadas da referência, **sem contar a medição aceita pela trilha** (para a bola parada não virar desordem). A ocupação é somada na vizinhança 3×3, porque um objeto parado na borda de células divide a ocupação.
- **Uso na inicialização:** pontos com ocupação > 0,5 perdem prioridade.
- **Trilha presa:** uma trilha **parada** (posição dentro de um raio de 4 px) há mais de 4 s, com outro candidato disponível, é descartada (provavelmente está presa numa folha) e o lugar é marcado como desordem.

**Parâmetros padrão:**

| Parâmetro | Valor |
|---|---|
| σa | 4000 px/s² |
| σm | 1,5 px |
| σv0 | 300 px/s |
| gate | 0,99 |
| max_dark | 20 |
| confirm_hits | 3 |
| max_gate_px | 60 |
| score_weight | 2 |

## Alternativas e experimentos

**Banco de teste:** primeiros 30 s do vídeo lateral (150 quadros marcados com a bola visível; recall do detector nesse trecho = 80%). Métrica: % dos quadros com a bola visível em que a estimativa fica a ≤ 10 px da marcação ("na bola"). Script de varredura: `python -m experiments.sweep_tracker videos/campo_lateral.MOV --end 30`, que reproduz a linha σa = 4000 / gate 0,99 da tabela v2 (64,0%). As tabelas v1 e "salvaguardas" foram geradas com versões anteriores de `src/tracker.py`; os valores de v1 estão nos padrões descritos acima.

**v1** (vizinho mais próximo puro, teto 120 px, max_dark 30, "parada" por velocidade < 15 px/s):

| σa (px/s²) | 800 | 2000 | 4000 | 8000 | 15000 |
|---|---|---|---|---|---|
| gate 0,99 | 44,0% | 45,3% | **58,0%** | 40,7% | 41,3% |
| gate 0,999 | 41,3% | 44,0% | 50,7% | 43,3% | 38,7% |

Com σa = 800 o filtro passava **45,8% do tempo no escuro**: chutes mudam a velocidade bruscamente e a bola saía da região de busca.

**Problemas de v1, vistos na tabela quadro a quadro:**

1. **Bola parada descartada.** No começo do vídeo a bola fica ~2 s parada. A regra de "trilha presa em objeto estático" a tratava como folha e trocava de alvo (quadro 65).
2. **Região de busca grande demais.** No escuro, a elipse crescia até 120 px e aceitava um pé. A velocidade estimada ficava absurda (~800 px/s) e a predição saía da imagem.
3. **Aparência ignorada.** A associação só olhava a distância.

**v2 (atual):**

- associação com score;
- teto de 60 px;
- max_dark 20;
- mapa de desordem sem a medição aceita;
- "parada" por deslocamento (raio de 4 px) e não por velocidade. Com σa alto, o ruído da medição vira ±15 px/s de velocidade estimada, então o critério por velocidade não funcionava.

| σa (px/s²) | 800 | 2000 | 4000 | 8000 | 15000 |
|---|---|---|---|---|---|
| gate 0,99 | 49,3% | 52,0% | **64,0%** | 46,0% | 40,7% |
| gate 0,999 | 18,0% | 58,7% | 46,0% | 42,0% | 42,0% |

**Salvaguardas físicas testadas e descartadas:**

| Variante (v2 +) | σa = 2000 | σa = 4000 | σa = 8000 |
|---|---|---|---|
| amortecimento 0,9/quadro no escuro + velocidade máx. 700 px/s | 49,3% | 35,3% | 32,0% |
| só velocidade máx. 700 px/s | 52,0% | 35,3% | 30,7% |
| nenhuma (atual) | 52,0% | **64,0%** | 46,0% |

O amortecimento ainda **quebra a predição durante oclusões** (falhou no teste de oclusão de 20 quadros), que é requisito obrigatório. As duas salvaguardas ficaram nos parâmetros (`dark_damping = 1.0`, `max_speed = 1e9`), desligadas.

**Observação importante:** os números variam de forma "caótica": uma única associação errada leva a trilha para outro objeto e muda o resultado em 10–20 pontos. Com só 150 quadros, diferenças pequenas entre configurações não são significativas.

## Evidência no vídeo inteiro

Comando: `python -m src.main videos/campo_lateral.MOV --save-video`, parâmetros padrão. Lateral, 810 quadros marcados, 510 com a bola visível:

| Métrica | Valor |
|---|---|
| Estimativa na bola (≤ 10 px) | **30,0%** |
| Medição aceita na bola | 26,1% |
| Medições aceitas com a bola **não visível** | 209 quadros |
| Erro da estimativa quando na bola (média / mediana / p90) | 2,65 / 2,39 / 4,03 px |
| Erro da medição quando na bola (média / mediana) | 2,55 / 2,39 px |
| Erro da predição no escuro (mediana) | 252 px (a trilha geralmente já estava no objeto errado) |
| Estados | RASTREANDO 63,7%, NO_ESCURO 31,7%, TENTATIVA 3,4%, BUSCANDO 1,2% |
| Trilhas iniciadas | 52 |
| Velocidade do pipeline | ~105 fps (estabilização + detector + trilha) |

No vídeo do centro: ~105 fps e 41 trilhas iniciadas. Ainda não há marcações para ele.

## Consequências

- **A trilha está abaixo do baseline trivial:** fica na bola em 30% dos quadros, contra **38%** de simplesmente pegar o melhor candidato do detector. O filtro está correto (validado); o gargalo é a **associação numa cena com muitos falsos positivos**. Quando a bola some (sai do quadro, de propósito, nos testes) ou leva um chute, a trilha prende num pé, numa folha ou na base da rede.
- **Mexer só em parâmetros do filtro não resolve.** O próximo passo é reduzir os candidatos falsos antes do filtro ([006](006-estado-atual-e-proximos-passos.md)).
- **Experimentos E2 e E3** (três σa, oclusão proposital) devem ser feitos **depois** de a associação estar estável, senão o efeito de σa fica mascarado pelas trocas de alvo.

## Revisão (21/11) — score combinado na entrada da trilha

O gargalo da associação foi atacado **antes** da trilha: os candidatos passam por um score combinado e os abaixo do limiar são descartados (ver [007](007-candidatos-falsos-e-detectores.md)). A trilha e o filtro não mudaram.

No **teste** (quadros ≥ 2012, não usados no ajuste do score):

| | Antes | Depois |
|---|---|---|
| Estimativa na bola | 14,8% | **63,7%** |
| Medições aceitas com a bola invisível | 99 | **5** |
| Erro da predição no escuro (mediana) | 256 px | **38 px** |

Os parâmetros da trilha (σa = 4000, gate 0,99, max_dark 20, ...) **não foram reajustados** depois dessa mudança. A varredura de σa e o E2 devem ser refeitos com `--scorer`.
