# 016 — E1: trajetória bruta do detector × trajetória filtrada

**Data:** 23/11/2026

**Código:** `experiments/e1_raw_vs_filtered.py` (novo).

**Testes:** `tests/test_e1_e3.py` (3 do E1: detector bruto, posição da bola interpolada, tremor e saltos). 66 no total.

## Contexto

- **O E1 é obrigatório:** trajetória bruta do detector × trajetória filtrada, quantificada (erro contra a marcação, variância da posição, atraso).
- **O [014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md) já mediu medição × estimativa dentro de cada σa**, mas sem o detector sozinho e sem fechar as medidas no formato do relatório.
- **Parâmetros:** os padrões atuais ([006](006-estado-atual-e-proximos-passos.md)): F1 σa = 2000 px/s² e F2 σa = 40 m/s², com manobra, σm = 1,5 px, gate 0,99, `max_dark` 20, score combinado com `cool`.

## Método

**Três trajetórias, no mesmo vídeo e sobre os mesmos candidatos** (cache do `replay_tracker`, [013](013-limites-fisicos-no-f2.md)):

- **detector bruto:** em cada quadro, o candidato de **maior score combinado**, sem filtro e sem gating. É o que se teria usando só o "sensor". O sensor já inclui o score: sem ele, o HSV dá ~6 falsos por quadro ([007](007-candidatos-falsos-e-detectores.md)) e a comparação seria trivial;
- **medição aceita:** o candidato que o gating do filtro aceitou, a medição crua que entra na correção;
- **estimativa filtrada:** o x̂ do Kalman, que inclui a predição nos quadros no escuro.

**Medidas** (px a 640; erros no quadro atual, como `src/evaluation.py`; saltos, tremor e σ local no quadro de referência estabilizado):

| Medida | Definição |
|---|---|
| Cobertura | % dos quadros marcados com a bola visível em que a trajetória tem um ponto |
| Na bola | % desses quadros com o ponto a ≤ 10 px da marcação |
| Erro | média e mediana da distância à marcação, nos quadros "na bola" |
| Saídas com a bola invisível | quadros marcados como bola invisível em que o detector devolve um ponto ou o filtro aceita uma medição (falsos positivos) |
| Saltos | deslocamentos > 20 px entre quadros consecutivos (troca de alvo) |
| Tremor | norma da 2ª diferença da posição (mediana e p90), em trechos de 3 quadros seguidos |
| σ local ("variância da posição") | desvio robusto (MAD) do resíduo de uma parábola em janelas de 7 quadros, como no `experiments/sigma_m.py` (revisão do [004](004-filtro-de-kalman.md)) |
| Atraso | deslocamento s (quadros) que minimiza o erro mediano entre a trajetória em f + s e a marcação em f (`lag_frames` do [014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md)) |

Medição e estimativa têm o tremor e o σ local medidos **nos mesmos quadros** (trechos com medição). O detector bruto é medido nos próprios trechos.

Comandos:

```bash
python -m experiments.e1_raw_vs_filtered videos/campo_lateral.MOV
python -m experiments.e1_raw_vs_filtered videos/campo_lateral.MOV --metric
```

(idem para `campo_centro`; tabelas em `outputs/experiments/<vídeo>/e1_F1.csv` e `e1_F2.csv`, detalhes em `.json`)

## Evidência

### F1, vídeo todo (entre parênteses, "na bola" na metade de teste)

| | Detector bruto | Medição aceita | **Estimativa filtrada** |
|---|---|---|---|
| **Lateral:** cobertura | 68,6% | 65,9% | **91,0%** |
| Na bola | 66,3% (64,6) | 64,5% (62,9) | **77,8% (71,7)** |
| Erro na bola, média / mediana | 2,73 / 2,38 px | 2,71 / 2,38 | 2,96 / 2,51 |
| Saídas com a bola invisível | 6 de 300 | 5 de 300 | — |
| Saltos > 20 px | 37 | 3 | 46 |
| Tremor, mediana / p90 | 0,72 / 2,53 px | 0,68 / 2,35 | **0,54 / 1,84** |
| σ local | 0,22 px | 0,21 | **0,16** |
| Atraso | 0,12 quadro | 0,11 | 0,21 |
| **Centro:** cobertura | 66,4% | 63,1% | **85,8%** |
| Na bola | 66,1% (59,5) | 62,6% (55,8) | **73,6% (68,3)** |
| Erro na bola, média / mediana | 3,20 / 2,98 px | 3,20 / 2,98 | 3,40 / 3,07 |
| Saídas com a bola invisível | 0 de 128 | 0 de 128 | — |
| Saltos > 20 px | 7 | 0 | 17 |
| Tremor, mediana / p90 | 1,02 / 2,87 px | 1,00 / 2,75 | **0,77 / 2,22** |
| σ local | 0,37 px | 0,37 | **0,28** |
| Atraso | 0,14 quadro | 0,15 | 0,06 |

**Medição × estimativa nos mesmos quadros** (medição aceita na bola): erro mediano 2,38 × 2,39 px no lateral (329 quadros) e 2,98 × 2,99 px no centro (375).

### F2 (metros), vídeo todo

| | Lateral: medição → estimativa | Centro: medição → estimativa |
|---|---|---|
| Cobertura | 65,7 → **91,2%** | 65,3 → **86,1%** |
| Na bola | 64,3 → **78,6%** (teste 73,8) | 64,9 → **76,8%** (teste 71,1) |
| Tremor, mediana | 0,69 → **0,49 px** | 1,01 → **0,74 px** |
| σ local | 0,21 → **0,16 px** | 0,37 → **0,29 px** |
| Saltos > 20 px | 10 → 60 | 4 → 66 |

Os números da estimativa batem com a referência do [006](006-estado-atual-e-proximos-passos.md) (mesmos parâmetros, mesmo cache).

### Gráficos

`outputs/experiments/<vídeo>/e1_<F1|F2>_trecho_<quadro>.png`: x(t), y(t), plano da imagem e distância à bola, no trecho de 90 quadros em que o detector bruto mais deixa de estar na bola (sem candidato ou candidato errado) enquanto a estimativa fica nela. Exemplo para o relatório: `campo_lateral/e1_F1_trecho_660.png`. O detector some várias vezes por 3–8 quadros com a bola andando; a predição (vermelho) atravessa os buracos a menos de 10 px da bola.

## Leitura

- **O ganho do filtro sobre o detector é preencher as falhas.** A cobertura sobe 20–23 pontos e o "na bola" 7–12 pontos (lateral 66,3 → 77,8%, centro 66,1 → 73,6%). São os quadros em que o detector não acha a bola e a predição segue a menos de 10 px dela.
- **A suavização é real, mas pequena, e está no lugar certo.** O tremor cai ~25% (0,72 → 0,54 px no lateral, 1,02 → 0,77 no centro), assim como o σ local (0,22 → 0,16 px; 0,37 → 0,28). É o esperado para K ≈ 0,81 em regime ([011](011-regua-de-diagnostico-da-trilha.md)), com o filtro no lado "peixinho-dourado" moderado do [014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md). O F2 suaviza mais (0,49 px no lateral), porque no fundo do campo o Σm em metros é maior e o K cai.
- **O erro contra a marcação não separa as trajetórias** (2,38 × 2,39 px nos mesmos quadros). O ruído do detector (σ local 0,2–0,4 px) é muito menor que a diferença entre o centro detectado e o ponto marcado à mão (~2,4–3 px). Parte dela é sistemática: a medição fica, na mediana, 1,7 px (lateral) e 2,2 px (centro) **acima** da marcação, provavelmente porque o detector pega a parte de cima da bola, mais iluminada pelos refletores, e quem marca clica no centro. A suavização se mede pelo tremor e pelo σ local, não pelo erro.
- **O modelo de velocidade constante não atrasa** com a bola em velocidade constante (0,1–0,2 quadro, igual para detector e estimativa). O atraso aparece como perda da bola nos chutes, que é o E2 ([014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md)).
- **O gating tira as trocas de alvo da medição:** 37 saltos no detector bruto contra 3 na medição aceita (lateral F1). Os saltos da estimativa (46) são de outro tipo: retomadas depois do escuro e trilhas novas, classificadas na régua do [011](011-regua-de-diagnostico-da-trilha.md). Eles aparecem porque a estimativa existe em todo quadro, inclusive quando o detector não tem nada.
- **Com o score combinado, quase não há falsos positivos com a bola invisível** no próprio detector (6 de 300 quadros no lateral, 0 de 128 no centro). Hoje o papel do filtro é cobrir as falhas, suavizar e associar, não limpar falsos: quem limpa é o score ([007](007-candidatos-falsos-e-detectores.md), [015](015-falsos-positivos-de-pele-e-cor-da-bola.md)). Sem o score, a trilha ficava na bola em 14,8% dos quadros de teste do lateral (histórico do [006](006-estado-atual-e-proximos-passos.md)).

## Decisão

- **O E1 do relatório usa este experimento:** a tabela F1 (vídeo todo, com o teste entre parênteses), a linha "mesmos quadros" para o erro e o gráfico `campo_lateral/e1_F1_trecho_660.png`.
- **O F2 entra como complemento** (mesma conclusão, um pouco mais suave).
- **Nenhum padrão muda.**

## Alternativas

- **Detector bruto sem o score combinado** (só o HSV): não dá para escolher "o" candidato sem um critério, e qualquer regra (o maior, o mais redondo) seria arbitrária. A comparação com e sem score já está no [007](007-candidatos-falsos-e-detectores.md).
- **Erro contra a marcação como medida principal:** não separa nada (ver Leitura). Fica na tabela, com a explicação.

## Limitações

- **A marcação é a cada 5 quadros.** Erro, "na bola" e atraso só usam os quadros marcados. A posição interpolada entre marcações só serve para escolher o trecho do gráfico e para o E3 ([017](017-e3-oclusao.md)).
- **O detector bruto já é bom** porque inclui o score treinado nas marcações da 1ª metade de cada vídeo. A metade de teste (entre parênteses) é a comparação justa.
