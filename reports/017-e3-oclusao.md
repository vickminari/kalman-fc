# 017 — E3: oclusão de 10 a 30 quadros (simulada e natural)

**Data:** 23/11/2026

**Código:**
- `experiments/e3_occlusion.py` (novo);
- `experiments/replay_tracker.py`: `replay(..., steps_out=)` guarda o `TrackStep` de cada quadro (região de busca e predição);
- `experiments/e1_raw_vs_filtered.py`: `ball_path` (posição da bola em todo quadro, interpolada entre marcações), usada aqui.

**Testes:** `tests/test_e1_e3.py` (4 do E3: apagar a bola só no buraco, crescimento da região de busca, oclusão simulada de ponta a ponta com 10 e 30 quadros, episódio natural). 66 no total.

## Contexto

- **O E3 é obrigatório:** "pelo menos uma sequência com oclusão proposital de 10 a 30 quadros", com o número de quadros recuperados durante a oclusão e a retomada da correção.
- **O [008](008-videos-de-teste-como-base.md) decidiu** que as oclusões naturais dos vídeos de teste contam como a proposital, com oclusão simulada como complemento.
- **O [013](013-limites-fisicos-no-f2.md) manteve `max_dark` = 20** pensando no E3. O enunciado pede que o filtro siga só com a predição "por um número máximo de quadros", e que a região de busca venha da covariância prevista.

## Método

### 1. Oclusão simulada (controlada)

**Onde:** em trechos com a bola visível, detectada e rastreada. A bola é "tapada" por N = 10, 20 e 30 quadros, apagando do cache os candidatos a até 20 px da posição da bola. A posição é interpolada entre as marcações, que são a cada 5 quadros. Cada início de oclusão precisa cumprir, nas trilhas normais (F1 e F2, `max_dark` 20 e 30):

- a posição da bola é conhecida de 3 quadros antes até 10 depois do buraco;
- no quadro anterior, a trilha está RASTREANDO, com a medição na bola;
- a bola era medida em pelo menos metade dos quadros do buraco (a oclusão tira de fato uma bola que estava sendo vista);
- a bola volta a ser medida até 10 quadros depois (senão a falha seria do detector, não da oclusão);
- há 60 quadros livres entre uma oclusão e a próxima.

**Resultado:** 20–28 oclusões por vídeo e N. Todas as oclusões de uma rodada são aplicadas juntas, e a trilha roda sobre o cache modificado. Uma oclusão só conta se, naquela rodada, a trilha chegou a ela rastreando a bola; nos dois vídeos, todas chegaram.

**Dois modos:**

- **`bola`:** só a bola some e os outros candidatos continuam (a bola atrás de um jogador);
- **`tudo`:** nenhum candidato (apagão do detector).

**Medidas por oclusão:**

- **quadros recuperados:** quadros do buraco com a predição da MESMA trilha a ≤ 10 px e a ≤ 20 px da bola;
- **erro da predição em k** = 5, 10, 20 e 30 quadros depois do início (é também a "predição k quadros à frente" do enunciado);
- **trilha mantida:** sem perder nem trocar de trilha durante o buraco;
- **associações erradas:** medições aceitas no buraco (todas falsas, porque a bola foi apagada);
- **reaceitação:** quadros até a primeira medição na bola depois do buraco, e se é a mesma trilha (o filtro retomou a correção) ou uma trilha nova (reinício).

**Região de busca:** o raio maior da elipse de 99%, a partir da covariância em regime, depois de k predições sem medição (`gate_growth`).

### 2. Oclusões naturais

Episódios no escuro de 10–30 quadros da trilha normal (padrões atuais). Para cada um:

- **motivo**, pelas marcações: bola marcada como invisível (oclusão ou saída do quadro) ou visível (falha do detector);
- **se a trilha estava na bola antes** (última medição a ≤ 10 px da bola);
- **se a bola foi reaceita**.

A lista tem o tempo em mm:ss, para escolher o trecho do vídeo de entrega.

Comando:

```bash
python -m experiments.e3_occlusion videos/campo_lateral.MOV --max-dark 20 30
```

(idem para `campo_centro`; saídas em `outputs/experiments/<vídeo>/`: `e3_resumo.csv`, `e3_simulada.csv`, `e3_naturais.csv`, `e3.json`, `e3_erro_k.png`, `e3_exemplo_<quadro>.png`)

## Evidência

### Oclusão simulada, padrões atuais (`max_dark` 20), modo `bola`

| | Lateral F1 | Lateral F2 | Centro F1 | Centro F2 |
|---|---|---|---|---|
| **N = 10:** oclusões | 27 | 27 | 28 | 28 |
| Trilha mantida | 100% | 100% | 100% | 100% |
| Quadros recuperados ≤ 10 px / ≤ 20 px | 78 / 93% | 79 / 93% | 66 / 83% | 68 / 88% |
| Erro da predição em k = 5 / 10 | 4,5 / 10,2 px | 4,8 / 10,9 | 5,7 / 11,7 | 6,5 / 13,2 |
| Retomada na mesma trilha | 96% | 85% | 96% | 100% |
| **N = 20:** oclusões | 20 | 20 | 24 | 24 |
| Trilha mantida | 100% | 100% | 100% | 100% |
| Quadros recuperados ≤ 10 px / ≤ 20 px | 49 / 72% | 51 / 70% | 48 / 64% | 48 / 64% |
| Erro em k = 10 / 20 | 13,6 / 25,1 px | 13,6 / 30,1 | 12,0 / 37,0 | 11,1 / 37,8 |
| Retomada na mesma trilha | 85% | 80% | 54% | 75% |
| **N = 30:** oclusões | 20 | 20 | 21 | 21 |
| Trilha mantida | 0% (perdida no 21º quadro) | 5% | 0% | 0% |
| Quadros recuperados ≤ 10 px / ≤ 20 px | 37 / 52% | 37 / 50% | 35 / 46% | 37 / 46% |
| Erro em k = 20 | 19,5 px | 20,5 | 28,9 | 27,9 |
| Associações erradas (oclusões com alguma) | 3 (2) | 4 (3) | 1 (1) | 1 (1) |
| Bola reaceita depois do buraco | 100%, em trilha nova, no 1º quadro | 100% | 100% | 100% |

**Em todos os casos, a bola volta a ser medida no 1º quadro depois do buraco** (mediana 0 quadro): ou a mesma trilha a aceita, ou uma trilha nova começa nela.

**Modo `tudo`:** os mesmos números do modo `bola`, exceto as associações erradas (zero). Com o score combinado, sobram ~0,04 falsos por quadro ([015](015-falsos-positivos-de-pele-e-cor-da-bola.md)), então tirar só a bola ou tirar tudo quase não muda nada.

### `max_dark` = 30 nas oclusões de 30 quadros (F1, modo `bola`)

| | Lateral | Centro |
|---|---|---|
| Trilha mantida | 100% | 100% |
| Quadros recuperados ≤ 20 px | 62% (52% com `max_dark` 20) | 54% (46%) |
| Erro em k = 30 | 37,3 px | 56,8 px |
| Retomada na mesma trilha | 55% | 48% |
| Retomada (na mesma trilha ou em trilha nova) | 100% | 100% |

Nas oclusões de 10 e 20 quadros, `max_dark` 30 dá os mesmos números que 20.

### Região de busca no escuro (F1, σa = 2000, σm = 1,5 px, 99%)

| k (quadros sem medição) | 1 | 5 | 6 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| Raio pela covariância | 10,5 px | 54 | 69 | 138 | 369 | 665 |
| Raio usado (teto `max_gate_px` = 60) | 10,5 | 54 | **60** | 60 | 60 | 60 |

### Oclusões naturais (trilha normal, F1; episódios no escuro de 10–30 quadros)

| | Lateral | Centro |
|---|---|---|
| Episódios | 29 | 25 |
| Com a trilha na bola antes do escuro | 13 (2 com a bola invisível, 11 com o detector falhando) | 15 (2 e 13) |
| … recuperados / reaceitos na bola | 7 / 6 | 3 / 3 |
| Com a trilha num falso antes do escuro | 6 | 8 |
| Sem marcação para saber | 10 (8 com a bola invisível; 4 recuperados) | 2 (2 recuperados) |

No F2: 15 com a trilha na bola antes no lateral (7 recuperados, 6 na bola) e 17 no centro (6 / 6).

**Para o vídeo de entrega** (lateral, F1, `e3_naturais.csv`):

- bola **invisível** (escondida ou fora do quadro) e recuperada: 0:25,6 (13 quadros), 0:52,0 (18), 1:50,9 (17);
- detector falhando com a bola andando, reaceita na bola: 0:11,2 (20 quadros), 0:23,2 (13), 1:25,4 (15), 1:29,0 (13).

No centro, a bola invisível e reaceita aparece em 1:40,5 (12 quadros).

### Gráficos

- **`e3_erro_k.png`:** erro da predição × k (mediana e quartis, N = 30) para F1 e F2, com o raio da região de busca (pela covariância e com o teto) e a linha do `max_dark`.
- **`e3_exemplo_<quadro>.png`:** a oclusão de 20 quadros (F1) em que a bola mais anda, com as elipses da região de busca a cada 5 quadros. Lateral, quadro 491; centro, quadro 2169.
  - **No exemplo do centro, a bola acelera durante a oclusão** (de ~3 para ~5,5 px/quadro). A predição segue a velocidade antiga e chega a 47 px de erro, mas a bola ainda cai dentro do teto de 60 px e a mesma trilha a reaceita no 1º quadro.
  - **Esse é o pior caso escolhido de propósito:** o maior deslocamento da bola.

## Leitura

- **Até ~10 quadros, a predição atravessa a oclusão na bola.** O erro mediano fica em 4–7 px em k = 5 e em 10–13 px em k = 10. Nas oclusões de 10 quadros, 66–79% dos quadros ficam a ≤ 10 px e 83–93% a ≤ 20 px, e quase sempre a mesma trilha retoma a correção.
- **Depois, o erro cresce mais ou menos linearmente com k:** 25–38 px em k = 20. É o erro de velocidade, mais a mudança de velocidade da bola durante a oclusão, que o modelo de velocidade constante não prevê. No centro, a bola anda mais rápido na imagem (mais perto da câmera) e o erro cresce mais depressa.
- **A região de busca vem da covariância e cresce rápido.** Com σa = 2000, P cresce como σa² k³; o raio passa de 10 px (k = 1) a 54 px (k = 5), e **o teto de 60 px ([005](005-gating-e-trilha.md)) manda a partir de k = 6**. É por isso que a retomada na mesma trilha cai com o tamanho da oclusão: quando o erro da predição passa de ~60 px, a bola cai fora da região de busca.
  - **Sem o teto**, a região de 20 quadros teria 370 px de raio, mais que meia imagem, e aceitaria qualquer pé.
  - **No relatório:** a região de busca é derivada da covariância, como o enunciado pede, e o teto é um limite de segurança justificado.
- **O `max_dark` = 20 se confirma.**
  - **Oclusão de 30 quadros com `max_dark` 20:** a trilha é perdida no 21º quadro, como manda o enunciado, e a bola é reaceita numa trilha nova no 1º quadro em que volta, porque com o score combinado ela é o melhor candidato.
  - **Com `max_dark` 30:** a mesma trilha segue, mas a predição chega a 37–57 px de erro em k = 30 e só metade das retomadas é na mesma trilha. O ganho é pequeno (≤ 20 px: 52 → 62% e 46 → 54%), e uma predição longa e errada é justamente a "trajetória louca" do [010](010-diagnostico-do-kalman-no-video.md).
- **F1 × F2: empate.** O F2 tem um pouco mais de associações erradas nas oclusões longas (3–4 contra 1–3) e retoma menos na mesma trilha no lateral, e mais no centro.
- **O gating segura os falsos durante a oclusão:** 0 associações erradas em 10 e 20 quadros, nos dois vídeos e nos dois filtros. Elas só aparecem nas oclusões de 30 quadros, com a região já no teto (1–4 em 20 oclusões).
- **Nas oclusões naturais, o gargalo é o detector, não o filtro.**
  - **A maioria dos episódios de 10–30 quadros é falha do detector com a bola visível**, e muitos terminam exatamente em 20 quadros: a bola não volta a ser detectada a tempo.
  - **Com a trilha na bola antes do escuro**, 7 de 13 (lateral) e 3 de 15 (centro) são recuperados.
  - **No centro, o baixo recall da metade de teste** ([015](015-falsos-positivos-de-pele-e-cor-da-bola.md)) aparece aqui também.

## Decisão

- **O E3 do relatório tem duas partes:**
  - **a oclusão simulada** (controlada, 10/20/30 quadros, ~20–28 repetições por vídeo), com a tabela, o `e3_erro_k.png` e um `e3_exemplo`;
  - **as oclusões naturais**, como "oclusões de verdade" e para o vídeo.
- **`max_dark` continua 20.** O teto de 60 px da região de busca continua.
- **Nenhum padrão muda.**

## Alternativas

- **Só as oclusões naturais (o que o [008](008-videos-de-teste-como-base.md) previa):** não dá controle sobre a duração e mistura falhas do detector com oclusões. Também não se sabe onde a bola estava quando ela está invisível. A simulação resolve isso; as naturais ficam como complemento e para o vídeo.
- **Oclusão simulada com `max_dark` maior:** testada (30). Ver Leitura.
- **Gravar um clipe com oclusão proposital (alguém tapando a bola):** é a leitura mais literal do enunciado. Fica como recomendação, se houver ida ao campo (ver o risco no [006](006-estado-atual-e-proximos-passos.md)).

## Limitações

- **A posição da bola dentro do buraco é interpolada entre marcações a cada 5 quadros.** Num chute ou quique, a interpolação linear erra alguns px, e isso entra no "erro da predição".
- **Apagar os candidatos a ≤ 20 px da bola simula a bola escondida, mas não o jogador que a esconde.** Como o score já tira quase todos os falsos, os modos `bola` e `tudo` dão quase o mesmo, e o teste de "o gating evita o falso perto da bola" é fraco. O E7 (gating ligado × desligado, opcional) mediria isso melhor.
- **Os trechos escolhidos são aqueles em que a bola estava sendo bem detectada.** É a situação da oclusão proposital, mas não mede o que acontece quando a oclusão começa com a trilha já fraca.
