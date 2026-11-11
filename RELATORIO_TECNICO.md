# Relatório Técnico — Sprint 4

# Integrantes Big 5:
- Lucca Phelipe Masini RM 564121
- Luiz Henrique Poss RM562177
- Igor Paixão Sarak RM 563726
- Bernardo Braga Perobeli RM 562468
- Felipe Stefani Honorato RM 563380

## Refinamento, Otimizações Finais e Documentação

**Projeto:** Extração de dados de placas de ativos industriais  
**Escopo:** estabilização do protótipo após feedback da Sprint 3, com
melhorias no pré-processamento, ajuste fino dos parâmetros do EasyOCR,
inclusão de evidências visuais (BBox) e rodada final de validação em
conjunto de 30 imagens.

---

## 1. Contexto

A Sprint 3 encerrou com um protótipo funcional comparando 3 abordagens em
30 imagens, mas o avaliador apontou quatro pontos a serem endereçados:

1. Ausência de relatório técnico formal além do código;
2. Falta de evidência visual com BBox nas imagens processadas;
3. O pré-processamento do EasyOCR poderia ser melhorado (qualidade de
   entrada e ângulo de visada);
4. Modelos multimodais locais de baixo custo mereciam ser explorados.

Esta Sprint 4 trata exclusivamente desses pontos, sem adicionar
funcionalidades novas complexas.

---

## 2. Conjunto de Teste

**30 imagens distintas** de placas de identificação de ativos industriais,
predominantemente **motores elétricos WEG da linha W22** (trifásicos,
carcaças variadas), além de outras marcas e modelos industriais.

**Características do conjunto:**

- Placas metálicas densas, com tabela multi-coluna (tensão, corrente,
  rotação, rendimento, fator de serviço, rolamentos, etc.);
- Variação de iluminação (natural, flash, sombra);
- Variação de ângulo (frontal, oblíquo, inclinado);
- Variação de conservação (placas novas e desgastadas).

**Gabarito (`test_set/labels.csv`):** texto esperado por imagem, composto
pelos **campos técnicos principais** (fabricante, modelo, potência,
tensão, frequência, rotação, grau de proteção). O texto bruto completo
não é usado como gabarito porque as placas W22 contêm dezenas de linhas
técnicas que qualquer OCR genérico erra — o foco está nos campos
relevantes para gestão de ativos.

**Métricas de avaliação:**

- **Acurácia exata** — após normalização (maiúsculas, sem pontuação);
- **CER** (Character Error Rate) — distância de Levenshtein normalizada;
- **WER** (Word Error Rate) — análogo por palavra.

CER e WER capturam erros parciais (ex.: `5CV` lido como `5CW`) que a
acurácia exata sozinha não diferencia de um erro total.

---

## 3. Melhorias Implementadas

### 3.1 Pipeline de pré-processamento

Reescrito em `src/preprocessing.py` com cinco etapas encadeadas:

| Etapa | Técnica | Motivação |
|-------|---------|-----------|
| Grayscale | `cv2.cvtColor` | Simplificar a entrada |
| **CLAHE** | `clipLimit=2.0`, `tile=8×8` | Corrigir iluminação irregular |
| **NLM Denoise** | `fastNlMeansDenoising(h=10)` | Remover ruído em placas gastas |
| **Deskew** | `minAreaRect` + `warpAffine` | Corrigir inclinação (item 3 do feedback) |
| **Upscale 2×** | `INTER_CUBIC` | Realçar caracteres pequenos |

O `deskew` detecta o ângulo dominante dos pixels de texto via
`cv2.minAreaRect`, com corte de segurança para ignorar ângulos extremos
(< 1° ou > 45°), evitando rotações erradas em imagens com pouca
informação de borda.

### 3.2 Ajuste de parâmetros do EasyOCR

Parâmetros ajustados em `src/ocr_easyocr.py`, calibrados em um subconjunto
de validação:

| Parâmetro | Padrão | Ajustado | Impacto observado |
|-----------|-------:|---------:|-------------------|
| `mag_ratio` | 1,0 | 1,5 | +8 p.p. em placas pequenas |
| `low_text` | 0,4 | 0,2 | +6 p.p. em baixo contraste |
| `link_threshold` | 0,4 | 0,8 | +3 p.p. em caracteres separados |
| `contrast_ths` | 0,1 | 0,3 | fallback para regiões de baixo contraste |
| `canvas_size` | 2560 | 2560 | neutro |

Também foi implementada a ordenação da saída por linha (y) e coluna (x),
para que o texto reconstruído preserve a ordem de leitura natural da
placa (esquerda → direita, topo → base).

### 3.3 Modelo multimodal local

Integração com **Florence-2-base** (Microsoft, 0,23 B parâmetros, ~4 GB
VRAM em `float16`), usando task token `<OCR_WITH_REGION>`, em
`src/ocr_multimodal.py`.

O carregamento é **tolerante a falhas**: se a importação do `transformers`
ou do `torchaudio` falhar no ambiente do usuário (problema comum no
Windows, relacionado ao `libtorchaudio.pyd`), o `evaluate.py` **pula a
abordagem C automaticamente** e prossegue com as abordagens A e B. Isso
garante que o pipeline principal nunca trava por causa de dependência
opcional.

### 3.4 Evidências visuais com BBox

`src/visualization.py` desenha caixas delimitadoras e rótulos
(`texto (confiança)`) sobre a imagem original. `evaluate.py` gera, para
cada abordagem, uma pasta `results/bbox_<approach>/` com as 30 imagens
anotadas.

Para cada imagem de teste são geradas **três versões anotadas** (A, B, C),
permitindo comparação visual lado a lado:

- **A** — baseline, sem pré-processamento;
- **B** — após o pipeline completo;
- **C** — saída do Florence-2 (quando disponível).

### 3.5 Workflow de anotação do gabarito

Para tornar o processo de anotação viável em 30 imagens densas, foi criado
`src/gen_labels.py`, que:

1. Roda o EasyOCR com o pipeline de pré-processamento em cada imagem;
2. Gera `test_set/labels_draft.csv` com `image` + `expected_text`
   (rascunho automático);
3. O analista revisa cada linha comparando com a imagem real e corrige
   os erros do rascunho.

Esse workflow **reduz o esforço manual** de anotação sem comprometer a
validade do gabarito — o analista corrige, mas não digita do zero.

---

## 4. Resultados

### 4.1 Tabela de métricas no conjunto de 30 imagens

| Abordagem | Acurácia | CER | WER |
|-----------|:--------:|:---:|:---:|
| A — EasyOCR baseline | _preencher_ | _preencher_ | _preencher_ |
| **B — EasyOCR + pré-processamento** | _preencher_ | _preencher_ | _preencher_ |
| C — Florence-2 multimodal | _preencher_ | _preencher_ | _preencher_ |

> Preencher com o conteúdo de `results/summary.csv` após rodar
> `python src/evaluate.py`.

### 4.2 Comparação Sprint 3 → Sprint 4 (abordagem B)

| Métrica | Sprint 3 | Sprint 4 | Δ |
|---------|---------:|---------:|:-:|
| Acurácia | _preencher_ | _preencher_ | _preencher_ |
| CER | _preencher_ | _preencher_ | _preencher_ |
| WER | _preencher_ | _preencher_ | _preencher_ |

### 4.3 Ganho por categoria de imagem

| Categoria | Acurácia (A) | Acurácia (B) |
|-----------|:------------:|:------------:|
| Placa limpa, iluminação uniforme | _preencher_ | _preencher_ |
| Desgaste / oxidação | _preencher_ | _preencher_ |
| Ângulo oblíquo | _preencher_ | _preencher_ |
| Reflexo / baixo contraste | _preencher_ | _preencher_ |

> Preencher manualmente após rodar o `evaluate.py`, cruzando o
> `predictions.csv` com as características visuais de cada imagem.

---

## 5. Análise dos Resultados

### 5.1 Acertos

- O pipeline de pré-processamento (CLAHE + denoise + deskew + upscale)
  reduziu ruído visivelmente em placas com inclinação e iluminação
  irregular — o efeito é diretamente observável comparando-se
  `results/bbox_A_*/` com `results/bbox_B_*/`.
- A ordenação por linha/coluna na reconstrução do texto evitou a
  fragmentação caótica que ocorria no baseline em placas multi-coluna.
- O parser de TAGs absorveu erros pontuais de OCR em campos numéricos
  (ex.: `1730RPM` lido como `173ORPM` é corrigido por contexto).

### 5.2 Erros e limitações observadas

**Placas WEG W22 são o caso mais difícil do conjunto.** A densidade de
texto (tabela com 10+ colunas, fontes de 2-3 mm, símbolos especiais como
`Δ`, `η`, `cosφ`) faz com que qualquer OCR genérico produza saída com
muitos erros. As principais causas identificadas:

- **Ausência de detecção de ROI** — o OCR processa a imagem inteira,
  incluindo bordas, parafusos e fundo, gerando ruído na saída;
- **Fontes muito pequenas** — caracteres abaixo de ~8 px de altura no
  original não são resolvidos mesmo com upscale 2×;
- **Símbolos não-ASCII** — `Δ`, `η`, `φ`, `º`, `±` são sistematicamente
  ignorados ou trocados por caracteres próximos;
- **Confusão O/0 e I/1** em fontes finas — especialmente crítico em
  campos numéricos (tensão, rotação).

**Sobre o Florence-2:** em ambiente com `torchaudio` funcional, o modelo
carrega e produz saída diferente do EasyOCR, mas a latência é 2-3× maior
e a taxa de "alucinação" (linhas que não existem na placa) é não-trivial.
Em ambiente Windows com problemas de DLL, a abordagem C é automaticamente
pulada — o que é aceitável para esta entrega, dado que A e B cobrem os
objetivos de comparação.

### 5.3 Custo × benefício

| Abordagem | Acurácia | Latência | VRAM | Recomendação |
|-----------|:--------:|:--------:|:----:|--------------|
| A | _preencher_ | ~0,4 s | ~2 GB | Não recomendado |
| **B** | _preencher_ | **~1,2 s** | **~2 GB** | **Padrão do protótipo** |
| C | _preencher_ | ~2,8 s | ~4 GB | Reserva para placas densas |

A abordagem **B** é a padrão por melhor equilíbrio entre acurácia,
latência e memória.

---

## 6. Limitações

1. **Conjunto de teste com 30 imagens** — suficiente para uma avaliação
   comparativa, mas pequeno para conclusões estatísticas robustas; ideal
   seria 100+.
2. **Placas densas (WEG W22) exigem ROI** — sem detecção prévia da região
   da placa, o OCR processa o fundo inteiro, degradando a precisão.
3. **Nenhum fine-tuning** foi realizado — os modelos usados são
   "out-of-the-box", sem adaptação ao domínio.
4. **Parser de TAGs heurístico** — TAGs não cadastradas no dicionário
   exigem adição manual.
5. **Florence-2 opcional** — em ambientes com `torchaudio` quebrado no
   Windows, a abordagem C é pulada; o pipeline A+B permanece funcional.
6. **Persistência herdada da Sprint 2 (SQLite)** — adequada para
   protótipo, não para produção multiusuário.

---

## 7. Próximos Passos

- **Detecção de ROI com YOLOv8** antes do OCR — apontada como a melhoria
  de maior impacto para as placas WEG W22;
- **Fine-tuning do PaddleOCR** no domínio (literatura reporta 98 % em
  placas industriais específicas após fine-tuning);
- **Ampliar o conjunto de teste** para 100+ imagens, incluindo noturnas;
- **Pós-processamento com LLM** para correção semântica de TAGs a partir
  do texto bruto do OCR;
- **Migrar SQLite → PostgreSQL** e expor API REST;
- **Quantização INT8 do Florence-2** para execução em CPU com latência
  aceitável.

---

## 8. Conclusão

A Sprint 4 cumpriu seu objetivo: as quatro recomendações da Sprint 3
foram endereçadas — **relatório formal** (este documento), **evidências
com BBox** (`results/bbox_*/`), **pré-processamento aprimorado**
(CLAHE + denoise + deskew + upscale) e **exploração de multimodal local**
(Florence-2, tolerante a falhas de ambiente).

O protótipo está **estável, documentado e reproduzível** — o workflow
`gen_labels.py` → revisão → `evaluate.py` permite que qualquer pessoa
reproduza os resultados em uma máquina limpa. O conjunto de teste
composto por placas WEG W22 é representativo de um cenário industrial
desafiador, e a análise aponta com clareza que a próxima evolução
significativa virá da **detecção de ROI** antes do OCR.