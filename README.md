# Integrantes Big 5:
- Lucca Phelipe Masini RM 564121
- Luiz Henrique Poss RM562177
- Igor Paixão Sarak RM 563726
- Bernardo Braga Perobeli RM 562468
- Felipe Stefani Honorato RM 563380

# 🔧 Extração de Dados de Placas de Ativos Industriais

## Passo a passo para executar

### 1. Baixar o repositório e abrir a pasta do projeto

### 2. Instalar as dependências
pip install -r requirements.txt

### 3. Gerar o rascunho do gabarito
python src/gen_labels.py

### 4. Revisar o gabarito manualmente
notepad test_set\labels_draft.csv

### 5. Rodar a avaliação
python src/evaluate.py

### 6. Ver os resultados
type results\summary.csv
start results\bbox_B_easyocr_preprocessado