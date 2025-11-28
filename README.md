
# Projeto: Reconhecimento/Identificação Facial Local

/Diogo Julio - RM553837/
/Vinicius Silva - RM553240/
/Victor Didoff - RM552965/
/Matheus Zottis - RM94119/
/Jonata Rafael - RM552939/

## Objetivo
Aplicação local (desktop/notebook) que realiza detecção e identificação facial usando OpenCV (Haar Cascade + LBPH).  
O sistema coleta imagens, treina um modelo e realiza reconhecimento em tempo real, exibindo retângulos, identificação e confiança.

## Execução
1. Instale dependências:  
   ```
   pip install -r requirements.txt
   ```

2. Colete imagens de uma pessoa:  
   ```
   python collect_faces.py --name "SeuNome" --num 60
   ```

3. Treine e execute o reconhecimento:  
   ```
   python train_and_recognize.py
   ```

## Parâmetros importantes

### Haar Cascade
- **scaleFactor**: controla sensibilidade (menor = mais detecções).  
- **minNeighbors**: controla precisão (maior = menos falsos positivos).  
- **minSize**: tamanho mínimo do rosto detectado.

### LBPH
- **radius, neighbors, grid_x, grid_y**: impactam detalhamento e robustez da identificação.  
- **threshold de confidence**: controla quando um rosto é considerado "Desconhecido".

## Estrutura do Projeto
```
face_id_project/
│ README.md
│ requirements.txt
│ ethics.md
│ collect_faces.py
│ train_and_recognize.py
└─ dataset/   (gerado automaticamente)
└─ models/    (modelo LBPH salvo após treinamento)
```

## Nota ética
Vide arquivo *ethics.md*.
