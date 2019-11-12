# BILSTM_CRF_Chinese_NER
BiLTSM+CRF序列标注/命名实体识别代码
## 采用模型：
  bilstm+crf
## 贡献者
  liguochao   
## Requirements
  python3.6  
  tensorflow1.13  
  numpy
 ## Basic Usage
 ### Default parameters:
 * batch size: 20
 * gradient clip: 5
 * embedding size: 300
 * optimizer: Adam
 * dropout rate: 0.5
 * learning rate: 1e-3
 ### Train the model with default parameters:
 ```
    python3 main.py --train=True --clean=True
 ```
  ## 参考文献
   1. Named Entity Recognition with Bidirectional LSTM-CNNs.
   2. Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition.