# EDESC-py

Implementação em PyTorch do **EDESC** (*Efficient Deep Embedded Subspace Clustering*, Cai et al., CVPR 2022), voltada para bases de sensoriamento remoto:

- **EDESC-SS**: versão semi-supervisionada (termo extra de perda com peso `--gamma`).

O repositório inclui o pipeline para bases de sensoriamento remoto agrícola (Sentinel-2 e MODIS: Pampa, Cerrado, SSF) e para benchmarks clássicos (Reuters, MNIST, Fashion-MNIST, CIFAR-10/100, STL-10).

---

## Estrutura

| Arquivo | Função |
|---|---|
| `EDESC.py` | EDESC original. Treino *full-batch*, 200 épocas por execução, 10 execuções. |
| `EDESC-SS.py` | EDESC semi-supervisionado. Mesma estrutura, mais `--gamma`. Usa `utils_ss.py`. |
| `AutoEncoder.py` | Autoencoder (`AE`) usado no pré-treino e como extrator do espaço latente. |
| `InitializeD.py` | Inicialização das bases dos subespaços (`D`) a partir dos clusters do K-means. |
| `Constraint.py` | Restrições sobre `D` (`D_constraint1`, `D_constraint2`). |
| `utils.py` | Carregamento/normalização das bases e métrica `cluster_acc`. |
| `utils_ss.py` | Igual ao `utils.py`, mas separa uma fração de amostras rotuladas (`labeled_ratio=0.05`). |
| `data/` | Bases e pesos pré-treinados. |
| `pretrain/` | Pesos pré-treinados. |

## Requisitos

- Python 3.9+
- Pacotes: `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `openpyxl` (leitura dos `.xlsx`), `h5py`
- `tensorflow`/`keras`: os scripts importam `tensorflow` e `keras.backend`, mas **não os utilizam**.
- GPU é opcional (`torch.cuda.is_available()` é detectado automaticamente). O `EDESC.py` e o `EDESC-SS.py` carregam a base inteira no dispositivo.

```bash
git clone https://github.com/HeitorAl/EDESC-py.git
cd EDESC-py
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
!git clone https://github.com/HeitorAl/EDESC-py.git
%cd EDESC-py
!pip install -r requirements.txt
```


---

## Dados

Os scripts esperam os arquivos nas pastas abaixo, relativas à raiz do repositório.

### Bases de sensoriamento remoto (`data/bases/`)

Cada base é uma planilha `.xlsx` com uma coluna `label` e as colunas de atributos:

- **Sentinel**: colunas no padrão `B02.01`, `B8A.01`, `NDVI.01`, `EVI.01`, ...
- **MODIS**: colunas no padrão `ndvi01`, `evi01`, `nir01`, `mir01`, ...

| `--dataset` | Arquivo em `data/bases/` | Sensor | Clusters | Atributos | Amostras |
|---|---|---|---|---|---|
| `ssf` | `ssf.sentinel.raw.xlsx` | Sentinel | 8 | 279 | 10000 |
| `ssf_filtered` | `ssf.sentinel.filtered.xlsx` | Sentinel | 8 | 279 | 8763 |
| `pampa` | `pampa.sentinel.raw.xlsx` | Sentinel | 9 | 184 | 6746 |
| `pampa_filtered` | `pampa.sentinel.filtered.xlsx` | Sentinel | 9 | 184 | 5886 |
| `cerrado_sentinel` | `cerrado.ii.sentinel.raw.xlsx` | Sentinel | 5 | 286 | 33544 |
| `cerrado_modis` | `cerrado.i.modis.raw.xlsx` | MODIS | 12 | 92 | 50160 |
| `cerrado_filtered` | `cerrado.i.modis.filtered.xlsx` | MODIS | 12 | 92 | 43910 |

Na primeira execução, o `utils.py` processa a planilha e salva um cache `*_features.npy` na mesma pasta. As execuções seguintes leem o cache. Para reprocessar, apague o `.npy`.

**Pré-processamento** (aplicado uma vez, no cache): winsorização de cada coluna entre os percentis 1 e 99, `RobustScaler`, *clip* em [-10, 10] e reescala para [0, 1]. O código imprime a fração de células com |x| > 1e5 antes da limpeza (valores fisicamente implausíveis) e emite um aviso se ela passar de 50%.

> `n_input` e `num_sample` são fixos no código para cada base. Se a sua planilha tiver outro número de colunas ou linhas, ajuste os valores no bloco "Auto-configure parameters" do script.

### Benchmarks clássicos

| `--dataset` | Origem | Clusters | `n_input` |
|---|---|---|---|
| `mnist` | baixado automaticamente (torchvision) para `data/mnist/` | 10 | 784 |
| `fashion_mnist` | baixado automaticamente para `data/fmnist/` | 10 | 784 |
| `cifar10` | baixado e convertido em *features* ResNet-50 (`data/cifar10/`) | 10 | 2048 |
| `cifar100` | idem (`data/cifar100/`) | 20 | 2048 |
| `stl10` | idem (`data/stl10/`) | 10 | 2048 |
| `reuters` | **manual**: `data/reuters/reutersidf10k.npy` | 4 | 2000 |

---

## Como executar

### Parâmetros

| Parâmetro | Significado | Padrão (`EDESC.py`) |
|---|---|---|
| `--dataset` | base de dados | `reuters` |
| `--d` | dimensão de cada subespaço | `2` |
| `--n_z` | dimensão do espaço latente | `20` |
| `--eta` | parâmetro da afinidade de subespaço | `2` |
| `--beta` | peso da perda KL (afinidade refinada) | `1` |
| `--lr` | taxa de aprendizado (Adam) | `0.001` |
| `--pretrain_path` | caminho dos pesos do autoencoder | automático por base |
| `--gamma` | peso da perda supervisionada (**só EDESC-SS**) | `0.1` |
| `--batch_size` | tamanho do lote (**só EDESC-MB**) | `256` |

`--n_clusters` é sobrescrito pelo valor fixo de cada base.

**Regra para `--n_z`:** use `n_z = n_clusters × d`. No `EDESC-SS.py` isso é verificado por `assert` (que também exige `eta == d`). Os padrões do `EDESC-SS.py` (`d=2`, `n_z=20`, `eta=2`) só passam o `assert` quando a base tem 10 clusters.

Para cada base agrícola existem os hiperparâmetros ideais seguindo as regras citadas acima, elas estão inclusas no caminho padrão de cada modelo

- `EDESC.py`: `data/EDESC/<base>/best_edesc_params.json`
- `EDESC-SS.py`: `data/EDESC-SS/<base>/best_edesc_params.json`

Na mesma pasta é possível ver um arquivo `runs.csv` o qual tem as informações de execução com os hiperparâmetros específicos daquele modelo e base agrícola

| Base | Clusters | `n_z` com `d=2` |
|---|---|---|
| `reuters` | 4 | 8 |
| `ssf`, `ssf_filtered` | 8 | 16 |
| `pampa`, `pampa_filtered` | 9 | 18 |
| `cerrado_sentinel` | 5 | 10 |
| `cerrado_modis`, `cerrado_filtered` | 12 | 24 |
| `mnist`, `fashion_mnist`, `cifar10`, `stl10` | 10 | 20 |
| `cifar100` | 20 | 40 |

### Exemplos

```bash
# EDESC na base Pampa (9 clusters, d=2 -> n_z=18)
mkdir -p data/EDESC/pampa
python EDESC.py --dataset pampa --d 2 --eta 2 --n_z 18 --beta 1

# EDESC-SS no Cerrado MODIS (12 clusters, d=2 -> n_z=24)
mkdir -p data/EDESC-SS/cerrado_modis
python EDESC-SS.py --dataset cerrado_modis --d 2 --eta 2 --n_z 24 --beta 0.1 --gamma 0.1
```

É recomendado o uso de uma GPU para a execução dos scripts, além disso, podem ser executados tanto em algum ambiente de desenvolvimento na nuvem quanto local.

### Pré-treino do autoencoder

- Se `--pretrain_path` (ou o caminho padrão da base) **não existir**, o script pré-treina o autoencoder por 50 épocas (MSE, Adam, lote de 512) e salva os pesos nesse caminho.
- Se existir, os pesos são carregados e o pré-treino é pulado.
- O diretório de destino precisa existir antes (`mkdir -p`), porque `torch.save` não o cria.

Caminhos padrão para as bases de sensoriamento remoto:

- `EDESC.py`: `data/EDESC/<base>/<base>.npz`
- `EDESC-SS.py`: `data/EDESC-SS/<base>/<base>.npz`

### O que o script faz

1. Carrega a base e o autoencoder (pré-treinado).
2. Roda K-means (`n_init=10`) no espaço latente para obter os clusters iniciais.
3. Inicializa as bases dos subespaços `D` com `Initialization_D`.
4. Treina por 200 épocas (100 no MB). Cada época calcula a afinidade de subespaço `s`, a afinidade refinada `s̃` e a perda total: reconstrução (MSE) + `beta` × KL(`s`, `s̃`) + restrições de `D` (+ `gamma` × perda supervisionada no SS).
5. Repete todo o processo **10 vezes** e imprime o melhor ACC, NMI e ARI.

---

## Saída e métricas

A cada época é impresso algo como:

```
Iter 17 :Current Acc 0.6123 :Max Acc 0.6240 , Current nmi ... :Max nmi ... , Current ari ... :Max ari ...
```

No fim das 10 execuções:

```
Best ACC 0.xxxx Best NMI 0.xxxx Best ARI 0.xxxx
```

Pontos importantes para interpretar os números:

- ACC usa o pareamento húngaro entre clusters e rótulos (`cluster_acc`).
- O valor reportado em cada execução é o **máximo ao longo das épocas**, e o `Best` final é o **máximo entre as 10 execuções**. Ambos usam os rótulos verdadeiros para a seleção.
- Não há semente fixa (o K-means e a inicialização não recebem `random_state`), então os resultados variam entre execuções. Para reportar média e desvio-padrão, guarde o `acc, nmi, ari` de cada uma das 10 execuções em vez de apenas o máximo.

---

## Referência

J. Cai, J. Fan, W. Guo, S. Wang, Y. Zhang, Z. Zhang. *Efficient Deep Embedded Subspace Clustering*. CVPR, 2022.
