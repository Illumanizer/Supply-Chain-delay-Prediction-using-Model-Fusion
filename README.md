# Supply Chain Disruption Prediction

A multi-source fusion deep learning system for predicting shipment delays in global supply chain networks. This project combines Graph Neural Networks (GNN), Long Short-Term Memory (LSTM) networks, and Gradient Boosted Trees (XGBoost) to achieve high-accuracy delay prediction with full explainability.

## Overview

Supply chain disruptions can cause significant financial losses and operational challenges. This project addresses the problem of predicting whether a shipment will be delayed by analyzing:

- **Network Structure**: Relationships between suppliers, warehouses, ports, and customers
- **Temporal Patterns**: Historical trends in delays, congestion, and weather
- **Shipment Characteristics**: Product type, priority, route complexity, and environmental factors

## Key Results

| Metric | Score |
|--------|-------|
| Accuracy | 94.0% |
| F1-Score | 95.7% |
| ROC-AUC | 98.7% |

## Architecture

The system employs a multi-source fusion approach with three main components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   GNN Module    │     │   LSTM Module   │     │ Tabular Module  │
│  (Graph Embed)  │     │ (Temporal Embed)│     │   (Features)    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Feature Fusion       │
                    │   (Concatenation/       │
                    │    Stacking/Weighted)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      XGBoost            │
                    │    Classifier           │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   SHAP Explainability   │
                    └─────────────────────────┘
```

### 1. Graph Neural Network (GNN) Module

Compares multiple GNN architectures for learning node embeddings:

- **GAT (Graph Attention Network)**: Multi-head attention mechanism
- **GraphSAGE**: Sample and aggregate neighborhood information
- **Graph Transformer**: Transformer-based message passing
- **DeepGCN**: Deep GCN with skip connections

Best performing architecture: **GATv2 with Skip Connections** (AUC: 0.7684)

### 2. LSTM Module

Bidirectional LSTM with attention for capturing temporal patterns:

- **Architecture**: 2-layer Bidirectional LSTM (128 hidden units)
- **Attention**: Learns to focus on relevant time steps
- **Input**: 12-week historical sequences
- **Features**: Shipment counts, delays, weather, congestion, etc.

### 3. Tabular Feature Module

Engineered features capturing shipment and route characteristics:

- **Environmental**: Weather severity, port congestion
- **Route/Path**: Distance, transit days, border crossings, customs complexity
- **Product**: Volume, value, priority, fragility, temperature sensitivity
- **Supplier**: Reliability scores, quality ratings, infrastructure
- **Temporal**: Season, holidays, demand surge indicators

## Dataset

The system processes four interconnected datasets:

| Dataset | Records | Features |
|---------|---------|----------|
| Nodes (Suppliers, Warehouses, Ports, Customers) | 1,035 | 41 |
| Edges (Routes) | 2,480 | 16 |
| Shipments | 10,000 | 39 |
| Time Series | 135,585 | 17 |

**Node Distribution:**
- Customers: 500
- Suppliers: 420
- Warehouses: 100
- Ports: 15

**Delay Rate:** 70.1% of shipments experience delays

## Feature Importance (SHAP Analysis)

Top factors driving delay predictions:

1. **path_length** - Number of hops in the supply chain route
2. **weather_severity** - Environmental conditions along the route
3. **total_customs_complexity** - Cumulative customs processing requirements
4. **port_congestion** - Current congestion levels at ports
5. **path_reliability** - Historical reliability of the route

### Category Breakdown

| Category | Importance (%) |
|----------|----------------|
| Route/Path | 45.2% |
| Environmental | 24.1% |
| Supplier | 12.8% |
| Product | 10.5% |
| Demand/Time | 7.4% |

## Robustness Analysis

The model was tested for robustness under various conditions:

### Noise Tolerance
| Noise Level (σ) | Accuracy | Accuracy Drop |
|-----------------|----------|---------------|
| 0.0 | 94.0% | - |
| 0.1 | 87.7% | -6.3% |
| 0.3 | 86.2% | -7.8% |
| 0.5 | 85.3% | -8.8% |

### Missing Data Tolerance
| Missing (%) | Accuracy | Accuracy Drop |
|-------------|----------|---------------|
| 5% | 92.9% | -1.2% |
| 10% | 92.2% | -1.8% |
| 20% | 89.1% | -4.9% |

## Installation

```bash
pip install torch torch_geometric xgboost scikit-learn pandas matplotlib shap
```

## Usage

```python
# Load the notebook
jupyter notebook model.ipynb

# Or run sequentially:
# 1. Load and explore data
# 2. Preprocess features
# 3. Train GNN architectures
# 4. Train LSTM module
# 5. Feature fusion and XGBoost training
# 6. SHAP explainability analysis
# 7. Robustness testing
```

## Requirements

- Python 3.12+
- PyTorch 2.9+
- PyTorch Geometric 2.7+
- XGBoost 3.1+
- scikit-learn 1.7+
- pandas 2.2+
- SHAP (for explainability)
- matplotlib (for visualization)

## Project Structure

```
├── model.ipynb                    # Main notebook with full pipeline
├── supply_chain_nodes.csv         # Node data (suppliers, warehouses, etc.)
├── supply_chain_edges.csv         # Edge data (routes between nodes)
├── supply_chain_shipments.csv     # Shipment records with labels
├── supply_chain_timeseries.csv    # Historical time series data
└── README.md                      # This file
```

## Key Findings

1. **Tabular features dominate prediction performance** - Direct disruption indicators (weather, congestion, reliability) capture the primary signal

2. **GNN contribution is context-dependent** - More valuable with larger, more complex networks; network structure less predictive than direct features in this dataset

3. **Temporal patterns are partially captured in engineered features** - Month, season, and holiday indicators reduce marginal value of LSTM

4. **Model is robust to moderate noise and missing data** - Maintains >85% accuracy with 30% Gaussian noise or 20% missing values

## Fusion Strategies Compared

| Strategy | Accuracy | F1-Score | ROC-AUC |
|----------|----------|----------|---------|
| Tabular Only (Best) | 94.4% | 96.0% | 98.7% |
| Full Fusion (159 features) | 93.7% | 95.5% | 98.6% |
| Stacking Ensemble | 93.8% | 95.6% | 97.5% |
| Feature Selection (Top 50) | 87.5% | 91.3% | 91.0% |



## Acknowledgments

Built using PyTorch Geometric for graph neural networks and SHAP for model interpretability.
