import argparse
import gc
import json
import os
from config import *

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier 
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class RouteLLMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, dim=64, use_proj=True, epochs=100, 
                 learning_rate=0.001, weight_decay=0.0001, alpha=0.05, 
                 batch_size=32, random_state=42):
        self.dim = dim
        self.use_proj = use_proj
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        self.prompt_scaler = StandardScaler()
        X_scaled = self.prompt_scaler.fit_transform(X)
        
        self.model = self._create_model(X.shape[1]).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.astype(np.float32)).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X, alpha=self.alpha)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
    
    def _create_model(self, input_dim):
        class RouteLLMNet(nn.Module):
            def __init__(self, input_dim, dim, use_proj):
                super().__init__()
                self.use_proj = use_proj
                self.strong_embed = nn.Parameter(torch.randn(dim))
                self.weak_embed = nn.Parameter(torch.randn(dim))
                
                if use_proj:
                    self.text_proj = nn.Linear(input_dim, dim, bias=False)
                else:
                    assert input_dim == dim
                
                self.classifier = nn.Linear(dim, 1, bias=False)
                
            def forward(self, prompt_embeddings, alpha=0.05):
                strong_embed = F.normalize(self.strong_embed.unsqueeze(0), p=2, dim=1)
                weak_embed = F.normalize(self.weak_embed.unsqueeze(0), p=2, dim=1)
                
                prompt_embed = prompt_embeddings
                if self.training:
                    prompt_embed += torch.randn_like(prompt_embed) * alpha
                
                if self.use_proj:
                    prompt_embed = self.text_proj(prompt_embed)
                
                model_diff = strong_embed - weak_embed
                return self.classifier((model_diff * prompt_embed)).squeeze(-1)
        
        return RouteLLMNet(input_dim, self.dim, self.use_proj)
    
    def predict_proba(self, X):
        X = np.array(X, dtype=np.float32)
        X_scaled = self.prompt_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.sigmoid(logits)
            proba_array = np.column_stack([1 - probabilities.cpu().numpy(), probabilities.cpu().numpy()])
            return proba_array
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

def load_embeddings_with_labels(benchmark, strong_model_name, weak_model_name, embedding_model, prompt_type):
    def sort_embeddings(embeddings, pairwise_data):
        return [embeddings[elem['key']] for elem in pairwise_data]
    
    embeddings_path = f'./data/{benchmark.lower()}/route_data/{embedding_model}_{prompt_type}_embeddings.pt'
    embeddings = torch.load(embeddings_path)
    labels_path = f'./data/{benchmark.lower()}/route_data/pairwise/{strong_model_name}_vs_{weak_model_name}.json'
    with open(labels_path) as f:
        data = json.load(f)
        labels = [1 if item['winner'] == 'model_b' else 0 for item in data] # 1 if the weak model wins, 0 otherwise
        keys = [item['key'] for item in data]

    return sort_embeddings(embeddings, data), labels, keys

def load_internal_states_with_layer(benchmark, strong_model_name, weak_model_name, layer, prompt_type):
    def sort_internal_states(internal_states, pairwise_data):
        tensor_states = [internal_states[elem['key']][layer] for elem in pairwise_data]
        return [x.float().numpy() for x in tensor_states]

    internal_states_path = f'./data/{benchmark.lower()}/route_data/{weak_model_name}_{prompt_type}_internal_states.pt'
    internal_states = torch.load(internal_states_path)
    labels_path = f'./data/{benchmark.lower()}/route_data/pairwise/{strong_model_name}_vs_{weak_model_name}.json'
    with open(labels_path) as f:
        data = json.load(f)
        labels = [1 if item['winner'] == 'model_b' else 0 for item in data]
        keys = [item['key'] for item in data]

    return sort_internal_states(internal_states, data), labels, keys

def find_best_model_with_hyperparameter_tuning(X, y, categories=None):
    if not isinstance(X, np.ndarray): X = np.array(X)
    if not isinstance(y, np.ndarray): y = np.array(y)

    if categories is not None:
        unique_groups = np.unique(categories)
        n_splits = min(5, len(unique_groups))
        if n_splits < 2:
            raise ValueError(f"Number of categories are smaller than 2: {n_splits}")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"\n--- Using Stratified Group {n_splits}-Fold CV ---")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("\n--- Using 5-Fold Stratified CV ---")

    pipelines = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(random_state=42)),
            ('model', LogisticRegression(random_state=42, max_iter=2000, solver='liblinear', penalty='l1', C=0.1))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(random_state=42)),
            ('model', SVC(probability=True, random_state=42, C=10, gamma='scale'))
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(early_stopping=True, n_iter_no_change=10, random_state=42, max_iter=200))
        ]), 
        # Skip tuning for Ensemble model
        'RouteLLM': RouteLLMClassifier(random_state=42, dim=4, epochs=30, learning_rate=0.01, batch_size=8),
    }

    param_grids = {
        'Logistic Regression': {
            'pca__n_components': [24, 32],
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.01, 0.1, 1.0]
        },
        'SVM': {
            'pca__n_components': [24, 32],
            'model__C': [0.1, 1, 10],
            'model__gamma': ['scale', 'auto']
        },
        'MLP': {
            'model__hidden_layer_sizes': [(32,), (16,), (8,), (32, 16), (16, 8), (8, 4)],
            'model__learning_rate_init': [0.001, 0.005, 0.01],
            'model__batch_size': [16, 32],
            'model__max_iter': [200, 300]
        },
        'RouteLLM': {
            'dim': [2, 4, 8],
            'learning_rate': [0.001, 0.005, 0.01],
            'batch_size': [4, 8, 16],
            'epochs': [20, 30]
        },
    }

    results = []
    print("\n--- Model & Hyperparameter Search (Optimizing for PR-AUC) ---")

    for name, pipeline in pipelines.items():
        print(f"\n[+] Tuning {name}...")
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=cv, scoring='average_precision', n_jobs=10)
        grid_search.fit(X, y, groups=categories)

        print(f"    Generating cross-validated probabilities...")

        results.append({
            'Model': name,
            'Best Score (CV PR-AUC)': grid_search.best_score_,
            'Best Params': grid_search.best_params_,
        })
        print(f"    Best CV PR-AUC Score: {grid_search.best_score_:.4f}")

    results_df = pd.DataFrame(results).sort_values(by='Best Score (CV PR-AUC)', ascending=False).reset_index(drop=True)
    print("\n\n--- Final Report: Model Comparison ---")
    print(results_df[['Model', 'Best Score (CV PR-AUC)', 'Best Params']].to_string())

    return results_df

def find_best_model_with_hyperparameter_preset(X, y, categories=None):
    if not isinstance(X, np.ndarray): X = np.array(X) 
    if not isinstance(y, np.ndarray): y = np.array(y)
    if categories is not None:
        unique_groups = np.unique(categories)
        n_splits = min(5, len(unique_groups))
        if n_splits < 2:
            raise ValueError(f"Number of categories are smaller than 2: {n_splits}")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"\n--- Using Stratified Group {n_splits}-Fold CV ---")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("\n--- Using 5-Fold Stratified CV ---")
    n_components = min(24, X.shape[1] - 1)
        
    pipelines = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('model', LogisticRegression(random_state=42, max_iter=2000, solver='liblinear', penalty='l1', C=0.1))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('model', SVC(probability=True, random_state=42, C=0.1, gamma='scale'))
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(early_stopping=True, n_iter_no_change=10, random_state=42, hidden_layer_sizes=(32,), learning_rate_init=0.005, batch_size=16, max_iter=200))
        ]), 
        'Ensemble': VotingClassifier(
            estimators=[
                ('lr', Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components, random_state=42)),
                    ('model', LogisticRegression(random_state=42, max_iter=2000, solver='liblinear', penalty='l1', C=0.1))
                ])),
                ('svm', Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components, random_state=42)),
                    ('model', SVC(probability=True, random_state=42, C=0.1, gamma='scale'))
                ])),
                ('mlp', Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', MLPClassifier(early_stopping=True, n_iter_no_change=10, random_state=42, hidden_layer_sizes=(32,), learning_rate_init=0.005, batch_size=16, max_iter=200))
                ]))
            ],
            voting='soft',
        ),
        'RouteLLM': RouteLLMClassifier(random_state=42, dim=2, epochs=20, learning_rate=0.01, batch_size=8),
    }

    results = []
    print("\n--- Model Evaluation (Fixed Parameters) ---")
    for name, pipeline in pipelines.items():
        print(f"\n[+] Evaluating {name}...")
        cv_probabilities = cross_val_predict(
            pipeline, X, y, cv=cv, method='predict_proba', n_jobs=10, groups=categories
        )
        results.append({
            'Model': name,
            'Best Params': dict(pipeline.named_steps['model'].get_params()) if hasattr(pipeline, 'named_steps') else dict(pipeline.get_params()),
            'CV Probabilities': cv_probabilities
        })
        
    pipelines.clear()
    del pipelines
    gc.collect()

    results_df = pd.DataFrame(results)
    del results
    gc.collect()
    
    return results_df

def find_best_model(X, y, categories=None, skip_hyperparameter_tuning=True):
    if skip_hyperparameter_tuning:
        return find_best_model_with_hyperparameter_preset(X, y, categories)
    else:
        return find_best_model_with_hyperparameter_tuning(X, y, categories)

def iterate_over_embeddings(benchmark, strong_model_name, weak_model_name, skip_hyperparameter_tuning, prompt_type):
    for embedding_model in tqdm.tqdm(EMBEDDING_MODELS):
        if skip_hyperparameter_tuning:
            pkl_path = f'./results/{benchmark.lower()}/preset/{embedding_model}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl'
        else: 
            pkl_path = f'./results/{benchmark.lower()}/tuning/{embedding_model}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl'
        if os.path.isfile(pkl_path):
            print(f"{pkl_path} already exists!")
            continue
        embeddings, labels, keys = load_embeddings_with_labels(benchmark, strong_model_name, weak_model_name, embedding_model, prompt_type)
        if benchmark.startswith('TestEval') or benchmark.startswith('LIBRO'):
             # Extract xource problem id for TestEval, source project for LIBRO - D4J
            categories = list(map(lambda id: id.split('_')[0], keys))
            embedding_df = find_best_model(embeddings, labels, categories=categories, skip_hyperparameter_tuning=skip_hyperparameter_tuning)
        else:
            embedding_df = find_best_model(embeddings, labels, skip_hyperparameter_tuning=skip_hyperparameter_tuning)
        pd.to_pickle(embedding_df, pkl_path)

def iterate_over_internal_states(benchmark, strong_model_name, weak_model_name, skip_hyperparameter_tuning, total_layers, prompt_type):
    layers = [round(total_layers / 2), round(total_layers * 2 / 3), round(total_layers * 3 / 4)]
    
    for layer in tqdm.tqdm(layers):
        if skip_hyperparameter_tuning:
            pkl_path = f'./results/{benchmark.lower()}/preset/internal_state_layer{layer}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl'
        else:
            pkl_path = f'./results/{benchmark.lower()}/tuning/internal_state_layer{layer}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl'
        if os.path.isfile(pkl_path):
            print(f"{pkl_path} already exists!")
            continue
        internal_states, labels, keys = load_internal_states_with_layer(benchmark, strong_model_name, weak_model_name, layer, prompt_type)
        if benchmark.startswith('TestEval') or benchmark.startswith('LIBRO'):
            categories = list(map(lambda id: id.split('_')[0], keys))
            internal_state_df = find_best_model(internal_states, labels, categories=categories, skip_hyperparameter_tuning=skip_hyperparameter_tuning)
        else:
            internal_state_df = find_best_model(internal_states, labels, skip_hyperparameter_tuning=skip_hyperparameter_tuning)
        pd.to_pickle(internal_state_df, pkl_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt_type', default='entire', choices=PROMPT_TYPES)
    parser.add_argument('--skip_hyperparameter_tuning', action='store_true')
    parser.add_argument('--code_generation', action='store_true')
    args = parser.parse_args()

    if args.code_generation:
        benchmarks = CODE_GENERATION_BENCHMARKS
    else:
        benchmarks = TESTING_BENCHMARKS

    prompt_type = args.prompt_type 
    for benchmark in benchmarks:
        print(f'Now processing {benchmark}')
        for strong_model_name in STRONG_MODELS:
            for weak_model_name in WEAK_MODELS:
                if args.skip_hyperparameter_tuning:
                    os.makedirs(f'./results/{benchmark.lower()}/preset', exist_ok=True)
                else:
                    os.makedirs(f'./results/{benchmark.lower()}/tuning', exist_ok=True)
                iterate_over_embeddings(
                    benchmark=benchmark,
                    strong_model_name=strong_model_name,
                    weak_model_name=weak_model_name,
                    skip_hyperparameter_tuning=args.skip_hyperparameter_tuning,
                    prompt_type=prompt_type,
                )
                iterate_over_internal_states(
                    benchmark=benchmark,
                    strong_model_name=strong_model_name,
                    weak_model_name=weak_model_name,
                    skip_hyperparameter_tuning=args.skip_hyperparameter_tuning,
                    total_layers=total_number_of_layers_of(weak_model_name),
                    prompt_type=prompt_type,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
