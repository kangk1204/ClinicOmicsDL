#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_ml_ultimate_progress.py

궁극적으로 최적화된 TCGA 유방암 데이터 분류 파이프라인
- 상세한 진행과정 출력
- 실시간 성능 모니터링
- 색상 있는 시각적 출력
- 단계별 시간 측정

주요 특징:
1. 다중 GPU 백엔드 지원 (CUDA, MPS, CPU)
2. 고급 앙상블 방법 (Voting + Stacking)
3. 자동 피처 엔지니어링
4. 베이지안 + Optuna 하이퍼파라미터 최적화
5. Boruta 피처 선택
6. 메모리 효율성 극대화
7. 고급 모델 해석성
8. 상세한 진행과정 표시

Usage:
  python 01_ml_ultimate_progress.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
import time
import gc
import warnings
import platform
import psutil
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any

import joblib
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# Progress / colour utilities
from tqdm import tqdm
try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    class Fore:  RED=GREEN=BLUE=YELLOW=MAGENTA=CYAN=WHITE=RESET = ""
    class Back:  RED=GREEN=BLUE=YELLOW=MAGENTA=CYAN=WHITE=RESET = ""
    class Style: BRIGHT=DIM=RESET_ALL = ""

# ML libraries
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, train_test_split,
    cross_val_score, cross_validate
)
from sklearn.preprocessing import (
    RobustScaler, LabelEncoder, label_binarize, PowerTransformer
)
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, VarianceThreshold
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    fbeta_score, roc_auc_score, roc_curve, auc,
    precision_score, recall_score,
    balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression

# Hyper‑parameter optimisation
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Boruta
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False

# GPU support
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

# Interpretability
import shap
try:
    import eli5
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False
try:
    from pdpbox import pdp
    PDP_AVAILABLE = True
except ImportError:
    PDP_AVAILABLE = False

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
RANDOM_STATE       = 42
N_JOBS             = -1
OUTPUT_DIR         = "./output_results_ultimate"
TOP_N_FEATURES     = 200
CV_FOLDS           = 5
CV_REPEATS         = 2
OPTUNA_TRIALS      = 100
BAYESIAN_TRIALS    = 50


def extract_feature_names_from_pipeline(pipe:Pipeline,
                                        original_names:list[str])->list[str]:
    """
    Walk through a fitted *preprocessing* pipeline and update `original_names`
    whenever a step drops columns (VarianceThreshold / SelectKBest).

    Parameters
    ----------
    pipe : Pipeline
        Fitted preprocessing pipeline (`pipe.fit(...)` already called).
    original_names : list[str]
        List of column names entering the pipeline.

    Returns
    -------
    list[str]
        Column names that actually reach the classifier.
    """
    names = list(original_names)
    for step_name, step in pipe.named_steps.items():
        # steps that keep shape leave names untouched
        if isinstance(step, VarianceThreshold):
            mask = step.get_support()
            names = [n for n, keep in zip(names, mask) if keep]
        elif isinstance(step, SelectKBest):
            mask = step.get_support()
            names = [n for n, keep in zip(names, mask) if keep]
        else:
            # PowerTransformer / RobustScaler / Imputer do not change #columns
            pass
    return names

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------
class ProgressTracker:
    """Console‑based step‑timer & progress reporter."""
    def __init__(self):
        self.start_time      = time.time()
        self.step_times      = {}
        self.step_end_times  = {}
        self.current_step    = None
        self.total_steps     = 12
        self.completed_steps = 0

    # -------------- internal utilities -----------------------------------
    def _format_time(self, sec: float) -> str:
        if sec < 60:               return f"{sec:.1f}s"
        elif sec < 3600:           return f"{int(sec//60)}m {int(sec%60)}s"
        else:                      return f"{int(sec // 3600)}h {int((sec % 3600)//60)}m"

    def _print_memory_status(self):
        mem  = psutil.virtual_memory()
        cpu  = psutil.cpu_percent(interval=1)
        mcol = Fore.RED if mem.percent>80 else (Fore.YELLOW if mem.percent>60 else Fore.GREEN)
        print(f"{Fore.WHITE}💾 Memory: {mcol}{mem.percent:.1f}%{Style.RESET_ALL} | "
              f"🖥️  CPU: {Fore.BLUE}{cpu:.1f}%{Style.RESET_ALL}")

    # -------------- public API -------------------------------------------
    def start_step(self, name:str, desc:str=""):
        self.current_step = name
        self.step_times[name] = time.time()
        prog = (self.completed_steps / self.total_steps) * 100
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[{self.completed_steps+1}/{self.total_steps}] {name.upper()}{Style.RESET_ALL}")
        if desc: print(f"{Fore.WHITE}{desc}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Progress: {prog:.1f}% | Elapsed: {self._format_time(time.time()-self.start_time)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        self._print_memory_status()

    def end_step(self, success:bool=True, details:str=""):
        if not self.current_step: return
        self.step_end_times[self.current_step] = time.time()
        dur   = self.step_end_times[self.current_step] - self.step_times[self.current_step]
        icon  = f"{Fore.GREEN}✓{Style.RESET_ALL}" if success else f"{Fore.RED}✗{Style.RESET_ALL}"
        text  = f"{Fore.GREEN}COMPLETED{Style.RESET_ALL}" if success else f"{Fore.RED}FAILED{Style.RESET_ALL}"
        print(f"\n{icon} {self.current_step}: {text}")
        print(f"{Fore.MAGENTA}⏱️  Duration: {self._format_time(dur)}{Style.RESET_ALL}")
        if details: print(f"{Fore.WHITE}📋 Details: {details}{Style.RESET_ALL}")
        if success: self.completed_steps += 1
        self.current_step = None

    def print_progress_bar(self, cur:int, total:int, prefix:str="", length:int=50):
        pct   = (cur/total)*100
        filled= int(length*cur/total)
        bar   = '█'*filled + '░'*(length-filled)
        print(f"\r{Fore.BLUE}{prefix} |{bar}| {pct:.1f}% ({cur}/{total}){Style.RESET_ALL}",end='')
        if cur == total: print()

    def print_summary(self):
        tot = time.time() - self.start_time
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🎉 PIPELINE EXECUTION SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Total Steps Completed: {self.completed_steps}/{self.total_steps}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}⏱️  Total Execution Time: {self._format_time(tot)}{Style.RESET_ALL}")
        if self.step_times:
            print(f"\n{Fore.YELLOW}📊 Step‑by‑Step Timing:{Style.RESET_ALL}")
            for step in self.step_times:
                if step in self.step_end_times:
                    dur = self.step_end_times[step] - self.step_times[step]
                    print(f"  {Fore.WHITE}• {step}: {self._format_time(dur)}{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.WHITE}• {step}: (no end time recorded){Style.RESET_ALL}")

class DeviceManager:
    @staticmethod
    def get_optimal_device():
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available(): return "mps"
            if torch.cuda.is_available():         return "cuda"
        return "cpu"

    @staticmethod
    def get_xgb_params_for_device(dev:str)->Dict[str,Any]:
        if dev=="cuda" and hasattr(xgb,"gpu_hist"):
            return {"tree_method":"gpu_hist","gpu_id":0,"predictor":"gpu_predictor"}
        return {"tree_method":"hist","predictor":"cpu_predictor"}

    @staticmethod
    def optimize_for_platform():
        if platform.system()=="Darwin":
            os.environ["OMP_NUM_THREADS"]=str(min(8, os.cpu_count()))
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                torch.mps.set_per_process_memory_fraction(0.8)
        elif platform.system()=="Linux":
            os.environ["OMP_NUM_THREADS"]=str(os.cpu_count())

class AdvancedFeatureEngineering:
    """Stat‑based, DR‑based, and interaction features."""
    # statistical
    @staticmethod
    def create_statistical_features(X:pd.DataFrame, tracker:ProgressTracker)->pd.DataFrame:
        print(f"{Fore.BLUE}📊 통계적 피처 생성 중...{Style.RESET_ALL}")
        feats=pd.DataFrame(index=X.index)
        ops=[('mean',lambda x:x.mean(axis=1)),
             ('std', lambda x:x.std(axis=1)),
             ('median',lambda x:x.median(axis=1)),
             ('q25', lambda x:x.quantile(0.25,axis=1)),
             ('q75', lambda x:x.quantile(0.75,axis=1)),
             ('skew', lambda x:x.skew(axis=1)),
             ('kurtosis',lambda x:x.kurtosis(axis=1)),
             ('mad', lambda x:np.mean(np.abs(x - x.mean(axis=1)),axis=1))]
        for _,(name,op) in enumerate(tqdm(ops,desc="Statistical Features",
                                          bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.GREEN,Style.RESET_ALL))):
            feats[name]=op(X)
        feats['iqr']    = feats['q75']-feats['q25']
        feats['range']  = X.max(axis=1)-X.min(axis=1)
        feats['cv']     = feats['std']/(feats['mean']+1e-8)
        feats['entropy']= -((X+1e-8).apply(lambda x:x*np.log(x+1e-8),axis=1)).sum(axis=1)
        print(f"{Fore.GREEN}✓ 통계적 피처 {feats.shape[1]}개 생성 완료{Style.RESET_ALL}")
        return feats
    # dimensionality reduction
    @staticmethod
    def create_dimensionality_reduction_features(X:pd.DataFrame,n_components:int,
                                                 tracker:ProgressTracker)->pd.DataFrame:
        print(f"{Fore.BLUE}🔍 차원 축소 피처 생성 중... (n_components={n_components}){Style.RESET_ALL}")
        X_no_nan = X.fillna(0)
        print(f"{Fore.YELLOW}  • PCA 수행 중...{Style.RESET_ALL}")
        pca=PCA(n_components=n_components,random_state=RANDOM_STATE)
        pca_df=pd.DataFrame(pca.fit_transform(X_no_nan),index=X.index,
                            columns=[f"pca_{i}" for i in range(n_components)])
        print(f"{Fore.YELLOW}  • TruncatedSVD 수행 중...{Style.RESET_ALL}")
        svd=TruncatedSVD(n_components=n_components,random_state=RANDOM_STATE)
        svd_df=pd.DataFrame(svd.fit_transform(X_no_nan),index=X.index,
                            columns=[f"svd_{i}" for i in range(n_components)])
        res=pd.concat([pca_df,svd_df],axis=1)
        print(f"{Fore.GREEN}✓ 차원 축소 피처 {res.shape[1]}개 생성 완료{Style.RESET_ALL}")
        return res
    # interaction
    @staticmethod
    def create_interaction_features(X:pd.DataFrame,top_n:int,
                                    tracker:ProgressTracker)->pd.DataFrame:
        print(f"{Fore.BLUE}🔗 상호작용 피처 생성 중... (top_n={top_n}){Style.RESET_ALL}")
        top_feats=X.var().nlargest(top_n).index.tolist()
        X_top=X[top_feats]
        inter_df=pd.DataFrame(index=X.index)
        inters=[]
        for i,f1 in enumerate(top_feats[:10]):
            for f2 in top_feats[i+1:10]:
                inters.append((f1,f2))
        for f1,f2 in tqdm(inters,desc="Interaction Features",
                          bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.MAGENTA,Style.RESET_ALL)):
            inter_df[f"{f1}_x_{f2}"]=X_top[f1]*X_top[f2]
        print(f"{Fore.GREEN}✓ 상호작용 피처 {inter_df.shape[1]}개 생성 완료{Style.RESET_ALL}")
        return inter_df

# ---------------------------------------------------------------------------
# Main Pipeline class
# ---------------------------------------------------------------------------
class UltimateMLPipeline:
    def __init__(self, random_state:int=RANDOM_STATE):
        self.random_state    = random_state
        self.device          = DeviceManager.get_optimal_device()
        self.label_encoder   = None
        self.voting_ensemble = None
        self.stacking_ensemble=None
        self.feature_names:List[str]=[]
        self.class_names     = None
        self.selected_features:List[str]=[]
        self.progress_tracker= ProgressTracker()
        DeviceManager.optimize_for_platform()
        print(f"{Fore.GREEN}🚀 Pipeline initialized with device: {self.device}{Style.RESET_ALL}")

    # -----------------------------------------------------------------------
    # Utility to align columns between train and predict
    # -----------------------------------------------------------------------
    def _align_features(self, X:pd.DataFrame)->pd.DataFrame:
        """Ensure X has identical column set/order as self.feature_names."""
        if not self.feature_names:
            raise RuntimeError("Feature names not yet set – train models first.")
        return X.reindex(columns=self.feature_names, fill_value=0)

    # -----------------------------------------------------------------------
    # Data loading / preprocessing
    # -----------------------------------------------------------------------
    def load_and_preprocess_data(self)->Tuple[pd.DataFrame,pd.Series]:
        tr=self.progress_tracker
        tr.start_step("Data Loading","Loading and preprocessing TCGA breast cancer data")
        paths={"clinical_sample":"./data_clinical_sample.txt",
               "clinical_patient":"./data_clinical_patient.txt",
               "mrna":"./data_mrna_seq_v2_rsem.txt"}
        dfs={}
        for i,(name,pth) in enumerate(paths.items()):
            print(f"{Fore.YELLOW}📂 Loading {name}...{Style.RESET_ALL}")
            tr.print_progress_bar(i,len(paths),f"Loading {name}")
            dfs[name]=pd.read_csv(pth,sep='\t',comment='#',low_memory=False)
            print(f"{Fore.WHITE}   Shape: {dfs[name].shape}{Style.RESET_ALL}")
        tr.print_progress_bar(len(paths),len(paths),"Data loading")

        # mapping
        sample2patient=dict(zip(dfs["clinical_sample"]["SAMPLE_ID"],
                                dfs["clinical_sample"]["PATIENT_ID"]))
        patient2subtype=dict(zip(dfs["clinical_patient"]["PATIENT_ID"],
                                 dfs["clinical_patient"]["SUBTYPE"]))

        # mRNA tidy
        print(f"{Fore.BLUE}🧬 Preprocessing mRNA data...{Style.RESET_ALL}")
        dfm=dfs["mrna"].copy()
        dfm["Hugo_Symbol"]=dfm["Hugo_Symbol"].replace('',np.nan).fillna("UNKNOWN")
        dfm["Entrez_Gene_Id"]=dfm["Entrez_Gene_Id"].fillna(0).astype(int).astype(str)
        dfm["GeneID"]=dfm["Hugo_Symbol"]+"|"+dfm["Entrez_Gene_Id"]
        if dfm["GeneID"].duplicated().any():
            dup=dfm["GeneID"].duplicated().sum()
            print(f"{Fore.YELLOW}   Removing {dup} duplicate genes{Style.RESET_ALL}")
        dfm=dfm.drop_duplicates(subset=["GeneID"],keep='first')
        dfm.set_index("GeneID",inplace=True)
        dfm.drop(columns=["Hugo_Symbol","Entrez_Gene_Id"],inplace=True)

        # keep only samples with subtype
        print(f"{Fore.BLUE}🔍 Filtering valid samples...{Style.RESET_ALL}")
        valid=[s for s in dfm.columns if s in sample2patient and
               sample2patient[s] in patient2subtype]
        dfm=dfm[valid]

        labels=[patient2subtype[sample2patient[s]] for s in dfm.columns]
        y=pd.Series(labels,index=dfm.columns,name="SUBTYPE")
        X=dfm.T
        mask=~y.isnull()
        X=X[mask]; y=y[mask]
        print(f"{Fore.GREEN}📊 Final dataset shape: X {X.shape}, y {y.shape}{Style.RESET_ALL}")
        dist=y.value_counts()
        print(f"{Fore.CYAN}📈 Class distribution:{Style.RESET_ALL}")
        for c,cnt in dist.items():
            print(f"  {Fore.WHITE}• {c}: {cnt} ({cnt/len(y)*100:.1f}%){Style.RESET_ALL}")

        del dfs,dfm; gc.collect()
        tr.end_step(True,f"Dataset: {X.shape[0]} samples, {X.shape[1]} genes, {len(dist)} classes")
        return X,y

    # -----------------------------------------------------------------------
    # Feature engineering wrapper
    # -----------------------------------------------------------------------
    def advanced_feature_engineering(self,X:pd.DataFrame)->pd.DataFrame:
        tr=self.progress_tracker
        tr.start_step("Feature Engineering",
                      "Creating statistical, dimensionality reduction, and interaction features")
        print(f"{Fore.BLUE}🔄 Applying log transformation...{Style.RESET_ALL}")
        Xproc=np.log1p(X.clip(lower=0))
        stat = AdvancedFeatureEngineering.create_statistical_features(Xproc,tr)
        dr   = AdvancedFeatureEngineering.create_dimensionality_reduction_features(
                Xproc, n_components=min(50, X.shape[1]//10), tracker=tr)
        if X.shape[1]<5000:
            inter=AdvancedFeatureEngineering.create_interaction_features(Xproc,20,tr)
            Xenh=pd.concat([Xproc,stat,dr,inter],axis=1)
        else:
            print(f"{Fore.YELLOW}⚠️  Skipping interaction features due to high dimensionality{Style.RESET_ALL}")
            Xenh=pd.concat([Xproc,stat,dr],axis=1)

        Xenh=Xenh.replace([np.inf,-np.inf],np.nan).fillna(Xenh.mean())
        tr.end_step(True,f"Enhanced features: {Xenh.shape[1]} (original: {X.shape[1]})")
        return Xenh

    # -----------------------------------------------------------------------
    # Pre‑processing pipeline builder
    # -----------------------------------------------------------------------
    def create_advanced_preprocessing_pipeline(self,X:pd.DataFrame)->Pipeline:
        tr=self.progress_tracker
        tr.start_step("Preprocessing Pipeline","Creating advanced preprocessing pipeline")
        pipe=Pipeline([
            ("imputer",SimpleImputer(strategy="mean")),
            ("variance",VarianceThreshold(threshold=0.01)),
            ("power",PowerTransformer(method="yeo-johnson",standardize=True)),
            ("scale",RobustScaler()),
            ("kbest",SelectKBest(mutual_info_classif,
                                 k=min(TOP_N_FEATURES*2, X.shape[1]//3)))
        ])
        tr.end_step(True,"5‑stage preprocessing pipeline created")
        return pipe

    # -----------------------------------------------------------------------
    # Boruta selection
    # -----------------------------------------------------------------------
    def boruta_feature_selection(self,X:pd.DataFrame,y:pd.Series)->List[str]:
        tr=self.progress_tracker
        tr.start_step("Boruta Feature Selection","Selecting most important features")
        if not BORUTA_AVAILABLE:
            sel=X.var().nlargest(TOP_N_FEATURES).index.tolist()
            tr.end_step(True,f"Variance‑based selection: {len(sel)} features")
            return sel

        # sample for memory reasons
        if len(X)>1000:
            idx=np.random.choice(len(X),1000,replace=False)
            Xs,y_s=X.iloc[idx],y.iloc[idx]
        else:
            Xs,y_s=X,y
        rf=RandomForestClassifier(n_estimators=100, n_jobs=N_JOBS,
                                  random_state=self.random_state,max_depth=5)
        y_enc=LabelEncoder().fit_transform(y_s)
        try:
            print(f"{Fore.BLUE}🌳 Running Boruta...{Style.RESET_ALL}")
            bor=BorutaPy(rf,n_estimators='auto',max_iter=50,verbose=1,
                         random_state=self.random_state)
            with tqdm(total=50,desc="Boruta Iterations",
                      bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.GREEN,Style.RESET_ALL)) as pbar:
                bor.fit(Xs.values,y_enc); pbar.update(50)
            sel=X.columns[bor.support_].tolist()
            tr.end_step(True,f"Boruta selected {len(sel)} features")
            return sel
        except Exception as e:
            print(f"{Fore.RED}❌ Boruta failed: {e}{Style.RESET_ALL}")
            sel=X.var().nlargest(TOP_N_FEATURES).index.tolist()
            tr.end_step(False,f"Fallback variance selection: {len(sel)} features")
            return sel

    # -----------------------------------------------------------------------
    # XGBoost optimisation (Optuna) — unchanged except for print shortening
    # -----------------------------------------------------------------------
    def optimize_xgboost_optuna(self,X:pd.DataFrame,y:pd.Series)->Dict[str,Any]:
        tr=self.progress_tracker
        tr.start_step("XGBoost Optimization",
                      f"Optimizing XGBoost hyperparameters ({OPTUNA_TRIALS} trials)")
        le=LabelEncoder(); y_enc=le.fit_transform(y)
        uniq=np.unique(y_enc); n_cls=len(uniq)
        best=0; device_params=DeviceManager.get_xgb_params_for_device(self.device)

        def objective(trial):
            nonlocal best
            cls_w=compute_class_weight("balanced",classes=uniq,y=y_enc)
            spw=cls_w[1]/cls_w[0] if n_cls==2 else 1
            params={ 'n_estimators': trial.suggest_int("n_estimators",100,2000),
                     'max_depth':   trial.suggest_int("max_depth",3,12),
                     'learning_rate': trial.suggest_float("learning_rate",0.01,0.5,log=True),
                     'subsample':   trial.suggest_float("subsample",0.5,1.0),
                     'colsample_bytree':trial.suggest_float("colsample_bytree",0.5,1.0),
                     'colsample_bylevel':trial.suggest_float("colsample_bylevel",0.5,1.0),
                     'colsample_bynode': trial.suggest_float("colsample_bynode",0.5,1.0),
                     'reg_alpha':   trial.suggest_float("reg_alpha",1e-8,100,log=True),
                     'reg_lambda':  trial.suggest_float("reg_lambda",1e-8,100,log=True),
                     'min_child_weight':trial.suggest_int("min_child_weight",1,20),
                     'gamma':       trial.suggest_float("gamma",1e-8,100,log=True),
                     'max_delta_step': trial.suggest_int("max_delta_step",0,10),
                     'scale_pos_weight':spw,'random_state':self.random_state,
                     'n_jobs':N_JOBS,'eval_metric':'mlogloss','num_class':n_cls,
                     **device_params}
            pipe=self.create_advanced_preprocessing_pipeline(X)
            model=xgb.XGBClassifier(**params)
            full=Pipeline([("pre",pipe),("clf",model)])
            cv=RepeatedStratifiedKFold(n_splits=CV_FOLDS,random_state=trial.number,n_repeats=1)
            try:
                scr=np.mean(cross_val_score(full,X,y_enc,cv=cv,scoring="balanced_accuracy",
                                            n_jobs=N_JOBS,error_score='raise'))
                best=max(best,scr)
                return scr
            except Exception: return 0.0

        study=optuna.create_study(direction="maximize",
                                  sampler=TPESampler(seed=self.random_state),
                                  pruner=HyperbandPruner(min_resource=1,max_resource=CV_FOLDS))
        with tqdm(total=OPTUNA_TRIALS,desc="XGB Opt",
                  bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.BLUE,Style.RESET_ALL)) as pbar:
            study.optimize(objective,n_trials=OPTUNA_TRIALS,timeout=7200,
                           callbacks=[lambda s,t: pbar.update(1)])
        tr.end_step(True,f"Best score {study.best_value:.4f}")
        return study.best_params

    # -----------------------------------------------------------------------
    # LightGBM optimisation (Bayesian then fallback Optuna) — unchanged
    # -----------------------------------------------------------------------
    def optimize_lightgbm_bayesian(self,X:pd.DataFrame,y:pd.Series)->Dict[str,Any]:
        tr=self.progress_tracker
        tr.start_step("LightGBM Optimization",
                      f"Optimizing LightGBM ({BAYESIAN_TRIALS} trials)")
        if not BAYESIAN_AVAILABLE:
            return self.optimize_lightgbm_optuna(X,y)
        le=LabelEncoder(); y_enc=le.fit_transform(y)
        spaces={
            "clf__n_estimators":Integer(100,2000),
            "clf__max_depth":   Integer(3,12),
            "clf__learning_rate":Real(0.01,0.5,prior='log-uniform'),
            "clf__subsample":   Real(0.5,1.0),
            "clf__colsample_bytree":Real(0.5,1.0),
            "clf__reg_alpha":   Real(1e-8,100,prior='log-uniform'),
            "clf__reg_lambda":  Real(1e-8,100,prior='log-uniform'),
            "clf__min_child_samples":Integer(10,200),
            "clf__num_leaves":  Integer(10,200)
        }
        pre=self.create_advanced_preprocessing_pipeline(X)
        lgb=LGBMClassifier(class_weight='balanced',random_state=self.random_state,
                           n_jobs=N_JOBS,verbose=-1)
        pipe=Pipeline([("pre",pre),("clf",lgb)])
        cv=StratifiedKFold(n_splits=CV_FOLDS,shuffle=True,random_state=self.random_state)
        opt=BayesSearchCV(pipe,spaces,n_iter=BAYESIAN_TRIALS,cv=cv,
                          scoring="balanced_accuracy",n_jobs=N_JOBS,
                          random_state=self.random_state,verbose=0)
        with tqdm(total=BAYESIAN_TRIALS,desc="BayesOpt",
                  bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.MAGENTA,Style.RESET_ALL)) as pbar:
            opt.fit(X,y_enc); pbar.update(BAYESIAN_TRIALS)
        tr.end_step(True,f"Best score {opt.best_score_:.4f}")
        return {k.replace("clf__",""):v for k,v in opt.best_params_.items()}

    def optimize_lightgbm_optuna(self,X:pd.DataFrame,y:pd.Series)->Dict[str,Any]:
        print(f"{Fore.YELLOW}📱 Using Optuna for LightGBM optimization{Style.RESET_ALL}")
        le=LabelEncoder(); y_enc=le.fit_transform(y); best=0
        def obj(trial):
            nonlocal best
            params={'n_estimators':trial.suggest_int('n_estimators',100,2000),
                    'max_depth':trial.suggest_int('max_depth',3,12),
                    'learning_rate':trial.suggest_float('learning_rate',0.01,0.5,log=True),
                    'subsample':trial.suggest_float('subsample',0.5,1.0),
                    'colsample_bytree':trial.suggest_float('colsample_bytree',0.5,1.0),
                    'reg_alpha':trial.suggest_float('reg_alpha',1e-8,100,log=True),
                    'reg_lambda':trial.suggest_float('reg_lambda',1e-8,100,log=True),
                    'min_child_samples':trial.suggest_int('min_child_samples',10,200),
                    'num_leaves':trial.suggest_int('num_leaves',10,200),
                    'class_weight':'balanced','random_state':self.random_state,
                    'n_jobs':N_JOBS,'verbose':-1}
            pre=self.create_advanced_preprocessing_pipeline(X)
            model=LGBMClassifier(**params)
            pipe=Pipeline([("pre",pre),("clf",model)])
            cv=StratifiedKFold(n_splits=CV_FOLDS,shuffle=True,random_state=trial.number)
            scr=np.mean(cross_val_score(pipe,X,y_enc,cv=cv,scoring="balanced_accuracy",n_jobs=N_JOBS))
            best=max(best,scr)
            return scr
        study=optuna.create_study(direction="maximize",
                                  sampler=CmaEsSampler(seed=self.random_state),
                                  pruner=MedianPruner(n_startup_trials=10))
        with tqdm(total=OPTUNA_TRIALS//2,desc="LGB Optuna",
                  bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.CYAN,Style.RESET_ALL)) as pbar:
            study.optimize(obj,n_trials=OPTUNA_TRIALS//2,timeout=3600,
                           callbacks=[lambda s,t:pbar.update(1)])
        return study.best_params

    # -----------------------------------------------------------------------
    # Ensemble creators
    # -----------------------------------------------------------------------
    def create_voting_ensemble(self,X:pd.DataFrame,y:pd.Series)->VotingClassifier:
        tr=self.progress_tracker; tr.start_step("Voting Ensemble","Building ensemble")
        xgb_par=self.optimize_xgboost_optuna(X,y)
        lgb_par=self.optimize_lightgbm_bayesian(X,y)
        xgb_par.update(DeviceManager.get_xgb_params_for_device(self.device))
        xgb_par['random_state']=self.random_state
        lgb_par.update({'random_state':self.random_state,'class_weight':'balanced',
                        'n_jobs':N_JOBS,'verbose':-1})
        p1=self.create_advanced_preprocessing_pipeline(X)
        p2=self.create_advanced_preprocessing_pipeline(X)
        p3=self.create_advanced_preprocessing_pipeline(X)
        xgb_pipe=Pipeline([("pre",p1),("clf",xgb.XGBClassifier(**xgb_par))])
        lgb_pipe=Pipeline([("pre",p2),("clf",LGBMClassifier(**lgb_par))])
        rf_pipe =Pipeline([("pre",p3),("clf",RandomForestClassifier(
                            n_estimators=500,max_depth=8,class_weight='balanced',
                            random_state=self.random_state,n_jobs=N_JOBS))])
        ens=VotingClassifier(estimators=[('xgb',xgb_pipe),('lgb',lgb_pipe),
                                         ('rf',rf_pipe)],voting='soft',n_jobs=N_JOBS)
        tr.end_step(True,"Voting ensemble ready")
        return ens

    def create_stacking_ensemble(self,X:pd.DataFrame,y:pd.Series)->StackingClassifier:
        tr=self.progress_tracker; tr.start_step("Stacking Ensemble","Building stacking")
        vote=self.create_voting_ensemble(X,y)
        base=vote.estimators
        p4=self.create_advanced_preprocessing_pipeline(X)
        et_pipe=Pipeline([("pre",p4),("clf",ExtraTreesClassifier(
                             n_estimators=300,max_depth=6,class_weight='balanced',
                             random_state=self.random_state,n_jobs=N_JOBS))])
        all_est=base+[('et',et_pipe)]
        meta=Pipeline([("imp",SimpleImputer(strategy="mean")),
                       ("log",LogisticRegression(class_weight='balanced',
                                 random_state=self.random_state,max_iter=1000,n_jobs=N_JOBS))])
        stack=StackingClassifier(estimators=all_est,final_estimator=meta,
                                 cv=CV_FOLDS,n_jobs=N_JOBS,passthrough=True)
        tr.end_step(True,"Stacking ensemble ready")
        return stack

    # -----------------------------------------------------------------------
    # Nested CV (optional; unchanged)
    # -----------------------------------------------------------------------
    def nested_cross_validation(self,X:pd.DataFrame,y:pd.Series)->Dict[str,Dict[str,float]]:
        tr=self.progress_tracker
        tr.start_step("Nested Cross-Validation",
                      f"Running {CV_FOLDS}-fold CV ×{CV_REPEATS}")
        le=LabelEncoder(); y_enc=le.fit_transform(y)
        outer=RepeatedStratifiedKFold(n_splits=CV_FOLDS,n_repeats=CV_REPEATS,
                                      random_state=self.random_state)
        models={'voting':self.create_voting_ensemble(X,y),
                'stacking':self.create_stacking_ensemble(X,y)}
        scoring={'accuracy':'accuracy','balanced_accuracy':'balanced_accuracy',
                 'f1_weighted':'f1_weighted','roc_auc_ovr':'roc_auc_ovr',
                 'precision_weighted':'precision_weighted',
                 'recall_weighted':'recall_weighted'}
        res={}
        for name,model in models.items():
            print(f"{Fore.BLUE}📊 Evaluating {name}...{Style.RESET_ALL}")
            with tqdm(total=CV_FOLDS*CV_REPEATS,desc=f"CV {name}",
                      bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.GREEN,Style.RESET_ALL)) as pb:
                cvres=cross_validate(model,X,y_enc,cv=outer,scoring=scoring,
                                     n_jobs=N_JOBS,return_train_score=False)
                pb.update(CV_FOLDS*CV_REPEATS)
            res[name]={m:{'mean':np.mean(cvres[f'test_{m}']),
                          'std': np.std(cvres[f'test_{m}'])}
                       for m in scoring}
        tr.end_step(True,"Nested CV complete")
        return res

    # -----------------------------------------------------------------------
    # Training final models
    # -----------------------------------------------------------------------
    def train_final_models(self,X:pd.DataFrame,y:pd.Series)->None:
        """
        *UNCHANGED logic* except:
          • After fitting we extract **final_feature_names_**
            (names seen by the classifiers after preprocessing)
          • These names are pushed into XGBoost’s booster so that SHAP
            recognises them.
        """
        tr=self.progress_tracker
        tr.start_step("Final Model Training","Training ensembles on full training set")

        self.label_encoder=LabelEncoder()
        y_enc=self.label_encoder.fit_transform(y)
        self.selected_features=self.boruta_feature_selection(X,y)

        Xsel=X[self.selected_features]
        Xenh=self.advanced_feature_engineering(Xsel)

        # ---- Train Voting ensemble ------------------------------------------------
        print(f"{Fore.BLUE}🏋️ Training voting ensemble...{Style.RESET_ALL}")
        self.voting_ensemble=self.create_voting_ensemble(Xenh,y)
        with tqdm(total=1,desc="Voting Train",
                  bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.GREEN,Style.RESET_ALL)) as pb:
            self.voting_ensemble.fit(Xenh,y_enc); pb.update(1)

        # ==== NEW / CHANGED ==== #
        pre_pipe_vote = self.voting_ensemble.named_estimators_['xgb'].named_steps['pre']
        self.final_feature_names_ = extract_feature_names_from_pipeline(
                                        pre_pipe_vote, list(Xenh.columns))
        # Push names into the underlying booster
        booster = self.voting_ensemble.named_estimators_['xgb'].named_steps['clf'].get_booster()
        booster.feature_names = self.final_feature_names_

        # ---- Train Stacking ensemble ---------------------------------------------
        print(f"{Fore.BLUE}🏋️ Training stacking ensemble...{Style.RESET_ALL}")
        self.stacking_ensemble=self.create_stacking_ensemble(Xenh,y)
        with tqdm(total=1,desc="Stacking Train",
                  bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.MAGENTA,Style.RESET_ALL)) as pb:
            self.stacking_ensemble.fit(Xenh,y_enc); pb.update(1)

        self.feature_names=Xenh.columns.tolist()          # original
        self.class_names=self.label_encoder.classes_
        tr.end_step(True,f"Models trained on {len(self.final_feature_names_)} final features")


    # -----------------------------------------------------------------------
    # Evaluation (aligned feature sets)
    # -----------------------------------------------------------------------
    def evaluate_models(self,Xt:pd.DataFrame,yt:pd.Series,out_dir:str)->Dict[str,Dict[str,float]]:
        tr=self.progress_tracker
        tr.start_step("Model Evaluation","Evaluating on test set")
        Xt_sel=Xt[self.selected_features]
        Xt_enh=self.advanced_feature_engineering(Xt_sel)
        Xt_enh=self._align_features(Xt_enh)             # <NEW>
        yt_enc=self.label_encoder.transform(yt)
        models={'voting':self.voting_ensemble,
                'stacking':self.stacking_ensemble}
        res={}
        for name,model in models.items():
            print(f"\n{Fore.CYAN}📊 Evaluating {name}...{Style.RESET_ALL}")
            with tqdm(total=2,desc=f"Predict {name}",
                      bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.BLUE,Style.RESET_ALL)) as pb:
                y_pred = model.predict(Xt_enh);  pb.update(1)
                y_prob = model.predict_proba(Xt_enh); pb.update(1)
            metrics={
                'accuracy':accuracy_score(yt_enc,y_pred),
                'balanced_accuracy':balanced_accuracy_score(yt_enc,y_pred),
                'f1_weighted':fbeta_score(yt_enc,y_pred,beta=1,average='weighted'),
                'precision_weighted':precision_score(yt_enc,y_pred,average='weighted',zero_division=0),
                'recall_weighted':recall_score(yt_enc,y_pred,average='weighted',zero_division=0),
                'mcc':matthews_corrcoef(yt_enc,y_pred),
                'kappa':cohen_kappa_score(yt_enc,y_pred)}
            try:
                metrics['roc_auc_ovr']=roc_auc_score(yt_enc,y_prob,multi_class='ovr',average='weighted')
            except ValueError: metrics['roc_auc_ovr']=np.nan
            colour=lambda v:Fore.GREEN if v>0.8 else (Fore.YELLOW if v>0.6 else Fore.RED)
            print(f"{Fore.GREEN}✅ {name} metrics:{Style.RESET_ALL}")
            for m,v in metrics.items():
                print(f"  {colour(v)}• {m}: {v:.4f}{Style.RESET_ALL}")
            res[name]=metrics
            self._save_evaluation_results(Xt_enh,yt_enc,y_pred,y_prob,metrics,
                                          os.path.join(out_dir,name),name)
        tr.end_step(True,f"Evaluation complete for {len(models)} models")
        return res

    # -----------------------------------------------------------------------
    # Helpers for saving & interpretation (unchanged except Xt pass)
    # -----------------------------------------------------------------------
    def _save_evaluation_results(self,Xt:pd.DataFrame,yt:np.ndarray,
                                 y_pred:np.ndarray,y_prob:np.ndarray,
                                 metrics:Dict[str,float],out_dir:str,
                                 model_name:str)->None:
        os.makedirs(out_dir,exist_ok=True)
        with open(os.path.join(out_dir,"metrics.txt"),"w") as f:
            f.write(f"=== {model_name} metrics ===\n")
            for m,v in metrics.items(): f.write(f"{m}: {v:.4f}\n")
        rep=classification_report(yt,y_pred,target_names=self.class_names,digits=4)
        with open(os.path.join(out_dir,"classification_report.txt"),"w") as f:
            f.write(rep)
        self._create_visualizations(yt,y_pred,y_prob,out_dir)
        self._perform_shap_analysis(Xt,out_dir,model_name)
        self._advanced_interpretability(Xt,out_dir,model_name)

    # --- plotting helpers (unchanged) ---
    def _create_visualizations(self,y_true,y_pred,y_prob,out_dir):
        plt.figure(figsize=(10,8))
        ConfusionMatrixDisplay.from_predictions(y_true,y_pred,display_labels=self.class_names,
                                                cmap='Blues',normalize='true')
        plt.title('Confusion Matrix (Norm)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,'confusion_matrix.png'),dpi=300); plt.close()
        if len(self.class_names)>2: self._plot_multiclass_roc(y_true,y_prob,out_dir)
        self._plot_class_distribution(y_true,y_pred,out_dir)
    def _plot_multiclass_roc(self,y_true,y_prob,out_dir):
        n=len(self.class_names); y_bin=label_binarize(y_true,classes=range(n))
        plt.figure(figsize=(12,8))
        colours=plt.cm.Set1(np.linspace(0,1,n))
        for i,c in zip(range(n),colours):
            fpr,tpr,_=roc_curve(y_bin[:,i],y_prob[:,i]); auc_val=auc(fpr,tpr)
            plt.plot(fpr,tpr,color=c,label=f'{self.class_names[i]} (AUC={auc_val:.3f})')
        plt.plot([0,1],[0,1],'k--'); plt.xlim([0,1]); plt.ylim([0,1.05])
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Multi‑class ROC')
        plt.legend(loc="lower right"); plt.grid(True,alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,'roc_curves.png'),dpi=300); plt.close()
    def _plot_class_distribution(self,y_true,y_pred,out_dir):
        fig,ax=plt.subplots(1,2,figsize=(15,6))
        true_cnt=pd.Series(y_true).value_counts().sort_index()
        pred_cnt=pd.Series(y_pred).value_counts().sort_index()
        ax[0].bar([self.class_names[i] for i in true_cnt.index],true_cnt.values,color='skyblue')
        ax[0].set_title('True'); ax[1].bar([self.class_names[i] for i in pred_cnt.index],
                                           pred_cnt.values,color='lightcoral'); ax[1].set_title('Pred')
        for a in ax: a.tick_params(axis='x',rotation=45)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,'class_distribution.png'),
                                        dpi=300); plt.close()

    # -----------------------------------------------------------------------
    # SHAP & interpretability –‑‑ UPDATED
    # -----------------------------------------------------------------------
    def _perform_shap_analysis(self,X_sample:pd.DataFrame,
                               out_dir:str,model_name:str)->None:
        """
        Generate SHAP summary plot *and* write mean absolute importance to txt.
        Now uses true column names so the plot & text are human‑readable.
        """
        print(f"{Fore.BLUE}🔍 SHAP 분석 수행 중... ({model_name}){Style.RESET_ALL}")
        try:
            # 1) Pick a small sample to speed things up
            X_s = X_sample.sample(n=min(100, len(X_sample)),
                                  random_state=RANDOM_STATE)
            # 2) Grab the correct pipeline+model
            if model_name == 'voting':
                base_pipe = self.voting_ensemble.named_estimators_['xgb']
            else:   # stacking – first estimator (xgb) as before
                base_pipe = self.stacking_ensemble.estimators_[0]

            pre = base_pipe.named_steps['pre']
            clf = base_pipe.named_steps['clf']
            # 3) Transform X and wrap back into DataFrame with *true* names
            X_proc = pre.transform(X_s)
            feat_names = extract_feature_names_from_pipeline(pre,
                                                             list(X_s.columns))
            X_proc_df = pd.DataFrame(X_proc, columns=feat_names,
                                     index=X_s.index)

            # 4) SHAP
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_proc_df)

            # 5) Plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_proc_df,
                              class_names=self.class_names,
                              show=False, max_display=20)
            plt.title(f'SHAP Summary – {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir,
                                     f'shap_{model_name}.png'),
                        dpi=300)
            plt.close()

            # ==== NEW / CHANGED ==== #
            # 6) Text export of global importance
            if isinstance(shap_values, list):   # multi‑class
                abs_vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:                               # binary / regression
                abs_vals = np.abs(shap_values)
            global_imp = abs_vals.mean(axis=0)          # mean(|SHAP|) per feat
            imp_series = pd.Series(global_imp,
                                   index=feat_names).sort_values(ascending=False)
            txt_path = os.path.join(out_dir,
                                    f'shap_{model_name}_importance.txt')
            imp_series.to_csv(txt_path, sep='\t',
                              header=['mean(|SHAP|)'])

            print(f"{Fore.GREEN}✓ SHAP 분석 및 중요도 저장 완료 → {txt_path}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}❌ SHAP 실패: {e}{Style.RESET_ALL}")

    def _advanced_interpretability(self,Xt,out_dir,model_name): pass # optional

    # -----------------------------------------------------------------------
    # Saving models
    # -----------------------------------------------------------------------
    def save_models(self,out_dir:str)->None:
        tr=self.progress_tracker; tr.start_step("Model Saving","Saving models")
        mdl_dir=os.path.join(out_dir,'models'); os.makedirs(mdl_dir,exist_ok=True)
        with tqdm(total=3,desc="Save Models",
                  bar_format="{l_bar}%s{bar}%s{r_bar}"%(Fore.CYAN,Style.RESET_ALL)) as pb:
            joblib.dump(self.voting_ensemble,os.path.join(mdl_dir,'voting.pkl'));  pb.update(1)
            joblib.dump(self.stacking_ensemble,os.path.join(mdl_dir,'stacking.pkl'));pb.update(1)
            meta={'label_encoder':self.label_encoder,
                  'feature_names':self.feature_names,
                  'class_names':self.class_names,
                  'selected_features':self.selected_features,
                  'device':self.device,'random_state':self.random_state}
            joblib.dump(meta,os.path.join(mdl_dir,'metadata.pkl')); pb.update(1)
        tr.end_step(True,f"Models saved to {mdl_dir}")

# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------
def print_system_info():
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🖥️  SYSTEM INFORMATION{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    items=[("Platform",f"{platform.system()} {platform.release()}"),
           ("Python",sys.version.split()[0]),
           ("CPU Count",str(os.cpu_count())),
           ("Available Mem",f"{psutil.virtual_memory().total/(1024**3):.1f} GB"),
           ("Mem Usage",f"{psutil.virtual_memory().percent:.1f}%"),
           ("XGBoost",xgb.__version__),
           ("Device",DEVICE.upper())]
    if TORCH_AVAILABLE:
        items.append(("PyTorch",torch.__version__))
        if torch.backends.mps.is_available(): items.append(("MPS","✅"))
        if torch.cuda.is_available(): items.append(("CUDA",f"✅ ({torch.cuda.get_device_name()})"))
    for k,v in items:
        print(f"{Fore.WHITE}📋 {k:.<20} {Fore.GREEN}{v}{Style.RESET_ALL}")
    print()

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main():
    start=time.time()
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🚀 ULTIMATE ML PIPELINE WITH PROGRESS TRACKING (FIXED){Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print_system_info()
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    pipe=UltimateMLPipeline(random_state=RANDOM_STATE)
    try:
        X,y=pipe.load_and_preprocess_data()
        pipe.progress_tracker.start_step("Data Splitting","80/20 stratified split")
        Xtr,Xts,ytr,yts=train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE,
                                         stratify=y,shuffle=True)
        pipe.progress_tracker.end_step(True,f"Train {Xtr.shape}, Test {Xts.shape}")
        _=pipe.nested_cross_validation(Xtr,ytr)
        pipe.train_final_models(Xtr,ytr)
        test_metrics=pipe.evaluate_models(Xts,yts,OUTPUT_DIR)
        pipe.save_models(OUTPUT_DIR)
        best=max(test_metrics,key=lambda k:test_metrics[k]['balanced_accuracy'])
        print(f"{Fore.GREEN}🏆 BEST MODEL: {best}{Style.RESET_ALL} "
              f"{test_metrics[best]['balanced_accuracy']:.4f}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        import traceback; traceback.print_exc()
    finally:
        pipe.progress_tracker.print_summary()
        print(f"{Fore.CYAN}📁 RESULTS SAVED TO: {OUTPUT_DIR}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}💾 FINAL MEMORY USAGE: {psutil.virtual_memory().percent:.1f}%{Style.RESET_ALL}")
        if TORCH_AVAILABLE and torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__=="__main__":
    main()
