import os.path as osp
import numpy as np
import pandas as pd

from .manager import BaseManager

class Logging(BaseManager):
    def __call__(self):
        print("Logging")
        if self.get("log_flag"):
            if self.get("mlflow"):
                mlflow.set_tracking_uri(osp.join(self.WORK_DIR, "mlruns"))
                self.create_mlflow()

    def calc_cv(self):
        preds = []
        cv_scores = []
        train_df = pd.read_csv(osp.join(self.ROOT, "input", self.raw_dirname, "train.csv"))
        
        for seed in self.seeds:
            # cvを1ファイルずつに変更
            cv_df = pd.read_csv(osp.join(self.ROOT, "input", "cv", f"{self.cv_type}_{seed}.csv" ))
            #mask = train_y[cv_feat] != -1
            cv_df["pred"] = np.nan
            cols = [f"pred_{n}" for n in range(self.out_size)]
            for col in cols:
                cv_df[col] = np.nan

            for fold in range(self.n_splits):
                val_preds = pd.read_csv(osp.join(self.val_preds_path, f"preds_{seed}_{fold}.csv"))
                cv_df.loc[cv_df["fold"] == fold, cols] = val_preds[cols].values
            cv_df[cols].to_csv(osp.join(self.val_preds_path, f"oof_preds_{seed}.csv"), index=False) 
            cv_df["pred"] =  np.argmax(cv_df[cols].values, axis=1)
            cv_score = accuracy_score(cv_df["label"].values, cv_df["pred"].values)
            cv_scores.append(cv_score)
            print(f"seed {seed}, cv : {cv_score}")
            preds.append(cv_df[cols].values.copy()) # copy!!!!!
        preds = np.mean(np.array(preds), axis=0)
        preds = np.argmax(preds, axis=1)
        preds = pd.DataFrame(preds, columns=["pred"])
        preds.to_csv(osp.join(self.val_preds_path, "oof_preds.csv"), index=False)
        try:
            cv_score = accuracy_score(train_df["label"], preds["pred"])
        except:
            cv_score = np.mean(cv_scores)
            print("mean cv")
            
        print(f"final cv : {cv_score}")
        return cv_score, cv_scores

    def create_mlflow(self):
        with mlflow.start_run():
            mlflow.log_param("exp_name", self.exp_name)
            mlflow.log_param("model_param", self.model_param)
            mlflow.log_param("features", self.feats)
            mlflow.log_param("seeds", self.seeds)
            mlflow.log_param("nfolds", self.nfolds)
            mlflow.log_param("cv_type", self.cv)
            mlflow.log_param("cv_scores", self.cv_scores)

            mlflow.log_metric("cv_score", self.cv_score)
            #log_metric("cv_scores", self.cv_scores)

            try:
                mlflow.log_artifact(self.feature_importances_fname)
            except:
                pass
            mlflow.log_artifact(self.submission_fname)


    

