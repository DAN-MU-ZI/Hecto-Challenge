import mlflow

class MLflowCallback:
    def __init__(self, model, run_name, fold=None):
        self.model = model
        self.run_name = run_name
        self.best_val_logloss = float('inf')
        self.fold = fold
    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        train_logloss, train_acc = train_metrics
        val_logloss, val_acc = val_metrics
        
        mlflow.log_metrics({
            "train_logloss": train_logloss,
            "train_accuracy": train_acc,
            "val_logloss": val_logloss,
            "val_accuracy": val_acc
        }, step=epoch)
