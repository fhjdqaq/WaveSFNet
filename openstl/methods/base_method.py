import numpy as np
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric
from torchmetrics import MeanMetric

class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        """if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None"""

        if 'weather' in args['dataname']:
            self.metric_list = args['metrics']
            self.spatial_norm = True

            # WeatherBench-M: r, t, u, v
            data_name = args.get('data_name', None)

            # single
            if data_name is None:
                self.channel_names = None

            # mv
            elif isinstance(data_name, str) and data_name == 'mv':
                self.channel_names = ['r', 't', 'u', 'v']

            elif isinstance(data_name, (list, tuple)):
                self.channel_names = list(data_name)
            else:
                self.channel_names = None

        else:
            self.metric_list = args['metrics']
            self.spatial_norm = False
            self.channel_names = None


        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []

        self.train_loss_epoch_metric = MeanMetric()
        self.val_loss_epoch_metric   = MeanMetric()
        self._train_avg_for_epoch = None


    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    

    def lr_scheduler_step(self, scheduler, metric=None):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
            return

        step_count = getattr(scheduler, "_step_count", 0)
        total = getattr(scheduler, "total_steps", None)
        if total is not None and step_count >= total:
            return

        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def on_fit_start(self):

        est = getattr(self.trainer, "estimated_stepping_batches", None)
        if est is None:
            return 

        try:
            from openstl.utils import print_log
            print_log(f"[LR-CHECK] estimated_stepping_batches = {est}")
        except Exception:
            pass

        for cfg in getattr(self.trainer, "lr_scheduler_configs", []):
            sch = getattr(cfg, "scheduler", None)
            if sch is None:
                continue
            if getattr(cfg, "interval", "epoch") != "step":
                continue
            if hasattr(sch, "total_steps"):
                ts = getattr(sch, "total_steps", None)
                if ts is None:
                    continue
                if int(ts) != int(est):
                    try:
                        from openstl.utils import print_log
                        print_log(f"[LR-CHECK] mismatch: scheduler.total_steps={ts} != estimated={est} -> FIXED")
                    except Exception:
                        pass
                    sch.total_steps = int(est)
                    if getattr(sch, "_step_count", 0) > sch.total_steps:
                        setattr(sch, "_step_count", sch.total_steps)
                else:
                    try:
                        from openstl.utils import print_log
                        print_log(f"[LR-CHECK] OK: scheduler.total_steps matches estimated ({est})")
                    except Exception:
                        pass




    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    """def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss"""
  
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)

        self.log('val_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.val_loss_epoch_metric.update(loss.detach())
        return loss

    def on_validation_epoch_end(self):
        self.log('epoch', int(self.current_epoch + 1),
             on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        val_avg = self.val_loss_epoch_metric.compute()
        self.log('val_loss', val_avg, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)


        if self.trainer.is_global_zero:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            epoch_display = self.current_epoch + 1  
            train_avg = self._train_avg_for_epoch  

            if train_avg is None:
                t = self.trainer.callback_metrics.get('train_loss_epoch')
                train_avg = float(t.detach().cpu()) if t is not None else float('nan')

            from openstl.utils import print_log
            try:
                val_avg_float = float(val_avg.detach().cpu())
            except Exception:
                val_avg_float = float(val_avg)

            print_log(
                f"Epoch {epoch_display}: Lr: {lr:.7f} | "
                f"Train Loss: {train_avg:.7f} | "
                f"Vali  Loss: {val_avg_float:.7f}"
            )
        self.val_loss_epoch_metric.reset()
        self._train_avg_for_epoch = None


        
    def on_validation_start(self):
        train_avg = self.train_loss_epoch_metric.compute()

        try:
            train_avg_float = float(train_avg.detach().cpu())
        except Exception:
            train_avg_float = float(train_avg)

        self.log(
            'train_loss_epoch', train_avg,
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )

        self._train_avg_for_epoch = train_avg_float

        self.train_loss_epoch_metric.reset()






    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
            channel_names=self.channel_names, spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None))
        
        #results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])
 
        results_all['metrics'] = np.array([eval_res[m] for m in self.metric_list], dtype=np.float32)


        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        return results_all
        