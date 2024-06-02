import argparse
from omegaconf import OmegaConf, DictConfig
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn

from dataset import TPMixerDataModule
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import warnings
from model import TPMixerSeqCls, TPMixerTokenCls
# import torch.quantization as quantization
warnings.filterwarnings("ignore", category=UserWarning, message="No audio backend is available.")
torch.set_float32_matmul_precision('medium')


class TPModulePLStyle(LightningModule):
    def __init__(self, optimizer_cfg: DictConfig, model_cfg: DictConfig, **kwargs):
        super(TPModulePLStyle, self).__init__(**kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model_cfg = model_cfg

        self.model = TPMixerSeqCls(
            model_cfg.bottleneck,
            model_cfg.mixer,
            model_cfg.sequence_cls,
            model_cfg.topic
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def metrics(self, outputs):
        ACC, Macro_Precision, Macro_F1, Macro_recall, length = 0, 0, 0, 0, len(outputs)
        for output in outputs:
            ACC += accuracy_score(output['targets'], output['predicts'])
            Macro_Precision += precision_score(output['targets'], output['predicts'], average='macro', zero_division=1)
            Macro_F1 += f1_score(output['targets'], output['predicts'], average='macro', zero_division=1)
            Macro_recall += recall_score(output['targets'], output['predicts'], average='macro', zero_division=1)
        return {
            "ACC": ACC / length,
            "Macro_Precision": Macro_Precision / length,
            "Macro_F1": Macro_F1 / length,
            "Macro_Recall": Macro_recall / length
        }

    def share_eval(self, batch, batch_idx):
        x1,x2, targets = batch["inputs1"], batch["inputs2"], batch["targets"]
        outs = self.model(x1,x2)
        loss = F.cross_entropy(outs, targets.long())
        predict = torch.argmax(outs, dim=1)
        return {
            'batch_ids': batch_idx,
            'predicts': predict.detach().cpu().numpy(),
            'targets': targets.detach().cpu().numpy(),
            'loss': loss
        }

    def training_step(self, batch: dict, batch_idx: int):
        results = self.share_eval(batch, batch_idx)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.training_step_outputs.append(results)
        return results

    def on_training_epoch_end(self) -> None:
        accuracy = self.metrics(self.training_step_outputs)
        self.log('train_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        results = self.share_eval(batch, batch_idx)
        self.log('val_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self) -> None:
        accuracy = self.metrics(self.validation_step_outputs)
        self.log('val_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch: dict, batch_idx: int):
        results = self.share_eval(batch, batch_idx)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.append(results)
        return results

    def on_test_epoch_end(self) -> None:
        accuracy = self.metrics(self.test_step_outputs)
        self.log('test_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer


class TPModuleTokenPLStyle(LightningModule):
    def __init__(self, optimizer_cfg: DictConfig, model_cfg: DictConfig, **kwargs):
        super(TPModuleTokenPLStyle, self).__init__(**kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model_cfg = model_cfg

        self.model = TPMixerTokenCls(
            model_cfg.bottleneck,
            model_cfg.mixer,
            model_cfg.token_cls,
            model_cfg.topic
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def metrics(self, outputs):
        corr = 0
        all = 0
        for output in outputs:
            corr += output['corr']
            all += output['all']
        return {
            'acc': corr / all
        }


    def share_eval(self, batch, batch_idx):
        x1, x2, targets = batch["inputs1"], batch["inputs2"], batch["targets"]
        # qqq=torch.quantization.quantize_dynamic(
        #     self.model,
        #     qconfig_spec=None,
        #     dtype=torch.qint8,
        #     mapping=None,
        #     inplace=False)
        # logits = qqq(x1, x2)

        # print(qqq)


        logits = self.model(x1, x2)
        loss = F.cross_entropy(logits.transpose(-1, -2), targets, ignore_index=-1)
        corr = torch.sum(torch.logical_and(logits.argmax(dim=-1) == targets, targets > 0))
        all = torch.sum(targets > 0)

        return {
                'loss': loss,
                'corr': corr,
                'all': all
        }

    def training_step(self, batch: dict, batch_idx: int):
        results = self.share_eval(batch, batch_idx)
        self.training_step_outputs.append(results)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return results

    def on_training_epoch_end(self) -> None:
        accuracy = self.metrics(self.training_step_outputs)
        self.log('train_acc', accuracy['acc'], prog_bar=True, logger=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        results = self.share_eval(batch, batch_idx)
        # self.log('val_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self) -> None:
        accuracy = self.metrics(self.validation_step_outputs)
        self.log('val_acc', accuracy['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('val_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('val_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch: dict, batch_idx: int):
        results = self.share_eval(batch, batch_idx)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.append(results)
        return results

    def on_test_epoch_end(self) -> None:
        accuracy = self.metrics(self.test_step_outputs)
        self.log('test_acc', accuracy['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-d', "--cfg", type=str, default="sst2")
    args.add_argument('-s', "--sub", type=str)
    args.add_argument('-p', '--ckpt', type=str)
    args.add_argument('-t', '--train', type=str, default="train")
    return args.parse_args()


def get_module_cls(type: str):
    if type in ['imdb', '20NG',  'ag_news', 'sst2',
                'cola',  'qqp', 'hyperpartisan', 'dbpedia', 'amazon']:
        return TPModulePLStyle
    else:
        return TPModuleTokenPLStyle


if __name__ == "__main__":
    pl.seed_everything(1)
    args = parse_args()
    cfg = OmegaConf.load("./cfg/{}.yml".format(args.cfg))
    vocab_cfg = cfg.vocab
    train_cfg = cfg.train
    model_cfg = cfg.model
    print(model_cfg)
    module_cls = get_module_cls(train_cfg.type)
    if args.ckpt:
        train_module = module_cls.load_from_checkpoint(args.ckpt, optimizer_cfg=train_cfg.optimizer,
                                                       model_cfg=model_cfg)  # .to('mps')
    else:
        train_module = module_cls(optimizer_cfg=train_cfg.optimizer, model_cfg=model_cfg)




    # Load int8 model
    # state_dict = torch.load('./openpose_vgg_quant.pth')

    # model_fp32 = get_pose_model()
    # model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # model_fp32_prepared = torch.quantization.prepare(model_fp32)
    # model_int8 = torch.quantization.convert(model_fp32_prepared)
    # model_int8.load_state_dict(state_dict)
    # model = model_int8
    # model.eval()

    # print(train_module)
    # params = train_module.state_dict()
    # for k, v in params.items():
    #     print(k)  # 打印网络中的变量名



    # print(params['model.TP_mixer.mixer.mixers.0.mlp_2.layers.0.weight'])  # 打印conv1的weight
    # vars(train_module.fc)
    # print(params['conv1.bias'])  # 打印conv1的bias


    data_module = TPMixerDataModule(cfg.vocab, train_cfg, model_cfg)

    # torch.save(train_module, 'quantized_model.ckpt')

    trainer = pl.Trainer(
        # accelerator='ddp',
        # amp_backend='native',
        # amp_level='O2',
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_acc',
                filename=args.cfg + args.sub + '-best-news-{epoch:03d}-{val_acc:.4f}-{val_f1:.4f}',
                save_top_k=1,
                mode='max',
                save_last=True
            ),
        ],
        enable_checkpointing=True,
        accelerator='gpu',
        devices=1,

        # accelerator='cpu',

        log_every_n_steps=train_cfg.log_interval_steps,
        logger=pl.loggers.TensorBoardLogger(train_cfg.tensorboard_path, args.cfg),
        max_epochs=train_cfg.epochs,
        check_val_every_n_epoch=1,
        # limit_train_batches=0.5,
        # limit_val_batches=0.5
    )
    if args.train == 'train':
        trainer.fit(train_module, data_module)
    if args.train == 'test':

        trainer.test(train_module, data_module)
        # trainer.save_checkpoint('./logs/mtop/quantized_model_new/MTOP-th-multi-1127-quantized_model.ckpt')