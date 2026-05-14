###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace

import numpy as np
from requests import head
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.cli.visualise_train import TrainingPlotter
from mace.modules.loss import spectral_loss

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    filter_nonzero_weight,
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    eval_metrics["head"] = valid_loader_name
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_stress={error_stress:8.2f} meV / A^3",
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_virials_per_atom={error_virials:8.2f} meV",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_stress={error_stress:8.2f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_virials={error_virials:8.2f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye",
        )
    elif log_errors == "DipolePolarRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} me A, RMSE_polarizability_per_atom={error_polarizability:.2f} me A^2 / V",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        )

def spectral_err_log(spectral_loss,epoch):

    # print validatin spectral loss
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    logging.info(f"{inintial_phrase}: spectral loss={spectral_loss:8.8f}")

def average_loss_heads(average_loss,epoch):

    # print average validation loss for epoch
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    logging.info(f"{inintial_phrase}: average loss of all heads={average_loss:8.8f}")

def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    sigma: float,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    plotter: TrainingPlotter = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========TRAINING===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    epoch = start_epoch

    # log validation loss before _any_ training

    # properties for all heads to calculate spectral_loss for validation data
    all_heads = []
    all_energy_ref = []
    all_tdm_ref = []
    all_pos = []
    all_energy_pred = []
    all_tdm_pred = []

    # validation losses for all heads 
    valid_loss_heads = []

    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics, spectral_properties = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
        )

        if loss_fn.__class__.__name__ == "SpectralLoss":
            heads, energy_ref, tdm_ref, energy_pred, tdm_pred, pos = spectral_properties
            all_heads.append(heads)
            all_energy_ref.append(energy_ref)
            all_tdm_ref.append(tdm_ref)
            all_energy_pred.append(energy_pred)
            all_tdm_pred.append(tdm_pred)
            all_pos.append(pos)

        valid_loss_heads.append(valid_loss_head)

        valid_err_log(
            valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
        )

    # spectral loss
    if loss_fn.__class__.__name__ == "SpectralLoss":
        spectral_loss_val = validation_batches_spectral_loss([all_heads,all_energy_ref,all_tdm_ref,all_energy_pred,all_tdm_pred,all_pos],sigma=sigma)

    # best average loss used for checkpointing
    # for each epoch find average of all heads and add spectral loss
    valid_loss = torch.mean(torch.tensor(valid_loss_heads))+spectral_loss_val
    spectral_err_log(spectral_loss_val,epoch=None)
    average_loss_heads(valid_loss,epoch=None)

    # variable used for broadcast by rank == 0 if epoch loop is exited early, e.g. patience
    exit_now = torch.zeros(1, device=device) if distributed else None
    while epoch < max_num_epochs:
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on Stage Two Weights")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed=distributed,
            distributed_model=distributed_model,
            rank=rank,
        )
        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            with param_context:
                wandb_log_dict = {}

                all_heads = []
                all_energy_ref = []
                all_tdm_ref = []
                all_pos = []
                all_energy_pred = []
                all_tdm_pred = []

                valid_loss_heads = []
                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics, spectral_properties = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                    )

                    if loss_fn.__class__.__name__ == "SpectralLoss":
                        heads, energy_ref, tdm_ref, energy_pred, tdm_pred, pos = spectral_properties
                        all_heads.append(heads)
                        all_energy_ref.append(energy_ref)
                        all_tdm_ref.append(tdm_ref)
                        all_energy_pred.append(energy_pred)
                        all_tdm_pred.append(tdm_pred)
                        all_pos.append(pos)

                    valid_loss_heads.append(valid_loss_head)

                    if rank == 0:
                        valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                        
                        if log_wandb:
                            wandb_log_dict[valid_loader_name] = {
                                "epoch": epoch,
                                "valid_loss": valid_loss_head,
                                "valid_rmse_e_per_atom": eval_metrics[
                                    "rmse_e_per_atom"
                                ],
                                "valid_rmse_f": eval_metrics["rmse_f"],
                            }

                # spectral loss
                if loss_fn.__class__.__name__ == "SpectralLoss":
                    spectral_loss_val = validation_batches_spectral_loss([all_heads,all_energy_ref,all_tdm_ref,all_energy_pred,all_tdm_pred,all_pos],sigma=sigma)
                # best average loss used for checkpointing
                valid_loss = torch.mean(torch.tensor(valid_loss_heads))
                # print losses
                spectral_err_log(spectral_loss_val,epoch)
                average_loss_heads(valid_loss,epoch)         

                if plotter and epoch % plotter.plot_frequency == 0:
                    try:
                        plotter.plot(epoch, model_to_evaluate, rank)
                    except Exception as e:  # pylint: disable=broad-except
                        logging.debug(f"Plotting failed: {e}")

            if log_wandb:
                wandb.log(wandb_log_dict)
            if rank == 0:
                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if swa is not None and epoch < swa.start:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                            )
                            epoch = swa.start
                        else:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement"
                            )
                            if exit_now is not None:
                                exit_now.fill_(1)
                    if save_all_checkpoints:
                        param_context = (
                            ema.average_parameters()
                            if ema is not None
                            else nullcontext()
                        )
                        with param_context:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    param_context = (
                        ema.average_parameters() if ema is not None else nullcontext()
                    )
                    with param_context:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints
        if distributed:
            torch.distributed.barrier()
        if exit_now is not None:
            torch.distributed.broadcast(exit_now, src=0)
            if exit_now == 1:
                break

        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed: bool,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
) -> None:
    model_to_train = model if distributed_model is None else distributed_model

    if isinstance(optimizer, LBFGS):
        _, opt_metrics = take_step_lbfgs(
            model=model_to_train,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            distributed=distributed,
            rank=rank,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)

    elif loss_fn.__class__.__name__ == "SpectralLoss":

        # unpack args
        dls = {}
        for i in range(len(data_loader)):
            dls[f'es{i+1}'] = data_loader[f'es{i+1}']
       
        num_configs = len(dls['es1'].dataset)
        idx = range(0,num_configs)
        
        # define new sampler which ensures all states for each config are sampled together
        sampler = RandomSampler(idx)

        # calculate batch size based on batch_size/num_heads - raise error if not divisible 
        batch_size =  dls['es1'].batch_size
        
        assert batch_size % len(data_loader) == 0, "For spectral loss training, batch size must be divisible by the number of heads"
            
        num_configs_per_state = int(batch_size/len(data_loader))
        batch_sampler = BatchSampler(sampler, batch_size=num_configs_per_state, drop_last=False)

        all_batches = []
        for batch in batch_sampler: 
            all_batches.append(batch)       

        for batch_idx in all_batches:
            
            # alternate heads so batch is organised per config
            batch_list = [item for i in batch_idx for item in (dls[f'es{j+1}'].dataset[i] for j in range(len(data_loader)))]
           
            batch = dls['es1'].collate_fn(batch_list)

            _, opt_metrics = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if rank == 0:
                logger.log(opt_metrics)

    else: 
        for batch in data_loader:
            _, opt_metrics = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if rank == 0:
                logger.log(opt_metrics)

def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    batch_dict = batch.to_dict()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch_dict,
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        loss = loss_fn(pred=output, ref=batch)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        return loss

    loss = closure()
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    distributed: bool,
    rank: int,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    if distributed:
        global_sample_count = torch.tensor(total_sample_count, device=device)
        torch.distributed.all_reduce(
            global_sample_count, op=torch.distributed.ReduceOp.SUM
        )
        total_sample_count = global_sample_count.item()

    signal = torch.zeros(1, device=device) if distributed else None

    def closure():
        if distributed:
            if rank == 0:
                signal.fill_(1)
                torch.distributed.broadcast(signal, src=0)

            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if distributed:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        return total_loss

    if distributed:
        if rank == 0:
            loss = optimizer.step(closure)
            signal.fill_(0)
            torch.distributed.broadcast(signal, src=0)
        else:
            while True:
                # Other ranks wait for signals from rank 0
                torch.distributed.broadcast(signal, src=0)
                if signal.item() == 0:
                    break
                if signal.item() == 1:
                    loss = closure()

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    else:
        loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict

# Keep parameters frozen/active after evaluation
@contextmanager
def preserve_grad_state(model):
    # save the original requires_grad state for all parameters
    requires_grad_backup = {param: param.requires_grad for param in model.parameters()}
    try:
        # temporarily disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
        yield  # perform evaluation here
    finally:
        # restore the original requires_grad states
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad = requires_grad

def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:


    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()

    heads = []
    energy_ref = []
    tdm_ref = []
    pos = []
    energy_pred = []
    tdm_pred = []

    with preserve_grad_state(model):
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            
            if loss_fn.__class__.__name__ == "SpectralLoss":
                heads.append(batch.head)
                energy_ref.append(batch.energy)
                tdm_ref.append(batch.dipole)
                pos.append(batch.first_positions.detach())

            output = model(
                batch_dict,
                training=False,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            
            if loss_fn.__class__.__name__ == "SpectralLoss":
                energy_pred.append(output['energy'].detach())
                tdm_pred.append(output['dipole'].detach())

            avg_loss, aux = metrics(batch, output)
      
    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    spectral_properties = [heads, energy_ref, tdm_ref, energy_pred, tdm_pred, pos]

    return avg_loss, aux, spectral_properties

def flatten_property(prop):
    
    # stack all tensors for each head and batch
    stacked = torch.stack([torch.stack(batch)for batch in prop])

    # for heads and energies
    if stacked.ndim == 3:

        # (num_heads, num_batches, batch_size)
        stacked = stacked.permute(1, 2, 0)
        flattened = stacked.reshape(-1)

    # for dipoles and positions
    elif stacked.ndim == 4:

        # (num_heads, num_batches, batch_size, 3)
        stacked = stacked.permute(1, 2, 0, 3)
        flattened = stacked.reshape(-1, 3)

    else:
        raise ValueError(f"Unexpected shape {stacked.shape}")

    return flattened

def validation_batches_spectral_loss(prop,sigma):

    spectral_properties = [flatten_property(p) for p in prop]

    all_heads,all_energy_ref,all_tdm_ref,all_energy_pred,all_tdm_pred,all_positions=spectral_properties

    # reconstruct Batch for refs
    ref = SimpleNamespace(head=all_heads,energy=all_energy_ref,dipole=all_tdm_ref,positions=all_positions)

    # reconstruct TensorDict for preds
    pred = {"energy": all_energy_pred,"dipole": all_tdm_pred}

    return spectral_loss(ref,pred,sigma=sigma, grad=False)

class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "polarizability_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_polarizability_per_atom", default=[], dist_reduce_fx="cat"
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
            self.E_computed += filter_nonzero_weight(
                batch, self.delta_es, batch.weight, batch.energy_weight
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
            self.Fs_computed += filter_nonzero_weight(
                batch,
                self.delta_fs,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )
        if output.get("stress") is not None and batch.stress is not None:
            self.delta_stress.append(batch.stress - output["stress"])
            self.stress_computed += filter_nonzero_weight(
                batch, self.delta_stress, batch.weight, batch.stress_weight
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
            self.virials_computed += filter_nonzero_weight(
                batch, self.delta_virials, batch.weight, batch.virials_weight
            )
    
        if output.get("dipole") is not None and batch.dipole is not None and (self.loss_fn.__class__.__name__ == "WeightedEnergyForcesDipoleLoss" or self.loss_fn.__class__.__name__ == "DipoleSingleLoss"):
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )
            self.Mus_computed += filter_nonzero_weight(
                batch,
                self.delta_mus,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
        if (
            output.get("polarizability") is not None
            and batch.polarizability is not None
        ):
            self.delta_polarizability.append(
                batch.polarizability - output["polarizability"]
            )
            self.delta_polarizability_per_atom.append(
                (batch.polarizability - output["polarizability"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
            )
            self.polarizability_computed += filter_nonzero_weight(
                batch,
                self.delta_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )

        if output.get("dipole") is not None and batch.dipole is not None and (self.loss_fn.__class__.__name__ == "WeightedEnergyForcesDipolePhaseLessLoss" or self.loss_fn.__class__.__name__ == "DipoleSinglePhaseLessLoss" or self.loss_fn.__class__.__name__ == "SpectralLoss"):
            self.Mus_computed += 1.0
            self.mus.append(batch.dipole)   

            # calculate two different diffs and construct tensor from the min for each validation example
            minus_diff = batch.dipole - output["dipole"]
            sum_diff = batch.dipole + output["dipole"]
            norm_minus = torch.norm(minus_diff, dim=1, keepdim=True)
            norm_sum = torch.norm(sum_diff, dim=1, keepdim=True)
            opt_diff = torch.where(norm_minus <= norm_sum, minus_diff, sum_diff)            

            self.delta_mus.append(opt_diff)
            self.delta_mus_per_atom.append(
                (opt_diff)
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)
    
    def compute(self):

        class NoneMultiply:
            def __mul__(self, other):
                return NoneMultiply()

            def __rmul__(self, other):
                return NoneMultiply()

            def __imul__(self, other):
                return NoneMultiply()

            def __format__(self, format_spec):
                return str(None)

        aux = defaultdict(NoneMultiply)
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)
        if self.polarizability_computed:
            delta_polarizability = self.convert(self.delta_polarizability)
            delta_polarizability_per_atom = self.convert(
                self.delta_polarizability_per_atom
            )
            aux["mae_polarizability"] = compute_mae(delta_polarizability)
            aux["mae_polarizability_per_atom"] = compute_mae(
                delta_polarizability_per_atom
            )
            aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
            aux["rmse_polarizability_per_atom"] = compute_rmse(
                delta_polarizability_per_atom
            )
            aux["q95_polarizability"] = compute_q95(delta_polarizability)

        return aux["loss"], aux
