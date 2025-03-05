###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import ast
import glob
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

import torch.distributed
import torch.nn.functional
from e3nn.util import jit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, tools
from mace.calculators.foundations_models import mace_mp, mace_off
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric
from mace.tools.model_script_utils import configure_model
from mace.tools.multihead_tools import (
    HeadConfig,
    assemble_mp_data,
    dict_head_to_dataclass,
    prepare_default_head,
)
from mace.tools.scripts_utils import (
    LRScheduler,
    check_path_ase_read,
    convert_to_json_format,
    dict_to_array,
    extract_config_mace_model,
    get_atomic_energies,
    get_avg_num_neighbors,
    get_config_type_weights,
    get_dataset_from_xyz,
    get_files_with_suffix,
    get_loss_fn,
    get_optimizer,
    get_params_options,
    get_swa,
    print_git_commit,
    remove_pt_head,
    setup_wandb,
)
from mace.tools.slurm_distributed import DistributedEnvironment
from mace.tools.tables_utils import create_error_table
from mace.tools.utils import AtomicNumberTable


def main() -> None:
    """
    This script runs the training/fine tuning for mace
    """
    args = tools.build_default_arg_parser().parse_args()
    run(args)


def process_file_path(file_path: Union[str, List[str]]) -> List[str]:
    """
    Process a file path or list of file paths to ensure they are valid.

    Args:
        file_path: A single file path or list of file paths

    Returns:
        A list of valid file paths
    """
    if file_path is None:
        return []

    # Ensure we're working with a list
    if isinstance(file_path, str):
        file_paths = [file_path]
    else:
        file_paths = file_path

    # Validate paths
    for path in file_paths:
        if not os.path.exists(path) and not glob.glob(path):
            logging.warning(f"Path does not exist: {path}")

    return file_paths


def combine_xyz_datasets(file_paths: List[str], **kwargs) -> data.Configurations:
    """
    Load and combine multiple xyz files into a single dataset.

    Args:
        file_paths: List of xyz file paths
        **kwargs: Additional arguments for get_dataset_from_xyz

    Returns:
        Combined configurations
    """
    combined_train = []
    combined_valid = []
    combined_tests = []
    combined_atomic_energies = {}

    for path in file_paths:
        collections, atomic_energies_dict = get_dataset_from_xyz(
            train_path=path, **kwargs
        )

        # Merge collections
        combined_train.extend(collections.train)
        combined_valid.extend(collections.valid)

        # Merge tests, preserving test names
        for name, test_configs in collections.tests:
            existing = next(
                (i for i, (n, _) in enumerate(combined_tests) if n == name), None
            )
            if existing is not None:
                combined_tests[existing][1].extend(test_configs)
            else:
                combined_tests.append((name, test_configs))

        # Merge atomic energies (preferring later files if conflicts)
        combined_atomic_energies.update(atomic_energies_dict)

    # Create a combined collections object
    combined_collections = data.tools.SubsetCollection(
        train=combined_train, valid=combined_valid, tests=combined_tests
    )

    return combined_collections, combined_atomic_energies


def load_datasets_for_head(head_config, z_table, r_max, heads, args):
    """
    Load datasets for a head configuration, handling multiple files if provided.

    Args:
        head_config: Head configuration
        z_table: Atomic number table
        r_max: Maximum radius
        heads: List of all head names

    Returns:
        (train_dataset, valid_dataset): Tuple of training and validation datasets
    """
    train_datasets = []
    valid_datasets = []

    # Process train files
    train_files = process_file_path(head_config.train_file)
    valid_files = process_file_path(head_config.valid_file)

    # Handle XYZ files (need special processing to combine before converting to AtomicData)
    xyz_train_files = [f for f in train_files if check_path_ase_read(f)]
    non_xyz_train_files = [f for f in train_files if not check_path_ase_read(f)]

    if xyz_train_files:
        config_type_weights = get_config_type_weights(head_config.config_type_weights)

        # Process all XYZ files together to create a single dataset
        collections, atomic_energies_dict = combine_xyz_datasets(
            file_paths=xyz_train_files,
            work_dir=args.work_dir,
            valid_path=None,  # We'll handle valid separately
            valid_fraction=head_config.valid_fraction,
            config_type_weights=config_type_weights,
            test_path=head_config.test_file,
            seed=args.seed,
            energy_key=head_config.energy_key,
            forces_key=head_config.forces_key,
            stress_key=head_config.stress_key,
            virials_key=head_config.virials_key,
            dipole_key=head_config.dipole_key,
            charges_key=head_config.charges_key,
            head_name=head_config.head_name,
            keep_isolated_atoms=head_config.keep_isolated_atoms,
        )

        # Save collections to head_config for later use
        head_config.collections = collections
        head_config.atomic_energies_dict = atomic_energies_dict

        # Convert configurations to AtomicData
        train_atomic_data = [
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=r_max, heads=heads
            )
            for config in collections.train
        ]

        valid_atomic_data = [
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=r_max, heads=heads
            )
            for config in collections.valid
        ]

        if train_atomic_data:
            train_datasets.append(train_atomic_data)
        if valid_atomic_data:
            valid_datasets.append(valid_atomic_data)

    # Process each non-XYZ file
    for file_path in non_xyz_train_files:
        if file_path.endswith(".h5"):
            train_datasets.append(
                data.HDF5Dataset(
                    file_path,
                    r_max=r_max,
                    z_table=z_table,
                    heads=heads,
                    head=head_config.head_name,
                )
            )
        elif file_path.endswith("_lmdb"):
            train_datasets.append(
                data.LMDBDataset(
                    file_path,
                    r_max=r_max,
                    z_table=z_table,
                    heads=heads,
                    head=head_config.head_name,
                )
            )
        else:  # Assume it's a directory of sharded h5 files
            train_datasets.append(
                data.dataset_from_sharded_hdf5(
                    file_path,
                    r_max=r_max,
                    z_table=z_table,
                    heads=heads,
                    head=head_config.head_name,
                )
            )

    # Process validation files separately
    for file_path in valid_files:
        if check_path_ase_read(file_path):
            # For XYZ files, we need to load and convert
            config_type_weights = get_config_type_weights(
                head_config.config_type_weights
            )
            _, valid_configs = data.load_from_xyz(
                file_path=file_path,
                config_type_weights=config_type_weights,
                energy_key=head_config.energy_key,
                forces_key=head_config.forces_key,
                stress_key=head_config.stress_key,
                virials_key=head_config.virials_key,
                dipole_key=head_config.dipole_key,
                charges_key=head_config.charges_key,
                head_key="head",
                head_name=head_config.head_name,
                extract_atomic_energies=False,
                keep_isolated_atoms=head_config.keep_isolated_atoms,
            )
            valid_atomic_data = [
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=r_max, heads=heads
                )
                for config in valid_configs
            ]
            valid_datasets.append(valid_atomic_data)
        elif file_path.endswith(".h5"):
            valid_datasets.append(
                data.HDF5Dataset(
                    file_path,
                    r_max=r_max,
                    z_table=z_table,
                    heads=heads,
                    head=head_config.head_name,
                )
            )
        elif file_path.endswith("_lmdb"):
            valid_datasets.append(
                data.LMDBDataset(
                    file_path,
                    r_max=r_max,
                    z_table=z_table,
                    heads=heads,
                    head=head_config.head_name,
                )
            )
        else:  # Assume it's a directory of sharded h5 files
            valid_datasets.append(
                data.dataset_from_sharded_hdf5(
                    file_path,
                    r_max=r_max,
                    z_table=z_table,
                    heads=heads,
                    head=head_config.head_name,
                )
            )

    # Combine datasets
    train_dataset = _combine_datasets(train_datasets)
    valid_dataset = _combine_datasets(valid_datasets)

    return train_dataset, valid_dataset


def _combine_datasets(datasets):
    """Helper function to combine datasets of various types"""
    if not datasets:
        return []

    # Handle lists of atomic data (from XYZ files)
    flattened_datasets = []
    for dataset in datasets:
        if isinstance(dataset, list):
            flattened_datasets.extend(dataset)
        else:
            flattened_datasets.append(dataset)

    # If we only have lists of atomic data, return the flattened list
    if all(isinstance(dataset, data.AtomicData) for dataset in flattened_datasets):
        return flattened_datasets

    # Otherwise use ConcatDataset
    return ConcatDataset(flattened_datasets)


def run(args: argparse.Namespace) -> None:
    """
    This script runs the training/fine tuning for mace
    """
    tag = tools.get_tag(name=args.name, seed=args.seed)
    args, input_log_messages = tools.check_args(args)

    if args.device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError as e:
            raise ImportError(
                "Error: Intel extension for PyTorch not found, but XPU device was specified"
            ) from e
    if args.distributed:
        try:
            distr_env = DistributedEnvironment()
        except Exception as e:  # pylint: disable=W0703
            logging.error(f"Failed to initialize distributed environment: {e}")
            return
        world_size = distr_env.world_size
        local_rank = distr_env.local_rank
        rank = distr_env.rank
        if rank == 0:
            print(distr_env)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = int(0)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=rank)
    logging.info("===========VERIFYING SETTINGS===========")
    for message, loglevel in input_log_messages:
        logging.log(level=loglevel, msg=message)

    if args.distributed:
        torch.cuda.set_device(local_rank)
        logging.info(f"Process group initialized: {torch.distributed.is_initialized()}")
        logging.info(f"Processes: {world_size}")

    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.debug(f"Configuration: {args}")

    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)
    commit = print_git_commit()
    model_foundation: Optional[torch.nn.Module] = None
    if args.foundation_model is not None:
        if args.foundation_model in ["small", "medium", "large"]:
            logging.info(
                f"Using foundation model mace-mp-0 {args.foundation_model} as initial checkpoint."
            )
            calc = mace_mp(
                model=args.foundation_model,
                device=args.device,
                default_dtype=args.default_dtype,
            )
            model_foundation = calc.models[0]
        elif args.foundation_model in ["small_off", "medium_off", "large_off"]:
            model_type = args.foundation_model.split("_")[0]
            logging.info(
                f"Using foundation model mace-off-2023 {model_type} as initial checkpoint. ASL license."
            )
            calc = mace_off(
                model=model_type,
                device=args.device,
                default_dtype=args.default_dtype,
            )
            model_foundation = calc.models[0]
        else:
            model_foundation = torch.load(
                args.foundation_model, map_location=args.device
            )
            logging.info(
                f"Using foundation model {args.foundation_model} as initial checkpoint."
            )
        args.r_max = model_foundation.r_max.item()
        if (
            args.foundation_model not in ["small", "medium", "large"]
            and args.pt_train_file is None
        ):
            logging.warning(
                "Using multiheads finetuning with a foundation model that is not a Materials Project model, need to provied a path to a pretraining file with --pt_train_file."
            )
            args.multiheads_finetuning = False
        if args.multiheads_finetuning:
            assert (
                args.E0s != "average"
            ), "average atomic energies cannot be used for multiheads finetuning"
            # check that the foundation model has a single head, if not, use the first head
            if not args.force_mh_ft_lr:
                logging.info(
                    "Multihead finetuning mode, setting learning rate to 0.001 and EMA to True. To use a different learning rate, set --force_mh_ft_lr=True."
                )
                args.lr = 0.0001
                args.ema = True
                args.ema_decay = 0.99999
            logging.info(
                "Using multiheads finetuning mode, setting learning rate to 0.001 and EMA to True"
            )
            if hasattr(model_foundation, "heads"):
                if len(model_foundation.heads) > 1:
                    logging.warning(
                        "Mutlihead finetuning with models with more than one head is not supported, using the first head as foundation head."
                    )
                    model_foundation = remove_pt_head(
                        model_foundation, args.foundation_head
                    )
    else:
        args.multiheads_finetuning = False

    if args.heads is not None:
        args.heads = ast.literal_eval(args.heads)
    else:
        args.heads = prepare_default_head(args)

    logging.info("===========LOADING INPUT DATA===========")
    heads = list(args.heads.keys())
    logging.info(f"Using heads: {heads}")
    head_configs: List[HeadConfig] = []
    for head, head_args in args.heads.items():
        logging.info(f"=============    Processing head {head}     ===========")
        head_config = dict_head_to_dataclass(head_args, head, args)
        if head_config.statistics_file is not None:
            with open(head_config.statistics_file, "r") as f:  # pylint: disable=W1514
                statistics = json.load(f)
            logging.info("Using statistics json file")
            head_config.r_max = (
                statistics["r_max"] if args.foundation_model is None else args.r_max
            )
            head_config.atomic_numbers = statistics["atomic_numbers"]
            head_config.mean = statistics["mean"]
            head_config.std = statistics["std"]
            head_config.avg_num_neighbors = statistics["avg_num_neighbors"]
            head_config.compute_avg_num_neighbors = False
            if isinstance(statistics["atomic_energies"], str) and statistics[
                "atomic_energies"
            ].endswith(".json"):
                with open(statistics["atomic_energies"], "r", encoding="utf-8") as f:
                    atomic_energies = json.load(f)
                head_config.E0s = atomic_energies
                head_config.atomic_energies_dict = ast.literal_eval(atomic_energies)
            else:
                head_config.E0s = statistics["atomic_energies"]
                head_config.atomic_energies_dict = ast.literal_eval(
                    statistics["atomic_energies"]
                )

        # Data preparation for single XYZ files - for compatibility with old code paths
        # New multi-file processing is done later
        single_train_file = (
            head_config.train_file if isinstance(head_config.train_file, str) else None
        )
        if single_train_file and check_path_ase_read(single_train_file):
            single_valid_file = (
                head_config.valid_file
                if isinstance(head_config.valid_file, str)
                else None
            )
            if single_valid_file is not None:
                assert check_path_ase_read(
                    single_valid_file
                ), "valid_file if given must be same format as train_file"
            config_type_weights = get_config_type_weights(
                head_config.config_type_weights
            )
            collections, atomic_energies_dict = get_dataset_from_xyz(
                work_dir=args.work_dir,
                train_path=single_train_file,
                valid_path=single_valid_file,
                valid_fraction=head_config.valid_fraction,
                config_type_weights=config_type_weights,
                test_path=head_config.test_file,
                seed=args.seed,
                energy_key=head_config.energy_key,
                forces_key=head_config.forces_key,
                stress_key=head_config.stress_key,
                virials_key=head_config.virials_key,
                dipole_key=head_config.dipole_key,
                charges_key=head_config.charges_key,
                head_name=head_config.head_name,
                keep_isolated_atoms=head_config.keep_isolated_atoms,
            )
            head_config.collections = collections
            head_config.atomic_energies_dict = atomic_energies_dict
            logging.info(
                f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
                f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}],"
            )
        head_configs.append(head_config)

    # Check if we have valid training data sizes
    total_train_size = 0
    total_valid_size = 0
    for head_config in head_configs:
        # Skip XYZ files that were already processed
        single_train_file = (
            head_config.train_file if isinstance(head_config.train_file, str) else None
        )
        if single_train_file and check_path_ase_read(single_train_file):
            total_train_size += len(head_config.collections.train)
            total_valid_size += len(head_config.collections.valid)

    if total_train_size > 0 and total_train_size < args.batch_size:
        logging.error(
            f"Batch size ({args.batch_size}) is larger than the number of training data ({total_train_size})"
        )
    if total_valid_size > 0 and total_valid_size < args.valid_batch_size:
        logging.warning(
            f"Validation batch size ({args.valid_batch_size}) is larger than the number of validation data ({total_valid_size})"
        )

    if args.multiheads_finetuning:
        logging.info(
            "==================Using multiheads finetuning mode=================="
        )
        args.loss = "universal"
        if (
            args.foundation_model in ["small", "medium", "large"]
            or args.pt_train_file == "mp"
        ):
            logging.info(
                "Using foundation model for multiheads finetuning with Materials Project data"
            )
            heads = list(dict.fromkeys(["pt_head"] + heads))
            head_config_pt = HeadConfig(
                head_name="pt_head",
                E0s="foundation",
                statistics_file=args.statistics_file,
                compute_avg_num_neighbors=False,
                avg_num_neighbors=model_foundation.interactions[0].avg_num_neighbors,
            )
            collections = assemble_mp_data(args, tag, head_configs)
            head_config_pt.collections = collections
            head_config_pt.train_file = f"mp_finetuning-{tag}.xyz"
            head_configs.append(head_config_pt)
        else:
            logging.info(
                f"Using foundation model for multiheads finetuning with {args.pt_train_file}"
            )
            heads = list(dict.fromkeys(["pt_head"] + heads))
            collections, atomic_energies_dict = get_dataset_from_xyz(
                work_dir=args.work_dir,
                train_path=args.pt_train_file,
                valid_path=args.pt_valid_file,
                valid_fraction=args.valid_fraction,
                config_type_weights=None,
                test_path=None,
                seed=args.seed,
                energy_key=args.energy_key,
                forces_key=args.forces_key,
                stress_key=args.stress_key,
                virials_key=args.virials_key,
                dipole_key=args.dipole_key,
                charges_key=args.charges_key,
                head_name="pt_head",
                keep_isolated_atoms=args.keep_isolated_atoms,
            )
            head_config_pt = HeadConfig(
                head_name="pt_head",
                train_file=args.pt_train_file,
                valid_file=args.pt_valid_file,
                E0s="foundation",
                statistics_file=args.statistics_file,
                valid_fraction=args.valid_fraction,
                config_type_weights=None,
                energy_key=args.energy_key,
                forces_key=args.forces_key,
                stress_key=args.stress_key,
                virials_key=args.virials_key,
                dipole_key=args.dipole_key,
                charges_key=args.charges_key,
                keep_isolated_atoms=args.keep_isolated_atoms,
                collections=collections,
                avg_num_neighbors=model_foundation.interactions[0].avg_num_neighbors,
                compute_avg_num_neighbors=False,
            )
            head_config_pt.collections = collections
            head_configs.append(head_config_pt)

        if total_train_size > 0:  # Only check if we have processed some training data
            ratio_pt_ft = total_train_size / len(head_config_pt.collections.train)
            if ratio_pt_ft < 0.1:
                logging.warning(
                    f"Ratio of the number of configurations in the training set and the in the pt_train_file is {ratio_pt_ft}, "
                    f"increasing the number of configurations in the pt_train_file by a factor of {int(0.1 / ratio_pt_ft)}"
                )
                for head_config in head_configs:
                    if head_config.head_name == "pt_head":
                        continue
                    # Only replicate if we have collections
                    if hasattr(head_config, "collections") and head_config.collections:
                        head_config.collections.train += (
                            head_config.collections.train * int(0.1 / ratio_pt_ft)
                        )
            logging.info(
                f"Total number of configurations in pretraining: train={len(head_config_pt.collections.train)}, valid={len(head_config_pt.collections.valid)}"
            )

    # Atomic number table
    # yapf: disable
    for head_config in head_configs:
        # Check if train_file is a list of files
        is_train_file_list = isinstance(head_config.train_file, list)
        train_file_ref = head_config.train_file[0] if is_train_file_list else head_config.train_file
        
        if head_config.atomic_numbers is None:
            assert check_path_ase_read(train_file_ref), "Must specify atomic_numbers when using .h5 train_file input"
            z_table_head = tools.get_atomic_number_table_from_zs(
                z
                for configs in (head_config.collections.train, head_config.collections.valid)
                for config in configs
                for z in config.atomic_numbers
            )
            head_config.atomic_numbers = z_table_head.zs
            head_config.z_table = z_table_head
        else:
            if head_config.statistics_file is None:
                logging.info("Using atomic numbers from command line argument")
            else:
                logging.info("Using atomic numbers from statistics file")
            zs_list = ast.literal_eval(head_config.atomic_numbers)
            assert isinstance(zs_list, list)
            z_table_head = tools.AtomicNumberTable(zs_list)
            head_config.atomic_numbers = zs_list
            head_config.z_table = z_table_head
    # yapf: enable
    all_atomic_numbers = set()
    for head_config in head_configs:
        all_atomic_numbers.update(head_config.atomic_numbers)
    z_table = AtomicNumberTable(sorted(list(all_atomic_numbers)))
    if args.foundation_model_elements and model_foundation:
        z_table = AtomicNumberTable(sorted(model_foundation.atomic_numbers.tolist()))
    logging.info(f"Atomic Numbers used: {z_table.zs}")

    # Atomic energies
    atomic_energies_dict = {}
    for head_config in head_configs:
        if (
            head_config.atomic_energies_dict is None
            or len(head_config.atomic_energies_dict) == 0
        ):
            assert head_config.E0s is not None, "Atomic energies must be provided"
            if (
                hasattr(head_config, "collections")
                and head_config.collections
                and head_config.E0s.lower() != "foundation"
            ):
                atomic_energies_dict[head_config.head_name] = get_atomic_energies(
                    head_config.E0s, head_config.collections.train, head_config.z_table
                )
            elif head_config.E0s.lower() == "foundation":
                assert args.foundation_model is not None
                z_table_foundation = AtomicNumberTable(
                    [int(z) for z in model_foundation.atomic_numbers]
                )
                foundation_atomic_energies = (
                    model_foundation.atomic_energies_fn.atomic_energies
                )
                if foundation_atomic_energies.ndim > 1:
                    foundation_atomic_energies = foundation_atomic_energies.squeeze()
                    if foundation_atomic_energies.ndim == 2:
                        foundation_atomic_energies = foundation_atomic_energies[0]
                        logging.info(
                            "Foundation model has multiple heads, using the first head as foundation E0s."
                        )
                atomic_energies_dict[head_config.head_name] = {
                    z: foundation_atomic_energies[
                        z_table_foundation.z_to_index(z)
                    ].item()
                    for z in z_table.zs
                }
            else:
                atomic_energies_dict[head_config.head_name] = get_atomic_energies(
                    head_config.E0s, None, head_config.z_table
                )
        else:
            atomic_energies_dict[head_config.head_name] = (
                head_config.atomic_energies_dict
            )

    # Atomic energies for multiheads finetuning
    if args.multiheads_finetuning:
        assert (
            model_foundation is not None
        ), "Model foundation must be provided for multiheads finetuning"
        z_table_foundation = AtomicNumberTable(
            [int(z) for z in model_foundation.atomic_numbers]
        )
        foundation_atomic_energies = model_foundation.atomic_energies_fn.atomic_energies
        if foundation_atomic_energies.ndim > 1:
            foundation_atomic_energies = foundation_atomic_energies.squeeze()
            if foundation_atomic_energies.ndim == 2:
                foundation_atomic_energies = foundation_atomic_energies[0]
                logging.info(
                    "Foundation model has multiple heads, using the first head as foundation E0s."
                )
        atomic_energies_dict["pt_head"] = {
            z: foundation_atomic_energies[z_table_foundation.z_to_index(z)].item()
            for z in z_table.zs
        }

    # Padding atomic energies if keeping all elements of the foundation model
    if args.foundation_model_elements and model_foundation:
        atomic_energies_dict_padded = {}
        for head_name, head_energies in atomic_energies_dict.items():
            energy_head_padded = {}
            for z in z_table.zs:
                energy_head_padded[z] = head_energies.get(z, 0.0)
            atomic_energies_dict_padded[head_name] = energy_head_padded
        atomic_energies_dict = atomic_energies_dict_padded

    if args.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        args.compute_dipole = True
        args.compute_energy = False
        args.compute_forces = False
        args.compute_virials = False
        args.compute_stress = False
    else:
        dipole_only = False
        if args.model == "EnergyDipolesMACE":
            args.compute_dipole = True
            args.compute_energy = True
            args.compute_forces = True
            args.compute_virials = False
            args.compute_stress = False
        else:
            args.compute_energy = True
            args.compute_dipole = False
        # atomic_energies: np.ndarray = np.array(
        #     [atomic_energies_dict[z] for z in z_table.zs]
        # )
        atomic_energies = dict_to_array(atomic_energies_dict, heads)
        for head_config in head_configs:
            try:
                logging.info(
                    f"Atomic Energies used (z: eV) for head {head_config.head_name}: "
                    + "{"
                    + ", ".join(
                        [
                            f"{z}: {atomic_energies_dict[head_config.head_name][z]}"
                            for z in head_config.z_table.zs
                        ]
                    )
                    + "}"
                )
            except KeyError as e:
                raise KeyError(
                    f"Atomic number {e} not found in atomic_energies_dict for head {head_config.head_name}, add E0s for this atomic number"
                ) from e

    # Load datasets using the new multi-file approach
    valid_sets = {head: [] for head in heads}
    train_sets = {head: [] for head in heads}
    for head_config in head_configs:
        # Skip heads that were already processed with the old code path
        single_train_file = (
            head_config.train_file if isinstance(head_config.train_file, str) else None
        )
        if (
            single_train_file
            and check_path_ase_read(single_train_file)
            and hasattr(head_config, "collections")
        ):
            # Use already processed collections
            train_sets[head_config.head_name] = [
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=args.r_max, heads=heads
                )
                for config in head_config.collections.train
            ]
            valid_sets[head_config.head_name] = [
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=args.r_max, heads=heads
                )
                for config in head_config.collections.valid
            ]
        else:
            # Process multiple files
            train_dataset, valid_dataset = load_datasets_for_head(
                head_config, z_table, args.r_max, heads, args
            )
            train_sets[head_config.head_name] = train_dataset
            valid_sets[head_config.head_name] = valid_dataset

        # Create a train loader for each head
        train_loader_head = torch_geometric.dataloader.DataLoader(
            dataset=train_sets[head_config.head_name],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        head_config.train_loader = train_loader_head

    # concatenate all the trainsets
    # Handle different types of datasets
    flat_train_sets = []
    for head in heads:
        if isinstance(train_sets[head], list):
            flat_train_sets.extend(train_sets[head])
        else:
            flat_train_sets.append(train_sets[head])

    if all(isinstance(dataset, data.AtomicData) for dataset in flat_train_sets):
        train_set = flat_train_sets
    else:
        train_set = ConcatDataset(flat_train_sets)

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        valid_samplers = {}
        for head, valid_set in valid_sets.items():
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )
            valid_samplers[head] = valid_sampler
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=(train_sampler is None),
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loaders = {heads[i]: None for i in range(len(heads))}
    if not isinstance(valid_sets, dict):
        valid_sets = {"Default": valid_sets}
    for head, valid_set in valid_sets.items():
        valid_loaders[head] = torch_geometric.dataloader.DataLoader(
            dataset=valid_set,
            batch_size=args.valid_batch_size,
            sampler=(
                valid_samplers[head]
                if args.distributed and head in valid_samplers
                else None
            ),
            shuffle=False,
            drop_last=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )

    loss_fn = get_loss_fn(args, dipole_only, args.compute_dipole)
    args.avg_num_neighbors = get_avg_num_neighbors(
        head_configs, args, train_loader, device
    )

    # Model
    model, output_args = configure_model(
        args, train_loader, atomic_energies, model_foundation, heads, z_table
    )
    model.to(device)

    logging.debug(model)
    logging.info(f"Total number of parameters: {tools.count_parameters(model)}")
    logging.info("")
    logging.info("===========OPTIMIZER INFORMATION===========")
    logging.info(f"Using {args.optimizer.upper()} as parameter optimizer")
    logging.info(f"Batch size: {args.batch_size}")
    if args.ema:
        logging.info(f"Using Exponential Moving Average with decay: {args.ema_decay}")
    logging.info(
        f"Number of gradient updates: {int(args.max_num_epochs*len(train_set)/args.batch_size)}"
    )
    logging.info(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
    logging.info(loss_fn)

    # Cueq
    if args.enable_cueq:
        logging.info("Converting model to CUEQ for accelerated training")
        assert model.__class__.__name__ in ["MACE", "ScaleShiftMACE"]
        model = run_e3nn_to_cueq(deepcopy(model), device=device)
    # Optimizer
    param_options = get_params_options(args, model)
    optimizer: torch.optim.Optimizer
    optimizer = get_optimizer(args, param_options)
    if args.device == "xpu":
        logging.info("Optimzing model and optimzier for XPU")
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    logger = tools.MetricsLogger(
        directory=args.results_dir, tag=tag + "_train"
    )  # pylint: disable=E1123

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        swa, swas = get_swa(args, model, optimizer, swas, dipole_only)

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception:  # pylint: disable=W0703
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    if args.wandb:
        setup_wandb(args)
    if args.distributed:
        distributed_model = DDP(model, device_ids=[local_rank])
    else:
        distributed_model = None

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        save_all_checkpoints=args.save_all_checkpoints,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
        distributed=args.distributed,
        distributed_model=distributed_model,
        train_sampler=train_sampler,
        rank=rank,
    )

    logging.info("")
    logging.info("===========RESULTS===========")
    logging.info("Computing metrics for training, validation, and test sets")

    train_valid_data_loader = {}
    for head_config in head_configs:
        data_loader_name = "train_" + head_config.head_name
        train_valid_data_loader[data_loader_name] = head_config.train_loader
    for head, valid_loader in valid_loaders.items():
        data_load_name = "valid_" + head
        train_valid_data_loader[data_load_name] = valid_loader

    test_sets = {}
    stop_first_test = False
    test_data_loader = {}
    if (
        all(
            head_config.test_file == head_configs[0].test_file
            for head_config in head_configs
        )
        and head_configs[0].test_file is not None
    ):
        stop_first_test = True
    if (
        all(
            head_config.test_dir == head_configs[0].test_dir
            for head_config in head_configs
        )
        and head_configs[0].test_dir is not None
    ):
        stop_first_test = True
    for head_config in head_configs:
        if hasattr(head_config, "collections") and head_config.collections.tests:
            for name, subset in head_config.collections.tests:
                test_sets[name] = [
                    data.AtomicData.from_config(
                        config, z_table=z_table, cutoff=args.r_max, heads=heads
                    )
                    for config in subset
                ]
        if head_config.test_dir is not None:
            if not args.multi_processed_test:
                test_files = get_files_with_suffix(head_config.test_dir, "_test.h5")
                for test_file in test_files:
                    name = os.path.splitext(os.path.basename(test_file))[0]
                    test_sets[name] = data.HDF5Dataset(
                        test_file,
                        r_max=args.r_max,
                        z_table=z_table,
                        heads=heads,
                        head=head_config.head_name,
                    )
            else:
                test_folders = glob.glob(head_config.test_dir + "/*")
                for folder in test_folders:
                    name = os.path.splitext(os.path.basename(folder))[0]
                    test_sets[name] = data.dataset_from_sharded_hdf5(
                        folder,
                        r_max=args.r_max,
                        z_table=z_table,
                        heads=heads,
                        head=head_config.head_name,
                    )
        for test_name, test_set in test_sets.items():
            test_sampler = None
            if args.distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_set,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                    seed=args.seed,
                )
            try:
                drop_last = test_set.drop_last
            except AttributeError as e:  # pylint: disable=W0612
                drop_last = False
            test_loader = torch_geometric.dataloader.DataLoader(
                test_set,
                batch_size=args.valid_batch_size,
                shuffle=(test_sampler is None),
                drop_last=drop_last,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            test_data_loader[test_name] = test_loader
        if stop_first_test:
            break

    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=swa_eval,
            device=device,
        )
        model.to(device)
        if args.distributed:
            distributed_model = DDP(model, device_ids=[local_rank])
        model_to_evaluate = model if not args.distributed else distributed_model
        if swa_eval:
            logging.info(f"Loaded Stage two model from epoch {epoch} for evaluation")
        else:
            logging.info(f"Loaded Stage one model from epoch {epoch} for evaluation")

        for param in model.parameters():
            param.requires_grad = False
        table_train_valid = create_error_table(
            table_type=args.error_table,
            all_data_loaders=train_valid_data_loader,
            model=model_to_evaluate,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
            distributed=args.distributed,
        )
        logging.info("Error-table on TRAIN and VALID:\n" + str(table_train_valid))

        if test_data_loader:
            table_test = create_error_table(
                table_type=args.error_table,
                all_data_loaders=test_data_loader,
                model=model_to_evaluate,
                loss_fn=loss_fn,
                output_args=output_args,
                log_wandb=args.wandb,
                device=device,
                distributed=args.distributed,
            )
            logging.info("Error-table on TEST:\n" + str(table_test))

        if rank == 0:
            # Save entire model
            if swa_eval:
                model_path = Path(args.checkpoints_dir) / (tag + "_stagetwo.model")
            else:
                model_path = Path(args.checkpoints_dir) / (tag + ".model")
            logging.info(f"Saving model to {model_path}")
            model_to_save = deepcopy(model)
            if args.enable_cueq:
                logging.info("Converting CUEQ model back to E3NN for saving")
                model_to_save = run_cueq_to_e3nn(deepcopy(model), device=device)
            if args.save_cpu:
                model_to_save = model_to_save.to("cpu")
            torch.save(model_to_save, model_path)
            extra_files = {
                "commit.txt": commit.encode("utf-8") if commit is not None else b"",
                "config.yaml": json.dumps(
                    convert_to_json_format(extract_config_mace_model(model))
                ),
            }
            if swa_eval:
                torch.save(
                    model_to_save,
                    Path(args.model_dir) / (args.name + "_stagetwo.model"),
                )
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_stagetwo_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0703
                    logging.warning(f"Failed to compile model: {e}")
                    pass
            else:
                torch.save(model_to_save, Path(args.model_dir) / (args.name + ".model"))
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata to {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0703
                    logging.warning(f"Failed to compile model: {e}")
                    pass

        if args.distributed:
            torch.distributed.barrier()

    logging.info("Done")
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
