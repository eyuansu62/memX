import sys
import os
from pathlib import Path
import logging
import argparse
import json as _json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.service.belief_memory_service import BeliefMemoryService, BeliefConfig
from memrl.service.memory_service import MemoryService
from memrl.service.strategies import BuildStrategy, RetrieveStrategy, UpdateStrategy, StrategyConfiguration
from memrl.run.locomo_runner import LoCoMoRunner, LoCoMoSelection


def setup_logging(project_root: Path, name: str):
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Logging configured. Log file: {log_filepath}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LoComo benchmark with memory-agent")
    p.add_argument(
        "--config",
        type=str,
        default=str(
            (project_root / "configs" / "rl_locomo_config.local.yaml")
            if (project_root / "configs" / "rl_locomo_config.local.yaml").exists()
            else (project_root / "configs" / "rl_locomo_config.yaml")
        ),
    )
    p.add_argument(
        "--data",
        type=str,
        default=str(project_root / "3rdparty" / "locomo" / "data" / "locomo10.json"),
        help="Path to the LoComo dataset JSON file.",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument(
        "--categories",
        type=int,
        nargs="+",
        default=None,
        help="Filter QA items to these category numbers (1-5).",
    )
    p.add_argument(
        "--memory_service",
        type=str,
        choices=["original", "belief"],
        default="belief",
        help="Which memory service class to use.",
    )
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to load before running.")
    return p.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = parse_args()
    try:
        cfg = MempConfig.from_yaml(args.config)
        setup_logging(project_root, cfg.experiment.experiment_name)

        out_dir = Path(cfg.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime('%Y%m%d-%H%M%S')
        log_dir = out_dir / "locomo" / f"exp_{cfg.experiment.experiment_name}_{run_id}" / "local_cache"
        log_dir.mkdir(parents=True, exist_ok=True)

        llm = OpenAILLM(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            default_temperature=cfg.llm.temperature,
            default_max_tokens=cfg.llm.max_tokens,
            token_log_dir=str(log_dir),
        )
        embedder = OpenAIEmbedder(
            api_key=cfg.embedding.api_key,
            base_url=cfg.embedding.base_url,
            model=cfg.embedding.model,
            max_text_len=getattr(cfg.embedding, "max_text_len", 4096),
            token_log_dir=str(log_dir),
        )

        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="memp_locomo_run_")
        user_id = f"locomo_{os.getpid()}"
        mos_config = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": cfg.llm.model,
                    "api_key": cfg.llm.api_key,
                    "api_base": cfg.llm.base_url,
                },
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": cfg.llm.model,
                            "api_key": cfg.llm.api_key,
                            "api_base": cfg.llm.base_url,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "model_name_or_path": cfg.embedding.model,
                            "api_key": cfg.embedding.api_key,
                            "base_url": cfg.embedding.base_url,
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
            "user_manager": {"backend": "sqlite", "config": {"db_path": os.path.join(temp_dir, "users.db")}},
            "top_k": 5,
        }
        mos_config_path = os.path.join(temp_dir, "mos_config.json")
        with open(mos_config_path, "w", encoding="utf-8") as f:
            _json.dump(mos_config, f)

        common_svc_kwargs = dict(
            mos_config_path=mos_config_path,
            llm_provider=llm,
            embedding_provider=embedder,
            strategy_config=StrategyConfiguration(
                BuildStrategy(cfg.memory.build_strategy),
                RetrieveStrategy(cfg.memory.retrieve_strategy),
                UpdateStrategy(cfg.memory.update_strategy),
            ),
            user_id=user_id,
            num_workers=cfg.experiment.batch_size,
            max_keywords=cfg.memory.max_keywords,
            add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
            enable_value_driven=cfg.experiment.enable_value_driven,
            rl_config=cfg.rl_config,
            db_max_concurrency=4,
            sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", 0.1856827586889267),
            sim_norm_std=getattr(cfg.memory, "sim_norm_std", 0.09407906234264374),
            vector_dimension=cfg.embedding.vector_dimension,
            memory_budget=getattr(cfg.memory, "memory_budget", 0),
            budget_policy=getattr(cfg.memory, "budget_policy", "q_weighted"),
            budget_check_interval=getattr(cfg.memory, "budget_check_interval", 1),
            budget_utilization_threshold=getattr(cfg.memory, "budget_utilization_threshold", 0.8),
        )

        use_belief = getattr(args, "memory_service", "belief") != "original"
        if use_belief:
            belief_cfg = cfg.belief.to_dataclass() if hasattr(cfg, "belief") else BeliefConfig()
            memsvc = BeliefMemoryService(**common_svc_kwargs, belief_config=belief_cfg)
        else:
            memsvc = MemoryService(**common_svc_kwargs)

        # Load checkpoint if specified
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if not ckpt_path.is_absolute():
                ckpt_path = project_root / ckpt_path
            logger.info("Loading checkpoint from %s", ckpt_path)
            memsvc.load_checkpoint_snapshot(str(ckpt_path), mem_cube_id=memsvc.default_cube_id)

        sel = LoCoMoSelection(
            data_path=args.data,
            categories=args.categories,
        )

        runner = LoCoMoRunner(
            name=cfg.experiment.experiment_name,
            llm=llm,
            selection=sel,
            output_dir=out_dir,
            memory_service=memsvc,
            run_id=run_id,
            temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
            max_tokens=(args.max_tokens if args.max_tokens is not None else (cfg.llm.max_tokens or 256)),
            retrieve_k=cfg.memory.k_retrieve,
            num_sections=cfg.experiment.num_sections,
            batch_size=cfg.experiment.batch_size,
            dataset_ratio=getattr(cfg.experiment, "dataset_ratio", 1.0),
            random_seed=getattr(cfg.experiment, "random_seed", 42) or 42,
            train_valid_split=getattr(cfg.experiment, "train_valid_split", 0.8),
            ckpt_eval_enabled=getattr(cfg.experiment, "ckpt_eval_enabled", False),
            ckpt_eval_path=getattr(cfg.experiment, "ckpt_eval_path", None),
            ckpt_resume_enabled=getattr(cfg.experiment, "ckpt_resume_enabled", False),
            ckpt_resume_path=getattr(cfg.experiment, "ckpt_resume_path", None),
            ckpt_resume_epoch=getattr(cfg.experiment, "ckpt_resume_epoch", None),
            baseline_mode=getattr(cfg.experiment, "baseline_mode", False),
            baseline_k=getattr(cfg.experiment, "baseline_k", 0),
            state_first=getattr(cfg.experiment, "state_first", False),
        )
        runner.run()
    except Exception as e:
        logger.error(f"LoComo run failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
