"""Entry script for MemoryAgentBench (MAB) baseline."""
import argparse
import json as _json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.service.memory_service import MemoryService
from memrl.service.strategies import (
    BuildStrategy,
    RetrieveStrategy,
    UpdateStrategy,
    StrategyConfiguration,
)
from memrl.run.mab_runner import MABRunner, MABSelection
from memrl.mab_eval.task_wrappers import SPLIT_NAMES


def setup_logging(project_root: Path, name: str) -> None:
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = log_dir / log_filename

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Logging configured. Log file: {log_filepath}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MemoryAgentBench baseline")
    p.add_argument(
        "--config",
        type=str,
        default=str(
            (project_root / "configs" / "rl_mab_config.local.yaml")
            if (project_root / "configs" / "rl_mab_config.local.yaml").exists()
            else (project_root / "configs" / "rl_mab_config.yaml")
        ),
    )
    p.add_argument(
        "--split",
        type=str,
        default="Accurate_Retrieval",
        choices=list(SPLIT_NAMES),
    )
    p.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Limit to N examples (default: all).",
    )
    p.add_argument(
        "--max_questions_per_example",
        type=int,
        default=None,
        help="Limit to N questions per example (default: all).",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--retrieve_k", type=int, default=None)
    p.add_argument(
        "--retrieve_threshold",
        type=float,
        default=None,
        help="Override similarity threshold for MemoryService.retrieve_query.",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace datasets cache dir.",
    )
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(project_root, "mab")
    logger = logging.getLogger(__name__)

    cfg = MempConfig.from_yaml(args.config)

    out_root = Path(args.output_dir or cfg.experiment.output_dir or "./results").resolve()
    out_dir = out_root / "mab_eval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg.llm.model.replace('/', '_')}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Providers
    token_log_dir = str((project_root / "logs" / "mab").resolve())
    llm = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
        default_max_tokens=(args.max_tokens if args.max_tokens is not None else cfg.llm.max_tokens),
        token_log_dir=token_log_dir,
    )
    embedder = OpenAIEmbedder(
        api_key=cfg.embedding.api_key,
        base_url=cfg.embedding.base_url,
        model=cfg.embedding.model,
        max_text_len=getattr(cfg.embedding, "max_text_len", 4096),
    )

    # MemOS config
    temp_dir = tempfile.mkdtemp(prefix="memp_mab_run_")
    user_id = f"mab_{os.getpid()}"
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
                        "provider": cfg.embedding.provider,
                        "model_name_or_path": cfg.embedding.model,
                        "api_key": cfg.embedding.api_key,
                        "base_url": cfg.embedding.base_url,
                    },
                },
                "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
            },
        },
        "user_manager": {
            "backend": "sqlite",
            "config": {"db_path": os.path.join(temp_dir, "users.db")},
        },
        "top_k": int(args.retrieve_k if args.retrieve_k is not None else cfg.memory.k_retrieve),
    }
    mos_config_path = os.path.join(temp_dir, "mos_config.json")
    with open(mos_config_path, "w", encoding="utf-8") as f:
        _json.dump(mos_config, f)

    memsvc = MemoryService(
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
        sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", 0.31),
        sim_norm_std=getattr(cfg.memory, "sim_norm_std", 0.10),
        vector_dimension=cfg.embedding.vector_dimension,
        memory_budget=getattr(cfg.memory, "memory_budget", 0),
        budget_policy=getattr(cfg.memory, "budget_policy", "q_weighted"),
        budget_check_interval=getattr(cfg.memory, "budget_check_interval", 1),
        budget_utilization_threshold=getattr(cfg.memory, "budget_utilization_threshold", 0.8),
    )

    sel = MABSelection(
        split=args.split,
        num_examples=args.num_examples,
        max_questions_per_example=args.max_questions_per_example,
        cache_dir=args.cache_dir,
    )

    tb_dir = str((project_root / "logs" / "tensorboard" / f"exp_mab_{run_id}").resolve())

    runner = MABRunner(
        selection=sel,
        llm=llm,
        memory_service=memsvc,
        output_dir=str(run_dir),
        model_name=cfg.llm.model,
        temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
        max_tokens=(args.max_tokens if args.max_tokens is not None else (cfg.llm.max_tokens or 256)),
        retrieve_k=int(args.retrieve_k if args.retrieve_k is not None else cfg.memory.k_retrieve),
        retrieve_threshold=args.retrieve_threshold,
        tb_log_dir=tb_dir,
    )

    logger.info("MAB run_dir: %s", run_dir)
    metrics = runner.run()
    logger.info("MAB metrics: %s", _json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
