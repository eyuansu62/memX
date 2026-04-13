import sys
import os
from pathlib import Path
import logging
import tempfile
import shutil
import json
import argparse
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.service.belief_memory_service import BeliefMemoryService, BeliefConfig
from memrl.service.memory_service import MemoryService
from memrl.service.strategies import BuildStrategy, RetrieveStrategy, UpdateStrategy, StrategyConfiguration
from memrl.analysis.logged_services import LoggedMemoryService, LoggedBeliefMemoryService
from memrl.analysis.memory_logger import MemoryEventLogger
from memrl.service.llm_judge import ALFWorldJudge
from memrl.agent.memp_agent import MempAgent
from memrl.run.alfworld_rl_runner import AlfworldRunner


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
    return log_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AlfWorld benchmark with memory-agent")
    p.add_argument(
        "--config",
        type=str,
        default=str(
            (project_root / "configs" / "rl_alf_config.local.yaml")
            if (project_root / "configs" / "rl_alf_config.local.yaml").exists()
            else (project_root / "configs" / "rl_alf_config.yaml")
        ),
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint snapshot dir. Implies mode=test (inference-only).")
    p.add_argument("--memory_service", type=str, default="belief",
                   choices=["belief", "original"],
                   help="Which memory service to use: 'belief' (BeliefMemoryService) or 'original' (MemoryService).")
    p.add_argument("--log_path", type=str, default=None,
                   help="Path to JSONL event log file. Enables diagnostic logging when set.")
    return p.parse_args()


logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    try:
        cfg = MempConfig.from_yaml(args.config)
        setup_logging(project_root, cfg.experiment.experiment_name)

        out_dir = Path(cfg.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime('%Y%m%d-%H%M%S')
        log_dir = out_dir / "alfworld" / f"exp_{cfg.experiment.experiment_name}_{run_id}" / "local_cache"
        log_dir.mkdir(parents=True, exist_ok=True)

        llm_provider = OpenAILLM(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            default_temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
            default_max_tokens=(args.max_tokens if args.max_tokens is not None else cfg.llm.max_tokens),
            token_log_dir=str(log_dir),
        )
        embedding_provider = OpenAIEmbedder(
            api_key=cfg.embedding.api_key,
            base_url=cfg.embedding.base_url,
            model=cfg.embedding.model,
            max_text_len=getattr(cfg.embedding, "max_text_len", 4096),
            token_log_dir=str(log_dir),
        )

        temp_dir = tempfile.mkdtemp(prefix="memp_alfworld_run_")
        logger.info(f"Using temporary directory for runtime artifacts: {temp_dir}")

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
            json.dump(mos_config, f)

        build_strategy = BuildStrategy(cfg.memory.build_strategy)
        retrieve_strategy = RetrieveStrategy(cfg.memory.retrieve_strategy)
        update_strategy = UpdateStrategy(cfg.memory.update_strategy)

        enable_value_driven = cfg.experiment.enable_value_driven
        rl_config = cfg.rl_config

        user_id = f"alf_{os.getpid()}"

        common_svc_kwargs = dict(
            mos_config_path=mos_config_path,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            strategy_config=StrategyConfiguration(build_strategy, retrieve_strategy, update_strategy),
            user_id=user_id,
            num_workers=cfg.experiment.batch_size,
            max_keywords=cfg.memory.max_keywords,
            add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
            enable_value_driven=enable_value_driven,
            rl_config=rl_config,
            db_max_concurrency=4,
            sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", None),
            sim_norm_std=getattr(cfg.memory, "sim_norm_std", None),
            vector_dimension=cfg.embedding.vector_dimension,
            memory_budget=getattr(cfg.memory, "memory_budget", 0),
            budget_policy=getattr(cfg.memory, "budget_policy", "q_weighted"),
            budget_check_interval=getattr(cfg.memory, "budget_check_interval", 1),
            budget_utilization_threshold=getattr(cfg.memory, "budget_utilization_threshold", 0.8),
        )
        use_belief = getattr(args, "memory_service", "belief") != "original"
        if use_belief:
            SvcClass = LoggedBeliefMemoryService if args.log_path else BeliefMemoryService
            belief_cfg = cfg.belief.to_dataclass() if hasattr(cfg, "belief") else BeliefConfig()
            memory_service = SvcClass(**common_svc_kwargs, belief_config=belief_cfg)
        else:
            SvcClass = LoggedMemoryService if args.log_path else MemoryService
            memory_service = SvcClass(**common_svc_kwargs)

        if args.log_path:
            log_path = Path(args.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            memory_service._mem_event_logger = MemoryEventLogger(str(log_path))
            logger.info("Memory event logging enabled → %s", log_path)

        with open(project_root / cfg.experiment.few_shot_path, "r", encoding="utf-8") as f:
            few_shot_examples = json.load(f)
        agent = MempAgent(llm_provider=llm_provider, few_shot_examples=few_shot_examples)

        llm_judge = None
        if getattr(cfg.experiment, "use_llm_judge", False):
            llm_judge = ALFWorldJudge(llm=llm_provider)
            logger.info(
                "LLM-as-judge enabled (alpha=%.2f). Env reward will be blended with judge score.",
                cfg.experiment.llm_judge_alpha,
            )

        # --checkpoint flag overrides config to force inference-only mode
        ckpt_resume_enabled = getattr(cfg.experiment, "ckpt_resume_enabled", False)
        ckpt_resume_path = getattr(cfg.experiment, "ckpt_resume_path", None)
        ckpt_resume_epoch = getattr(cfg.experiment, "ckpt_resume_epoch", None)
        run_mode = cfg.experiment.mode

        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint path not found: {args.checkpoint}")
            run_mode = "test"
            ckpt_resume_enabled = True
            # If user points to a numbered snapshot dir (e.g. .../snapshot/8),
            # extract the epoch; otherwise leave it to auto-detect.
            if ckpt_path.name.isdigit():
                ckpt_resume_epoch = int(ckpt_path.name)
                ckpt_resume_path = str(ckpt_path.parent.parent)
            else:
                ckpt_resume_path = str(ckpt_path)
            logger.info("Inference mode: loading checkpoint from %s (epoch=%s)", ckpt_resume_path, ckpt_resume_epoch)

        # Build compiler LLM (for resolve/summary modes)
        compiler_llm = None
        compiler_cfg = getattr(cfg, "compiler", None)
        if compiler_cfg and not getattr(compiler_cfg, "use_actor_model", True):
            compiler_llm = OpenAILLM(
                api_key=compiler_cfg.api_key,
                base_url=compiler_cfg.base_url,
                model=compiler_cfg.model,
                default_temperature=compiler_cfg.temperature,
                default_max_tokens=compiler_cfg.max_tokens,
                token_log_dir=str(log_dir),
            )
            logger.info("Using separate compiler LLM: %s @ %s", compiler_cfg.model, compiler_cfg.base_url)
        else:
            logger.info("Compiler will reuse actor LLM")

        alfworld_config_path = project_root / "configs" / "envs" / "alfworld.yaml"
        runner = AlfworldRunner(
            agent=agent,
            root=project_root,
            env_config=alfworld_config_path,
            memory_service=memory_service,
            exp_name=cfg.experiment.experiment_name,
            ck_dir=log_dir,
            random_seed=cfg.experiment.random_seed,
            num_section=cfg.experiment.num_sections,
            batch_size=cfg.experiment.batch_size,
            max_steps=cfg.experiment.max_steps,
            rl_config=rl_config,
            bon=cfg.experiment.bon,
            retrieve_k=cfg.memory.k_retrieve,
            mode=run_mode,
            valid_interval=cfg.experiment.valid_interval,
            test_interval=cfg.experiment.test_interval,
            dataset_ratio=cfg.experiment.dataset_ratio,
            ckpt_resume_enabled=ckpt_resume_enabled,
            ckpt_resume_path=ckpt_resume_path,
            ckpt_resume_epoch=ckpt_resume_epoch,
            baseline_mode=getattr(cfg.experiment, "baseline_mode", None),
            baseline_k=getattr(cfg.experiment, "baseline_k", 10),
            llm_judge=llm_judge,
            llm_judge_alpha=getattr(cfg.experiment, "llm_judge_alpha", 0.3),
            state_first=getattr(cfg.experiment, "state_first", False),
            compile_mode=getattr(cfg.experiment, "compile_mode", "off"),
            compiler_use_belief=getattr(cfg.experiment, "compiler_use_belief", True),
            compiler_llm=compiler_llm,
        )
        runner.run()

    except Exception as e:
        logger.error(f"An unhandled error occurred during the experiment: {e}", exc_info=True)
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
