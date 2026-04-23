from __future__ import annotations
import json
from pathlib import Path
from dotenv import load_dotenv
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = typer.Option("mock", help="Runtime mode: mock or llm."),
    llm_model: str | None = typer.Option(None, help="Model name for llm mode."),
    llm_base_url: str | None = typer.Option(None, help="OpenAI-compatible base URL for llm mode."),
    llm_api_key: str | None = typer.Option(None, help="API key for llm mode."),
) -> None:
    load_dotenv()
    if mode not in {"mock", "llm"}:
        raise typer.BadParameter("mode must be either 'mock' or 'llm'.")
    examples = load_dataset(dataset)
    react = ReActAgent(runtime_mode=mode, llm_model=llm_model, llm_base_url=llm_base_url, llm_api_key=llm_api_key)
    reflexion = ReflexionAgent(
        max_attempts=reflexion_attempts,
        runtime_mode=mode,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
    )
    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
