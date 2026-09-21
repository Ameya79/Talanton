"""Talanton Command-Line Interface (CLI).

Provides commands:
    talanton count "prompt" --model gpt-4o
    talanton cost "prompt" --model claude-sonnet-4.5 --output-tokens 200
    talanton compare "prompt" --models gpt-4o,claude-sonnet-4.5,gpt-4o-mini
    talanton forecast --model gpt-4o --calls-per-day 500 --avg-input 800 --avg-output 300 --months 6 --growth 0.15
    talanton compare-at-scale --models gpt-4o,gpt-4o-mini --calls-per-day 500 --avg-input 800 --avg-output 300
    talanton track summary --period day
    talanton track calls --limit 50 --model gpt-4o
    talanton track export --format csv --output spend.csv
    talanton budget set --soft 50 --hard 100 --period day
    talanton budget status
"""

from __future__ import annotations

import sys
import click

from talanton.cost import calculate_cost, compare_models
from talanton.counting import count_tokens
from talanton.forecast import compare_at_scale, forecast
from talanton.pricing import PricingNotFoundError


def _render_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Renders table using rich if available, falling back to plain ASCII."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(title=f"\n[bold gold1]{title}[/bold gold1]", box=box.ROUNDED, header_style="bold cyan")

        for h in headers:
            table.add_column(h)

        for r in rows:
            formatted_row = []
            for cell in r:
                if cell == "exact":
                    formatted_row.append("[bold green]exact[/bold green]")
                elif cell == "estimated":
                    formatted_row.append("[bold yellow]estimated[/bold yellow]")
                else:
                    formatted_row.append(str(cell))
            table.add_row(*formatted_row)

        console.print(table)
        print()
    except ImportError:
        # Fallback to plain text table
        print(f"\n=== {title} ===")
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        fmt = " | ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("-+-".join("-" * w for w in widths))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))
        print()


def _parse_models_list(models_arg: str | list[str]) -> list[str]:
    """Parses a comma-separated models string or list of models."""
    if isinstance(models_arg, list):
        items = []
        for m in models_arg:
            items.extend([x.strip() for x in m.split(",") if x.strip()])
        return items
    return [x.strip() for x in models_arg.split(",") if x.strip()]


@click.group()
@click.version_option(version="0.2.1", prog_name="Talanton")
def cli() -> None:
    """Talanton — Universal token counting, cost calculation, scale forecasting, and live cost tracking."""
    pass


@cli.command("count")
@click.argument("text", required=False)
@click.option("--model", "-m", default="gpt-4o", help="Model name (e.g. gpt-4o, claude-sonnet-4.5).")
def cmd_count(text: str | None, model: str) -> None:
    """Count tokens for text on a specific model."""
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            raise click.UsageError("Please provide prompt text as an argument or via stdin.")

    res = count_tokens(text, model)
    flag = "exact" if res["exact"] else "estimated"

    headers = ["Model", "Provider", "Tokens", "Status"]
    rows = [[res["model"], res["provider"], str(res["tokens"]), flag]]
    _render_table("Talanton Token Count", headers, rows)


@cli.command("cost")
@click.argument("text", required=False)
@click.option("--model", "-m", required=True, help="Model name.")
@click.option("--output-tokens", "-o", default=0, type=int, help="Expected completion tokens.")
def cmd_cost(text: str | None, model: str, output_tokens: int) -> None:
    """Calculate dollar cost for a prompt and completion."""
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            raise click.UsageError("Please provide prompt text as an argument or via stdin.")

    try:
        res = calculate_cost(text, model, expected_output_tokens=output_tokens)
    except PricingNotFoundError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    flag = "exact" if res["exact"] else "estimated"
    headers = ["Model", "Input Tokens", "Output Tokens", "Input Cost", "Output Cost", "Total Cost", "Status"]
    rows = [[
        res["model"],
        str(res["input_tokens"]),
        str(res["output_tokens"]),
        f"${res['input_cost']:.6f}",
        f"${res['output_cost']:.6f}",
        f"${res['total_cost']:.6f}",
        flag
    ]]
    _render_table("Talanton Cost Calculation", headers, rows)


@cli.command("compare")
@click.argument("text", required=False)
@click.option("--models", "-m", required=True, help="Comma-separated model names (e.g. gpt-4o,claude-sonnet-4.5,gpt-4o-mini).")
@click.option("--output-tokens", "-o", default=0, type=int, help="Expected completion tokens.")
def cmd_compare(text: str | None, models: str, output_tokens: int) -> None:
    """Compare prompt costs across candidate models, ranked cheapest first."""
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            raise click.UsageError("Please provide prompt text as an argument or via stdin.")

    model_list = _parse_models_list(models)
    if not model_list:
        raise click.UsageError("No models specified to compare.")

    try:
        results = compare_models(text, model_list, expected_output_tokens=output_tokens)
    except PricingNotFoundError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    headers = ["Rank", "Model", "Input Tokens", "Output Tokens", "Input Cost", "Output Cost", "Total Cost", "Status"]
    rows = []
    for idx, r in enumerate(results, start=1):
        flag = "exact" if r["exact"] else "estimated"
        rows.append([
            f"#{idx}",
            r["model"],
            str(r["input_tokens"]),
            str(r["output_tokens"]),
            f"${r['input_cost']:.6f}",
            f"${r['output_cost']:.6f}",
            f"${r['total_cost']:.6f}",
            flag
        ])
    _render_table("Talanton Model Cost Comparison (Cheapest First)", headers, rows)


@cli.command("forecast")
@click.option("--model", "-m", required=True, help="Model name.")
@click.option("--calls-per-day", "-c", required=True, type=float, help="Initial daily API calls.")
@click.option("--avg-input", "-i", required=True, type=int, help="Average prompt tokens per call.")
@click.option("--avg-output", "-o", required=True, type=int, help="Average completion tokens per call.")
@click.option("--months", default=6, type=int, help="Forecast horizon in months.")
@click.option("--growth", "-g", default=0.0, type=float, help="Monthly compounding growth rate (e.g. 0.15 for 15%).")
def cmd_forecast(model: str, calls_per_day: float, avg_input: int, avg_output: int, months: int, growth: float) -> None:
    """Forecast cumulative spend over months under volume and growth assumptions."""
    try:
        results = forecast(
            model=model,
            calls_per_day=calls_per_day,
            avg_input_tokens=avg_input,
            avg_output_tokens=avg_output,
            months=months,
            growth_rate=growth,
        )
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    headers = ["Month", "Calls / Day", "Projected Cost (USD)"]
    rows = []
    total = 0.0
    for r in results:
        rows.append([str(r["month"]), f"{r['calls_per_day']:,.1f}", f"${r['cost']:,.2f}"])
        total += r["cost"]

    rows.append(["TOTAL", "-", f"${total:,.2f}"])
    _render_table(f"Spend Forecast: {model} ({months} Months @ {growth*100:.1f}% Growth)", headers, rows)
    click.echo("(Projection is based purely on compounding user-supplied volume assumptions; it does not predict market adoption).\n")


@cli.command("compare-at-scale")
@click.option("--models", "-m", required=True, help="Comma-separated model names.")
@click.option("--calls-per-day", "-c", required=True, type=float, help="Daily API calls.")
@click.option("--avg-input", "-i", required=True, type=int, help="Average prompt tokens.")
@click.option("--avg-output", "-o", required=True, type=int, help="Average completion tokens.")
@click.option("--months", default=6, type=int, help="Horizon in months.")
@click.option("--growth", "-g", default=0.0, type=float, help="Monthly growth rate (default 0.0).")
def cmd_compare_at_scale(models: str, calls_per_day: float, avg_input: int, avg_output: int, months: int, growth: float) -> None:
    """Compare long-term volume spend across models and quantify switching savings."""
    model_list = _parse_models_list(models)
    try:
        summary = compare_at_scale(
            models=model_list,
            calls_per_day=calls_per_day,
            avg_input_tokens=avg_input,
            avg_output_tokens=avg_output,
            months=months,
            growth_rate=growth,
        )
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    totals = summary["totals"]
    cheapest = summary["cheapest"]
    savings = summary["savings_by_switching"]

    headers = ["Model", f"Total Projected Spend ({months} mo)", "Difference vs Cheapest"]
    rows = []
    cheapest_cost = totals[cheapest]

    # Sort ascending
    sorted_models = sorted(totals.items(), key=lambda x: x[1])
    for model_name, cost in sorted_models:
        diff = cost - cheapest_cost
        diff_str = "Cheapest Option" if diff == 0 else f"+${diff:,.2f}"
        rows.append([model_name, f"${cost:,.2f}", diff_str])

    _render_table("Scale Comparison & Switch Analysis", headers, rows)
    click.secho(f"[*] Cheapest Model: {cheapest}", fg="green", bold=True)
    if savings > 0:
        click.secho(f"[*] Potential Savings by Choosing {cheapest}: ${savings:,.2f} over {months} months\n", fg="cyan", bold=True)


# ══════════════════════════════════════════════════════════════════════════════
# NEW v0.2.0: Track & Budget Command Groups
# ══════════════════════════════════════════════════════════════════════════════


@cli.group("track")
def cmd_track_group() -> None:
    """Live cost tracking — view spend summaries, recent calls, and export data."""
    pass


@cmd_track_group.command("summary")
@click.option("--period", "-p", default="day", type=click.Choice(["day", "week", "month"]), help="Summary period.")
@click.option("--team", "-t", default=None, help="Filter by team.")
def cmd_track_summary(period: str, team: str | None) -> None:
    """View aggregated spend summary for a period."""
    from talanton.tracker import TalantonTracker

    tracker = TalantonTracker()
    summary = tracker.get_summary(period=period, team=team)

    click.secho(f"\n{'═' * 60}", fg="cyan")
    click.secho(f"  Talanton Spend Summary — Last {period.upper()}", fg="cyan", bold=True)
    click.secho(f"{'═' * 60}", fg="cyan")
    click.secho(f"  Total Spend: ${summary['total_spend']:.6f}", fg="white", bold=True)

    if summary["by_model"]:
        headers = ["Model", "Calls", "Total Cost", "Input Tokens", "Output Tokens"]
        rows = []
        for entry in summary["by_model"]:
            rows.append([
                entry["model"],
                str(entry["total_calls"]),
                f"${entry['total_cost']:.6f}",
                f"{entry['total_input_tokens']:,}",
                f"{entry['total_output_tokens']:,}",
            ])
        _render_table("Spend by Model", headers, rows)

    if summary["by_team"]:
        headers = ["Team", "Calls", "Total Cost"]
        rows = []
        for entry in summary["by_team"]:
            rows.append([
                entry["team"],
                str(entry["total_calls"]),
                f"${entry['total_cost']:.6f}",
            ])
        _render_table("Spend by Team", headers, rows)

    if not summary["by_model"]:
        click.secho("  No tracked calls found for this period.\n", fg="yellow")


@cmd_track_group.command("calls")
@click.option("--limit", "-l", default=20, type=int, help="Number of recent calls to show.")
@click.option("--model", "-m", default=None, help="Filter by model.")
@click.option("--team", "-t", default=None, help="Filter by team.")
def cmd_track_calls(limit: int, model: str | None, team: str | None) -> None:
    """View recent tracked calls."""
    from talanton.tracker import TalantonTracker

    tracker = TalantonTracker()
    calls = tracker.get_calls(limit=limit, model=model, team=team)

    if not calls:
        click.secho("No tracked calls found.", fg="yellow")
        return

    headers = ["Timestamp", "Model", "In Tokens", "Out Tokens", "Cost", "Team"]
    rows = []
    for c in calls:
        ts = c.get("timestamp", "")[:19]  # Trim to seconds
        rows.append([
            ts,
            c.get("model", ""),
            str(c.get("input_tokens", 0)),
            str(c.get("output_tokens", 0)),
            f"${c.get('total_cost', 0):.6f}",
            c.get("team", "-") or "-",
        ])
    _render_table(f"Recent Tracked Calls (Last {limit})", headers, rows)


@cmd_track_group.command("export")
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "csv"]), help="Export format.")
@click.option("--output", "-o", default=None, help="Output file path. Prints to stdout if not specified.")
def cmd_track_export(fmt: str, output: str | None) -> None:
    """Export tracked call data to JSON or CSV."""
    from talanton.tracker import TalantonTracker

    tracker = TalantonTracker()
    data = tracker.export(fmt=fmt)

    if not data:
        click.secho("No data to export.", fg="yellow")
        return

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(data)
        click.secho(f"Exported to {output} ({fmt.upper()})", fg="green")
    else:
        click.echo(data)


@cli.group("budget")
def cmd_budget_group() -> None:
    """Budget guardrails — set and check spend limits."""
    pass


@cmd_budget_group.command("status")
@click.option("--period", "-p", default="day", type=click.Choice(["day", "week", "month"]), help="Budget period.")
@click.option("--team", "-t", default=None, help="Team scope.")
def cmd_budget_status(period: str, team: str | None) -> None:
    """Show current spend against configured limits."""
    from talanton.tracker import TalantonTracker

    tracker = TalantonTracker()
    spend = tracker.store.get_total_spend(period=period, team=team)

    scope = f"team={team}" if team else "global"
    click.secho(f"\n{'═' * 50}", fg="cyan")
    click.secho(f"  Budget Status — {period.upper()} ({scope})", fg="cyan", bold=True)
    click.secho(f"{'═' * 50}", fg="cyan")
    click.secho(f"  Current Spend: ${spend:.6f}", fg="white", bold=True)
    click.echo()

    # Show suggestion for setting limits
    if spend == 0:
        click.secho("  No spend recorded. Start tracking calls to see budget status.", fg="yellow")
    click.echo()



if __name__ == "__main__":
    cli()

