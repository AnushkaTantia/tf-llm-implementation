"""
Prompt Builder for TF-LLM.

Implements Section 3.2 of the paper:
    - Static prompt template (baseline, used for comparison in paper's
      Table 12 "P1: Static Prompt Only")
    - Dynamic prompt template (Eq. 5-6): context(X) + task(theta) + Statistics(X)
    - Digit tokenization workaround: numbers are split into individual
      digit tokens (space/comma separated) and decimal points are
      stripped, preventing GPT-2's BPE tokenizer from producing
      meaningless sub-word splits of numeric values.

Prompts are returned as plain strings; converting them into GPT-2 input
embeddings happens in Stage 6 once the backbone is loaded.
"""

import re


def format_number_for_prompt(value, decimals=2):
    """
    Digit tokenization workaround (Section 3.2):
    - Round to a fixed number of decimals
    - Strip the decimal point (redundant given fixed precision)
    - Insert spaces between digits so BPE tokenization treats each
      digit as a separate, consistent token rather than splitting
      arbitrarily across sub-word boundaries.

    Example: 27.787001 -> "2 7 7 8" (rounded to 2 decimals -> "27.79" -> "2779" -> "2 7 7 9")
    """
    rounded = round(float(value), decimals)
    # strip sign separately so it doesn't get mixed into digit spacing
    sign = "-" if rounded < 0 else ""
    rounded = abs(rounded)

    as_str = f"{rounded:.{decimals}f}".replace(".", "")
    spaced = " ".join(list(as_str))
    return f"{sign}{spaced}"


def compute_input_statistics(x, feature_names=None, decimals=2):
    """
    Compute per-feature mean/variance statistics from an input window,
    formatted as a natural-language statistics string.

    Args:
        x: (seq_len, num_features) tensor or array
        feature_names: optional list of feature names; defaults to
                        generic "feature_i" labels
        decimals: rounding precision for the digit tokenization workaround

    Returns:
        str, e.g. "Historical mean: OT=2 3 5 0, HUFL=1 2 0 0. ..."
    """
    import torch

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    num_features = x.shape[-1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    means = x.mean(dim=0)
    stds = x.std(dim=0)

    mean_parts = [
        f"{name}={format_number_for_prompt(means[i].item(), decimals)}"
        for i, name in enumerate(feature_names)
    ]
    std_parts = [
        f"{name}={format_number_for_prompt(stds[i].item(), decimals)}"
        for i, name in enumerate(feature_names)
    ]

    return (
        f"Historical mean: {', '.join(mean_parts)}. "
        f"Historical std: {', '.join(std_parts)}."
    )


# ---------------------------------------------------------------------------
# Dataset context templates
# ---------------------------------------------------------------------------

DATASET_CONTEXTS = {
    "ETTh1": (
        "This data is from an Electricity Transformer Temperature (ETT) monitoring "
        "system, recorded hourly. Features include High/Middle/Low UseFul Load "
        "(HUFL/MUFL/LUFL), High/Middle/Low UseLess Load (HULL/MULL/LULL), and Oil "
        "Temperature (OT), which reflects the transformer's operating condition "
        "under varying load."
    ),
}


TASK_INSTRUCTIONS = {
    "forecast": (
        "Your current task is multivariate forecasting. Based on the historical "
        "time points from {start} to {end}, predict the future values of variable "
        "{target} for the next {pred_len} steps."
    ),
    "impute": (
        "Your current task is missing value recovery. Based on the available "
        "historical data, fill the missing interval in variable {target}."
    ),
    "classify": (
        "Your current task is time series classification. Based on the input "
        "sequence, determine the correct category."
    ),
    "anomaly": (
        "Your current task is anomaly detection. Identify the time intervals "
        "in the input sequence where anomalous behavior occurs."
    ),
}


def build_static_prompt(task="forecast"):
    """
    Static prompt: fixed instruction text, no temporal reference or input
    statistics. Matches the paper's "P1: Static Prompt Only" baseline
    (Table 12), used for comparison against the dynamic prompt.
    """
    static_texts = {
        "forecast": "Predict future values based on past readings.",
        "impute": "Fill in the missing values based on past readings.",
        "classify": "Classify this time series based on its pattern.",
        "anomaly": "Detect anomalies in this time series.",
    }
    return static_texts.get(task, static_texts["forecast"])


def build_dynamic_prompt(
    dataset_name,
    task,
    x,
    feature_names,
    target="OT",
    pred_len=96,
    start_idx=0,
    end_idx=None,
    decimals=2,
):
    """
    Dynamic prompt combining context(X) + task(theta) + Statistics(X),
    per Eq. 5-6.

    Args:
        dataset_name: key into DATASET_CONTEXTS
        task: one of "forecast", "impute", "classify", "anomaly"
        x: (seq_len, num_features) input window, used to compute statistics
        feature_names: list of feature column names, in order
        target: target variable name (forecasting/imputation)
        pred_len: forecast horizon (forecasting only)
        start_idx, end_idx: temporal boundaries to reference in the prompt
        decimals: digit rounding precision for the tokenization workaround

    Returns:
        str, the full assembled prompt
    """
    if end_idx is None:
        end_idx = x.shape[0]

    context = DATASET_CONTEXTS.get(dataset_name, "This is a multivariate time series dataset.")

    task_template = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS["forecast"])
    task_text = task_template.format(
        start=start_idx, end=end_idx, target=target, pred_len=pred_len
    )

    stats_text = compute_input_statistics(x, feature_names=feature_names, decimals=decimals)

    return f"{context} {task_text} {stats_text}"


if __name__ == "__main__":
    # quick smoke test
    import torch

    feature_names = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    x = torch.randn(512, 7) * 5 + 20  # fake data resembling real-ish magnitudes

    static = build_static_prompt(task="forecast")
    dynamic = build_dynamic_prompt(
        dataset_name="ETTh1",
        task="forecast",
        x=x,
        feature_names=feature_names,
        target="OT",
        pred_len=96,
        start_idx=0,
        end_idx=512,
    )

    print("--- STATIC PROMPT ---")
    print(static)
    print()
    print("--- DYNAMIC PROMPT ---")
    print(dynamic)
    print()
    print("--- DIGIT TOKENIZATION EXAMPLE ---")
    print(f"27.787001 -> {format_number_for_prompt(27.787001)}")
    print(f"-3.14159  -> {format_number_for_prompt(-3.14159)}")
