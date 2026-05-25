import torch
import gradio as gr
import plotly.graph_objects as go
from models.singleton import ModelManager
from models.loader import get_lm_head
from analysis.metrics import extract_metrics

manager = ModelManager()


def _layer_keys(payload: dict) -> list[str]:
    return [k for k in payload if not k.startswith("_")]


def plot_cosine_similarity(payload: dict) -> go.Figure:
    layers = _layer_keys(payload)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=[payload[layer]["cosine_sim"] for layer in layers],
        mode="lines+markers", name="Cosine Similarity"
    ))
    fig.update_layout(
        title="Cosine Similarity per Layer",
        xaxis_title="Layer", yaxis_title="Cosine Similarity",
        yaxis_range=[-1, 1], height=300, margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def plot_mse(payload: dict) -> go.Figure:
    layers = _layer_keys(payload)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layers, y=[payload[layer]["mse"] for layer in layers],
        name="MSE"
    ))
    fig.update_layout(
        title="Mean Squared Error per Layer",
        xaxis_title="Layer", yaxis_title="MSE",
        height=300, margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def plot_velocity(payload: dict) -> go.Figure:
    layers = _layer_keys(payload)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=[payload[layer]["model_a_velocity"] for layer in layers],
        mode="lines+markers", name="Model A Velocity"
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=[payload[layer]["model_b_velocity"] for layer in layers],
        mode="lines+markers", name="Model B Velocity"
    ))
    fig.update_layout(
        title="Layer Velocity (MSE between consecutive layers)",
        xaxis_title="Layer", yaxis_title="Velocity",
        height=300, margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def plot_l2_magnitudes(payload: dict) -> go.Figure:
    layers = _layer_keys(payload)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=[payload[layer]["model_a_l2"] for layer in layers],
        mode="lines+markers", name="Model A L2"
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=[payload[layer]["model_b_l2"] for layer in layers],
        mode="lines+markers", name="Model B L2"
    ))
    fig.update_layout(
        title="Mean L2 Magnitude per Layer",
        xaxis_title="Layer", yaxis_title="L2 Norm",
        height=300, margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def plot_sparsity(payload: dict) -> go.Figure:
    layers = _layer_keys(payload)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=[payload[layer]["model_a_sparsity"] for layer in layers],
        mode="lines+markers", name="Model A Sparsity"
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=[payload[layer]["model_b_sparsity"] for layer in layers],
        mode="lines+markers", name="Model B Sparsity"
    ))
    fig.update_layout(
        title="Activation Sparsity (Hoyer) per Layer",
        xaxis_title="Layer", yaxis_title="Sparsity",
        height=300, margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def plot_token_heatmap(payload: dict, tokens: list[str]) -> go.Figure | None:
    if "_per_token_cosine" not in payload:
        return None
    matrix = payload["_per_token_cosine"]
    n_layers = len(matrix)
    n_tokens = len(matrix[0]) if matrix else 0
    layer_labels = [str(i) for i in range(n_layers)]
    dtick = max(1, n_tokens // 20)
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=tokens,
        y=layer_labels,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        hovertemplate="Token: %{x}<br>Layer: %{y}<br>Cosine: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Per-Token Cosine Similarity Heatmap",
        xaxis_title="Token Position",
        xaxis=dict(tickmode="linear", tick0=0, dtick=dtick),
        yaxis_title="Layer",
        height=max(300, n_layers * 12),
        margin=dict(l=40, r=20, t=40, b=80),
    )
    return fig


def plot_logit_lens(payload: dict) -> go.Figure | None:
    if "_logit_lens_a" not in payload and "_logit_lens_b" not in payload:
        return None
    entries_a = payload.get("_logit_lens_a")
    entries_b = payload.get("_logit_lens_b")
    n_layers = len(entries_a or entries_b or [])
    layer_labels = [str(i) for i in range(n_layers)]

    def top5_text(entries: list[dict] | None) -> list[str]:
        if entries is None:
            return ["No LM head"] * n_layers
        return [
            "  |  ".join(
                f"{t['token']}({t['prob']:.1%})" for t in layer_entries
            )
            for layer_entries in entries
        ]

    text_a = top5_text(entries_a)
    text_b = top5_text(entries_b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1] * n_layers,
        y=layer_labels,
        text=text_a,
        textposition="middle right",
        mode="text",
        name="Model A",
    ))
    fig.add_trace(go.Scatter(
        x=[2] * n_layers,
        y=layer_labels,
        text=text_b,
        textposition="middle left",
        mode="text",
        name="Model B",
    ))
    fig.update_layout(
        title="Logit Lens — Top-5 Predicted Tokens per Layer",
        xaxis=dict(
            tickvals=[1, 2],
            ticktext=["Model A", "Model B"],
            range=[0.5, 2.5],
        ),
        yaxis_title="Layer",
        height=max(300, n_layers * 24),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


def run_pipeline(
    model_a_path: str,
    model_b_path: str,
    adapter_path: str,
    quantization: str,
    allow_mismatch: bool,
    prompt: str,
):
    try:
        plan = manager.load_if_needed(
            model_a_path=model_a_path,
            model_b_path=model_b_path or None,
            adapter_path=adapter_path or None,
            quantization=quantization,
            allow_mismatch=allow_mismatch,
        )
    except ValueError as e:
        raise gr.Error(str(e))

    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")

    inputs_a = plan.tokenizer_a(prompt, return_tensors="pt").to(
        plan.model_a.device
    )
    tokens_a = plan.tokenizer_a.convert_ids_to_tokens(inputs_a["input_ids"][0])
    num_tokens_a = inputs_a["input_ids"].size(1)

    try:
        if plan.mode == "single_model_adapter_swap":
            with torch.no_grad():
                with plan.model_a.disable_adapter():
                    outputs_base = plan.model_a(**inputs_a, output_hidden_states=True)
            with torch.no_grad():
                outputs_adapter = plan.model_a(**inputs_a, output_hidden_states=True)
            hidden_a = outputs_base.hidden_states
            hidden_b = outputs_adapter.hidden_states
            is_matched = True
            hidden_dim_b = plan.model_a.config.hidden_size
            del outputs_base, outputs_adapter
        else:
            inputs_b = plan.tokenizer_b(prompt, return_tensors="pt").to(
                plan.model_b.device
            )
            num_tokens_b = inputs_b["input_ids"].size(1)

            with torch.no_grad():
                outputs_a = plan.model_a(**inputs_a, output_hidden_states=True)
            with torch.no_grad():
                outputs_b = plan.model_b(**inputs_b, output_hidden_states=True)

            hidden_a = outputs_a.hidden_states
            hidden_b = outputs_b.hidden_states

            has_dim_mismatch = any(
                "Hidden Dimension Mismatch" in w for w in plan.warnings
            )
            is_matched = not has_dim_mismatch and num_tokens_a == num_tokens_b
            hidden_dim_b = plan.model_b.config.hidden_size
            del outputs_a, outputs_b

    except RuntimeError as e:
        raise gr.Error(f"Forward pass failed (OOM or CUDA error): {e}")

    hidden_dim_a = plan.model_a.config.hidden_size

    lm_head_a = get_lm_head(plan.model_a)
    lm_head_b = get_lm_head(plan.model_b) if plan.model_b else lm_head_a
    tok_b = plan.tokenizer_b if plan.tokenizer_b else plan.tokenizer_a

    payload = extract_metrics(
        hidden_a, hidden_b, is_matched, hidden_dim_a, hidden_dim_b,
        lm_head_a=lm_head_a,
        lm_head_b=lm_head_b,
        tokenizer_a=plan.tokenizer_a,
        tokenizer_b=tok_b,
    )
    del hidden_a, hidden_b

    cosine_fig = plot_cosine_similarity(payload) if is_matched else None
    mse_fig = plot_mse(payload) if is_matched else None
    velocity_fig = plot_velocity(payload)
    l2_fig = plot_l2_magnitudes(payload)
    sparsity_fig = plot_sparsity(payload)
    heatmap_fig = plot_token_heatmap(payload, tokens_a) if is_matched else None
    logit_fig = plot_logit_lens(payload)

    warning_text = (
        "⚠️ **Dimension Mismatch Detected:** "
        "Direct vector comparison disabled. Showing structural metrics only."
        if not is_matched
        else ""
    )

    return [
        cosine_fig,
        mse_fig,
        velocity_fig,
        l2_fig,
        sparsity_fig,
        heatmap_fig,
        logit_fig,
        gr.update(visible=not is_matched, value=warning_text),
        gr.update(visible=is_matched),
        gr.update(visible=is_matched),
    ]


with gr.Blocks(
    title="LLM Activation Analyzer",
) as demo:
    gr.Markdown("# LLM Activation Analyzer")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            model_a_path = gr.Textbox(
                label="Model A Path",
                placeholder="e.g. meta-llama/Meta-Llama-3-8B",
            )
            model_b_path = gr.Textbox(
                label="Model B Path (or leave empty for LoRA mode)",
                placeholder="e.g. meta-llama/Meta-Llama-3-8B-Instruct",
            )
            adapter_path = gr.Textbox(
                label="LoRA Adapter Path",
                placeholder="Path to LoRA adapter (optional)",
            )
            quantization = gr.Dropdown(
                label="Quantization",
                choices=["4-bit", "8-bit", "None"],
                value="4-bit",
            )
            allow_mismatch = gr.Checkbox(
                label="Authorize Mismatched Models",
                value=False,
            )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=4,
            )
            run_btn = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=2):
            mismatch_banner = gr.Markdown(visible=False)

            with gr.Tabs():
                with gr.Tab("Matched Vectors") as matched_tab:
                    with gr.Column() as matched_column:
                        cosine_plot = gr.Plot(label="Cosine Similarity")
                        mse_plot = gr.Plot(label="Mean Squared Error")

                with gr.Tab("Token Heatmap") as heatmap_tab:
                    with gr.Column() as heatmap_column:
                        heatmap_plot = gr.Plot(label="Token Activation Heatmap")

                with gr.Tab("Logit Lens") as logit_tab:
                    logit_plot = gr.Plot(label="Logit Lens")

                with gr.Tab("Structural Integrity") as structural_tab:
                    velocity_plot = gr.Plot(label="Layer Velocity")
                    l2_plot = gr.Plot(label="L2 Magnitudes")
                    sparsity_plot = gr.Plot(label="Activation Sparsity")

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            model_a_path,
            model_b_path,
            adapter_path,
            quantization,
            allow_mismatch,
            prompt,
        ],
        outputs=[
            cosine_plot,
            mse_plot,
            velocity_plot,
            l2_plot,
            sparsity_plot,
            heatmap_plot,
            logit_plot,
            mismatch_banner,
            matched_column,
            heatmap_column,
        ],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(theme=gr.themes.Soft())
