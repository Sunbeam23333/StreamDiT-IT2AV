"""Compatibility shim for transformers API changes.

In transformers >= 4.47, SequenceSummary was removed from modeling_utils.
This module provides a drop-in replacement.
"""

try:
    from transformers.modeling_utils import SequenceSummary  # noqa: F401
except ImportError:
    import torch.nn as nn
    from transformers.activations import get_activation

    class SequenceSummary(nn.Module):  # type: ignore[no-redef]
        """Compute a single vector summary of a sequence hidden states (compatible shim)."""

        def __init__(self, config):
            super().__init__()
            self.summary_type = getattr(config, "summary_type", "last")
            self.summary = nn.Identity()
            if hasattr(config, "summary_use_proj") and config.summary_use_proj:
                if (
                    hasattr(config, "summary_proj_to_labels")
                    and config.summary_proj_to_labels
                    and config.num_labels > 0
                ):
                    num_classes = config.num_labels
                else:
                    num_classes = config.hidden_size
                self.summary = nn.Linear(config.hidden_size, num_classes)
            activation_string = getattr(config, "summary_activation", None)
            self.activation = (
                get_activation(activation_string) if activation_string else nn.Identity()
            )
            self.first_dropout = nn.Identity()
            if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
                self.first_dropout = nn.Dropout(config.summary_first_dropout)
            self.last_dropout = nn.Identity()
            if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
                self.last_dropout = nn.Dropout(config.summary_last_dropout)

        def forward(self, hidden_states, cls_index=None):
            if self.summary_type == "last":
                output = hidden_states[:, -1]
            elif self.summary_type == "first":
                output = hidden_states[:, 0]
            elif self.summary_type == "mean":
                output = hidden_states.mean(dim=1)
            elif self.summary_type == "cls_index":
                if cls_index is None:
                    cls_index = hidden_states.new_full(
                        hidden_states.shape[:-1],
                        hidden_states.shape[1] - 1,
                        dtype=hidden_states.long(),
                    )
                cls_index = (
                    cls_index.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, -1, hidden_states.size(-1))
                )
                output = hidden_states.gather(-2, cls_index).squeeze(-2)
            else:
                output = hidden_states
            output = self.first_dropout(output)
            output = self.summary(output)
            output = self.activation(output)
            output = self.last_dropout(output)
            return output
