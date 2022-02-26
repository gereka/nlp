import torch

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("dropout")
class DropoutEncoder(Seq2SeqEncoder):
    """
    Copied from AllenNLP PassThroughEncoder.
    This class allows you to add dropout to your layers.
    Use with compose seq2seq encoder.
    """

    def __init__(self, input_dim: int, bidirectional: bool, dropout: float) -> None:
        super().__init__()
        self._input_dim = input_dim
        self.bidir = bidirectional
        self.dropout = torch.nn.Dropout(p=dropout)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def is_bidirectional(self):
        return self.bidir

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        # Parameters
        inputs : `torch.Tensor`, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, timesteps).
        # Returns
        A tensor of shape (batch_size, timesteps, output_dim),
        where output_dim = input_dim.
        """

        inputs = self.dropout(inputs)
        if mask is None:
            return inputs
        else:
            # We should mask out the output instead of the input.
            # But here, output = input, so we directly mask out the input.
            return inputs * mask.unsqueeze(dim=-1)
