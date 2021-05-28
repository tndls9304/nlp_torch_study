# https://github.com/HIT-SCIR/ELMoForManyLangs/blob/master/elmoformanylangs/modules/lstm_cell_with_projection.py
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

from ._encoder_base import _EncoderBase
from .bi_lstm_pre import BidirectionalLSTM
from .ci_embedding import ContextIndependentEmbedding
from .lstm_cell import LstmCellWithProjection


class BidirectionalLM(_EncoderBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ci_embedding = ContextIndependentEmbedding(config)
        self.bi_lstm = BidirectionalLSTM(config)

        self.output_dim = config['output_dim']
        self.fc = nn.Linear(config['elmo_embedding_size'], self.output_dim)

        self.forward_layers = []
        self.backward_layers = []
        lstm_input_size = config['emb_dim']
        for layer_idx in range(self.config['enc_n_layers']):
            forward_layer = LstmCellWithProjection(lstm_input_size,
                                                   config['emb_dim'],
                                                   config['enc_hid_dim'],
                                                   True,
                                                   config['rnn_dropout'],
                                                   config['cell_clip'],
                                                   config['proj_clip'])
            backward_layer = LstmCellWithProjection(lstm_input_size,
                                                    config['emb_dim'],
                                                    config['enc_hid_dim'],
                                                    False,
                                                    config['rnn_dropout'],
                                                    config['cell_clip'],
                                                    config['proj_clip'])
            lstm_input_size = config['emb_dim']

            self.add_module('forward_layer_{}'.format(layer_idx), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_idx), backward_layer)

            self.forward_layers.append(forward_layer)
            self.backward_layers.append(backward_layer)

    def forward(self, input_batch, mask):
        batch_size, seq_len = mask.size()
        input_batch = self.ci_embedding(input_batch)
        stacked_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._lstm_forward, input_batch, mask)

        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()

        if num_valid < batch_size:
            zeros = stacked_sequence_output.data.new(num_layers,
                                                     batch_size - num_valid,
                                                     returned_timesteps,
                                                     encoder_dim).fill_(0)
            zeros = Variable(zeros)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
                zeros = Variable(zeros)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = seq_len - returned_timesteps
        if sequence_length_difference > 0:
          zeros = stacked_sequence_output.data.new(num_layers,
                                                   batch_size,
                                                   sequence_length_difference,
                                                   stacked_sequence_output[0].size(-1)).fill_(0)
          zeros = Variable(zeros)
          stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        return stacked_sequence_output.index_select(1, restoration_indices)

    def _lstm_forward(self, packed_input, initial_state=None):
        """
        Parameters
        ----------
        packed_input : ``PackedSequence``, required.
          A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
          A tuple (state, memory) representing the initial hidden state and memory
          of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
          (num_layers, batch_size, 2 * cell_size) respectively.
        Returns
        -------
        output_sequence : ``torch.FloatTensor``
          The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
        final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
          The per-layer final (state, memory) states of the LSTM, with shape
          (num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
          respectively. The last dimension is duplicated because it contains the state/memory
          for both the forward and backward layers.
        """

        if initial_state is None:
            hidden_states: List[Optional[Tuple[torch.Tensor,
                                               torch.Tensor]]] = [None] * len(self.forward_layers)
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise Exception("Initial states were passed to forward() but the number of "
                            "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        inputs, batch_lengths = pad_packed_sequence(packed_input, batch_first=True)
        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                                   batch_lengths,
                                                                   forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence,
                                                                      batch_lengths,
                                                                      backward_state)
            # Skip connections, just adding the input to the output.
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence, backward_output_sequence], -1))
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1)))

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple = (torch.cat(final_hidden_states, 0), torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple
