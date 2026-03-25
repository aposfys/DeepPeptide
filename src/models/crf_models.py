'''
The CRF state space models. Many parameters are hardcoded due to the complexity of the CRF configuration.
'''
import torch
import torch.nn as nn
from .multi_tag_crf import CRF
from .lstm_cnn import LSTMCNN



class CRFBaseModel(nn.Module):
    '''Extend this model by defining a feature_extractor.'''
    def __init__(
        self,
        num_labels: int = 3, #logits (=emissions) to produce by the NN (0: Mature, 1: Propeptide Body, 2: Cleavage Site)
        num_states = 101 # total number of states in the state space model (0: Mature, 1-99: Propeptide Body, 100: Cleavage Site)
        ) -> None:


        super().__init__()
        self.max_len = 100 # V5: Expand ruler to 100 states to handle long-tail propeptides accurately
        self.min_len = 5
        self.feature_extractor = None
        self.features_to_emissions = nn.Linear(64, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(self.max_len, self.min_len)
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)

    @staticmethod
    def get_crf_constraints(max_len: int = 100, min_len: int = 5):
        '''Build the V5 Precision Propeptide Specialist state space model.
           States:
           0: Mature
           1 to (max_len - 1): Propeptide Body
           max_len: Cleavage Site (The final residue of a propeptide)
        '''
        allowed_starts = list(range(max_len + 1)) # Architectural Relaxation: Anchor Release
        allowed_ends = [0, max_len]

        allowed_state_transitions = []

        # RULE 1: Mature state
        allowed_state_transitions.append((0, 0)) # Stay in Mature
        allowed_state_transitions.append((0, 1)) # Start first Propeptide

        # RULE 2: The Ladder (Propeptide Body 1 to 99)
        for i in range(1, max_len):
            # A. Progress through the body: i -> i+1
            if i < max_len - 1:
                allowed_state_transitions.append((i, i + 1))

            # B. Identify Cleavage: i -> max_len (Jump to the cleavage site state)
            # A propeptide must be at least `min_len` residues long total.
            if i >= (min_len - 1):
                allowed_state_transitions.append((i, max_len))

        # RULE 3: Body Overflow Safety
        # If a propeptide is >100 residues, it loops at 99 until it's ready to cleave.
        allowed_state_transitions.append((max_len - 1, max_len - 1))

        # Ensure long-tail propeptides can still exit to the cleavage site
        if (max_len - 1, max_len) not in allowed_state_transitions:
            allowed_state_transitions.append((max_len - 1, max_len))

        # RULE 4: Cleavage Exits (max_len -> 0 or 1)
        # Once at the cleavage site, the sequence MUST exit the propeptide.
        allowed_state_transitions.append((max_len, 0)) # Standard exit to Mature
        allowed_state_transitions.append((max_len, 1)) # Multi-propeptide adjacent jump

        return allowed_state_transitions, allowed_starts, allowed_ends

    def _debug_crf(self, targets):
        '''Check label sequences for incompatibilities with the defined state grammar.'''
        for i in range(targets.shape[0]):

            for j in range(1, targets.shape[1]):
                l = int(targets[i,j].item())
                l_prev = int(targets[i,j-1].item())

                if (l_prev, l) not in self.allowed_transitions:
                    print(f'Found invalid transition from {l_prev} to {l}.')


    
    def _repeat_emissions(self, emissions):
        '''Turn a (batch_size, seq_len, 3) tensor into (batch_size, seq_len, num_states) for V5.'''

        if emissions.shape[-1] == 3:
            emissions_out = torch.zeros(emissions.shape[0], emissions.shape[1], self.num_states, dtype=emissions.dtype, device=emissions.device)
            emissions_out[:,:,0] = emissions[:,:,0] # Mature State
            emissions_out[:,:, 1:self.max_len] = emissions[:,:,1].unsqueeze(-1).expand(-1, -1, self.max_len - 1) # Propeptide Body
            emissions_out[:,:, self.max_len] = emissions[:,:,2] # Cleavage Site
        else:
            raise NotImplementedError()
        
        return emissions_out


    def forward(self, embeddings, mask, targets = None, skip_marginals: bool = False, top_k: int = 1):

        features = self.feature_extractor(embeddings, mask) # (batch_size, seq_len, feature_dim)
        emissions = self.features_to_emissions(features) # (batch_size, seq_len, num_labels)

        # Inverted Class Weighting: Bias State 1 and 2 to penalize false negatives.
        # Body (Class 1) happens ~50 times. Cleavage (Class 2) happens exactly 1 time.
        # Massive class imbalance! We apply an enormous bias (5.0) to Cleavage to force the
        # model to stop being conservative and aggressively identify exact cleavage sites.
        body_bias = 0.5
        cleavage_bias = 5.0

        if emissions.shape[-1] == 3:
            emissions[:, :, 1] = emissions[:, :, 1] + body_bias
            emissions[:, :, 2] = emissions[:, :, 2] + cleavage_bias

        emissions = self._repeat_emissions(emissions) # (batch_size, seq_len, num_states)
        
        # viterbi_paths = self.crf.decode(emissions=emissions, mask = mask.byte())

        viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask = mask.byte(), top_k=top_k)

        #pad the viterbi paths
        # max_pad_len = max([len(x) for x in viterbi_paths])
        # pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 
        # pos_preds = torch.tensor(pos_preds, device = emissions.device) #Tensor conversion is just for compatibility with downstream metric functions

        probs = self.crf.compute_marginal_probabilities(emissions, mask.byte()) if not skip_marginals else torch.softmax(emissions, dim=-1)

        if targets is not None:
            loss = self.crf(emissions = emissions, tags=targets.long(), mask = mask.byte(), reduction='mean') *-1

            if loss.item()>10000:
                self._debug_crf(targets)
            return (probs, viterbi_paths, loss)
        else:
            return probs, viterbi_paths, path_probs

    @staticmethod
    def _esm_embed(sequence:str, device: torch.device, repr_layers: int=33) -> torch.Tensor:

        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

        model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        model.eval()

        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            # ESM3 encoding and inference
            from esm.sdk.api import LogitsConfig

            output = model.logits(
                protein,
                LogitsConfig(sequence=True, return_embeddings=True)
            )

            # Access the aligned embeddings and cast to float32 for MacOS / MPS compatibility
            seq_embedding = output.embeddings
            if seq_embedding.dim() == 3:
                seq_embedding = seq_embedding.squeeze(0)

            # Slice BOS/EOS tokens to align perfectly with the generator's saved `.pt` shape
            if seq_embedding.shape[0] == len(sequence) + 2:
                seq_embedding = seq_embedding[1:-1, :]

            seq_embedding = seq_embedding.to(torch.float32).cpu()

        return seq_embedding

    def predict_from_sequence(self, sequence: str, top_k: int = 5):
        self.eval()
        with torch.no_grad():
            device =  next(self.parameters()).device
            embedding = self._esm_embed(sequence, device)
            embedding = torch.unsqueeze(embedding.permute(1,0), 0)
            mask = torch.unsqueeze(torch.ones(embedding.shape[2]),0)
        

            pos_probs, pos_preds, path_probs = self(embedding, mask, top_k=top_k)

            return pos_probs.squeeze(), pos_preds[0], path_probs[0]

    @staticmethod
    def _make_tag_bitmap(length, start, end, start_state=1, min_len=5, max_len=50):
        '''Make a multi-tag bitmap for the given peptide positions where all other positions are flexible.'''
        with torch.no_grad():
            label = torch.zeros((length, max_len*2+1))

            peptide_length = end-start +1 # inclusive.
            peptide_label = torch.concat(
                [ 
                torch.arange(start_state, start_state+min_len-2),#np.arange(1, 4), # from start to first position with skip connections
                # (end_state -1) - (peptide_length - min_len)
                torch.arange((start_state+max_len-2 - (peptide_length - min_len)), start_state+max_len) #np.arange( 59-(peptide_length-5) ,61) 
                ]
            )

            # set the positions in the matrix to true.
            label[torch.arange(start-1,end), peptide_label] = 1
            label[:start-1,:] = 1
            label[end:, :] = 1

        return label

    def predict_peptide_probability(self, sequence:str, start: int, stop: int):
        '''Computes probability of a peptide given all possible paths.'''
        self.eval()
        with torch.no_grad():
            device =  next(self.parameters()).device
            embedding = self._esm_embed(sequence, device)
            embedding = torch.unsqueeze(embedding.permute(1,0), 0)
            mask = torch.unsqueeze(torch.ones(embedding.shape[2]),0)
        
            features = self.feature_extractor(embedding, mask) # (batch_size, seq_len, feature_dim)
            emissions = self.features_to_emissions(features) # (batch_size, seq_len, num_labels)
            emissions = self._repeat_emissions(emissions) # (batch_size, seq_len, num_states)

            targets = self._make_tag_bitmap(len(sequence), start, stop, start_state=1)
            targets = torch.unsqueeze(targets,0)
            llh_pep= self.crf(emissions = emissions, tag_bitmap=targets.long(), mask = mask.byte(), reduction='none')
            
            targets = self._make_tag_bitmap(len(sequence), start, stop, start_state=51)
            targets = torch.unsqueeze(targets,0)
            llh_pro= self.crf(emissions = emissions, tag_bitmap=targets.long(), mask = mask.byte(), reduction='none')

            return torch.exp(llh_pep[0]).item(), torch.exp(llh_pro[0]).item()


class LSTMCNNCRF(CRFBaseModel):
    '''LSTM-CNN feature extractor + multistate CRF.'''
    def __init__(
        self,
        input_size: int = 1536,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int =3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers : int = 1,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61 # total number of states in the state space model
        ) -> None:


        super().__init__(num_labels, num_states)

        self.feature_extractor = LSTMCNN(input_size=input_size, dropout_input=dropout_input, n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_lstm_layers=1, dropout_conv1=dropout_conv1, n_tissues=0)
        self.features_to_emissions = nn.Linear(n_filters*2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(self.max_len, self.min_len)
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)



class SimpleLSTMCNNCRF(CRFBaseModel):
    '''LSTM-CNN feature extractor with simple 2-state CRF model.'''
    def __init__(
        self,
        input_size: int = 1536,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int =3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers : int = 1,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 2 # total number of states in the state space model
        ) -> None:


        super().__init__()

        self.feature_extractor = LSTMCNN(input_size=input_size, dropout_input=dropout_input, n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_lstm_layers=1, dropout_conv1=dropout_conv1, n_tissues=0)
        self.features_to_emissions = nn.Linear(n_filters*2, num_labels)
        self.num_states = num_states
        self.crf = CRF(num_states, batch_first=True) # no constraints on CRF.


    # redefine forward because no emission repeating.
    def forward(self, embeddings, mask, targets = None, skip_marginals: bool = False):
        features = self.feature_extractor(embeddings, mask) # (batch_size, seq_len, feature_dim)
        emissions = self.features_to_emissions(features) # (batch_size, seq_len, num_labels)
        
        viterbi_paths, probs = self.crf.decode(emissions=emissions, mask = mask.byte())

        #pad the viterbi paths
        # max_pad_len = max([len(x) for x in viterbi_paths])
        # pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 
        # pos_preds = torch.tensor(pos_preds, device = emissions.device) #Tensor conversion is just for compatibility with downstream metric functions

        probs = self.crf.compute_marginal_probabilities(emissions, mask.byte()) if not skip_marginals else torch.softmax(emissions, dim=-1)

        if targets is not None:
            loss = self.crf(emissions = emissions, tags=targets.long(), mask = mask.byte(), reduction='mean') *-1
            return (probs, viterbi_paths, loss)
        else:
            return probs, viterbi_paths



class SelfAttentionFeatureNet(nn.Module):

    def __init__(self,
        input_size: float = 1536,
        hidden_size: float = 640,
        dropout_input: float = 0.25,
        n_heads: int = 4,
        attn_dropout: float = 0.1, 
        ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout_input)

        if hidden_size != input_size:
            self.projection = nn.Linear(input_size, hidden_size)
        self.mha = nn.MultiheadAttention(hidden_size,n_heads, attn_dropout, batch_first=True)


    def forward(self, inputs, mask):

        inputs = inputs.transpose(2,1)
        inputs = self.dropout(inputs)
        attn_mask = 1 -mask
        # key_padding_mask – If specified, a mask of shape (N, S)
        # indicating which elements within key to ignore for the purpose of attention (i.e. treat as “padding”). 
        # For unbatched query, shape should be (S)(S). Binary and byte masks are supported. 
        # For a binary mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention. 
        # For a byte mask, a non-zero value indicates that the corresponding key value will be ignored
        #attn_mask = torch.
        if self.hidden_size != self.input_size:
            inputs = self.projection(inputs)

        out, attn = self.mha(inputs, inputs, inputs, key_padding_mask = attn_mask)

        return out

class SelfAttentionCRF(CRFBaseModel):
    '''Attention feature extractor + multistate CRF.'''
    def __init__(
        self,
        input_size: int = 1536,
        hidden_size: int = 128,
        dropout_input: float = 0.25,
        n_heads: int = 4,
        attn_dropout: float = 0.15,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61 # total number of states in the state space model
        ) -> None:


        super().__init__(num_labels, num_states)

        self.feature_extractor = SelfAttentionFeatureNet(input_size, hidden_size, dropout_input, n_heads, attn_dropout)

        self.features_to_emissions = nn.Linear(hidden_size, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(self.max_len, self.min_len)
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)