import torch
import torch.nn as nn
import re
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class BiLSTM_Alt(nn.Module):
    def __init__(self, tag_to_ix, embedding_dim, hp_dict):
        super(BiLSTM_Alt, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hp_dict['hidden_dim']
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim // 2,
                            num_layers=hp_dict["num_layers"], bidirectional=True)
        
        self.hidden_to_tag = nn.Linear(self.hidden_dim, self.tagset_size)

        self.hidden = self.init_hidden(32)
        
    def init_hidden(self, batch_size):
        return (torch.randn(2 * self.lstm.num_layers, batch_size, self.hidden_dim // 2).to(device),
                torch.randn(2 * self.lstm.num_layers, batch_size, self.hidden_dim // 2).to(device))
        
    def forward(self, X, X_lengths):
        self.hidden = self.init_hidden(X.shape[0])

        batch_size, seq_len, _ = X.shape

        # ---------------------
        # Input is already embedding

        # ---------------------
        # 2. Run through RNN
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
        X = pack_padded_sequence(X, X_lengths, batch_first = True)

        X, self.hidden = self.lstm(X, self.hidden)

        X, _ = pad_packed_sequence(X, batch_first = True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden_to_tag(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size, seq_len, self.tagset_size)

        y_hat = X
        return y_hat

default_hp_dict = {"hidden_dim" : 40, "num_layers" : 1}
default_embedding_path = "word2vec_embedding.embed"
default_nn_path = "lstm_alt3_weights.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CitationTagger():
    def __init__(self, hp_dict = default_hp_dict, embedding_path = default_embedding_path,
        nn_path = default_nn_path):
        self.tag_to_ix = {"A": 0, "D": 1, "T": 2, "O": 3}
        self.ix_to_tag = {0: "A", 1: "D", 2: "T", 3: "O"}
        self.PAD_VALUE_LABEL = len(self.tag_to_ix)
        self.embedding_path = embedding_path
        self.nn_path = nn_path
        self.prepare_embedding()
        self.prepare_nn_model(hp_dict)

    def format_string(self, cit):
        '''
        Formats a specific string such that it fits our data.
        '''
        cit = re.sub(r'[,.;:\'\"]', '', cit)
        cit = re.sub(r'\(|[a-z]\)|\)', '', cit)
        cit = re.sub(r'\d\d\d\d(???|-)\d\d\d\d', 'dateRange', cit)
        cit = re.sub(r'\d\d\d\d', 'fourDigitNum', cit)
        cit = re.sub(r'[xiv]+(???|-)[xiv]+', 'numerals', cit)
        cit = re.sub(r'\d+(???|-)\d+', 'pageRange', cit)
        cit = re.sub(r'\d+', 'otherDigits', cit)
        cit = cit.lower()
        return cit.split()

    def prepare_embedding(self):
        '''
        Prepares the embedding that will be used for citation tagging.
        '''
        self.embedding_model = Word2Vec.load(self.embedding_path)

    def prepare_nn_model(self, hp_dict):
        '''
        Prepares the NN model that parses the citation tag.
        '''
        self.nn_model = BiLSTM_Alt(self.tag_to_ix, 100, hp_dict).to(device)
        self.nn_model.load_state_dict(torch.load(self.nn_path))

    def prepare_sequence(self, seq):
        '''
        Prepares a single sentence sequence to be analyzed.
        '''
        idxs = [torch.Tensor(self.embedding_model.wv[w]) if w in self.embedding_model.wv
        else torch.randn(self.embedding_model.wv[0].shape[0])
        for w in seq]
        return torch.vstack(idxs)

    def tag_citations(self, citations):
        '''
        Processes a list of citations by first preprocessing,
        then embedding them, then padding them,
        and finally tagging them.
        Returns a list of lists. Each list item in the big list is the
        tags for a single citation entry.
        '''
        # Preprocess
        citations = [self.format_string(c) for c in citations]
        # Embed
        citations = [self.prepare_sequence(c) for c in citations]
        # Pad
        citations = sorted(citations, key = lambda x: x.shape[0], reverse = True)
        lengths = torch.Tensor([x.shape[0] for x in citations])
        citations = torch.nn.utils.rnn.pad_sequence(citations, batch_first = True)
        # Tag
        self.nn_model.eval()
        with torch.no_grad():
            y_hat = self.nn_model(citations.to(device), lengths.cpu())
        predicted = y_hat.argmax(dim = 2)
        ret_list = []
        for idx, length in enumerate(lengths):
            ret_list.append([self.ix_to_tag[p.item()] for p in predicted[idx]])
        return ret_list

    def __call__(self, citations):
        '''
        Prints out the token-by-token citation as output by model.
        '''
        return self.print_token_and_tag(citations)

    def print_token_and_tag(self, citations):
        '''
        Prints out token-by-token citation as output by model.
        '''
        original_list = [c.split() for c in citations]
        tag_list = self.tag_citations(citations)
        for idx, (original, tag) in enumerate(zip(original_list, tag_list)):
            print(f"--- Citation #{idx + 1} ---")
            for token, output in zip(original, tag):
                print("%20s  ->  %s" % (token, output))
            print("-------------------")
            