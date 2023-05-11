import torch
import torchtext
from torchtext.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
import random
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import unicodedata
import datetime
import time
import copy
import spacy

ngpu = 1


device = torch.device("cuda:0")
print('device=', device)

data_df = pd.read_csv('./Datasets/eng-fra.txt',
                      encoding='UTF-8', sep='\t', header=None,
                      names=['source', 'target'], index_col=False)

# Data preprocessing

# Converts unicode strings to ASCII:
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# Normalizes the string
def normalizeString(s):
    s = s.lower().strip()
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)  # \1 means that group(1) is the first match that matches '.' or '! 'Or '? ', always replaced with 'space.' or' space! 'or' space? '
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # not letters and not.! ? Any other character is replaced with a space
    s = re.sub(r'[\s]+', " ", s)  # Replaces multiple Spaces that will appear with one space.
    return s

MAX_LENGTH = 10

pairs = [[normalizeString(s) for s in line] for line in data_df.values]

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [[pair[0], pair[1]] for pair in pairs if filterPair(pair)]

pairs = filterPairs(pairs)

#Partition data set: training set and validation set
train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=1234)

tokenizer = lambda x: x.split()
#tokenizer = spacy.load('en_core_web_sm')
SRC = torchtext.data.Field(
                                tokenize='spacy',
                                tokenizer_language='en_core_web_sm',
                                lower=True,
                                fix_length=MAX_LENGTH + 2,
                                preprocessing=lambda x: ['<start>'] + x + ['<end>'],
                                # after tokenizing but before numericalizing
                                # postprocessing # after numericalizing but before the numbers are turned into a Tensor
                                )
TRG = torchtext.data.Field(
                                 tokenize='spacy',
                                 tokenizer_language='de_core_news_sm',
                                 lower=True,
                                 fix_length=MAX_LENGTH + 2,
                                 preprocessing=lambda x: ['<start>'] + x + ['<end>'],
                                 )

def get_dataset(pairs, src, trg):
    fields = [('src', src), ('trg', trg)]  # filed information fields dict[str, Field])
    examples = []  # list(Example)
    for eng, ger in tqdm(pairs): # Progress bar
        examples.append(torchtext.data.Example.fromlist([eng, ger], fields))
    return examples, fields

ds_train = torchtext.data.Dataset(*get_dataset(train_pairs, SRC, TRG))
ds_val = torchtext.data.Dataset(*get_dataset(val_pairs, SRC, TRG))

#  Build vocabulary
SRC.build_vocab(ds_train)  # Create a glossary and establish the mapping relationship between token and ID
TRG.build_vocab(ds_train)

BATCH_SIZE = 512

# Build a data pipeline iterator
train_iter, val_iter = torchtext.data.Iterator.splits(
    (ds_train, ds_val),
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    batch_sizes=(BATCH_SIZE, BATCH_SIZE)
)

for batch in train_iter:

    print(batch.src[:,0])
    print(batch.src.shape, batch.trg.shape)
    break

# To organize the torch from data pipeline. Utils. Data. The DataLoader similar inputs and output form of the targets
class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)

    def __len__(self):
        return self.length

    def __iter__(self):
        for batch in self.data_iter:
            yield (torch.transpose(batch.src, 0, 1), torch.transpose(batch.trg, 0, 1))


train_dataloader = DataLoader(train_iter)
val_dataloader = DataLoader(val_iter)
# View data pipeline
for batch_src, batch_trg in train_dataloader:
    print(batch_src.shape, batch_trg.shape)
    print(batch_src[0], batch_src.dtype)
    print(batch_trg[0], batch_trg.dtype)
    break


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)

pos_encoding = positional_encoding(50, 512)

pad = 1
def create_padding_mask(seq):
    seq = torch.eq(seq, torch.tensor(pad)).float()
    return seq[:, np.newaxis, np.newaxis, :]

x = torch.tensor([[7, 6, 0, 0, 1],
                  [1, 2, 3, 0, 0],
                  [0, 0, 0, 4, 5]])

mask = create_padding_mask(x)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask

x = torch.rand(1,3)
mask = create_look_ahead_mask(x.shape[1])

def scaled_dot_product_attention(q, k, v, mask=None):
    """
        # Compute attention weights.
        q, k, v must have matching antecedent dimensions. and dq = dk
        k, v must have matching penultimate dimensions, e.g. seq_len_k = seq_len_v.
        #While masks have different shapes depending on their type (padding or look-ahead).
        #but the mask must be able to perform a broadcast transformation in order to sum.

        #Parameter.
            q: shape of the request == (... , seq_len_q, depth)
            k: shape of primary key == (... , seq_len_k, depth)
            v: shape of the value == (... , seq_len_v, depth_v) seq_len_k = seq_len_v
            mask: Float tensor whose shape can be converted to
                  (... , seq_len_q, seq_len_k). Default is None.

        # return value.
            #output, attention weights
        """
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))

    # Scaling matmul_qk
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    # Add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

        self.final_linear = torch.nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads,
                   self.depth)
        return x.transpose(1, 2)

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]

        q = self.wq(q)  # =>[b, seq_len, d_model]
        k = self.wq(k)  # =>[b, seq_len, d_model]
        v = self.wq(v)  # =>[b, seq_len, d_model]

        q = self.split_heads(q, batch_size)  # =>[b, num_head=8, seq_len, depth=64]
        k = self.split_heads(k, batch_size)  # =>[b, num_head=8, seq_len, depth=64]
        v = self.split_heads(v, batch_size)  # =>[b, num_head=8, seq_len, depth=64]

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # => [b, num_head=8, seq_len_q, depth=64]

        scaled_attention = scaled_attention.transpose(1, 2)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = self.final_linear(concat_attention)
        return output, attention_weights


temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.rand(1, 60, 512)
print(x.shape)
out, attn_weights = temp_mha(x, x, x, mask=None)
print(out.shape, attn_weights.shape)

# forward net
def point_wise_feed_forward_network(d_model, dff):
    feed_forward_net = torch.nn.Sequential(
        torch.nn.Linear(d_model, dff),
        torch.nn.ReLU(),
        torch.nn.Linear(dff, d_model),
    )
    return feed_forward_net

sample_ffn = point_wise_feed_forward_network(512, 2048)
input = torch.rand(64, 50, 512)
print(sample_ffn(input).shape)

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)  # Multi-headed attention (padding mask) (self-attention)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


sample_encoder_layer = EncoderLayer(512, 8, 2048)
x = torch.rand(64, 50, 512)
sample_encoder_layer_output = sample_encoder_layer(x, None)
print(sample_encoder_layer_output.shape)

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model,
                                       num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)


    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x,
                                               look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        # Q: receives the output from decoder's first attention block
        # K V: V (value) and K (key) receive the encoder output as inputs
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output,
                                               padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,  # Dimensions of the inner layer of a point feedforward network fn
                 input_vocab_size,  # Input word list size
                 maximun_position_encoding,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        self.pos_encoding = positional_encoding(maximun_position_encoding,
                                                d_model)

        self.enc_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, mask):
        inp_seq_len = x.shape[-1]

        # adding embedding and position encoding
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_encoding = self.pos_encoding[:, :inp_seq_len, :]
        pos_encoding = pos_encoding.cuda()
        x += pos_encoding

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        return x


class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximun_position_encoding,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=d_model)
        self.pos_encoding = positional_encoding(maximun_position_encoding,
                                                d_model)

        self.dec_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        targ_seq_len = x.shape[-1]

        attention_weights = {}

        # adding embedding and position encoding
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_encoding = self.pos_encoding[:, :targ_seq_len, :]
        pos_encoding = pos_encoding.cuda()
        x += pos_encoding

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attn_block1, attn_block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)


            attention_weights[f'decoder_layer{i + 1}_block1'] = attn_block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = attn_block2

        return x, attention_weights


class Transformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 pe_input,  # input max_pos_encoding
                 pe_target,  # input max_pos_encoding
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               input_vocab_size,
                               pe_input,
                               rate)
        self.decoder = Decoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               target_vocab_size,
                               pe_target,
                               rate)
        self.final_layer = torch.nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, targ, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)

        dec_output, attention_weights = self.decoder(targ, enc_output, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = len(SRC.vocab)
target_vocab_size = len(TRG.vocab)
dropout_rate = 0.1

print(input_vocab_size, target_vocab_size)
class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warm_steps=4):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warm_steps

        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        """
                # The rsqrt function is used to calculate the reciprocal of the square root of an element x.  That is = 1 / sqrt{x}
                arg1 = torch.rsqrt(torch.tensor(self._step_count, dtype=torch.float32))
                arg2 = torch.tensor(self._step_count * (self.warmup_steps ** -1.5), dtype=torch.float32)
                dynamic_lr = torch.rsqrt(self.d_model) * torch.minimum(arg1, arg2)
        """
        arg1 = self._step_count ** (-0.5)
        arg2 = self._step_count * (self.warmup_steps ** -1.5)
        dynamic_lr = (self.d_model ** (-0.5)) * min(arg1, arg2)
        return [dynamic_lr for group in self.optimizer.param_groups]


loss_object = torch.nn.CrossEntropyLoss(reduction='none')

def mask_loss_func(real, pred):
    _loss = loss_object(pred.transpose(-1, -2), real)

    mask = torch.logical_not(real.eq(pad)).type(_loss.dtype)
    _loss *= mask
    return _loss.sum() / mask.sum().item()



def mask_accuracy_func(real, pred):
    _pred = pred.argmax(dim=-1)
    corrects = _pred.eq(real)
    mask = torch.logical_not(real.eq(pad))
    corrects *= mask
    return corrects.sum().float() / mask.sum().item()


def create_mask(inp, targ):
    # encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # decoder's first attention block(self-attention)
    look_ahead_mask = create_look_ahead_mask(targ.shape[-1])  # =>[targ_seq_len,targ_seq_len]
    dec_targ_padding_mask = create_padding_mask(targ)
    combined_mask = torch.max(look_ahead_mask, dec_targ_padding_mask)

    dec_padding_mask = create_padding_mask(inp)

    return enc_padding_mask, combined_mask, dec_padding_mask

save_dir = './save/'

transformer = Transformer(num_layers,
                          d_model,
                          num_heads,
                          dff,
                          input_vocab_size,
                          target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

print(transformer)

transformer = transformer.to(device)
if ngpu > 1:
    transformer = torch.nn.DataParallel(transformer,  device_ids=list(range(ngpu)))

optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = CustomSchedule(optimizer, d_model, warm_steps=4000)

def train_step(model, inp, targ):
    # The target is split into tar_inp and tar_real
    # tar_inp is passed as input to the decoder.
    # tar_real is the same input shifted by 1: at each position in tar_inp, tar_real contains the next token that should be predicted.
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp)

    inp = inp.to(device)
    targ_inp = targ_inp.to(device)
    targ_real = targ_real.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    combined_mask = combined_mask.to(device)
    dec_padding_mask = dec_padding_mask.to(device)

    model.train()  # set train mode

    optimizer.zero_grad()  # Gradient clearing

    # forward
    prediction, _ = transformer(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)


    loss = mask_loss_func(targ_real, prediction)
    metric = mask_accuracy_func(targ_real, prediction)

    # backward
    loss.backward()  # Calculating gradients by back propagation
    optimizer.step()  # Update parameters

    return loss.item(), metric.item()


batch_src, batch_targ = next(iter(train_dataloader))
print(train_step(transformer, batch_src, batch_targ))

def validate_step(model, inp, targ):
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp)

    inp = inp.to(device)
    targ_inp = targ_inp.to(device)
    targ_real = targ_real.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    combined_mask = combined_mask.to(device)
    dec_padding_mask = dec_padding_mask.to(device)

    model.eval()

    with torch.no_grad():
        # forward
        prediction, _ = model(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)

        val_loss = mask_loss_func(targ_real, prediction)
        val_metric = mask_accuracy_func(targ_real, prediction)

    return val_loss.item(), val_metric.item()

EPOCHS = 60
print_trainstep_every = 50

metric_name = 'acc'
df_history = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val_' + metric_name])


def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)

def train_model(model, epochs, train_dataloader, val_dataloader, print_every):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_acc = 0.
    for epoch in range(1, epochs + 1):

        loss_sum = 0.
        metric_sum = 0.

        for step, (inp, targ) in enumerate(train_dataloader, start=1):
            # inp [64, 10] , targ [64, 10]
            loss, metric = train_step(model, inp, targ)

            loss_sum += loss
            metric_sum += metric

            # print log
            if step % print_every == 0:
                print('*' * 8, f'[step = {step}] loss: {loss_sum / step:.3f}, {metric_name}: {metric_sum / step:.3f}')

            lr_scheduler.step()

        # test(model, train_dataloader)
        val_loss_sum = 0.
        val_metric_sum = 0.
        for val_step, (inp, targ) in enumerate(val_dataloader, start=1):
            loss, metric = validate_step(model, inp, targ)

            val_loss_sum += loss
            val_metric_sum += metric

        # record = (epoch, loss_sum/step, metric_sum/step)
        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record

        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
            record[0], record[1], metric_name, record[2], record[3], metric_name, record[4]))
        printbar()

        current_acc_avg = val_metric_sum / val_step
        if current_acc_avg > best_acc:
            best_acc = current_acc_avg
            checkpoint = save_dir + 'ckpt.tar'.format(epoch, current_acc_avg)
            if device.type == 'cuda' and ngpu > 1:
                model_sd = copy.deepcopy(model.module.state_dict())
            else:
                model_sd = copy.deepcopy(model.state_dict())
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, checkpoint)


    print('finishing training...')
    endtime = time.time()
    time_elapsed = endtime - starttime
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return df_history

# train
df_history = train_model(transformer, EPOCHS, train_dataloader, val_dataloader, print_trainstep_every)
print(df_history)

# Plotting training curves
def plot_metric(df_history, metric):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]  #

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')  #

    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig( metric + '.png')
    plt.show()


#plot_metric(df_history, 'loss')
#plot_metric(df_history, metric_name)

# load model
checkpoint = './save/ckpt.tar'
print('checkpoint:', checkpoint)
ckpt = torch.load(checkpoint)
transformer_sd = ckpt['net']


reload_model = Transformer(num_layers,
                           d_model,
                           num_heads,
                           dff,
                           input_vocab_size,
                           target_vocab_size,
                           pe_input=input_vocab_size,
                           pe_target=target_vocab_size,
                           rate=dropout_rate)

reload_model = reload_model.to(device)
if ngpu > 1:
    reload_model = torch.nn.DataParallel(reload_model,  device_ids=list(range(ngpu)))


print('Loading model ...')
if device.type == 'cuda' and ngpu > 1:
   reload_model.module.load_state_dict(transformer_sd)
else:
   reload_model.load_state_dict(transformer_sd)
print('Model loaded ...')

def test(model, dataloader):

    test_loss_sum = 0.
    test_metric_sum = 0.
    for test_step, (inp, targ) in enumerate(dataloader, start=1):
        loss, metric = validate_step(model, inp, targ)

        test_loss_sum += loss
        test_metric_sum += metric
    print('*' * 8,
          'Test: loss: {:.3f}, {}: {:.3f}'.format(test_loss_sum / test_step, 'test_acc', test_metric_sum / test_step))

print('*' * 8, 'final test...')
test(reload_model, val_dataloader)

def tokenizer_encode(tokenize, sentence, vocab):
    sentence = normalizeString(sentence)
    sentence = tokenize(sentence)  # list
    sentence = ['<start>'] + sentence + ['<end>']
    sentence_ids = [vocab.stoi[token] for token in sentence]
    return sentence_ids


def tokenzier_decode(sentence_ids, vocab):
    sentence = [vocab.itos[id] for id in sentence_ids if id<len(vocab)]
    return " ".join(sentence)

s = 'je pars en vacances pour quelques jours .'


s_ids = [3, 5, 251, 17, 365, 35, 492, 390, 4, 2]
print(tokenzier_decode(s_ids, SRC.vocab))
print(tokenzier_decode(s_ids, TRG.vocab))

def evaluate(model, inp_sentence):
    model.eval()  # set eval mode

    inp_sentence_ids = tokenizer_encode(tokenizer, inp_sentence, SRC.vocab)
    encoder_input = torch.tensor(inp_sentence_ids).unsqueeze(dim=0)

    decoder_input = [TRG.vocab.stoi['<start>']]
    decoder_input = torch.tensor(decoder_input).unsqueeze(0)

    with torch.no_grad():
        for i in range(MAX_LENGTH + 2):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input.cpu(), decoder_input.cpu()) ################

            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            enc_padding_mask = enc_padding_mask.to(device)
            combined_mask = combined_mask.to(device)
            dec_padding_mask = dec_padding_mask.to(device)

            # forward
            predictions, attention_weights = model(encoder_input,
                                                   decoder_input,
                                                   enc_padding_mask,
                                                   combined_mask,
                                                   dec_padding_mask)

            # Look at the last word and calculate its argmax
            prediction = predictions[:, -1:, :]
            prediction_id = torch.argmax(prediction, dim=-1)
            if prediction_id.squeeze().item() == TRG.vocab.stoi['<end>']:
                return decoder_input.squeeze(dim=0), attention_weights

            decoder_input = torch.cat([decoder_input, prediction_id],
                                      dim=-1)

    return decoder_input.squeeze(dim=0), attention_weights


def plot_attention_weights(attention, sentence, pred_sentence, layer):
    sentence = sentence.split()
    pred_sentence = pred_sentence.split()

    fig = plt.figure(figsize=(16, 8))

    attention = torch.squeeze(attention[layer], dim=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        cax = ax.matshow(attention[head].cpu(), cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(pred_sentence)))

        ax.set_ylim(len(pred_sentence) - 1.5, -0.5)

        ax.set_xticklabels(['<start>']+sentence+['<end>'], fontdict=fontdict, rotation=90)
        ax.set_yticklabels(pred_sentence, fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()

# Batch translation
sentence_pairs1 = [
    ['i m taking a couple of days off .','Ich fahre für ein paar tage in den urlaub.'],
    ['i m not panicking .','Ich habe keine angst, hector .'],
    ['i am looking for an assistant .','Ich suche einen assistenten .'],
    ['i m a long way from home .','Ich bin weit weg Von zu hause .'],
    ['you re very late .','Du bist spät dran..'],
    ['i am thirsty .','Ich habe durst.'],
    ['i m crazy about you .','Ich bin verrückt nach dir.'],
    [ 'you are naughty .','Du bist so ungezogen.'],
    ['he s old and ugly .','Er ist alt und hässlich.'],
    ['i m terrified .','Ich habe zu viel angst.'],
]

sentence_pairs = [
    ['i m taking a couple of days off .','je pars en vacances pour quelques jours .'],
    ['i m not panicking .','je ne me panique pas .' ],
    ['i am looking for an assistant .','je recherche un assistant .' ],
    ['i m a long way from home .','je suis loin de chez moi .'],
    ['you re very late .','vous etes en retard .' ],
    ['i am thirsty .','j ai soif .'],
    ['i m crazy about you .','je suis fou de vous .'],
    ['you are naughty .','vous etes vilain .' ],
    ['he s old and ugly .','il est vieux et laid .'],
    ['i m terrified .','je suis terrifiee .' ],
]

def batch_translate(sentence_pairs):
    for pair in sentence_pairs:
        print('input:', pair[0])
        print('target:', pair[1])
        pred_result, _ = evaluate(reload_model, pair[0])
        pred_sentence = tokenzier_decode(pred_result, TRG.vocab)
        print('pred:', pred_sentence)
        print('')

batch_translate(sentence_pairs)

def translate(sentence_pair, plot=None):
    print('input:', sentence_pair[0])
    print('target:', sentence_pair[1])
    pred_result, attention_weights = evaluate(reload_model, sentence_pair[0])
    print('attention_weights:', attention_weights.keys())
    pred_sentence = tokenzier_decode(pred_result, TRG.vocab)
    print('pred:', pred_sentence)
    print('')

    if plot:
        plot_attention_weights(attention_weights, sentence_pair[0], pred_sentence, plot)


#translate(sentence_pairs[0], plot='decoder_layer4_block2')

#translate(sentence_pairs[2], plot='decoder_layer4_block2')
