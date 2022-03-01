#!/usr/bin/env python
# coding: utf-8



from data import generate_batches
from data import prepare_data
from data import data_to_index
from data import DEP_LABELS

from model.graph import Sintactic_GCN
from model.encoder import Encoder
from model.decoder import Decoder_luong

from BLEU import BLEU

from utils import time_since

import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from stanfordcorenlp import StanfordCoreNLP 

import numpy as np
import time

from validation import Evaluator



# In[2]:


USE_CUDA = False
MAX_LENGTH = 100

SPLIT_TRAIN = 0.7
SPLIT_VALID = 0.15
# The rest is for test

# Configure models
hidden_size_rnn = 512
hidden_size_graph = 512
emb_size=300
n_layers = 2
dropout = 0.1
batch_size = 50

# Configure training/optimization
clip = 10.0
learning_rate_graph = 0.0002
n_epochs = 20
print_every = 10
validate_loss_every = 50
validate_acc_every = 2 * validate_loss_every
tf_ratio = 0.5
best_bleu = 0

criterion = nn.NLLLoss()

# Training
def pass_batch_luong(batch_size, input_batches, target_batches, train=True, adj_arc_in=None, adj_arc_out=None, adj_lab_in=None, adj_lab_out=None, mask_in=None, mask_out=None, mask_loop=None,encoder_optimizer=None,decoder_optimizer=None,gcn1_optimizer=None,encoder=None,decoder=None,gcn1=None,input_lang=None,output_lang=None):
        
    hidden = encoder.init_hidden(batch_size)
    encoder_outputs, encoder_hidden = encoder(input_batches, hidden)
    decoder_input = Variable(torch.LongTensor([input_lang.vocab.stoi["<sos>"]] * batch_size))
    if gcn1:
        encoder_outputs = gcn1(encoder_outputs,
                            adj_arc_in, adj_arc_out,
                            adj_lab_in, adj_lab_out,
                            mask_in, mask_out,  
                            mask_loop)
    
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size)) 
    
    all_decoder_outputs = Variable(torch.zeros(target_batches.data.size()[0], batch_size, len(output_lang.vocab.itos)))

    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
    
    if train:
        use_teacher_forcing = np.random.random() < tf_ratio
    else:
        use_teacher_forcing = False
    
    if use_teacher_forcing:        
        # Use targets as inputs
        for di in range(target_batches.shape[0]):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                decoder_input.unsqueeze(0), decoder_context, decoder_hidden, encoder_outputs)
            
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_batches[di]
    else:        
        # Use decoder output as inputs
        for di in range(target_batches.shape[0]):            
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                decoder_input.unsqueeze(0), decoder_context, decoder_hidden, encoder_outputs) 
            
            all_decoder_outputs[di] = decoder_output
            
            # Greedy approach, take the word with highest probability
            topv, topi = decoder_output.data.topk(1)            
            decoder_input = Variable(torch.LongTensor(topi.cpu()).squeeze())
            if USE_CUDA: decoder_input = decoder_input.cuda()
        
    del decoder_output
    del decoder_hidden
        
    return all_decoder_outputs, target_batches


def train_luong(input_batches, target_batches, batch_size, train=True, adj_arc_in=None, adj_arc_out=None, adj_lab_in=None, adj_lab_out=None, mask_in=None, mask_out=None, mask_loop=None,encoder_optimizer=None,decoder_optimizer=None,gcn1_optimizer=None,encoder=None,decoder=None,gcn1=None,input_lang=None,output_lang=None):

# Zero gradients of both optimizers
    if train:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    loss = 0 # Added onto for each word
    all_decoder_outputs, target_batches = pass_batch_luong(batch_size, input_batches, target_batches, train, adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop,encoder_optimizer,decoder_optimizer,gcn1_optimizer,encoder,decoder,gcn1,input_lang,output_lang)
    
    # Loss calculation and backpropagation
    loss = criterion(all_decoder_outputs.view(-1, decoder.output_size), target_batches.contiguous().view(-1))
    
    if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        if gcn1:
            torch.nn.utils.clip_grad_norm_(gcn1.parameters(), clip)
            gcn1_optimizer.step()

    del all_decoder_outputs
    del target_batches
    
    return loss.item()


def main():
    input_lang, output_lang, pairs = prepare_data('ques', 'paraques', max_length=MAX_LENGTH)

    #splitting the pairs into train,val,test
    pairs_train = pairs[:int(len(pairs) * SPLIT_TRAIN)]
    pairs_valid = pairs[int(len(pairs) * SPLIT_TRAIN):int(len(pairs) * (SPLIT_TRAIN + SPLIT_VALID))]
    pairs_test = pairs[int(len(pairs) * (SPLIT_TRAIN + SPLIT_VALID)):]

    #Get the adjacency matrix for the pairs
    arr_dep_train=np.load('arr_dep_train.npy')
    arr_dep_valid=np.load('arr_dep_valid.npy')
    arr_dep_test=np.load('arr_dep_test.npy')

    # Converting words to index in pairs

    pairs_train = data_to_index(pairs_train, input_lang, output_lang)
    pairs_valid = data_to_index(pairs_valid, input_lang, output_lang)
    pairs_test = data_to_index(pairs_test, input_lang, output_lang)



    # Initialize models
    encoder = Encoder(len(input_lang.vocab.itos), hidden_size_rnn, emb_size, n_layers=n_layers, dropout=dropout, USE_CUDA=USE_CUDA)
    decoder = Decoder_luong('general', hidden_size_graph, len(output_lang.vocab.itos), 300, n_layers=2 * n_layers, dropout=dropout, USE_CUDA=USE_CUDA)
    gcn1 = Sintactic_GCN(hidden_size_rnn, hidden_size_graph, num_labels=len(DEP_LABELS))

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    gcn1_optimizer = optim.Adam(gcn1.parameters())#, learning_rate_graph)


    # Move models to GPU
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        gcn1 = gcn1.cuda()
        
    # Keep track of time elapsed and running averages
    start = time.time()
    train_losses = []
    validation_losses = []
    validation_bleu = []

    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every


    # In[ ]:


    for epoch in range(1, n_epochs): 
        # Shuffle data
        id_aux = np.random.permutation(np.arange(len(pairs_train)))
        pairs_train = pairs_train[id_aux]
        arr_dep_train = arr_dep_train[id_aux]
        


        # Get the batches for this epoch
        input_batches, target_batches = generate_batches(input_lang, output_lang, batch_size, pairs_train, return_dep_tree=True, arr_dep=arr_dep_train, max_degree=6, USE_CUDA=USE_CUDA)

        
    
        print_loss_total = 0
        
        for batch_ix, (input_batch, target_var) in enumerate(zip(input_batches, target_batches)):
        
            encoder.train()
            decoder.train()
            gcn1.train()
        
            [input_var, adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop] = input_batch
            # Run the train function
            loss = train_luong(input_var, target_var, input_var.size(1), 
                        True, adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop,encoder_optimizer,decoder_optimizer,gcn1_optimizer,encoder,decoder,gcn1,input_lang,output_lang)
                
            torch.cuda.empty_cache()

            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss

            if batch_ix == 0: continue

            if batch_ix % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
                train_losses.append(loss)

                print(f'{time_since(start, batch_ix / len(input_batches))} ({batch_ix} {batch_ix / len(input_batches) * 100:.2f}%) train_loss: {print_loss_avg:.4f}')
        
        input_batches, target_batches = generate_batches(input_lang, output_lang, batch_size, pairs_valid, return_dep_tree=True, arr_dep=arr_dep_valid, max_degree=6, USE_CUDA=USE_CUDA)
        print_loss_total = 0
        for input_batch, target_var in zip(input_batches, target_batches):
        
            encoder.eval()
            decoder.eval()
            gcn1.eval()
        
            [input_var, adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop] = input_batch
            # Run the train function
            loss = train_luong(input_var, target_var, input_var.size(1), 
                        False, adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop,encoder_optimizer,decoder_optimizer,gcn1_optimizer,encoder,decoder,gcn1,input_lang,output_lang)
            
            print_loss_total += loss
        val_loss = print_loss_total / len(input_batches)
        validation_losses.append(val_loss)
        # Evaluating Bleu
        evaluator = Evaluator(encoder, decoder, gcn1, None, input_lang, output_lang, MAX_LENGTH, True)
        candidates, references = evaluator.get_candidates_and_references(pairs_test, arr_dep_test, k_beams=1)
        bleu = BLEU(candidates, [references])
        if bleu[0] > best_bleu:
            best_bleu = bleu[0]
            torch.save(encoder.state_dict(), 'encoder_graph.pkl')
            torch.save(decoder.state_dict(), 'decoder_graph.pkl')
            torch.save(gcn1.state_dict(), 'gcn_graph.pkl')
        validation_bleu.append(bleu)
        print(f'val_loss: {val_loss:.4f} - bleu: {bleu}', end=' ')

        # Prevent overflow gpu memory
        del evaluator



if __name__ == "__main__":
    main()


