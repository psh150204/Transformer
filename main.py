import os
import argparse

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
from model import Transformer

import numpy as np
import torch
import torch.nn.functional as F

def save_checkpoint(epoch, step_num, model, optimizer, path):
    state = {
        'epoch' : epoch,
        'step_num' : step_num,
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    
    torch.save(state, path)
    print('A check point has been generated : ' + path)

def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    # TODO: use these information.
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    num_enc = 6
    num_dec = 6
    d_model = 512
    d_k = 64
    d_v = 64
    d_ff = 2048
    h = 8

    # TODO: use these values to construct embedding layers
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    print(src_vocab_size)
    print(tgt_vocab_size)

    device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu")
    transformer = Transformer(6, 6, src_vocab_size, tgt_vocab_size, d_model, d_k, d_v, d_ff, h).to(device)
    
    checkpoint_path = 'checkpoints/final'
    if arg.test and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        transformer.load_state_dict(checkpoint['state_dict'])
        print("trained model is loaded from : " + checkpoint_path)

    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        # TODO: train
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        
        step_num = 1
        warmup_steps = 4000
        losses = []

        print('Start training ...')
        for epoch in range(100):
            start_epoch = time.time()
            i = 0

            for src_batch, tgt_batch in train_loader:
                start_batch = time.time()

                src_batch = torch.tensor(src_batch).to(device)
                trg_batch = torch.tensor(tgt_batch).to(device)
                
                trg_input = trg_batch[:,:-1]
                trg_output = trg_batch[:,1:].contiguous().view(-1)
                
                pred = transformer(src_batch, trg_input)

                # lr scheduling
                lr = (d_model ** -0.5) * min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
                for group in optimizer.param_groups:
                    group['lr'] = lr

                optimizer.zero_grad()

                loss = F.cross_entropy(pred.view(-1, pred.size(-1)), trg_output, ignore_index = 2)
                loss.backward()

                optimizer.step()
                step_num += 1

                i = i+1
                losses.append(loss.item())

                batch_time = time.time() - start_batch
                print('[%d/%d][%d/%d] train loss : %.4f | time : %.2fs'%(epoch+1, 100, i, train_loader.size//128 + 1, loss.item(), batch_time))
                
            i = 0
            # TODO: validation
            for src_batch, tgt_batch in valid_loader:
                src_batch = torch.tensor(src_batch).to(device)
                trg_batch = torch.tensor(tgt_batch).to(device)
                
                trg_input = trg_batch[:,:-1]
                trg_output = trg_batch[:,1:].contiguous().view(-1)
                
                pred = transformer(src_batch, trg_input)
                loss = F.cross_entropy(pred.view(-1, pred.size(-1)), trg_output, ignore_index = 2)

                i = i + 1
                print('[%d/%d][%d/%d] validation loss : %.4f'%(epoch+1, 100, i, valid_loader.size//128 + 1, loss.item()))
            
            epoch_time = time.time() - start_epoch
            print('Time taken for %d epoch : %.2fs'%(epoch+1, epoch_time))

            if (epoch+1) % 3 == 0:
                save_checkpoint(epoch, step_num, transformer, optimizer, 'checkpoints/epoch_%d'%(epoch+1))

        print('End of the training')
        save_checkpoint(epoch, step_num, transformer, optimizer, 'checkpoints/final')
    else:
        # test
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []
        for src_batch, tgt_batch in test_loader:
            # TODO: predict pred_batch from src_batch with your model.
            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]

            batch_size = len(tgt_batch)
            src_batch = torch.tensor(src_batch).to(device)
            pred_batch = torch.zeros(batch_size, 1, dtype = int).to(device) # [[0],[0],...,[0]]
            # eos_mask[i] = 1 means i-th sentence has eos
            eos_mask = torch.zeros(batch_size, dtype = int)

            for _ in range(max_length):
                output = transformer(src_batch, pred_batch) # batch_size * sentence_length * tgt_vocab_size
                output = torch.argmax(F.softmax(output, dim = -1), dim = -1) # batch_size * sentence_length
                predictions = output[:,-1].unsqueeze(1)
                
                pred_batch = torch.cat([pred_batch, predictions], dim = -1)

                for i in range(batch_size):
                    if predictions[i] == eos_idx:
                        eos_mask[i] = 1

                # every sentence has eos
                if eos_mask.sum() == batch_size :
                    break

            pred += seq2sen(pred_batch, tgt_vocab)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--test',
        action='store_true')
    args = parser.parse_args()

    main(args)
