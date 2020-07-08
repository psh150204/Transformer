import os
import argparse

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
from model import Transformer

import numpy as np

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

    # TODO: use these values to construct embedding layers
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    print(src_vocab_size)
    print(tgt_vocab_size)

    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        # TODO: train
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
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

                optimizer.zero_grad()

                loss = F.cross_entropy(pred.view(-1, pred.size(-1)), trg_output, ignore_index = 2)
                loss.backward()

                optimizer.step()

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

        print('End of the training')
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

            batch_size = tgt_batch.size(0)
            src_batch = torch.tensor(src_batch).to(device)
            pred_batch = torch.zeros(batch_size, 1).to(device) # [[0],[0],...,[0]]
            
            # eos_mask[i] = 1 means i-th sentence has eos
            eos_mask = torch.zeros(batch_size)

            for _ in range(max_length):
                output = transformer(src_batch, pred_batch) # batch_size * sentence_length * tgt_vocab_size
                output = torch.argmax(F.softmax(output, dim = -1), dim = -1) # batch_size * sentence_length
                pred_batch = torch.cat([pred_batch, output[:, -1]], dim = -1)

                for i in range(batch_size):
                    if output[:, -1][i] == eos_index:
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
