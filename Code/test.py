import argparse
import codecs

import torch
from torch import nn
from torch.autograd import Variable

from model.BIMPM import BIMPM
from model.CBIMPM import CBIMPM
from model.CONV import CONV

from model.utils import SNLI, Quora
import numpy as np
import json

def test_with_result(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    res = None

    for batch in iterator:
        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1:
                char_p = char_p.cuda(args.gpu)
                char_h = char_h.cuda(args.gpu)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        pred = model(**kwargs)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]

        _, pred = pred.max(dim=1)

        if res is None:
            res = pred.cpu().data.numpy()
        else:
            res = np.concatenate((res, pred.cpu().data.numpy()), axis=0)

        acc += (pred == batch.label).sum().float()
        size += len(pred)
    acc /= size
    acc = acc.cpu().data[0]
    return loss, acc, res


def test(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1:
                char_p = char_p.cuda(args.gpu)
                char_h = char_h.cuda(args.gpu)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        pred = model(**kwargs)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]

        _, pred = pred.max(dim=1)

        acc += (pred == batch.label).sum().float()
        size += len(pred)
    acc /= size
    acc = acc.cpu().data[0]
    return loss, acc

def load_model(args, data):
    if args.use_my_model:
        model = CBIMPM(args, data)
    elif args.use_only_conv:
        model = CONV(args, data)
    else:
        model = BIMPM(args, data)

    model.load_state_dict(torch.load(args.model_path))

    if args.gpu > -1:
        model.cuda(args.gpu)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--use-char-emb', default=False, action='store_true')
    parser.add_argument('--word-dim', default=300, type=int)

    parser.add_argument('--n_fm', default=50, type=int)
    parser.add_argument('--conv', default=True, type=bool)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--use_my_model', default=False, type=bool)
    parser.add_argument('--use_only_conv', default=False, type=bool)

    # these three are necessary
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', default="")
    parser.add_argument('--config', default="")

    args = parser.parse_args()

    argdic = json.load(open(args.config, "r"))
    setattr(args, 'n_fm', argdic['n_fm'])
    setattr(args, 'conv', argdic['conv'])
    setattr(args, 'kernel_size', argdic['kernel_size'])
    setattr(args, 'use_my_model', argdic['use_my_model'])
    setattr(args, 'use_only_conv', argdic['use_only_conv'])
    setattr(args, 'dropout', argdic['dropout'])
    setattr(args, 'epoch', argdic['epoch'])
    setattr(args, 'hidden_size', argdic['hidden_size'])
    setattr(args, 'learning_rate', argdic['learning_rate'])
    setattr(args, 'char_hidden_size', argdic['char_hidden_size'])
    setattr(args, 'batch_size', argdic['batch_size'])

    if args.data_type == 'SNLI':
        print('loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('loading Quora data...')
        data = Quora(args)

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)

    print('loading model...')
    model = load_model(args, data)

    _, acc, result = test_with_result(model, args, data)
    print(str(acc))

    np.savetxt(args.save_path, result, delimiter=",")

    print("test acc: "+str(acc))
