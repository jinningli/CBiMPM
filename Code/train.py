import argparse
import copy
import os
import torch
import time

from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from time import gmtime, strftime
from model.CBIMPM import CBIMPM
from model.CONV import CONV

from model.BIMPM import BIMPM
from model.utils import SNLI, Quora
from test import test

import codecs
import json

def train(args, data):
    if args.use_my_model:
        model = CBIMPM(args, data)
    elif args.use_only_conv:
        model = CONV(args, data)
    else:
        model = BIMPM(args, data)
    if args.gpu > -1:
        model.cuda(args.gpu)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_acc, max_test_acc = 0, 0

    iterator = data.train_iter
    savenow = False
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            savenow = True
            with codecs.open('saved_models/' + args.model_time + "/acc.txt", "a+", "utf-8") as output:
                output.write('\nEpoch: ' + str(present_epoch + 1))
            print('Epoch: ' + str(present_epoch + 1))
        last_epoch = present_epoch

        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s2 = getattr(batch, s1), getattr(batch, s2)

        # limit the lengths of input sentences up to max_sent_len
        if args.max_sent_len >= 0:
            if s1.size()[1] > args.max_sent_len:
                s1 = s1[:, :args.max_sent_len]
            if s2.size()[1] > args.max_sent_len:
                s2 = s2[:, :args.max_sent_len]

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

        optimizer.zero_grad()
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            dev_loss, dev_acc = test(model, args, data, mode='dev')
            test_loss, test_acc = test(model, args, data)
            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('acc/dev', dev_acc, c)
            writer.add_scalar('loss/test', test_loss, c)
            writer.add_scalar('acc/test', test_acc, c)

            print("[" + str(i) + "][loss] train: " + "{:.3f}".format(loss) + " dev: " + "{:.3f}".format(dev_loss) + " test: "+ "{:.3f}".format(test_loss) +
                  "\n[" + str(i) + "][acc]  dev: " + "{:.3f}".format(dev_acc) + " test: " + "{:.3f}".format(test_acc))

            with codecs.open('saved_models/' + args.model_time + "/acc.txt", "a+", "utf-8") as output:
                output.write("\n[" + str(i) + "][loss] train: " + "{:.3f}".format(loss) + " dev: " + "{:.3f}".format(dev_loss) + " test: "+ "{:.3f}".format(test_loss) +
                                  "\n[" + str(i) + "][acc]  dev: " + "{:.3f}".format(dev_acc) + " test: " + "{:.3f}".format(test_acc))

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)
                # with codecs.open('saved_models/' + args.model_time + "/best.json", "w+", "utf-8") as out:
                #     out.write(json.dumps(test_res))

            if savenow:
                print('Saving model...', present_epoch)
                torch.save(best_model.state_dict(), "saved_models/" + args.model_time + "/Epoch_" + str(present_epoch)
                           + "_" + "{:.5f}".format(max_test_acc) + "_" + str(args.model_time))
                savenow = False
            loss = 0
            model.train()

    writer.close()
    print("max dev acc: "+str(max_dev_acc) + " max test acc: "+str(max_test_acc))

    return best_model

def main():
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--max-sent-len', default=-1, type=int,
                        help='max length of input sentences model can accept, if -1, it accepts any length')
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--print-freq', default=500, type=int)
    parser.add_argument('--use-char-emb', default=False, action='store_true')
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--n_fm', default=50, type=int)
    parser.add_argument('--conv', default=True, type=bool)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--use_my_model', default=False, type=bool)
    parser.add_argument('--use_only_conv', default=False, type=bool)
    args = parser.parse_args()

    if args.data_type == 'SNLI':
        print('loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('loading Quora data...')
        data = Quora(args)
    else:
        raise NotImplementedError('only SNLI or Quora data is possible')

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))

    if not os.path.exists('saved_models/' + args.model_time):
        os.makedirs('saved_models/' + args.model_time)
        with codecs.open('saved_models/' + args.model_time + "/log.txt", "w+", "utf-8") as output:
            output.write(json.dumps(args.__dict__))
    start = time.time()
    print("Timer Start at " + str(start))
    print('training start!')
    best_model = train(args, data)
    end = time.time()
    print("Timer Stop at " + str(end) + "  Time cost: " + str(end - start))

    with codecs.open('saved_models/' + args.model_time + "/log.txt", "a+", "utf-8") as output:
        output.write("\nTime cost: " + str(end-start))

    torch.save(best_model.state_dict(), "saved_models/" + args.model_time + "/CBIMPM_" + str(args.data_type)+ "_" + str(args.model_time))

    print('training finished!')


if __name__ == '__main__':
    main()
