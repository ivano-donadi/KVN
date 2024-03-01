import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel
from lib.train.recorder import SmoothedValue

class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        network = DataParallel(network)
        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        if not recorder.loss_stats:
            prev_stats = None
        else:
            prev_stats = {}
            for k, v in recorder.loss_stats.items():
                if isinstance(v, SmoothedValue):
                    prev_stats.update({k: v.median})
                else:
                    prev_stats.update({k: v})

        self.network.module.update_loss_stats(prev_stats)

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1
            
            output, loss, loss_stats, image_stats = self.network(batch,epoch)
            self.network.module.update_loss_stats(loss_stats)
            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            #get_dot = output['gd']#tgt.register_hooks(loss)#
            loss.backward()
            #dot = get_dot()
            #dot.render('tmp',view=False)
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 1)
            optimizer.step()
            #quit()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader, total = data_size):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            with torch.no_grad():
                #output_L, loss, loss_stats, image_stats = self.network.module(batch_L, epoch)
                output, loss, loss_stats, image_stats = self.network.module(batch, epoch)
                if evaluator is not None:
                    evaluator.joint_evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
        
        return val_loss_stats, loss_state

