from transformers import GPT2LMHeadModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import sys,logging
import torch
import json, os
from matplotlib import pyplot as plt
logging.root.handlers = []
logging.basicConfig(level="INFO", format = '%(asctime)s:%(levelname)s: %(message)s' ,stream = sys.stdout)
logger = logging.getLogger(__name__)
def check_memory():
    logger.info('GPU memory: %.1f M' % (torch.cuda.memory_allocated() // 1024 ** 2))

def check_max_memory():
    memory=torch.cuda.max_memory_allocated() // 1024 ** 2
    #logger.info('Max GPU memory: %.1f M' % (memory))
    return memory

def get_optimizers(model):
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": 0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=10000)
    return optimizer, scheduler

if __name__=='__main__':
    length=[]
    memory=[]
    if not os.path.exists('analysis/memory.json'):
        for sequence_len in range(40, 1024, 20):
            #sequence_len=1024
            batch_size=4
            device=0
            model = GPT2LMHeadModel.from_pretrained('distilgpt2')
            model.to(device)
            model.train()
            optimizer, scheduler=get_optimizers(model)
            #logger.info('Initial memory')
            #check_memory()
            x = torch.randint(low =100, high = 50000 , size = (batch_size, sequence_len)).to(device)
            outputs=model(x, labels=x, return_dict=True)
            #logger.info('Forward memory with batch size:{}, sequence length:{}'.format(batch_size, sequence_len))
            #check_memory()
            loss=outputs.loss
            loss.backward()
            #logger.info('Backward memory with batch size:{}, sequence length:{}'.format(batch_size, sequence_len))
            #check_memory()
            optimizer.step()
            scheduler.step()
            #logger.info('Optimize memory with batch size:{}, sequence length:{}'.format(batch_size, sequence_len))
            #check_memory()
            length.append(sequence_len)
            memory.append(check_max_memory())
            model.zero_grad()
            optimizer.zero_grad()
        json.dump({'length':length, 'memory':memory}, open('analysis/memory.json', 'w'))
    else:
        data=json.load(open('analysis/memory.json', 'r'))
        length=data['length']
        memory=data['memory']
    plt.plot(length, memory)
    id1, id2 = length.index(240), len(length)-1
    x, y=length[id1], memory[id1]
    plt.plot(x, y, marker='o', color='green')
    plt.annotate("SGA-DS-SL", xy=(x, y), xytext=(-80, 0), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
    plt.annotate("(%s,%s)"%(x, y), xy=(x, y), xytext=(10, -10), textcoords='offset points')
    x, y=length[id2], memory[id2]
    plt.plot(x, y, marker='o', color='green')
    plt.annotate("SimpleTOD and UBAR", xy=(x, y), xytext=(-150, 0), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
    plt.annotate("(%s,%s)"%(1024, y), xy=(x, y), xytext=(-55, -15), textcoords='offset points')
    plt.xlabel('Sequence length /tokens')
    plt.ylabel('Memory Cost /MB')
    plt.savefig('analysis/memory.png')
