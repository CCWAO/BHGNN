from config import get_config
from torch_geometric.data import DataLoader
from models import *
from utils import *
import datetime
import time
import os
from sklearn.metrics import f1_score

now = int(time.time())
timeArray = time.localtime(now)
Time = time.strftime("%Y%m%d%H%M%S", timeArray)

config = get_config()
config.Time = Time
config.use_gpu = torch.cuda.is_available()
config.device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else "cpu")

path_save_log = config.save_dir + config.data_name + '/' + config.uncertainty + '/' + config.Time + '/'
if os.path.exists(path_save_log) == False:
    os.makedirs(path_save_log)

ex_log_file = path_save_log + 'log' + config.Time + '.txt'
log = open(ex_log_file, 'w')
print(ex_log_file)
print(ex_log_file, file=log)
print(config, file=log)
log.flush()

model_file_valid = path_save_log +'model_valid' + '.pkl'
model_file_test = path_save_log +'model_test' + '.pkl'

train_set = torch.load('/train.pt')
test_set = torch.load('/test.pt')

num_train = len(train_set)
num_test = len(test_set)

print('num_train', len(train_set))
print('num_test', len(test_set))

train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)

train_set_id = Batch.from_data_list(train_set).time_id
test_set_id = Batch.from_data_list(test_set).time_id

model_file = __import__('models')
model_cls_name = config.uncertainty
NET = getattr(model_file, model_cls_name)
print(NET)
model = NET(config)

if config.use_gpu:
    model = model.to(config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
critiren = un_loss(config)


def train(epoch, best_acc):
    start_time_epoch = time.time()
    model.train()
    train_loss = 0
    train_acc_classify = 0
    uncertainty_train = []
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data_t1, data_t2, data_t3 = get_train_3sample_batch(data, train_set_id, train_set)
        if config.use_gpu:
            data = data.to(config.device)
            data_t1, data_t2, data_t3 = data_t1.to(config.device), data_t2.to(config.device), data_t3.to(config.device)

        prediction, _, output = model(data)

        _, _, output1 = model(data_t1)
        _, _, output2 = model(data_t2)
        _, _, output3 = model(data_t3)
        time_output = (output1 + output2 + output3) / 3

        uncertainty_train_tem = model.test_un(data)
        uncertainty_train.append(uncertainty_train_tem)

        loss = critiren(prediction, data.y, output, time_output, uncertainty_train_tem['un'])

        prediction_mean = uncertainty_train_tem['mean']
        _, pred = prediction_mean.max(dim=1)
        acc = (pred == data.y).sum()
        train_loss += loss.detach().item()
        train_acc_classify += acc.item()
        
        loss.backward()
        optimizer.step()

    train_acc_classify = train_acc_classify / num_train

    test_loss, test_acc_classify, test_micro_f1, test_macro_f1, uncertainty_test = test(num_test,
        test_loader, test_set_id, test_set)

    if test_acc_classify > best_acc:
        torch.save(model.state_dict(), model_file_valid)
        print('best acc epoch so far, saving...', epoch + 1)

    epoch_time = time.time() - start_time_epoch
    still_time = epoch_time * (config.epochs - epoch)
    finish_time = datetime.datetime.now() + datetime.timedelta(seconds=int(still_time))
    finish_time_str = str(finish_time.strftime('%m%d-%H%M'))

    print('epoch{}, tr_Loss: {:.6f}, tr_Acc: {:.6f},'
          'te_loss: {:.6f}, te_mif1: {:.6f}, te_maf1: {:.6f}, finishi_time:{}'.format(
        epoch + 1,train_loss, train_acc_classify,
        test_loss,test_micro_f1, test_macro_f1, finish_time_str))
    print('epoch{}, tr_Loss: {:.6f}, tr_Acc: {:.6f},'
          'te_loss: {:.6f}, te_mif1: {:.6f}, te_maf1: {:.6f}'.format(
        epoch + 1, train_loss, train_acc_classify,
        test_loss, test_micro_f1, test_macro_f1), file=log)

    log.flush()
    
    torch.cuda.empty_cache()
    return test_micro_f1, test_macro_f1


def test(num_sample, data_loader, set_id, data_set):
    test_loss = 0
    acc_classify = 0
    preddd = torch.tensor([0])
    yy = torch.tensor([0])
    uncertainty_results = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            data_t1, data_t2, data_t3 = get_train_3sample_batch(data, set_id, data_set)
            if config.use_gpu:
                data = data.to(config.device)
                data_t1, data_t2, data_t3 = data_t1.to(config.device), data_t2.to(config.device), data_t3.to(config.device)

            prediction, _, output = model(data)
            _, _, output1 = model(data_t1)
            _, _, output2 = model(data_t2)
            _, _, output3 = model(data_t3)
                
            uncertainty_results_tem = model.test_un(data)
            uncertainty_results.append(uncertainty_results_tem)
            time_output = (output1 + output2 + output3) / 3
            loss = critiren(prediction, data.y, output, time_output, uncertainty_results_tem['un'])

            prediction_mean = uncertainty_results_tem['mean']
            _, pred = prediction_mean.max(dim=1)
            
            acc = (pred == data.y).sum()
            preddd = torch.cat([preddd, pred.cpu()])
            yy = torch.cat([yy, data.y.cpu()])
            test_loss += loss.detach().item()
            acc_classify += acc.item()

    acc_classify = acc_classify / num_sample

    preddd = preddd[1::]
    yy = yy[1::]
    micro_f1 = f1_score(preddd, yy, average='micro')
    macro_f1 = f1_score(preddd, yy, average='macro')
    
    torch.cuda.empty_cache()

    return test_loss, acc_classify, micro_f1, macro_f1, uncertainty_results


print('start training')
best_acc = 0
best_f1 = 0

lr = config.lr
patience = 100
for epoch in range(config.epochs):
    test_micro_f1, test_macro_f1 = train(epoch, best_acc)

    if test_micro_f1 > best_acc:
        best_acc = test_micro_f1
        best_f1 = test_macro_f1
        best_epoch_acc = epoch

print('best epoch mi-f1', best_acc)
print('best epoch ma-f1', best_f1)

now = int(time.time())
timeArray = time.localtime(now)
print(Time)
print(time.strftime("%Y%m%d_%H%M", timeArray))
print(time.strftime("%Y%m%d_%H%M", timeArray), file=log)
log.flush()



